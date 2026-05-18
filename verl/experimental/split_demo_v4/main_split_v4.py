"""split demo v4 主入口 — 3-stage pipeline。

运行命令：
    torchrun --nproc_per_node=3 -m verl.experimental.split_demo_v4.main_split_v4
"""

from __future__ import annotations

import json
import os
import random
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.trainer.ppo.core_algos import agg_loss, compute_grpo_outcome_advantage
from verl.utils.torch_functional import logprobs_from_logits, masked_mean

from .core.pipeline_engine import SplitPipelineEngine
from .core.transport import SHUTDOWN
from .reward import FunctionReward
from .reward.function_reward import math_exact_match


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl_samples(path):
    file_path = Path(to_absolute_path(os.path.expanduser(path)))
    if not file_path.exists():
        raise FileNotFoundError(f"Training file not found: {file_path}")
    samples = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "prompt" not in item or "answer" not in item:
                raise ValueError("Missing required keys 'prompt'/'answer'.")
            samples.append(item)
    if not samples:
        raise ValueError(f"No valid samples found in {file_path}")
    return samples


class SimplePipelineRollout:
    """3-stage pipeline 的 rollout 包装。"""

    def __init__(self, engine, tokenizer, temperature=1.0, top_p=1.0):
        self.engine = engine
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def _sample_next_token(self, logits):
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)
        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > self.top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled)
        return torch.multinomial(probs, num_samples=1)

    def _pipeline_forward(self, input_ids, attention_mask):
        """通过 3-stage pipeline 做一次完整前向。"""
        if self.engine.is_head:
            h, pos_ids, pos_emb, mask = self.engine.stage.forward_input(input_ids, attention_mask)
            # 发给 Middle
            B, S, H = h.shape
            for val in [0, B, S, H, 0]:  # FWD_ONLY flag
                self.engine.transport.send_int(val)
            self.engine.transport.send(h)
            # 等 Tail 的 logits（Middle 会转发）
            result_shape = torch.zeros(2, dtype=torch.int64, device="cuda")
            dist.recv(result_shape, src=2)
            logits = torch.empty(result_shape[0].item(), result_shape[1].item(), 151936,
                                dtype=torch.bfloat16, device="cuda")
            dist.recv(logits, src=2)
            return logits

        elif self.engine.is_middle:
            self.engine.stage.run()  # 不会返回
            return None

        elif self.engine.is_tail:
            # 从 Head 收 h（通过 Middle 转发，但 Middle 在 run 循环中直接转发）
            # 实际上 Tail 直接收 Middle 转发的数据
            B = self.engine.transport.recv_int()
            S = self.engine.transport.recv_int()
            H = self.engine.transport.recv_int()
            dtype_code = self.engine.transport.recv_int()
            tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
            h_middle = self.engine.transport.recv((B, S, H), tensor_dtype)
            with torch.no_grad():
                logits = self.engine.stage.forward_output(h_middle)
            # 发给 Head
            dist.send(torch.tensor([logits.shape[0], logits.shape[1]], dtype=torch.int64, device="cuda"), dst=0)
            dist.send(logits, dst=0)
            return logits

    def generate(self, prompt_ids, group_size, max_new_tokens, attention_mask=None):
        device = torch.device("cuda")
        if self.engine.is_middle:
            self.engine.stage.run()
            return None

        prompt_ids = prompt_ids.to(device)
        batch_size, prompt_len = prompt_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        prompt_ids = prompt_ids.unsqueeze(1).repeat(1, group_size, 1).reshape(batch_size * group_size, prompt_len)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, group_size, 1).reshape(batch_size * group_size, prompt_len)

        generated_tokens = []
        generated_masks = []
        finished = torch.zeros(prompt_ids.size(0), dtype=torch.bool, device=device)

        prefill_logits = self._pipeline_forward(prompt_ids, attention_mask)
        next_token = self._sample_next_token(prefill_logits[:, -1, :])
        current_valid = (~finished).float().unsqueeze(-1)
        generated_tokens.append(next_token)
        generated_masks.append(current_valid)
        finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
        current_ids = torch.cat([prompt_ids, next_token], dim=-1)
        current_mask = torch.cat([attention_mask, current_valid.long()], dim=-1)

        for _ in range(max_new_tokens - 1):
            next_logits = self._pipeline_forward(current_ids, current_mask)
            sampled = self._sample_next_token(next_logits[:, -1, :])
            sampled = torch.where(finished.unsqueeze(-1), torch.full_like(sampled, self.pad_token_id), sampled)
            current_valid = (~finished).float().unsqueeze(-1)
            generated_tokens.append(sampled)
            generated_masks.append(current_valid)
            finished = finished | (sampled.squeeze(-1) == self.eos_token_id)
            current_ids = torch.cat([current_ids, sampled], dim=-1)
            current_mask = torch.cat([current_mask, current_valid.long()], dim=-1)
            if bool(torch.all(finished)):
                break

        response_ids = torch.cat(generated_tokens, dim=-1)
        response_mask = torch.cat(generated_masks, dim=-1).long()
        sequences = torch.cat([prompt_ids, response_ids], dim=-1)
        full_mask = torch.cat([attention_mask, response_mask], dim=-1)

        class Out: pass
        out = Out()
        out.sequences = sequences
        out.attention_mask = full_mask
        out.response_ids = response_ids
        out.response_mask = response_mask
        out.prompt_len = prompt_len
        return out


@hydra.main(config_path="config", config_name="split_demo_v4", version_base=None)
def main(config: DictConfig):
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    engine = SplitPipelineEngine()
    engine.model_path = str(config.model.path)
    engine.front_end = int(config.split.front_end)
    engine.middle_end = int(config.split.middle_end)
    engine.lr = float(config.trainer.lr)
    engine.clip_grad = float(config.trainer.max_grad_norm)
    engine.initialize()

    # rank1 (Middle): 进入请求响应循环
    if rank == 1:
        try:
            engine.stage.run()
        finally:
            dist.destroy_process_group()
        return

    # rank0 (Head) + rank2 (Tail): 训练逻辑
    set_seed(int(config.trainer.seed))
    tokenizer = engine.tokenizer
    samples = load_jsonl_samples(config.data.train_file)

    total_steps = int(config.trainer.total_steps)
    batch_size = int(config.trainer.train_batch_size)
    group_size = int(config.rollout.group_size)
    max_new_tokens = int(config.rollout.max_new_tokens)
    mini_batch_size = int(config.algorithm.mini_batch_size)
    update_epochs = int(config.algorithm.update_epochs)
    loss_scale_factor = int(config.algorithm.loss_scale_factor)
    log_freq = int(config.trainer.log_freq)

    rng = random.Random(int(config.trainer.seed))

    for global_step in range(1, total_steps + 1):
        picked = [samples[rng.randrange(len(samples))] for _ in range(batch_size)]
        prompts = [item["prompt"] for item in picked]
        answers = [str(item["answer"]) for item in picked]

        encoded = tokenizer(prompts, padding=True, truncation=True,
                          max_length=int(config.data.max_prompt_length), return_tensors="pt")

        # rollout
        if engine.is_head:
            # Head 做 rollout 生成
            # 每次生成需要 Head → Middle → Tail → Head 的完整往返
            rollout = SimplePipelineRollout(engine, tokenizer)
            rollout_output = rollout.generate(encoded["input_ids"], group_size, max_new_tokens, encoded["attention_mask"])
            # 广播 rollout 结果给 Tail
            if rollout_output is not None:
                for name in ["sequences", "attention_mask", "response_ids", "response_mask"]:
                    tensor = getattr(rollout_output, name)
                    dist.broadcast(tensor, src=0)
                dist.broadcast(torch.tensor([rollout_output.prompt_len], dtype=torch.int64, device="cuda"), src=0)
        elif engine.is_tail:
            # Tail 等 Head 的 rollout 结果
            # 先让 Middle 启动
            pass  # Middle 已经在 run() 中

    # 简化：v4 第一步先验证 3-stage 通信，不跑完整 GRPO
    print(f"[rank{rank}] v4 3-stage pipeline engine initialized", flush=True)

    if engine.is_head:
        engine.transport.send_int(SHUTDOWN)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
