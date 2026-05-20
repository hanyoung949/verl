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
from .core.transport import PIPELINE_DONE, SHUTDOWN
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
    """3-stage pipeline 的 rollout 包装。

    通信路径:
        Head → Middle → Tail（前向）
        Tail → Head（logits 直接回传）
    """

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
        """Head→Middle→Tail 前向，Tail→Head 直接回传 logits。

        协议（与 v3 对齐）:
            Head → Middle: header(6-int tensor) + h + pos_ids + [mask]
            Middle → Tail: header + h + pos_ids（原样转发）
            Tail → Head:   logits 直接 dist.send
        """
        if self.engine.is_head:
            h, pos_ids, pos_emb, mask = self.engine.stage.forward_input(input_ids, attention_mask)
            B, S, H = h.shape
            has_mask = 0 if mask is None else 1
            header = torch.tensor([0, B, S, H, has_mask, 0], dtype=torch.int64, device="cuda")
            self.engine.transport.send(header)
            self.engine.transport.send(h)
            self.engine.transport.send(pos_ids)
            if mask is not None:
                self.engine.transport.send(mask)
            # 从 Tail 直接收 logits
            result_shape = torch.zeros(2, dtype=torch.int64, device="cuda")
            dist.recv(result_shape, src=2)
            logits = torch.empty(result_shape[0].item(), result_shape[1].item(), 151936,
                                dtype=torch.bfloat16, device="cuda")
            dist.recv(logits, src=2)
            return logits

        elif self.engine.is_middle:
            self.engine.stage.run()
            return None

        elif self.engine.is_tail:
            # 从 Middle 接收转发的 header
            header = torch.zeros(6, dtype=torch.int64, device="cuda")
            dist.recv(header, src=1)
            _flag, _B, S, H, _has_mask, dtype_code = [int(x) for x in header.tolist()]
            tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
            h = self.engine.transport.recv((_B, S, H), tensor_dtype)
            pos_ids = self.engine.transport.recv((_B, S), torch.int64)
            # Tail 自己的 rotary_emb 重算 position_embeddings
            pos_emb = None
            if self.engine.stage.rotary_emb is not None:
                try:
                    pos_emb = self.engine.stage.rotary_emb(h.to("cuda"), pos_ids.to("cuda"))
                except TypeError:
                    pos_emb = self.engine.stage.rotary_emb(h.to("cuda"), seq_len=S)
            with torch.no_grad():
                logits = self.engine.stage.forward_output(h, position_ids=pos_ids, position_embeddings=pos_emb)
            # 直接发给 Head（不经过 Middle）
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


def make_grpo_loss_fn(clip_low, clip_high, loss_agg_mode, loss_scale_factor):
    """GRPO loss 函数，传给 engine.train_batch()。"""
    def grpo_loss_fn(logits, data):
        prompt_len = logits.shape[1] - data["response_ids"].shape[1]
        log_prob_new = logprobs_from_logits(
            logits[:, prompt_len - 1:-1, :], data["response_ids"])
        neg_kl = torch.clamp(log_prob_new - data["old_log_probs"], min=-20.0, max=20.0)
        ratio = neg_kl.exp()
        adv = data["advantages"]
        pg_losses = torch.maximum(
            -adv * ratio,
            -adv * ratio.clamp(1.0 - clip_low, 1.0 + clip_high),
        )
        loss = agg_loss(
            loss_mat=pg_losses, loss_mask=data["response_mask"],
            loss_agg_mode=loss_agg_mode, loss_scale_factor=loss_scale_factor)
        approx_kl = masked_mean(-neg_kl, data["response_mask"])
        clipped = (ratio.detach() < 1.0 - clip_low) | (ratio.detach() > 1.0 + clip_high)
        clipfrac = masked_mean(clipped.float(), data["response_mask"])
        return {
            "loss": loss,
            "metrics": {"approx_kl": approx_kl.detach().item(), "clipfrac": clipfrac.detach().item()},
        }
    return grpo_loss_fn


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

    reward_fn = FunctionReward(fn=math_exact_match, tokenizer=tokenizer)

    rng = random.Random(int(config.trainer.seed))

    for global_step in range(1, total_steps + 1):
        picked = [samples[rng.randrange(len(samples))] for _ in range(batch_size)]
        prompts = [item["prompt"] for item in picked]
        answers = [str(item["answer"]) for item in picked]

        encoded = tokenizer(prompts, padding=True, truncation=True,
                          max_length=int(config.data.max_prompt_length), return_tensors="pt")

        # ===== Phase 1: Pipeline forward (rollout) =====
        if engine.is_head:
            rollout = SimplePipelineRollout(engine, tokenizer)
            rollout_output = rollout.generate(
                encoded["input_ids"], group_size, max_new_tokens, encoded["attention_mask"])
            header = torch.tensor([PIPELINE_DONE, 0, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
            engine.transport.send(header)

        elif engine.is_tail:
            while True:
                header = torch.zeros(6, dtype=torch.int64, device="cuda")
                dist.recv(header, src=1)
                flag = int(header[0].item())
                if flag == PIPELINE_DONE:
                    break
                _B, S, H_dim, _has_mask, dtype_code = [int(x) for x in header[1:].tolist()]
                tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
                h = engine.transport.recv((_B, S, H_dim), tensor_dtype)
                pos_ids = engine.transport.recv((_B, S), torch.int64)
                pos_emb = None
                if engine.stage.rotary_emb is not None:
                    try:
                        pos_emb = engine.stage.rotary_emb(h.to("cuda"), pos_ids.to("cuda"))
                    except TypeError:
                        pos_emb = engine.stage.rotary_emb(h.to("cuda"), seq_len=S)
                with torch.no_grad():
                    logits = engine.stage.forward_output(h, position_ids=pos_ids, position_embeddings=pos_emb)
                dist.send(torch.tensor([logits.shape[0], logits.shape[1]], dtype=torch.int64, device="cuda"), dst=0)
                dist.send(logits, dst=0)

        # ===== Phase 2: Sync + Reward + old_log_prob + Advantage + Train =====
        if engine.is_head and rollout_output is not None:
            # 2a: Sync rollout results to Tail
            for name in ["sequences", "attention_mask", "response_ids", "response_mask"]:
                tensor = getattr(rollout_output, name)
                shape_t = torch.tensor(list(tensor.shape), dtype=torch.int64, device="cuda")
                dist.send(shape_t, dst=2)
                dist.send(tensor.contiguous(), dst=2)
            dist.send(torch.tensor([rollout_output.prompt_len], dtype=torch.int64, device="cuda"), dst=2)

            # 2b: Reward
            repeated_answers = [a for a in answers for _ in range(group_size)]
            rewards = reward_fn.score(
                rollout_output.sequences, rollout_output.attention_mask,
                {"answers": repeated_answers, "prompt_len": rollout_output.prompt_len})
            resp_texts = tokenizer.batch_decode(rollout_output.response_ids[:4], skip_special_tokens=True)
            print(f"[step {global_step}] reward={rewards.mean().item():.4f} samples={resp_texts}", flush=True)

            # 2c: old_log_prob — re-run forward on full sequences
            logits_full = rollout._pipeline_forward(
                rollout_output.sequences, rollout_output.attention_mask)
            old_log_prob = logprobs_from_logits(
                logits_full[:, rollout_output.prompt_len - 1:-1, :],
                rollout_output.response_ids)
            # signal Tail end of old_log_prob forward
            header = torch.tensor([PIPELINE_DONE, 0, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
            engine.transport.send(header)
            print(f"[step {global_step}] old_log_prob mean={old_log_prob.mean().item():.4f}", flush=True)

            # 2d: Advantage
            token_level_scores = torch.zeros_like(rollout_output.response_mask, dtype=torch.float32)
            last_valid = rollout_output.response_mask.sum(dim=1).long() - 1
            token_level_scores[torch.arange(token_level_scores.size(0), device="cuda"), last_valid] = rewards
            rewards_grouped = rewards.view(batch_size, group_size)
            if bool(config.algorithm.dynamic_sampling):
                valid_groups = rewards_grouped.std(dim=1) > 1e-6
            else:
                valid_groups = torch.ones(batch_size, dtype=torch.bool, device="cuda")
            if int(valid_groups.sum().item()) == 0:
                print(f"[step {global_step}] all groups filtered, skip", flush=True)
                continue
            valid_seq = valid_groups.unsqueeze(1).expand(batch_size, group_size).reshape(-1)
            uid_array = [f"g{pi}" for pi in range(batch_size) for _ in range(group_size)]
            uid_valid = np.array(uid_array, dtype=object)[valid_seq.detach().cpu().numpy()]
            advantages_valid, _ = compute_grpo_outcome_advantage(
                token_level_rewards=token_level_scores[valid_seq],
                response_mask=rollout_output.response_mask[valid_seq],
                index=uid_valid,
                norm_adv_by_std_in_grpo=bool(config.algorithm.norm_adv_by_std_in_grpo))
            print(f"[step {global_step}] advantage mean={advantages_valid.mean().item():.4f} "
                  f"valid={int(valid_groups.sum().item())}/{batch_size}", flush=True)

            # 2e: Train — 先用 forward_only 验证通路，后续补 backward
            grpo_loss_fn = make_grpo_loss_fn(
                clip_low=float(config.algorithm.cliprange_low),
                clip_high=float(config.algorithm.cliprange_high),
                loss_agg_mode=str(config.algorithm.loss_agg_mode),
                loss_scale_factor=loss_scale_factor)
            train_data = TensorDict({
                "input_ids": rollout_output.sequences[valid_seq],
                "attention_mask": rollout_output.attention_mask[valid_seq],
                "response_ids": rollout_output.response_ids[valid_seq],
                "response_mask": rollout_output.response_mask[valid_seq],
                "old_log_probs": old_log_prob[valid_seq],
                "advantages": advantages_valid,
            }, batch_size=[valid_seq.sum().item()])
            # 用 infer_batch (forward_only) 代替 train_batch 测试通路
            infer_result = engine.infer_batch(train_data)
            logits_train = infer_result["logits"]
            loss_output = grpo_loss_fn(logits_train, train_data)
            loss_val = loss_output["loss"].item()
            metrics = loss_output.get("metrics", {})
            print(f"[step {global_step}] loss={loss_val:.4f} "
                  f"approx_kl={metrics.get('approx_kl', 0):.4f} "
                  f"clipfrac={metrics.get('clipfrac', 0):.4f}", flush=True)

            # 通知 Tail 训练结束（通过 Middle 转发）
            done_header = torch.tensor([PIPELINE_DONE, 0, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
            engine.transport.send(done_header)

        elif engine.is_tail:
            # 2a: Receive rollout results from Head
            for _ in range(4):
                shape_t = torch.zeros(2, dtype=torch.int64, device="cuda")
                dist.recv(shape_t, src=0)
                buf = torch.zeros(*[int(s) for s in shape_t.tolist()], dtype=torch.int64, device="cuda")
                dist.recv(buf, src=0)
            pl = torch.zeros(1, dtype=torch.int64, device="cuda")
            dist.recv(pl, src=0)

            # 2c: old_log_prob forward loop (matches Head's re-forward)
            while True:
                header = torch.zeros(6, dtype=torch.int64, device="cuda")
                dist.recv(header, src=1)
                flag = int(header[0].item())
                if flag == PIPELINE_DONE:
                    break
                _B, S, H_dim, _has_mask, dtype_code = [int(x) for x in header[1:].tolist()]
                tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
                h = engine.transport.recv((_B, S, H_dim), tensor_dtype)
                pos_ids = engine.transport.recv((_B, S), torch.int64)
                pos_emb = None
                if engine.stage.rotary_emb is not None:
                    try:
                        pos_emb = engine.stage.rotary_emb(h.to("cuda"), pos_ids.to("cuda"))
                    except TypeError:
                        pos_emb = engine.stage.rotary_emb(h.to("cuda"), seq_len=S)
                with torch.no_grad():
                    logits = engine.stage.forward_output(h, position_ids=pos_ids, position_embeddings=pos_emb)
                dist.send(torch.tensor([logits.shape[0], logits.shape[1]], dtype=torch.int64, device="cuda"), dst=0)
                dist.send(logits, dst=0)

            # 2d: Training forward+backward (Tail is the terminal stage for backward)
            # Head's _head_forward_backward sends data, Tail computes loss+backward
            while True:
                header = torch.zeros(6, dtype=torch.int64, device="cuda")
                dist.recv(header, src=1)
                flag = int(header[0].item())
                if flag == SHUTDOWN:
                    break
                if flag == PIPELINE_DONE:
                    break
                _B, S, H_dim, _has_mask, dtype_code = [int(x) for x in header[1:].tolist()]
                tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
                h = engine.transport.recv((_B, S, H_dim), tensor_dtype)
                pos_ids = engine.transport.recv((_B, S), torch.int64)
                pos_emb = None
                if engine.stage.rotary_emb is not None:
                    try:
                        pos_emb = engine.stage.rotary_emb(h.to("cuda"), pos_ids.to("cuda"))
                    except TypeError:
                        pos_emb = engine.stage.rotary_emb(h.to("cuda"), seq_len=S)
                # Tail as terminal: forward_output + send logits detached
                with torch.no_grad():
                    logits = engine.stage.forward_output(h, position_ids=pos_ids, position_embeddings=pos_emb)
                dist.send(torch.tensor([logits.shape[0], logits.shape[1]], dtype=torch.int64, device="cuda"), dst=0)
                dist.send(logits, dst=0)

    print(f"[rank{rank}] v4 3-stage pipeline engine initialized", flush=True)

    if engine.is_head:
        header = torch.tensor([SHUTDOWN, 0, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
        engine.transport.send(header)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
