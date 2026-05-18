"""split demo v3 主入口 — Phase 1 规范路径版。

所有 forward/backward/optimizer 通过 SplitEngine 接口调用：
- old_log_prob → engine.infer_batch()
- training     → engine.train_batch(data, grpo_loss_fn)
- checkpoint   → engine.save_checkpoint()

运行命令：
    torchrun --nproc_per_node=2 -m verl.experimental.split_demo_v3.main_split_v3
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

from .core.split_engine import SplitEngine
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


class SimpleSplitRollout:
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

    def generate(self, prompt_ids, group_size, max_new_tokens, attention_mask=None):
        # Rollout 仍直接调用 split_core（rollout 不需要 backward）
        # 未来 vLLM 替换时只改这里
        device = torch.device("cuda:0")
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

        with torch.no_grad():
            prefill_logits = self.engine.split_core.forward_full(prompt_ids, attention_mask)
        next_token = self._sample_next_token(prefill_logits[:, -1, :])
        current_valid = (~finished).float().unsqueeze(-1)
        generated_tokens.append(next_token)
        generated_masks.append(current_valid)
        finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
        current_ids = torch.cat([prompt_ids, next_token], dim=-1)
        current_mask = torch.cat([attention_mask, current_valid.long()], dim=-1)

        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                next_logits = self.engine.split_core.forward_full(current_ids, current_mask)
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
    """创建 GRPO loss 函数，传给 engine.train_batch()。
    prompt_len 从 data["prompt_len"] 读取。

    loss_fn(logits, data) → {"loss": tensor, "metrics": {...}}
    data 需要包含：response_ids, old_log_probs, advantages, response_mask
    """

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
            loss_mat=pg_losses,
            loss_mask=data["response_mask"],
            loss_agg_mode=loss_agg_mode,
            loss_scale_factor=loss_scale_factor,
        )

        approx_kl = masked_mean(-neg_kl, data["response_mask"])
        clipped = (ratio.detach() < 1.0 - clip_low) | (ratio.detach() > 1.0 + clip_high)
        clipfrac = masked_mean(clipped.float(), data["response_mask"])

        return {
            "loss": loss,
            "metrics": {
                "approx_kl": approx_kl.detach().item(),
                "clipfrac": clipfrac.detach().item(),
            },
        }

    return grpo_loss_fn


@hydra.main(config_path="config", config_name="split_demo_v3", version_base=None)
def main(config: DictConfig):
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    engine = SplitEngine()
    engine.model_path = str(config.model.path)
    engine.front_end = int(config.split.front_end)
    engine.middle_end = int(config.split.middle_end)
    engine.lr = float(config.trainer.lr)
    engine.clip_grad = float(config.trainer.max_grad_norm)
    engine.initialize()

    if rank == 1:
        try:
            engine.middle_worker.run()
        finally:
            dist.destroy_process_group()
        return

    set_seed(int(config.trainer.seed))
    tokenizer = engine.tokenizer
    samples = load_jsonl_samples(config.data.train_file)
    rollout = SimpleSplitRollout(engine, tokenizer, float(config.rollout.temperature), float(config.rollout.top_p))
    reward_fn = FunctionReward(fn=math_exact_match, tokenizer=tokenizer)

    total_steps = int(config.trainer.total_steps)
    batch_size = int(config.trainer.train_batch_size)
    group_size = int(config.rollout.group_size)
    max_new_tokens = int(config.rollout.max_new_tokens)
    mini_batch_size = int(config.algorithm.mini_batch_size)
    update_epochs = int(config.algorithm.update_epochs)
    loss_scale_factor = int(config.algorithm.loss_scale_factor)
    log_freq = int(config.trainer.log_freq)
    prompt_len = int(config.data.max_prompt_length)

    grpo_loss_fn = make_grpo_loss_fn(
        clip_low=float(config.algorithm.cliprange_low),
        clip_high=float(config.algorithm.cliprange_high),
        loss_agg_mode=str(config.algorithm.loss_agg_mode),
        loss_scale_factor=loss_scale_factor,
    )

    rng = random.Random(int(config.trainer.seed))
    engine.split_core.reset_runtime_stats()

    for global_step in range(1, total_steps + 1):
        torch.cuda.reset_peak_memory_stats(0)
        engine.split_core.reset_runtime_stats()

        picked = [samples[rng.randrange(len(samples))] for _ in range(batch_size)]
        prompts = [item["prompt"] for item in picked]
        answers = [str(item["answer"]) for item in picked]

        encoded = tokenizer(prompts, padding=True, truncation=True,
                          max_length=int(config.data.max_prompt_length), return_tensors="pt")
        prompt_ids = encoded["input_ids"]
        prompt_mask = encoded["attention_mask"]

        rollout_output = rollout.generate(prompt_ids, group_size, max_new_tokens, prompt_mask)

        # Phase 1: old_log_prob 通过 engine.infer_batch() 计算
        infer_data = TensorDict({
            "input_ids": rollout_output.sequences,
            "attention_mask": rollout_output.attention_mask,
        }, batch_size=[rollout_output.sequences.size(0)])
        infer_result = engine.infer_batch(infer_data)
        logits_old = infer_result["logits"]
        old_log_prob = logprobs_from_logits(
            logits_old[:, rollout_output.prompt_len - 1:-1, :], rollout_output.response_ids)

        repeated_answers = [a for a in answers for _ in range(group_size)]
        rewards = reward_fn.score(rollout_output.sequences, rollout_output.attention_mask,
                                 {"answers": repeated_answers, "prompt_len": rollout_output.prompt_len})

        token_level_scores = torch.zeros_like(rollout_output.response_mask, dtype=torch.float32)
        last_valid = rollout_output.response_mask.sum(dim=1).long() - 1
        token_level_scores[torch.arange(token_level_scores.size(0), device=token_level_scores.device), last_valid] = rewards

        rewards_grouped = rewards.view(batch_size, group_size)
        if bool(config.algorithm.dynamic_sampling):
            valid_groups = rewards_grouped.std(dim=1) > 1e-6
        else:
            valid_groups = torch.ones(batch_size, dtype=torch.bool, device=rewards_grouped.device)
        if int(valid_groups.sum().item()) == 0:
            if global_step % log_freq == 0:
                print(f"[step {global_step}] all groups filtered, skip")
            continue

        valid_seq = valid_groups.unsqueeze(1).expand(batch_size, group_size).reshape(-1)
        sequences_valid = rollout_output.sequences[valid_seq]
        attn_mask_valid = rollout_output.attention_mask[valid_seq]
        response_ids_valid = rollout_output.response_ids[valid_seq]
        old_lp_valid = old_log_prob[valid_seq]
        token_scores_valid = token_level_scores[valid_seq]
        resp_mask_valid = rollout_output.response_mask[valid_seq]

        uid_array = []
        for pi in range(batch_size):
            for _ in range(group_size):
                uid_array.append(f"group_{pi}")
        uid_valid = np.array(uid_array, dtype=object)[valid_seq.detach().cpu().numpy()]

        advantages_valid, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_scores_valid, response_mask=resp_mask_valid,
            index=uid_valid, norm_adv_by_std_in_grpo=bool(config.algorithm.norm_adv_by_std_in_grpo))

        n_valid = sequences_valid.size(0)
        last_loss = last_clipfrac = last_approx_kl = None

        for _ in range(update_epochs):
            perm = torch.randperm(n_valid, device=sequences_valid.device)
            for start in range(0, n_valid, mini_batch_size):
                mb = perm[start:start + mini_batch_size]

                # Phase 1: 训练通过 engine.train_batch()
                train_data = TensorDict({
                    "input_ids": sequences_valid[mb],
                    "attention_mask": attn_mask_valid[mb],
                    "response_ids": response_ids_valid[mb],
                    "response_mask": resp_mask_valid[mb],
                    "old_log_probs": old_lp_valid[mb],
                    "advantages": advantages_valid[mb],
                }, batch_size=[mb.size(0)])

                train_result = engine.train_batch(train_data, grpo_loss_fn)
                last_loss = train_result["loss"][0]
                last_clipfrac = train_result["metrics"].get("clipfrac", 0)
                last_approx_kl = train_result["metrics"].get("approx_kl", 0)

        if global_step % log_freq == 0:
            gpu_alloc = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_peak = torch.cuda.max_memory_allocated(0) / (1024**3)
            print(f"[step {global_step}] reward={rewards[valid_seq].mean().item():.4f} "
                  f"valid_group_ratio={valid_groups.float().mean().item():.4f} "
                  f"loss={last_loss:.4f} clipfrac={last_clipfrac:.4f} approx_kl={last_approx_kl:.4f} "
                  f"gpu0_alloc={gpu_alloc:.2f}GB gpu0_peak={gpu_peak:.2f}GB")

    # Phase 1: checkpoint 通过 engine.save_checkpoint()
    save_dir = Path(str(getattr(config.trainer, "save_dir", "verl/experimental/split_demo_v3/checkpoints")))
    engine.save_checkpoint(str(save_dir / "latest"), global_step=total_steps)
    engine.split_core.middle_executor.shutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
