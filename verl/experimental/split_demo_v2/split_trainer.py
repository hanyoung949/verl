"""Split GRPO 训练器 — v2 NCCL 版。

与 v1 基本相同。核心训练循环不变。
差异点：
- rank0 只有一张卡（GPU 0），显存统计只看 primary_device
- model.forward_full() 内部会触发 NCCL 通信，但 trainer 透明
- fit() 结束后会调用 shutdown() 终止 rank1 worker
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from verl.trainer.ppo.core_algos import agg_loss, compute_grpo_outcome_advantage
from verl.utils.torch_functional import logprobs_from_logits, masked_mean


@dataclass
class TrainerBatch:
    prompts: list[str]
    answers: list[str]


class SplitGRPOTrainer:
    """最小可运行的 Split GRPO 训练循环（NCCL 版）。"""

    def __init__(
        self,
        model,
        rollout,
        reward,
        tokenizer,
        config,
    ) -> None:
        self.model = model
        self.rollout = rollout
        self.reward = reward
        self.tokenizer = tokenizer
        self.config = config

        self.group_size = int(config.rollout.group_size)
        self.max_new_tokens = int(config.rollout.max_new_tokens)
        self.mini_batch_size = int(config.algorithm.mini_batch_size)
        self.update_epochs = int(config.algorithm.update_epochs)
        self.loss_scale_factor = int(config.algorithm.loss_scale_factor)
        self.log_gpu_memory = bool(getattr(config.trainer, "log_gpu_memory", False))
        self.log_transport = bool(getattr(config.trainer, "log_transport", False))
        self.save_dir = Path(str(getattr(config.trainer, "save_dir", "verl/experimental/split_demo_v2/checkpoints")))
        self.save_latest = bool(getattr(config.trainer, "save_latest", True))

        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=float(config.trainer.lr),
        )

        self._rng = random.Random(int(config.trainer.seed))

    def _sample_batch(self, samples: list[dict[str, Any]], batch_size: int) -> TrainerBatch:
        picked = [samples[self._rng.randrange(len(samples))] for _ in range(batch_size)]
        return TrainerBatch(
            prompts=[item["prompt"] for item in picked],
            answers=[str(item["answer"]) for item in picked],
        )

    def _tokenize_prompts(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=int(self.config.data.max_prompt_length),
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def _build_uid_array(self, prompt_batch_size: int) -> np.ndarray:
        uids: list[str] = []
        for prompt_idx in range(prompt_batch_size):
            group_uid = f"group_{prompt_idx}"
            for _ in range(self.group_size):
                uids.append(group_uid)
        return np.array(uids, dtype=object)

    def fit(self, samples: list[dict[str, Any]]) -> None:
        total_steps = int(self.config.trainer.total_steps)
        batch_size = int(self.config.trainer.train_batch_size)
        log_freq = int(self.config.trainer.log_freq)

        for global_step in range(1, total_steps + 1):
            self._reset_peak_memory_stats()
            if hasattr(self.model, "reset_runtime_stats"):
                self.model.reset_runtime_stats()

            batch = self._sample_batch(samples, batch_size)
            prompt_ids, prompt_mask = self._tokenize_prompts(batch.prompts)

            with torch.no_grad():
                rollout_output = self.rollout.generate(
                    prompt_ids=prompt_ids,
                    group_size=self.group_size,
                    max_new_tokens=self.max_new_tokens,
                    attention_mask=prompt_mask,
                )

            with torch.no_grad():
                logits_old = self.model.forward_full(rollout_output.sequences, rollout_output.attention_mask)
                old_log_prob = logprobs_from_logits(
                    logits_old[:, rollout_output.prompt_len - 1 : -1, :],
                    rollout_output.response_ids,
                )

            repeated_answers = [answer for answer in batch.answers for _ in range(self.group_size)]
            rewards = self.reward.score(
                sequences=rollout_output.sequences,
                attention_mask=rollout_output.attention_mask,
                metadata={
                    "answers": repeated_answers,
                    "prompt_len": rollout_output.prompt_len,
                },
            )

            token_level_scores = torch.zeros_like(rollout_output.response_mask, dtype=torch.float32)
            last_valid = rollout_output.response_mask.sum(dim=1).long() - 1
            token_level_scores[torch.arange(token_level_scores.size(0), device=token_level_scores.device), last_valid] = rewards

            rewards_grouped = rewards.view(batch_size, self.group_size)
            if bool(self.config.algorithm.dynamic_sampling):
                valid_groups = rewards_grouped.std(dim=1) > 1e-6
            else:
                valid_groups = torch.ones(
                    batch_size,
                    dtype=torch.bool,
                    device=rewards_grouped.device,
                )
            if int(valid_groups.sum().item()) == 0:
                if global_step % log_freq == 0:
                    print(f"[step {global_step}] all groups filtered out by dynamic sampling, skip")
                continue

            valid_seq = valid_groups.unsqueeze(1).expand(batch_size, self.group_size).reshape(-1)

            sequences_valid = rollout_output.sequences[valid_seq]
            attn_mask_valid = rollout_output.attention_mask[valid_seq]
            response_ids_valid = rollout_output.response_ids[valid_seq]
            old_lp_valid = old_log_prob[valid_seq]
            token_scores_valid = token_level_scores[valid_seq]
            resp_mask_valid = rollout_output.response_mask[valid_seq]

            uid_array = self._build_uid_array(batch_size)
            uid_valid = uid_array[valid_seq.detach().cpu().numpy()]

            advantages_valid, _ = compute_grpo_outcome_advantage(
                token_level_rewards=token_scores_valid,
                response_mask=resp_mask_valid,
                index=uid_valid,
                norm_adv_by_std_in_grpo=bool(self.config.algorithm.norm_adv_by_std_in_grpo),
            )

            n_valid = sequences_valid.size(0)
            last_loss = None
            last_clipfrac = None
            last_approx_kl = None

            for _ in range(self.update_epochs):
                perm = torch.randperm(n_valid, device=sequences_valid.device)

                for start in range(0, n_valid, self.mini_batch_size):
                    mb = perm[start : start + self.mini_batch_size]
                    logits_new = self.model.forward_full(sequences_valid[mb], attn_mask_valid[mb])
                    log_prob_new = logprobs_from_logits(
                        logits_new[:, rollout_output.prompt_len - 1 : -1, :],
                        response_ids_valid[mb],
                    )

                    neg_kl = torch.clamp(log_prob_new - old_lp_valid[mb], min=-20.0, max=20.0)
                    ratio = neg_kl.exp()
                    approx_kl = masked_mean(-neg_kl, resp_mask_valid[mb])

                    clip_low = float(self.config.algorithm.cliprange_low)
                    clip_high = float(self.config.algorithm.cliprange_high)
                    clipped = (ratio.detach() < 1.0 - clip_low) | (ratio.detach() > 1.0 + clip_high)
                    clipfrac = masked_mean(clipped.float(), resp_mask_valid[mb])

                    adv = advantages_valid[mb]
                    pg_losses = torch.maximum(
                        -adv * ratio,
                        -adv * ratio.clamp(1.0 - clip_low, 1.0 + clip_high),
                    )
                    loss = agg_loss(
                        loss_mat=pg_losses,
                        loss_mask=resp_mask_valid[mb],
                        loss_agg_mode=str(self.config.algorithm.loss_agg_mode),
                        loss_scale_factor=self.loss_scale_factor,
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.trainable_parameters(), max_norm=float(self.config.trainer.max_grad_norm))
                    self.optimizer.step()

                    last_loss = loss.detach().item()
                    last_clipfrac = clipfrac.detach().item()
                    last_approx_kl = approx_kl.detach().item()

            if global_step % log_freq == 0:
                message = (
                    f"[step {global_step}] "
                    f"reward={rewards[valid_seq].mean().item():.4f} "
                    f"valid_group_ratio={valid_groups.float().mean().item():.4f} "
                    f"loss={last_loss:.4f} "
                    f"clipfrac={last_clipfrac:.4f} "
                    f"approx_kl={last_approx_kl:.4f}"
                )
                if self.log_gpu_memory:
                    message += " " + self._format_memory_stats()
                if self.log_transport and hasattr(self.model, "get_runtime_stats"):
                    message += " " + self._format_transport_stats()
                print(message)

        self._save_training_artifacts(global_step=total_steps)

    def _reset_peak_memory_stats(self) -> None:
        if not torch.cuda.is_available():
            return
        # rank0 只有一张卡
        device_name = getattr(self.model, "primary_device", None)
        if not device_name or not str(device_name).startswith("cuda:"):
            return
        index = int(str(device_name).split(":")[1])
        torch.cuda.reset_peak_memory_stats(index)

    def _format_memory_stats(self) -> str:
        if not torch.cuda.is_available():
            return "gpu_mem=cpu_only"

        parts: list[str] = []
        device_name = getattr(self.model, "primary_device", None)
        if not device_name or not str(device_name).startswith("cuda:"):
            return "gpu_mem=unknown"

        index = int(str(device_name).split(":")[1])
        allocated = torch.cuda.memory_allocated(index) / (1024**3)
        reserved = torch.cuda.memory_reserved(index) / (1024**3)
        peak = torch.cuda.max_memory_allocated(index) / (1024**3)
        parts.append(
            f"gpu{index}_alloc={allocated:.2f}GB "
            f"gpu{index}_reserv={reserved:.2f}GB "
            f"gpu{index}_peak={peak:.2f}GB"
        )
        return " ".join(parts)

    def _format_transport_stats(self) -> str:
        stats = self.model.get_runtime_stats()
        middle = stats.get("middle_executor", {})

        def gb(nbytes: int) -> float:
            return float(nbytes) / (1024**3)

        parts = [
            f"fwd_calls={stats.get('forward_calls', 0)}",
            f"prefill_calls={stats.get('prefill_calls', 0)}",
            f"decode_calls={stats.get('decode_calls', 0)}",
            f"to_middle={gb(int(middle.get('forward_to_middle_bytes', 0))):.4f}GB",
            f"pos_ids={gb(int(middle.get('position_ids_bytes', 0))):.4f}GB",
            f"pos_emb={gb(int(middle.get('position_embeddings_bytes', 0))):.4f}GB",
            f"mask={gb(int(middle.get('attention_mask_bytes', 0))):.4f}GB",
            f"to_primary={gb(int(stats.get('return_to_primary_bytes', 0))):.4f}GB",
        ]

        if middle.get("last_hidden_shape") is not None:
            parts.append(f"last_hidden={middle['last_hidden_shape']}")
        if stats.get("last_logits_shape") is not None:
            parts.append(f"last_logits={stats['last_logits_shape']}")

        return " ".join(parts)

    def _save_training_artifacts(self, global_step: int) -> None:
        base_dir = self.save_dir
        step_dir = base_dir / f"global_step_{global_step}"
        latest_dir = base_dir / "latest"

        for target in [step_dir, latest_dir] if self.save_latest else [step_dir]:
            self.model.save_adapter(target, global_step=global_step)
            torch.save(self.optimizer.state_dict(), target / "optimizer.pt")
            trainer_state = {
                "global_step": global_step,
                "train_batch_size": int(self.config.trainer.train_batch_size),
                "total_steps": int(self.config.trainer.total_steps),
                "group_size": int(self.config.rollout.group_size),
                "max_new_tokens": int(self.config.rollout.max_new_tokens),
                "lr": float(self.config.trainer.lr),
                "loss_scale_factor": int(self.config.algorithm.loss_scale_factor),
            }
            (target / "trainer_state.json").write_text(
                json.dumps(trainer_state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
