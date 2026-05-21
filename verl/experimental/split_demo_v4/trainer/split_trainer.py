"""SplitTrainer — 3-stage pipeline 的 GRPO 训练循环。"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_grpo_outcome_advantage
from verl.utils.torch_functional import logprobs_from_logits, masked_mean

from ..core.transport import PIPELINE_DONE, SHUTDOWN, TRAIN_MB, TRAIN_DONE, STEP_SKIP, TRAIN_START


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
            "metrics": {"clipfrac": clipfrac.item(), "approx_kl": approx_kl.item()}
        }
    return grpo_loss_fn


class SplitTrainer:
    """3-stage pipeline 的 GRPO 训练器。

    把 main_split_v4.py 中的训练循环抽取出来，main 只负责组件组装。
    """

    def __init__(self, engine, rollout_backend, reward_fn, config):
        self.engine = engine
        self.rollout_backend = rollout_backend
        self.reward_fn = reward_fn
        self.config = config

        self.total_steps = int(config.trainer.total_steps)
        self.batch_size = int(config.trainer.train_batch_size)
        self.group_size = int(config.rollout.group_size)
        self.max_new_tokens = int(config.rollout.max_new_tokens)
        self.mini_batch_size = int(config.algorithm.mini_batch_size)
        self.update_epochs = int(config.algorithm.update_epochs)
        self.loss_scale_factor = int(config.algorithm.loss_scale_factor)
        self.log_freq = int(config.trainer.log_freq)

    def fit(self, samples):
        """主训练循环。"""
        set_seed(int(self.config.trainer.seed))
        rng = random.Random(int(self.config.trainer.seed))
        tokenizer = self.engine.tokenizer
        batch_size = self.batch_size
        group_size = self.group_size

        for global_step in range(1, self.total_steps + 1):
            picked = [samples[rng.randrange(len(samples))] for _ in range(batch_size)]
            prompts = [item["prompt"] for item in picked]
            answers = [str(item["answer"]) for item in picked]

            encoded = tokenizer(prompts, padding=True, truncation=True,
                                max_length=int(self.config.data.max_prompt_length), return_tensors="pt")

            # ===== Phase 1: Rollout =====
            rollout_output = None
            if self.engine.is_head:
                rollout_output = self.rollout_backend.generate(
                    encoded["input_ids"], group_size, self.max_new_tokens, encoded["attention_mask"])
            elif self.engine.is_tail:
                self.rollout_backend.run_tail_loop()

            # ===== Phase 2: Reward + old_log_prob + Advantage + Train =====
            if self.engine.is_head and rollout_output is not None:
                self._head_train_step(global_step, rollout_output, answers, batch_size, group_size)
            elif self.engine.is_tail:
                self._tail_train_step(global_step)

        self._shutdown()

    # ------------------------------------------------------------------
    # Head 侧私有方法
    # ------------------------------------------------------------------

    def _head_train_step(self, global_step, rollout_output, answers, batch_size, group_size):
        """Head: sync rollout results → reward → advantage → train."""
        # 2a: Sync rollout results to Tail
        for name in ["response_ids", "response_mask"]:
            tensor = getattr(rollout_output, name)
            shape_t = torch.tensor(list(tensor.shape), dtype=torch.int64, device="cuda")
            dist.send(shape_t, dst=2)
            dist.send(tensor.contiguous(), dst=2)
        dist.send(torch.tensor([rollout_output.prompt_len], dtype=torch.int64, device="cuda"), dst=2)

        # 2b: 等 Tail 确认已收到 rollout results
        tail_ready = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.recv(tail_ready, src=2)

        # 2c: old_log_prob（rollout 时已保存，无需 infer_batch 重算）
        old_log_prob = rollout_output.old_log_probs

        # 通知 Tail old_log_prob 阶段结束
        header = torch.tensor([PIPELINE_DONE, 0, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
        self.engine.transport.send(header)

        # 2d: Reward
        repeated_answers = [a for a in answers for _ in range(group_size)]
        rewards = self.reward_fn.score(
            rollout_output.sequences, rollout_output.attention_mask,
            {"answers": repeated_answers, "prompt_len": rollout_output.prompt_len})
        print(f"[step {global_step}] reward={rewards.mean().item():.4f}", flush=True)

        # 2d: token_level_scores + dynamic_sampling + advantage
        token_level_scores = torch.zeros_like(rollout_output.response_mask, dtype=torch.float32)
        last_valid = rollout_output.response_mask.sum(dim=1).long() - 1
        token_level_scores[torch.arange(token_level_scores.size(0), device="cuda"), last_valid] = rewards
        rewards_grouped = rewards.view(batch_size, group_size)
        if bool(self.config.algorithm.dynamic_sampling):
            valid_groups = rewards_grouped.std(dim=1) > 1e-6
        else:
            valid_groups = torch.ones(batch_size, dtype=torch.bool, device="cuda")
        if int(valid_groups.sum().item()) == 0:
            print(f"[step {global_step}] all groups filtered, skip", flush=True)
            skip_ctrl = torch.tensor([STEP_SKIP, 0], dtype=torch.int64, device="cuda")
            dist.send(skip_ctrl, dst=2)
            return
        valid_seq = valid_groups.unsqueeze(1).expand(batch_size, group_size).reshape(-1)
        uid_array = [f"g{pi}" for pi in range(batch_size) for _ in range(group_size)]
        uid_valid = np.array(uid_array, dtype=object)[valid_seq.detach().cpu().numpy()]
        advantages_valid, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_scores[valid_seq],
            response_mask=rollout_output.response_mask[valid_seq],
            index=uid_valid,
            norm_adv_by_std_in_grpo=bool(self.config.algorithm.norm_adv_by_std_in_grpo))
        print(f"[step {global_step}] advantage mean={advantages_valid.mean().item():.4f} "
              f"valid={int(valid_groups.sum().item())}/{batch_size}", flush=True)

        # 2e: 多 epoch × mini-batch 训练
        seq_v = rollout_output.sequences[valid_seq]
        mask_v = rollout_output.attention_mask[valid_seq]
        resp_ids_v = rollout_output.response_ids[valid_seq]
        resp_mask_v = rollout_output.response_mask[valid_seq]
        old_lp_v = old_log_prob[valid_seq]
        n_valid = seq_v.size(0)

        # 通知 Tail 开始接收训练数据
        dist.send(torch.tensor([TRAIN_START, 0], dtype=torch.int64, device="cuda"), dst=2)

        # 一次性发送完整训练数据给 Tail
        tensors_to_send = [resp_ids_v, resp_mask_v, old_lp_v, advantages_valid]
        for i, tensor in enumerate(tensors_to_send):
            print(f"[step {global_step}] sending tensor {i}: {tensor.shape} {tensor.dtype}", flush=True)
            shape_t = torch.tensor(list(tensor.shape), dtype=torch.int64, device="cuda")
            dtype_map = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2, torch.int64: 3}
            dtype_t = torch.tensor([dtype_map.get(tensor.dtype, 0)], dtype=torch.int64, device="cuda")
            dist.send(shape_t, dst=2)
            dist.send(dtype_t, dst=2)
            dist.send(tensor.contiguous(), dst=2)
        # 发 GRPO 配置
        grpo_cfg = torch.tensor([
            float(self.config.algorithm.cliprange_low),
            float(self.config.algorithm.cliprange_high),
            float(self.config.algorithm.loss_scale_factor),
        ], dtype=torch.float32, device="cuda")
        dist.send(grpo_cfg, dst=2)
        print(f"[step {global_step}] full data + GRPO config sent", flush=True)

        last_loss = last_clipfrac = last_approx_kl = 0.0
        grpo_loss_fn = make_grpo_loss_fn(
            clip_low=float(self.config.algorithm.cliprange_low),
            clip_high=float(self.config.algorithm.cliprange_high),
            loss_agg_mode=str(self.config.algorithm.loss_agg_mode),
            loss_scale_factor=self.loss_scale_factor)

        for _epoch in range(self.update_epochs):
            perm = torch.randperm(n_valid, device="cuda")
            for _start in range(0, n_valid, self.mini_batch_size):
                mb = perm[_start:_start + self.mini_batch_size]

                # Head 给 Tail 发 TRAIN_MB + mb indices
                ctrl_header = torch.tensor([TRAIN_MB, mb.size(0)], dtype=torch.int64, device="cuda")
                dist.send(ctrl_header, dst=2)
                dist.send(mb, dst=2)

                # Head 调用 train_batch
                mb_data = {
                    "input_ids": seq_v[mb],
                    "attention_mask": mask_v[mb],
                    "response_ids": resp_ids_v[mb],
                    "response_mask": resp_mask_v[mb],
                    "old_log_probs": old_lp_v[mb],
                    "advantages": advantages_valid[mb],
                }
                train_data = TensorDict(mb_data, batch_size=[mb.size(0)])
                self.engine.train_batch(train_data, grpo_loss_fn)

                # 从 Tail 接收 loss/metrics
                metrics_tensor = torch.zeros(3, dtype=torch.float32, device="cuda")
                dist.recv(metrics_tensor, src=2)
                last_loss = metrics_tensor[0].item()
                last_clipfrac = metrics_tensor[1].item()
                last_approx_kl = metrics_tensor[2].item()

        # 训练结束
        done_ctrl = torch.tensor([TRAIN_DONE, 0], dtype=torch.int64, device="cuda")
        dist.send(done_ctrl, dst=2)

        print(f"[step {global_step}] loss={last_loss:.4f} "
              f"approx_kl={last_approx_kl:.4f} "
              f"clipfrac={last_clipfrac:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Tail 侧私有方法
    # ------------------------------------------------------------------

    def _tail_train_step(self, global_step):
        """Tail: receive rollout results → train loop."""
        # 2a: Receive rollout results from Head
        for _ in range(2):
            shape_t = torch.zeros(2, dtype=torch.int64, device="cuda")
            dist.recv(shape_t, src=0)
            buf = torch.zeros(*[int(s) for s in shape_t.tolist()], dtype=torch.int64, device="cuda")
            dist.recv(buf, src=0)
        pl = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.recv(pl, src=0)

        # 2b: 先发 ready 给 Head
        dist.send(torch.tensor([1], dtype=torch.int64, device="cuda"), dst=0)

        # old_log_prob forward loop —— 统一走 engine API
        while True:
            result = self.engine.forward_backward_batch(TensorDict({}, batch_size=[0]), None, forward_only=True)
            if not result or "logits" not in result:
                break

        # 2c: 接收 Head 的控制消息
        ctrl = torch.zeros(2, dtype=torch.int64, device="cuda")
        dist.recv(ctrl, src=0)
        ctrl_flag = int(ctrl[0].item())
        if ctrl_flag == STEP_SKIP:
            return
        if ctrl_flag != TRAIN_START:
            raise RuntimeError(
                f"[Tail] expected TRAIN_START ({TRAIN_START}) or STEP_SKIP ({STEP_SKIP}), "
                f"got {ctrl_flag}. Protocol misalignment between Head and Tail."
            )

        # 一次性接收完整训练数据
        train_cache = {}
        for name in ["response_ids", "response_mask", "old_log_probs", "advantages"]:
            shape_t = torch.zeros(2, dtype=torch.int64, device="cuda")
            dist.recv(shape_t, src=0)
            dtype_t = torch.zeros(1, dtype=torch.int64, device="cuda")
            dist.recv(dtype_t, src=0)
            dtype_map = {0: torch.float32, 1: torch.bfloat16, 2: torch.float16, 3: torch.int64}
            buf = torch.empty(*[int(s) for s in shape_t.tolist()],
                            dtype=dtype_map[int(dtype_t.item())], device="cuda")
            dist.recv(buf, src=0)
            train_cache[name] = buf

        # 接收 GRPO 配置
        grpo_cfg = torch.zeros(3, dtype=torch.float32, device="cuda")
        dist.recv(grpo_cfg, src=0)
        tail_loss_fn = make_grpo_loss_fn(
            clip_low=float(grpo_cfg[0].item()), clip_high=float(grpo_cfg[1].item()),
            loss_agg_mode=str(self.config.algorithm.loss_agg_mode), loss_scale_factor=int(grpo_cfg[2].item()))
        print(f"[Tail] full data received, entering training loop", flush=True)

        # 2d: 训练循环
        while True:
            ctrl = torch.zeros(2, dtype=torch.int64, device="cuda")
            dist.recv(ctrl, src=0)
            ctrl_flag = int(ctrl[0].item())

            if ctrl_flag == TRAIN_DONE:
                break

            if ctrl_flag == TRAIN_MB:
                mb_size = int(ctrl[1].item())
                mb_indices = torch.zeros(mb_size, dtype=torch.int64, device="cuda")
                dist.recv(mb_indices, src=0)

                train_tensors = {k: v[mb_indices] for k, v in train_cache.items()}
                train_data = TensorDict(train_tensors, batch_size=[mb_size])

                train_result = self.engine.train_batch(train_data, tail_loss_fn)

                loss_val = train_result.get("loss", [0.0])[0] if train_result else 0.0
                metrics = train_result.get("metrics", {})
                metrics_tensor = torch.tensor([
                    float(loss_val),
                    float(metrics.get("clipfrac", 0.0)),
                    float(metrics.get("approx_kl", 0.0)),
                ], dtype=torch.float32, device="cuda")
                dist.send(metrics_tensor, dst=0)

    # ------------------------------------------------------------------
    # 通用辅助方法
    # ------------------------------------------------------------------

    def _shutdown(self):
        print(f"[rank{dist.get_rank() if dist.is_initialized() else 0}] v4 3-stage pipeline engine initialized", flush=True)
        if self.engine.is_head:
            header = torch.tensor([SHUTDOWN, 0, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
            self.engine.transport.send(header)
        if dist.is_initialized():
            dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
