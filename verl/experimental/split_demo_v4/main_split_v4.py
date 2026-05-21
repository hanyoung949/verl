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
from .core.transport import FWD_ONLY, FWD_WITH_BWD, PIPELINE_DONE, SHUTDOWN, TRAIN_MB, TRAIN_DONE, STEP_SKIP, TRAIN_START
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

    def _pipeline_forward(self, data):
        """Head→Middle→Tail 前向，Tail 本地采样后回传 next_token + log_prob。

        通信量从 O(B*S*vocab) 降到 O(B)。
        """
        if self.engine.is_head:
            result = self.engine.sample_next_token(data, temperature=self.temperature, pad_token_id=self.pad_token_id)
            return result

        elif self.engine.is_middle:
            self.engine.stage.run()
            return None

        # Tail 分支不会走到这里：generate() 在 Tail rank 提前返回 None。

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

        prefill_data = TensorDict({
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
        }, batch_size=prompt_ids.shape[0])
        prefill_result = self._pipeline_forward(prefill_data)
        next_token = prefill_result["next_token"]
        prefill_log_prob = prefill_result["log_prob"]
        current_valid = (~finished).float().unsqueeze(-1)
        generated_tokens = [next_token]
        generated_log_probs = [prefill_log_prob]
        generated_masks = [current_valid]
        finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
        current_ids = torch.cat([prompt_ids, next_token], dim=-1)
        current_mask = torch.cat([attention_mask, current_valid.long()], dim=-1)

        for _ in range(max_new_tokens - 1):
            decode_data = TensorDict({
                "input_ids": current_ids,
                "attention_mask": current_mask,
                "finished": finished,
            }, batch_size=current_ids.shape[0])
            result = self._pipeline_forward(decode_data)
            sampled = result["next_token"]
            step_log_prob = result["log_prob"]
            current_valid = (~finished).float().unsqueeze(-1)
            generated_tokens.append(sampled)
            generated_log_probs.append(step_log_prob)
            generated_masks.append(current_valid)
            finished = finished | (sampled.squeeze(-1) == self.eos_token_id)
            current_ids = torch.cat([current_ids, sampled], dim=-1)
            current_mask = torch.cat([current_mask, current_valid.long()], dim=-1)
            if bool(torch.all(finished)):
                break

        response_ids = torch.cat(generated_tokens, dim=-1)
        response_log_probs = torch.cat(generated_log_probs, dim=-1)
        response_mask = torch.cat(generated_masks, dim=-1).long()
        sequences = torch.cat([prompt_ids, response_ids], dim=-1)
        full_mask = torch.cat([attention_mask, response_mask], dim=-1)

        class Out: pass
        out = Out()
        out.sequences = sequences
        out.attention_mask = full_mask
        out.response_ids = response_ids
        out.response_mask = response_mask
        out.old_log_probs = response_log_probs
        out.prompt_len = prompt_len
        return out


def _tail_forward_logits(engine, src_rank=1, dst_rank=0):
    """Tail 从 Middle 接收 activation 并做无 grad 前向，返回 logits 和 flag。

    抽取重复逻辑，减少协议修改时遗漏的风险。
    """
    header = torch.zeros(6, dtype=torch.int64, device="cuda")
    dist.recv(header, src=src_rank)
    flag = int(header[0].item())
    if flag == PIPELINE_DONE:
        return None, flag
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
    dist.send(torch.tensor([logits.shape[0], logits.shape[1], logits.shape[2]], dtype=torch.int64, device="cuda"), dst=dst_rank)
    dist.send(logits, dst=dst_rank)
    return logits, flag


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
            # Phase 1 rollout —— 统一走 engine.sample_next_token()
            while True:
                result = engine.sample_next_token(TensorDict({}, batch_size=[0]))
                if not result or "next_token" not in result:
                    break

        # ===== Phase 2: Reward + old_log_prob + Advantage + Train =====
        if engine.is_head and rollout_output is not None:
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

            # 通知 Tail old_log_prob 阶段结束（协议对齐：Tail 仍需收 PIPELINE_DONE 才能进入训练）
            header = torch.tensor([PIPELINE_DONE, 0, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
            engine.transport.send(header)

            # 2d: Reward
            repeated_answers = [a for a in answers for _ in range(group_size)]
            rewards = reward_fn.score(
                rollout_output.sequences, rollout_output.attention_mask,
                {"answers": repeated_answers, "prompt_len": rollout_output.prompt_len})
            print(f"[step {global_step}] reward={rewards.mean().item():.4f}", flush=True)

            # 2d: token_level_scores + dynamic_sampling + advantage
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
                # 通知 Tail 跳过本轮训练，防止协议错位
                skip_ctrl = torch.tensor([STEP_SKIP, 0], dtype=torch.int64, device="cuda")
                dist.send(skip_ctrl, dst=2)
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

            # 2e: 多 epoch × mini-batch 训练（优化：一次性同步完整训练数据给 Tail）
            seq_v = rollout_output.sequences[valid_seq]
            mask_v = rollout_output.attention_mask[valid_seq]
            resp_ids_v = rollout_output.response_ids[valid_seq]
            resp_mask_v = rollout_output.response_mask[valid_seq]
            old_lp_v = old_log_prob[valid_seq]
            n_valid = seq_v.size(0)

            # 通知 Tail 开始接收训练数据
            dist.send(torch.tensor([TRAIN_START, 0], dtype=torch.int64, device="cuda"), dst=2)

            # 一次性发送完整训练数据给 Tail（response_ids, response_mask, old_log_probs, advantages）
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
                float(config.algorithm.cliprange_low),
                float(config.algorithm.cliprange_high),
                float(config.algorithm.loss_scale_factor),
            ], dtype=torch.float32, device="cuda")
            dist.send(grpo_cfg, dst=2)
            print(f"[step {global_step}] full data + GRPO config sent", flush=True)

            last_loss = last_clipfrac = last_approx_kl = 0.0
            grpo_loss_fn = make_grpo_loss_fn(
                clip_low=float(config.algorithm.cliprange_low),
                clip_high=float(config.algorithm.cliprange_high),
                loss_agg_mode=str(config.algorithm.loss_agg_mode),
                loss_scale_factor=loss_scale_factor)

            for _epoch in range(update_epochs):
                perm = torch.randperm(n_valid, device="cuda")
                for _start in range(0, n_valid, mini_batch_size):
                    mb = perm[_start:_start + mini_batch_size]

                    # Step 1: Head 给 Tail 发 TRAIN_MB + mb indices（不再发完整 tensor）
                    ctrl_header = torch.tensor([TRAIN_MB, mb.size(0)],
                                             dtype=torch.int64, device="cuda")
                    dist.send(ctrl_header, dst=2)
                    dist.send(mb, dst=2)

                    # Step 2: Head 调用 train_batch（统一 zero_grad / step 在 engine 内处理）
                    mb_data = {
                        "input_ids": seq_v[mb],
                        "attention_mask": mask_v[mb],
                        "response_ids": resp_ids_v[mb],
                        "response_mask": resp_mask_v[mb],
                        "old_log_probs": old_lp_v[mb],
                        "advantages": advantages_valid[mb],
                    }
                    train_data = TensorDict(mb_data, batch_size=[mb.size(0)])
                    engine.train_batch(train_data, grpo_loss_fn)
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

        elif engine.is_tail:
            # 2a: Receive rollout results from Head (response_ids, response_mask + prompt_len)
            for _ in range(2):
                shape_t = torch.zeros(2, dtype=torch.int64, device="cuda")
                dist.recv(shape_t, src=0)
                buf = torch.zeros(*[int(s) for s in shape_t.tolist()], dtype=torch.int64, device="cuda")
                dist.recv(buf, src=0)
            pl = torch.zeros(1, dtype=torch.int64, device="cuda")
            dist.recv(pl, src=0)

            # 2b: 先发 ready 给 Head，再进入 old_log_prob loop
            # 顺序至关重要：必须先 ready，否则 Head 等 ready、Tail 等 infer_batch → 死锁
            dist.send(torch.tensor([1], dtype=torch.int64, device="cuda"), dst=0)

            # old_log_prob forward loop —— 统一走 engine API
            while True:
                result = engine.forward_backward_batch(TensorDict({}, batch_size=[0]), None, forward_only=True)
                if not result or "logits" not in result:
                    break

            # 2c: 接收 Head 的控制消息
            ctrl = torch.zeros(2, dtype=torch.int64, device="cuda")
            dist.recv(ctrl, src=0)
            ctrl_flag = int(ctrl[0].item())
            if ctrl_flag == STEP_SKIP:
                # Head 通知本 step 跳过训练（dynamic_sampling 全过滤）
                continue

            # 一次性接收完整训练数据（response_ids, response_mask, old_log_probs, advantages）
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
            # Tail 和 Head 使用相同的 loss_agg_mode，避免配置不一致
            tail_loss_fn = make_grpo_loss_fn(
                clip_low=float(grpo_cfg[0].item()), clip_high=float(grpo_cfg[1].item()),
                loss_agg_mode=str(config.algorithm.loss_agg_mode), loss_scale_factor=int(grpo_cfg[2].item()))
            print(f"[Tail] full data received, entering training loop", flush=True)

            # 2d: 训练循环 — 先收 Head 的控制消息，再调用 engine.train_batch()
            while True:
                # 先等 Head 的控制消息
                ctrl = torch.zeros(2, dtype=torch.int64, device="cuda")
                dist.recv(ctrl, src=0)
                ctrl_flag = int(ctrl[0].item())

                if ctrl_flag == TRAIN_DONE:
                    break

                if ctrl_flag == TRAIN_MB:
                    # 收 mb indices，从缓存中切片
                    mb_size = int(ctrl[1].item())
                    mb_indices = torch.zeros(mb_size, dtype=torch.int64, device="cuda")
                    dist.recv(mb_indices, src=0)

                    train_tensors = {k: v[mb_indices] for k, v in train_cache.items()}
                    train_data = TensorDict(train_tensors, batch_size=[mb_size])

                    # 统一调用 engine.train_batch() 处理 forward/backward/optimizer step
                    train_result = engine.train_batch(train_data, tail_loss_fn)

                    # 将 loss/metrics 回传给 Head
                    loss_val = train_result.get("loss", [0.0])[0] if train_result else 0.0
                    metrics = train_result.get("metrics", {})
                    metrics_tensor = torch.tensor([
                        float(loss_val),
                        float(metrics.get("clipfrac", 0.0)),
                        float(metrics.get("approx_kl", 0.0)),
                    ], dtype=torch.float32, device="cuda")
                    dist.send(metrics_tensor, dst=0)

    # Checkpoint（placeholder）
    print(f"[rank{rank}] v4 3-stage pipeline engine initialized", flush=True)

    if engine.is_head:
        header = torch.tensor([SHUTDOWN, 0, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
        engine.transport.send(header)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
