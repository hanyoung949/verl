"""朴素 rollout 实现。

这个版本不做 KV cache，也不做 speculative decoding。
它的意义不是快，而是：
1. 接口完整
2. 行为可控
3. 易于验证 split actor 的正确性
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import RolloutOutput, SplitRolloutBackend


class NaiveSplitRollout(SplitRolloutBackend):
    """最朴素的 rollout 后端。"""

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if self.pad_token_id is None:
            # 很多开源 CausalLM 默认没有 pad token，这里直接回退到 eos。
            self.pad_token_id = self.eos_token_id

    def _sample_next_token(self, logits: Tensor) -> Tensor:
        """对最后一步 logits 做采样。"""

        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)

        # 当前先实现最基础的 top-p。top_p=1.0 时等价于不裁剪。
        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > self.top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_in_sorted)
            return next_token

        return torch.multinomial(probs, num_samples=1)

    def generate(
        self,
        prompt_ids: Tensor,
        group_size: int,
        max_new_tokens: int,
        attention_mask: Optional[Tensor] = None,
    ) -> RolloutOutput:
        """执行朴素自回归生成。

        这里虽然调用的是传入的 `model`，但接口层面并不要求 rollout 一定和训练实例是同一个对象。
        当前 demo 为了节省实现复杂度，允许 main 脚本先把同一个 core 注入进来。
        """

        # rollout 的输入 prompt 很可能来自 tokenizer，默认还在 CPU。
        # 这里统一把它搬到训练主卡，避免后面和 GPU 上采样出来的 token 拼接时报设备不一致。
        device = torch.device(getattr(self.model, "primary_device", prompt_ids.device))
        prompt_ids = prompt_ids.to(device)
        batch_size, prompt_len = prompt_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        # 将每条 prompt 复制 G 份，用于 GRPO 的 group sampling。
        prompt_ids = prompt_ids.unsqueeze(1).repeat(1, group_size, 1).reshape(batch_size * group_size, prompt_len)
        attention_mask = (
            attention_mask.unsqueeze(1).repeat(1, group_size, 1).reshape(batch_size * group_size, prompt_len)
        )

        generated_tokens: list[Tensor] = []
        generated_masks: list[Tensor] = []
        finished = torch.zeros(prompt_ids.size(0), dtype=torch.bool, device=device)

        # 先做一次 prefill，拿到第一个位置的 logits。
        prefill_logits = self.model.prefill(prompt_ids, attention_mask)
        next_token = self._sample_next_token(prefill_logits[:, -1, :])
        current_valid = (~finished).float().unsqueeze(-1)
        generated_tokens.append(next_token)
        generated_masks.append(current_valid)
        finished = finished | (next_token.squeeze(-1) == self.eos_token_id)

        current_ids = torch.cat([prompt_ids, next_token], dim=-1)
        current_mask = torch.cat([attention_mask, current_valid.long()], dim=-1)

        for _ in range(max_new_tokens - 1):
            next_logits = self.model.decode_step(current_ids, current_mask)
            sampled = self._sample_next_token(next_logits)

            # 已经结束的序列后续全部补 pad，确保 shape 整齐。
            sampled = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(sampled, self.pad_token_id),
                sampled,
            )

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
        full_attention_mask = torch.cat([attention_mask, response_mask], dim=-1)

        return RolloutOutput(
            sequences=sequences,
            attention_mask=full_attention_mask,
            response_ids=response_ids,
            response_mask=response_mask,
            prompt_len=prompt_len,
        )
