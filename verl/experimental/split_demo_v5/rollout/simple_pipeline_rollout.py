"""3-stage pipeline 的 rollout 实现。

通信路径:
    Head → Middle → Tail（前向）
    Tail → Head（next_token + log_prob + token_mask 直接回传）
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from ..core.transport import PIPELINE_DONE
from .base import BaseRolloutBackend


class SimplePipelineRollout(BaseRolloutBackend):
    def __init__(self, engine, tokenizer, temperature=1.0, top_p=1.0):
        self.engine = engine
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def _pipeline_forward(self, data):
        """Head→Middle→Tail 前向，Tail 本地采样后回传 accepted tokens。"""
        if self.engine.is_head:
            return self.engine.sample_next_token(
                data, temperature=self.temperature, pad_token_id=self.pad_token_id)
        elif self.engine.is_middle:
            self.engine.stage.run()
            return None
        # Tail 分支不会走到这里：generate() / run_tail_loop() 在 Tail rank 有各自的处理。

    def generate(self, prompt_ids, group_size, max_new_tokens, attention_mask=None):
        device = torch.device("cuda")
        prompt_ids = prompt_ids.to(device)
        batch_size, prompt_len = prompt_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        prompt_ids = prompt_ids.unsqueeze(1).repeat(1, group_size, 1).reshape(
            batch_size * group_size, prompt_len)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, group_size, 1).reshape(
            batch_size * group_size, prompt_len)

        finished = torch.zeros(prompt_ids.size(0), dtype=torch.bool, device=device)

        generated_tokens = []
        generated_log_probs = []
        generated_masks = []
        current_ids = prompt_ids
        current_mask = attention_mask
        generated_len = 0

        while generated_len < max_new_tokens:
            decode_data = TensorDict({
                "input_ids": current_ids,
                "attention_mask": current_mask,
                "finished": finished,
            }, batch_size=current_ids.shape[0])
            result = self._pipeline_forward(decode_data)
            sampled = result["next_token"]
            step_log_prob = result["log_prob"]
            token_mask = result.get("token_mask")
            if token_mask is None:
                token_mask = (~finished).long().unsqueeze(-1).expand_as(sampled)

            remaining = max_new_tokens - generated_len
            if sampled.size(1) > remaining:
                sampled = sampled[:, :remaining]
                step_log_prob = step_log_prob[:, :remaining]
                token_mask = token_mask[:, :remaining]

            generated_tokens.append(sampled)
            generated_log_probs.append(step_log_prob)
            generated_masks.append(token_mask.long())
            finished = finished | (((sampled == self.eos_token_id) & token_mask.bool()).any(dim=-1))
            current_ids = torch.cat([current_ids, sampled], dim=-1)
            current_mask = torch.cat([current_mask, token_mask.long()], dim=-1)
            generated_len += sampled.size(1)
            if bool(torch.all(finished)):
                break

        response_ids = torch.cat(generated_tokens, dim=-1)
        response_log_probs = torch.cat(generated_log_probs, dim=-1)
        response_mask = torch.cat(generated_masks, dim=-1).long()
        sequences = torch.cat([prompt_ids, response_ids], dim=-1)
        full_mask = torch.cat([attention_mask, response_mask], dim=-1)

        # Head 端 rollout 完成后，发送 PIPELINE_DONE 让 Tail 退出服务循环
        header = torch.tensor(
            [PIPELINE_DONE, 0, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
        self.engine.transport.send(header)

        class Out:
            pass

        out = Out()
        out.sequences = sequences
        out.attention_mask = full_mask
        out.response_ids = response_ids
        out.response_mask = response_mask
        out.old_log_probs = response_log_probs
        out.prompt_len = prompt_len
        return out

    def run_tail_loop(self):
        while True:
            result = self.engine.sample_next_token(TensorDict({}, batch_size=[0]))
            if not result or "next_token" not in result:
                break
