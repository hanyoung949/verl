"""3-stage pipeline 的 rollout 实现。

通信路径:
    Head → Middle → Tail（前向）
    Tail → Head（next_token + log_prob 直接回传）
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
        """Head→Middle→Tail 前向，Tail 本地采样后回传 next_token + log_prob。"""
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
