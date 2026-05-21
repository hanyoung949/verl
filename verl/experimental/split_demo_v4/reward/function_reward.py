"""规则函数 reward。"""

from __future__ import annotations

import re
from typing import Callable

import torch
from torch import Tensor

from .base import RewardAdapter


def extract_last_number(text: str) -> str:
    """从文本中提取最后一个数字串。

    这是一个非常朴素的示例实现，只适合先验证 demo 通路。
    真正做数学任务时，往往还需要更细致的 normalize 逻辑。
    """

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else ""


def math_exact_match(text: str, meta: dict) -> float:
    """示例规则 reward：只看 response 文本的最后一个数字是否等于答案。"""

    return 1.0 if extract_last_number(text) == str(meta["answer"]) else 0.0


class FunctionReward(RewardAdapter):
    """基于 Python 函数的 reward adapter。"""

    def __init__(self, fn: Callable[[str, dict], float], tokenizer,
                 overlong_cfg=None, max_resp_len=None) -> None:
        self.fn = fn
        self.tokenizer = tokenizer
        self.overlong_cfg = overlong_cfg
        self.max_resp_len = max_resp_len

    def score(
        self,
        sequences: Tensor,
        attention_mask: Tensor,
        metadata: dict,
    ) -> Tensor:
        """只 decode response 部分，然后逐条调用规则函数。

        这里故意不 decode 整个 sequence，避免 prompt 内容污染 reward。
        若启用 overlong penalty，response 长度超过阈值时会额外扣分。
        """

        prompt_len = int(metadata["prompt_len"])
        answers = metadata["answers"]
        response_ids = sequences[:, prompt_len:]
        texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        scores = []
        for text, answer in zip(texts, answers, strict=True):
            scores.append(self.fn(text, {"answer": answer}))

        rewards = torch.tensor(scores, dtype=torch.float32, device=sequences.device)

        # DAPO overlong penalty
        if self.overlong_cfg and self.overlong_cfg.get("enable", False):
            buffer_len = self.overlong_cfg["buffer_len"]
            penalty_factor = self.overlong_cfg["penalty_factor"]
            expected_len = self.max_resp_len - buffer_len

            resp_mask = attention_mask[:, prompt_len:]
            valid_lengths = resp_mask.sum(dim=1).long()

            for i in range(len(valid_lengths)):
                exceed_len = int(valid_lengths[i].item()) - expected_len
                if exceed_len > 0:
                    overlong_reward = min(-exceed_len / buffer_len * penalty_factor, 0)
                    rewards[i] += overlong_reward

        return rewards

