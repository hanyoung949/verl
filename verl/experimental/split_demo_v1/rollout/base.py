"""rollout 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from torch import Tensor


@dataclass
class RolloutOutput:
    """rollout 输出结构。

    使用 dataclass 的原因：
    - 字段访问更安全，不依赖字符串 key；
    - 后续扩展 speculative / cache 信息时，加 Optional 字段即可；
    - trainer 对新增字段天然向后兼容。
    """

    sequences: Tensor
    attention_mask: Tensor
    response_ids: Tensor
    response_mask: Tensor
    prompt_len: int
    draft_acceptance_rate: Optional[float] = None
    kv_cache: Optional[Any] = None


class SplitRolloutBackend(ABC):
    """rollout 抽象接口。"""

    @abstractmethod
    def generate(
        self,
        prompt_ids: Tensor,
        group_size: int,
        max_new_tokens: int,
        attention_mask: Optional[Tensor] = None,
    ) -> RolloutOutput:
        """根据 prompt 生成完整 response。"""

