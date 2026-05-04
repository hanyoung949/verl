"""reward 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class RewardAdapter(ABC):
    """打分策略抽象接口。"""

    @abstractmethod
    def score(
        self,
        sequences: Tensor,
        attention_mask: Tensor,
        metadata: dict,
    ) -> Tensor:
        """对每条序列输出一个标量 reward。"""

