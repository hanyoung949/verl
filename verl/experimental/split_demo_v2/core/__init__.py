"""模型执行核心层（NCCL 版）。"""

from .middle_executor import FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, NCCLMiddleExecutor
from .split_actor import SplitActorCore

__all__ = [
    "NCCLMiddleExecutor",
    "SplitActorCore",
    "FWD_ONLY",
    "FWD_WITH_BWD",
    "SHUTDOWN",
]
