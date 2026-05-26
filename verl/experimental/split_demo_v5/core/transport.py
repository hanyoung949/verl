"""StageTransport — 两个相邻 stage 之间的 NCCL P2P 通信。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


# 通信协议常量
FWD_ONLY = 0
FWD_WITH_BWD = 1
SHUTDOWN = 2
PIPELINE_DONE = 3
TRAIN_MB = 4      # Head 通知 Tail：开始一个 mini-batch
TRAIN_DONE = 5    # Head 通知 Tail：训练结束
STEP_SKIP = 6     # Head 通知 Tail：本 step 跳过训练（dynamic_sampling 全过滤）
TRAIN_START = 7   # Head 通知 Tail：开始接收训练数据

_DTYPE_CODE = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}

_DTYPE_MAP = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}


class StageTransport:
    """两个相邻 stage 之间的 NCCL P2P 通信。"""

    def __init__(self, local_rank, remote_rank, device):
        self.local_rank = local_rank
        self.remote_rank = remote_rank
        self.device = torch.device(device)

    def send(self, tensor):
        """发送 tensor 到 remote rank。"""
        dist.send(tensor.contiguous(), dst=self.remote_rank)

    def recv(self, shape, dtype):
        """从 remote rank 接收 tensor。"""
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=self.remote_rank)
        return buf

    def send_int(self, value):
        """发送一个整数。"""
        t = torch.tensor([value], dtype=torch.int64, device=self.device)
        dist.send(t, dst=self.remote_rank)

    def recv_int(self):
        """接收一个整数。"""
        t = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(t, src=self.remote_rank)
        return t.item()
