# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""StageTransport — NCCL P2P between adjacent split stages."""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


# Control flags
FWD_ONLY = 0
FWD_WITH_BWD = 1
SHUTDOWN = 2
PIPELINE_DONE = 3
TRAIN_MB = 4
TRAIN_DONE = 5
STEP_SKIP = 6
TRAIN_START = 7

_DTYPE_CODE = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}

_DTYPE_MAP = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}


class StageTransport:
    """Thin NCCL P2P wrapper between two adjacent split stages."""

    def __init__(self, local_rank: int, remote_rank: int, device: torch.device | str):
        self.local_rank = local_rank
        self.remote_rank = remote_rank
        self.device = torch.device(device)

    def send(self, tensor: Tensor) -> None:
        tensor_bytes = tensor.numel() * tensor.element_size()
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        dist.send(tensor.contiguous(), dst=self.remote_rank)
        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(
            f"STAGE_TRANSPORT send local_rank={self.local_rank} remote_rank={self.remote_rank} "
            f"bytes={tensor_bytes} shape={list(tensor.shape)} dtype={tensor.dtype} time_ms={elapsed_ms:.3f}",
            flush=True,
        )

    def recv(self, shape, dtype):
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        dist.recv(buf, src=self.remote_rank)
        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        tensor_bytes = buf.numel() * buf.element_size()
        print(
            f"STAGE_TRANSPORT recv local_rank={self.local_rank} remote_rank={self.remote_rank} "
            f"bytes={tensor_bytes} shape={list(buf.shape)} dtype={buf.dtype} time_ms={elapsed_ms:.3f}",
            flush=True,
        )
        return buf

    def send_int(self, value: int) -> None:
        t = torch.tensor([value], dtype=torch.int64, device=self.device)
        dist.send(t, dst=self.remote_rank)

    def recv_int(self) -> int:
        t = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(t, src=self.remote_rank)
        return t.item()
