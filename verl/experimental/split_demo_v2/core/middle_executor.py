"""NCCL 中段执行器。

rank0 侧使用。通过 NCCL P2P 将 hidden states 发送到 rank1，
由 rank1 的 MiddleWorker 执行 middle layers，结果返回 rank0。

可微性通过 _NCCLMiddleFunction(torch.autograd.Function) 保证：
- forward: 发送 h_front，接收 h_middle
- backward: 发送 grad_h_middle，接收 grad_h_front
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


# 协议常量
FWD_ONLY = 0
FWD_WITH_BWD = 1
SHUTDOWN = 2

# dtype 编码：与 rank1 的 _DTYPE_MAP 对应
_DTYPE_CODE = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}


class _NCCLMiddleFunction(torch.autograd.Function):
    """跨进程 middle 执行的 autograd Function。

    forward: 把 h_front 发给 rank1，接收 h_middle。
    backward: 把 grad_h_middle 发给 rank1，rank1 做本地 backward 后把 grad_h_front 发回。
    """

    @staticmethod
    def forward(ctx, h_front, position_ids, attention_mask, executor):
        # type: (Any, Tensor, Tensor, Optional[Tensor], NCCLMiddleExecutor) -> Tensor
        ctx.executor = executor

        executor._send_fwd_request(
            h_front, position_ids, attention_mask, flag=FWD_WITH_BWD,
        )
        h_middle = executor._recv_tensor(h_front.shape, h_front.dtype)
        return h_middle

    @staticmethod
    def backward(ctx, grad_h_middle):
        # type: (Any, Tensor) -> tuple[Tensor, None, None, None]
        grad_h_front = ctx.executor._exchange_backward(grad_h_middle)
        return grad_h_front, None, None, None


class NCCLMiddleExecutor:
    """通过 NCCL P2P 通信执行中段层的执行器。

    替代 v1 的 LocalMiddleExecutor（tensor.to('cuda:1')），
    实现真正的跨进程 middle 执行。
    """

    def __init__(self, middle_rank: int = 1, device: str = "cuda") -> None:
        self.middle_rank = middle_rank
        self.device = torch.device(device)
        self._cached_mask_shape: Optional[tuple[int, ...]] = None
        self.reset_stats()

    # ── 传输统计（与 v1 LocalMiddleExecutor.stats 接口对齐） ──────

    @staticmethod
    def _tensor_nbytes(tensor: Optional[Tensor]) -> int:
        if tensor is None:
            return 0
        return tensor.numel() * tensor.element_size()

    def reset_stats(self) -> None:
        self.stats = {
            "calls": 0,
            "forward_to_middle_bytes": 0,
            "position_ids_bytes": 0,
            "position_embeddings_bytes": 0,
            "attention_mask_bytes": 0,
            "last_hidden_shape": None,
            "last_position_ids_shape": None,
            "last_attention_mask_shape": None,
        }

    def get_stats(self) -> dict:
        return dict(self.stats)

    # ── 底层 NCCL 操作 ─────────────────────────────────────────

    def _send_tensor(self, tensor: Tensor) -> None:
        """发送一个 contiguous tensor 到 middle_rank。"""
        dist.send(tensor.contiguous(), dst=self.middle_rank)

    def _recv_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        """从 middle_rank 接收一个 tensor。"""
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=self.middle_rank)
        return buf

    def _send_fwd_request(
        self,
        h_front: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor],
        flag: int,
    ) -> None:
        """发送一次 forward 请求的完整数据（header + tensors）。"""

        B, S, H = h_front.shape
        has_mask = 0 if attention_mask is None else 1
        dtype_code = _DTYPE_CODE.get(h_front.dtype, 0)

        header = torch.tensor(
            [flag, B, S, H, has_mask, dtype_code],
            dtype=torch.int64,
            device=self.device,
        )
        self._send_tensor(header)
        self._send_tensor(h_front)
        self._send_tensor(position_ids)

        if attention_mask is not None:
            self._send_tensor(attention_mask)
            self._cached_mask_shape = tuple(attention_mask.shape)
        else:
            self._cached_mask_shape = None

    def _exchange_backward(self, grad_h_middle: Tensor) -> Tensor:
        """发送 grad_h_middle 给 rank1，接收 grad_h_front 回来。"""
        self._send_tensor(grad_h_middle)
        grad_h_front = self._recv_tensor(grad_h_middle.shape, grad_h_middle.dtype)
        return grad_h_front

    # ── 公开接口 ───────────────────────────────────────────────

    def execute(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        position_embeddings: Optional[tuple[Tensor, Tensor]],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """执行中段层。根据当前是否在梯度上下文中选择 FWD_ONLY 或 FWD_WITH_BWD。

        注意：调用方必须在不做 backward 的路径（rollout / old_log_prob）上显式
        使用 torch.no_grad()。否则 rank1 会走 FWD_WITH_BWD 分支并等待 grad，
        导致死锁。
        """

        self.stats["calls"] += 1
        self.stats["last_hidden_shape"] = tuple(hidden_states.shape)
        self.stats["last_position_ids_shape"] = tuple(position_ids.shape)
        self.stats["last_attention_mask_shape"] = None if attention_mask is None else tuple(attention_mask.shape)
        self.stats["forward_to_middle_bytes"] += self._tensor_nbytes(hidden_states)
        self.stats["position_ids_bytes"] += self._tensor_nbytes(position_ids)
        # position_embeddings 不再跨卡发送，rank1 本地重算
        self.stats["attention_mask_bytes"] += self._tensor_nbytes(attention_mask)

        if torch.is_grad_enabled():
            return _NCCLMiddleFunction.apply(
                hidden_states, position_ids, attention_mask, self,
            )
        else:
            return self._fwd_only(hidden_states, position_ids, attention_mask)

    def _fwd_only(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """FWD_ONLY 模式：只做前向，不缓存上下文，不等 backward。"""
        self._send_fwd_request(
            hidden_states, position_ids, attention_mask, flag=FWD_ONLY,
        )
        return self._recv_tensor(hidden_states.shape, hidden_states.dtype)

    def shutdown(self) -> None:
        """发送 SHUTDOWN 信号给 rank1，让其退出 worker loop。"""
        header = torch.tensor(
            [SHUTDOWN, 0, 0, 0, 0, 0],
            dtype=torch.int64,
            device=self.device,
        )
        dist.send(header, dst=self.middle_rank)
