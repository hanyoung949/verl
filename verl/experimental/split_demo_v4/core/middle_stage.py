"""MiddleStage — 冻结的中间 stage，响应 Head 和 Tail 的请求。"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .transport import FWD_ONLY, FWD_WITH_BWD, SHUTDOWN


class MiddleStage:
    """Middle stage: 冻结的 middle_layers，纯请求响应循环。"""

    def __init__(self, layers, rotary_emb, device):
        self.layers = nn.ModuleList(layers)
        self.rotary_emb = rotary_emb
        self.device = torch.device(device)

        # 冻结所有参数
        for p in self.layers.parameters():
            p.requires_grad = False
        if self.rotary_emb is not None:
            for p in self.rotary_emb.parameters():
                p.requires_grad = False
        self.layers.to(self.device)
        if self.rotary_emb is not None:
            self.rotary_emb.to(self.device)

    def _call_layer(self, layer, h, **kwargs):
        try:
            output = layer(h, **kwargs)
        except TypeError:
            filtered = {k: v for k, v in kwargs.items() if k in ("attention_mask", "position_ids")}
            output = layer(h, **filtered)
        return output[0] if isinstance(output, tuple) else output

    def forward(self, h, **kwargs):
        for layer in self.layers:
            h = self._call_layer(layer, h, **kwargs)
        return h

    def _recv_tensor(self, shape, dtype, src):
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=src)
        return buf

    def _send_tensor(self, tensor, dst):
        dist.send(tensor.contiguous(), dst=dst)

    def _recv_int(self, src):
        t = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(t, src=src)
        return t.item()

    def run(self):
        """请求响应循环。从前一个 stage 接收，处理后发给下一个 stage。"""
        prev_rank = 0  # Head
        next_rank = 2  # Tail

        while True:
            # 从 Head 接收 header
            flag = self._recv_int(prev_rank)
            if flag == SHUTDOWN:
                break

            B = self._recv_int(prev_rank)
            S = self._recv_int(prev_rank)
            H = self._recv_int(prev_rank)
            has_pos_ids = self._recv_int(prev_rank)
            has_mask = self._recv_int(prev_rank)
            dtype_code = self._recv_int(prev_rank)
            tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]

            # 从 Head 接收数据
            h_in = self._recv_tensor((B, S, H), tensor_dtype, prev_rank)
            position_ids = self._recv_tensor((B, S), torch.int64, prev_rank) if has_pos_ids else None
            attention_mask = self._recv_tensor((B, 1, S, S), tensor_dtype, prev_rank) if has_mask else None

            # 本地重算 pos_emb
            pos_emb = None
            if self.rotary_emb is not None and position_ids is not None:
                try:
                    pos_emb = self.rotary_emb(h_in, position_ids)
                except TypeError:
                    pos_emb = self.rotary_emb(h_in, seq_len=S)

            if flag == FWD_ONLY:
                with torch.no_grad():
                    h_out = self.forward(h_in, position_ids=position_ids,
                                       position_embeddings=pos_emb, attention_mask=attention_mask)
                # 发给 Tail
                self._send_tensor(h_out, next_rank)

            elif flag == FWD_WITH_BWD:
                h_in_detached = h_in.detach().requires_grad_(True)
                h_out = self.forward(h_in_detached, position_ids=position_ids,
                                   position_embeddings=pos_emb, attention_mask=attention_mask)
                # 发给 Tail
                self._send_tensor(h_out.detach(), next_rank)

                # 等 Tail 的 grad 回来
                grad_out = self._recv_tensor(h_out.shape, h_out.dtype, next_rank)
                torch.autograd.backward(h_out, grad_out)

                # 把 grad 发给 Head
                self._send_tensor(h_in_detached.grad, prev_rank)
