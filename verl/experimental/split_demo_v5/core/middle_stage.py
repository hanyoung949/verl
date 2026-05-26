"""MiddleStage — 冻结的中间 stage，响应 Head 的请求并转发给 Tail。"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .transport import FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE, _DTYPE_MAP


class MiddleStage:
    """Middle stage: 冻结的 middle_layers，Head→Tail 转发循环。"""

    def __init__(self, layers, rotary_emb, device):
        self.layers = nn.ModuleList(layers)
        self.rotary_emb = rotary_emb
        self.device = torch.device(device)

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

    def run(self, topology=None):
        """转发循环: 从 Head 接收 header+data，原样转发给 Tail。

        注意：当前运行实现仅支持单 Head / 单 Middle / 单 Tail。
        topology 格式预留了多 rank 扩展，但多 middle rank 的 PP 调度尚未实现。
        """
        t = topology or {"head": [0], "middle": [1], "tail": [2]}
        head_rank = t["head"][0]
        tail_rank = t["tail"][0]
        _VALID_FLAGS = {FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE}

        while True:
            # 从 Head 接收 header（6-int tensor）
            header = torch.zeros(6, dtype=torch.int64, device=self.device)
            dist.recv(header, src=head_rank)
            flag = int(header[0].item())

            if flag not in _VALID_FLAGS:
                raise RuntimeError(
                    f"[MiddleStage] received illegal flag {flag} from rank {head_rank}. "
                    f"Valid flags: {_VALID_FLAGS}. "
                    f"This usually means a Head→Tail direct message leaked into the Middle pipeline."
                )

            if flag == SHUTDOWN:
                break

            # 原样转发 header 给 Tail
            dist.send(header.clone(), dst=tail_rank)

            if flag == PIPELINE_DONE:
                continue

            # 转发 h + pos_ids + mask
            B, S, H, has_mask, dtype_code = [int(x) for x in header[1:].tolist()]
            tensor_dtype = _DTYPE_MAP[dtype_code]
            h = self._recv_tensor((B, S, H), tensor_dtype, head_rank)
            pos_ids = self._recv_tensor((B, S), torch.int64, head_rank)
            if has_mask:
                mask = self._recv_tensor((B, 1, S, S), tensor_dtype, head_rank)
            else:
                mask = None

            # 重算 position_embeddings
            pos_emb = None
            if self.rotary_emb is not None:
                try:
                    pos_emb = self.rotary_emb(h, pos_ids)
                except TypeError:
                    pos_emb = self.rotary_emb(h, seq_len=S)

            if flag == FWD_ONLY:
                # 推理模式：经过 Middle 层计算（但不需要梯度）
                with torch.no_grad():
                    pos_emb = None
                    if self.rotary_emb is not None:
                        try:
                            pos_emb = self.rotary_emb(h, pos_ids)
                        except TypeError:
                            pos_emb = self.rotary_emb(h, seq_len=S)
                    h_out = self.forward(h, position_ids=pos_ids,
                                       position_embeddings=pos_emb, attention_mask=mask)
                self._send_tensor(h_out, tail_rank)
                self._send_tensor(pos_ids, tail_rank)
                if mask is not None:
                    self._send_tensor(mask, tail_rank)

            elif flag == FWD_WITH_BWD:
                # 训练用：经过 Middle 层计算，保留计算图用于反向
                h_in = h.detach().requires_grad_(True)
                h_out = self.forward(h_in, position_ids=pos_ids,
                                   position_embeddings=pos_emb, attention_mask=mask)
                self._send_tensor(h_out.detach(), tail_rank)
                self._send_tensor(pos_ids, tail_rank)
                if mask is not None:
                    self._send_tensor(mask, tail_rank)

                # 从 Tail 接收 grad_h_out
                grad_out = self._recv_tensor(h_out.shape, h_out.dtype, tail_rank)
                # 反向传播
                torch.autograd.backward(h_out, grad_out)
                # 把 grad_h_in 发给 Head
                self._send_tensor(h_in.grad, head_rank)
