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

"""Frozen middle stage (stage_1) that forwards activations and gradients."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .transport import FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE, _DTYPE_MAP


class Stage1:
    """Stage_1 (Cloud Body): frozen middle layers that relay between stage_0 and stage_2.

    Supports multi-rank pipeline-split: when stage_1 has multiple ranks, each
    rank handles a subset of layers.  ``src_rank`` is where activations come
    from (stage_0 or previous stage_1 rank), ``dst_rank`` is where they go
    (stage_2 or next stage_1 rank).
    """

    def __init__(self, layers, rotary_emb, device, src_rank=None, dst_rank=None):
        self.layers = nn.ModuleList(layers)
        self.rotary_emb = rotary_emb
        self.device = torch.device(device)
        self.src_rank = src_rank
        self.dst_rank = dst_rank

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

    def forward(self, h: Tensor, **kwargs):
        for layer in self.layers:
            h = self._call_layer(layer, h, **kwargs)
        return h

    def _recv_tensor(self, shape, dtype, src, label=""):
        return self._timed_recv(shape, dtype, src, label=label)

    def _send_tensor(self, tensor: Tensor, dst, label=""):
        self._timed_send(tensor, dst, label=label)

    def _timed_send(self, tensor: Tensor, dst: int, label: str = "") -> None:
        tensor_bytes = tensor.numel() * tensor.element_size()
        t0 = time.perf_counter()
        dist.send(tensor.contiguous(), dst=dst)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(
            f"STAGE_TRANSPORT send src=? dst={dst} bytes={tensor_bytes} "
            f"shape={list(tensor.shape)} dtype={tensor.dtype} label={label} time_ms={elapsed_ms:.3f}",
            flush=True,
        )

    def _timed_recv(self, shape, dtype, src: int, label: str = ""):
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        t0 = time.perf_counter()
        dist.recv(buf, src=src)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        tensor_bytes = buf.numel() * buf.element_size()
        print(
            f"STAGE_TRANSPORT recv dst=? src={src} bytes={tensor_bytes} "
            f"shape={list(buf.shape)} dtype={buf.dtype} label={label} time_ms={elapsed_ms:.3f}",
            flush=True,
        )
        return buf

    def _run_once_impl(self, topology, blocking_wait_for_header: bool):
        t = topology or {"stage_0": [0], "stage_1": [1], "stage_2": [2]}
        # Use explicit src/dst if set (multi-rank stage_1), otherwise derive from topology.
        recv_rank = self.src_rank if self.src_rank is not None else t["stage_0"][0]
        send_rank = self.dst_rank if self.dst_rank is not None else t["stage_2"][0]
        _VALID_FLAGS = {FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE}

        header = self._timed_recv((6,), torch.int64, recv_rank, label="header")
        flag = int(header[0].item())

        if flag not in _VALID_FLAGS:
            raise RuntimeError(
                f"[Stage1] received illegal flag {flag} from rank {recv_rank}. "
                f"Valid flags: {_VALID_FLAGS}. "
                f"This usually means a stage_0 -> stage_2 direct message leaked into the stage_1 pipeline."
            )

        if flag == SHUTDOWN:
            return False

        self._timed_send(header.clone(), send_rank, label="header")

        if flag == PIPELINE_DONE:
            return True

        B, S, H, has_mask, dtype_code = [int(x) for x in header[1:].tolist()]
        tensor_dtype = _DTYPE_MAP[dtype_code]
        h = self._recv_tensor((B, S, H), tensor_dtype, recv_rank, label="input_h")
        pos_ids = self._recv_tensor((B, S), torch.int64, recv_rank, label="input_pos")
        if has_mask:
            mask = self._recv_tensor((B, 1, S, S), tensor_dtype, recv_rank, label="input_mask")
        else:
            mask = None

        pos_emb = None
        if self.rotary_emb is not None:
            try:
                pos_emb = self.rotary_emb(h, pos_ids)
            except TypeError:
                pos_emb = self.rotary_emb(h, seq_len=S)

        if flag == FWD_ONLY:
            with torch.no_grad():
                pos_emb = None
                if self.rotary_emb is not None:
                    try:
                        pos_emb = self.rotary_emb(h, pos_ids)
                    except TypeError:
                        pos_emb = self.rotary_emb(h, seq_len=S)
                h_out = self.forward(
                    h,
                    position_ids=pos_ids,
                    position_embeddings=pos_emb,
                    attention_mask=mask,
                )
            self._send_tensor(h_out, send_rank, label="output_h")
            self._send_tensor(pos_ids, send_rank, label="output_pos")

        elif flag == FWD_WITH_BWD:
            h_in = h.detach().requires_grad_(True)
            h_out = self.forward(
                h_in,
                position_ids=pos_ids,
                position_embeddings=pos_emb,
                attention_mask=mask,
            )
            self._send_tensor(h_out.detach(), send_rank, label="output_h")
            self._send_tensor(pos_ids, send_rank, label="output_pos")

            grad_out = self._recv_tensor(h_out.shape, h_out.dtype, send_rank, label="grad_out")
            torch.autograd.backward(h_out, grad_out)
            self._send_tensor(h_in.grad, recv_rank, label="grad_input")

        return True

    def run_once(self, topology=None):
        """Run a single forward/backward pipeline iteration.

        Used by the Ray driver, which invokes stage_0/stage_1/stage_2 concurrently.
        """
        return self._run_once_impl(topology, blocking_wait_for_header=True)

    def has_grad(self):
        """Return True if any stage_1 parameter has a gradient (should be False)."""
        for p in self.layers.parameters():
            if p.grad is not None:
                return True
        if self.rotary_emb is not None:
            for p in self.rotary_emb.parameters():
                if p.grad is not None:
                    return True
        return False

    def run(self, topology=None):
        """Infinite relay loop (not used by the Ray driver)."""
        while True:
            should_continue = self._run_once_impl(topology, blocking_wait_for_header=True)
            if not should_continue:
                break
