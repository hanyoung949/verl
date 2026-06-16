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

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .transport import FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE, _DTYPE_MAP


class Stage1:
    """Stage_1 (Cloud Body): frozen middle layers that relay between stage_0 and stage_2."""

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

    def forward(self, h: Tensor, **kwargs):
        for layer in self.layers:
            h = self._call_layer(layer, h, **kwargs)
        return h

    def _recv_tensor(self, shape, dtype, src):
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=src)
        return buf

    def _send_tensor(self, tensor: Tensor, dst):
        dist.send(tensor.contiguous(), dst=dst)

    def _run_once_impl(self, topology, blocking_wait_for_header: bool):
        t = topology or {"stage_0": [0], "stage_1": [1], "stage_2": [2]}
        stage_0_rank = t["stage_0"][0]
        stage_2_rank = t["stage_2"][0]
        _VALID_FLAGS = {FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE}

        header = torch.zeros(6, dtype=torch.int64, device=self.device)
        dist.recv(header, src=stage_0_rank)
        flag = int(header[0].item())

        if flag not in _VALID_FLAGS:
            raise RuntimeError(
                f"[Stage1] received illegal flag {flag} from rank {stage_0_rank}. "
                f"Valid flags: {_VALID_FLAGS}. "
                f"This usually means a stage_0 -> stage_2 direct message leaked into the stage_1 pipeline."
            )

        if flag == SHUTDOWN:
            return False

        dist.send(header.clone(), dst=stage_2_rank)

        if flag == PIPELINE_DONE:
            return True

        B, S, H, has_mask, dtype_code = [int(x) for x in header[1:].tolist()]
        tensor_dtype = _DTYPE_MAP[dtype_code]
        h = self._recv_tensor((B, S, H), tensor_dtype, stage_0_rank)
        pos_ids = self._recv_tensor((B, S), torch.int64, stage_0_rank)
        if has_mask:
            mask = self._recv_tensor((B, 1, S, S), tensor_dtype, stage_0_rank)
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
            self._send_tensor(h_out, stage_2_rank)
            self._send_tensor(pos_ids, stage_2_rank)

        elif flag == FWD_WITH_BWD:
            h_in = h.detach().requires_grad_(True)
            h_out = self.forward(
                h_in,
                position_ids=pos_ids,
                position_embeddings=pos_emb,
                attention_mask=mask,
            )
            self._send_tensor(h_out.detach(), stage_2_rank)
            self._send_tensor(pos_ids, stage_2_rank)

            grad_out = self._recv_tensor(h_out.shape, h_out.dtype, stage_2_rank)
            torch.autograd.backward(h_out, grad_out)
            self._send_tensor(h_in.grad, stage_0_rank)

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
