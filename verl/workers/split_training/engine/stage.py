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

"""Trainable stage containers for stage_0 (head) and stage_2 (tail)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .transport import StageTransport


class Stage(nn.Module):
    """Generic container for a contiguous set of transformer layers."""

    def __init__(self, stage_id: str, layers, device: torch.device | str, trainable: bool = False):
        super().__init__()
        self.stage_id = stage_id
        self.layers = nn.ModuleList(layers)
        self.device = torch.device(device)
        self.trainable = trainable
        self.prev_transport: Optional[StageTransport] = None
        self.next_transport: Optional[StageTransport] = None

    def forward(self, h: Tensor, **kwargs):
        for layer in self.layers:
            h = self._call_layer(layer, h, **kwargs)
        return h

    def _call_layer(self, layer, hidden_states, **kwargs):
        try:
            output = layer(hidden_states, **kwargs)
        except TypeError:
            # Compatibility shim for differing transformers signatures.
            filtered = {k: v for k, v in kwargs.items() if k in ("attention_mask", "position_ids")}
            output = layer(hidden_states, **filtered)
        return output[0] if isinstance(output, tuple) else output

    def trainable_parameters(self):
        if not self.trainable:
            return []
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_parameters(self):
        if self.trainable:
            return []
        return list(self.parameters())


class Stage0(Stage):
    """Stage_0 (Edge Head): embed_tokens + front transformer layers + LoRA."""

    def __init__(self, embed_tokens, layers, rotary_emb, device, lr: float = 1e-4, clip_grad: float = 1.0):
        super().__init__("stage_0", layers, device, trainable=True)
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        self.optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=lr)
        self.clip_grad = clip_grad

    def forward_input(self, input_ids: Tensor, attention_mask: Tensor):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        h = self.embed_tokens(input_ids)
        position_ids = self._build_position_ids(attention_mask)

        try:
            position_embeddings = self.rotary_emb(h, position_ids)
        except TypeError:
            position_embeddings = self.rotary_emb(h, seq_len=input_ids.size(1))

        causal_mask = self._build_causal_mask(attention_mask, h.dtype)

        for layer in self.layers:
            h = self._call_layer(
                layer,
                h,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
            )

        return h, position_ids, position_embeddings, causal_mask

    def _build_position_ids(self, attention_mask: Tensor) -> Tensor:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.clamp(min=0)

    def _build_causal_mask(self, attention_mask: Tensor, dtype: torch.dtype):
        if bool(torch.all(attention_mask == 1)):
            return None
        batch_size, seq_len = attention_mask.shape
        key_mask = attention_mask[:, None, None, :].bool()
        causal = torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool)).view(
            1, 1, seq_len, seq_len
        )
        full_mask = key_mask & causal
        additive_mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=self.device, dtype=dtype)
        additive_mask = additive_mask.masked_fill(~full_mask, torch.finfo(dtype).min)
        return additive_mask

    def trainable_parameters(self):
        params = [p for p in self.embed_tokens.parameters() if p.requires_grad]
        for layer in self.layers:
            params.extend(p for p in layer.parameters() if p.requires_grad)
        return params

    def get_trainable_state_dict(self):
        state = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                state[name] = p.detach().cpu().clone()
        return state

    def load_trainable_state_dict(self, state):
        for name, p in self.named_parameters():
            if name in state:
                p.data.copy_(state[name].to(p.device))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), self.clip_grad)
        self.optimizer.step()
        return 0.0


class Stage2(Stage):
    """Stage_2 (Edge Tail): tail transformer layers + norm + lm_head + LoRA."""

    def __init__(
        self,
        layers,
        norm,
        lm_head,
        device,
        lr: float = 1e-4,
        clip_grad: float = 1.0,
        rotary_emb=None,
    ):
        super().__init__("stage_2", layers, device, trainable=True)
        self.norm = norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb
        self.optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=lr)
        self.clip_grad = clip_grad

    def forward_output(self, h: Tensor, position_ids=None, position_embeddings=None, attention_mask=None, **kwargs):
        h = h.to(self.device)
        if position_embeddings is None and position_ids is not None and self.rotary_emb is not None:
            try:
                position_embeddings = self.rotary_emb(h, position_ids)
            except TypeError:
                position_embeddings = self.rotary_emb(h, seq_len=h.size(1))
        for layer in self.layers:
            h = self._call_layer(
                layer,
                h,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        h = self.norm(h)
        return self.lm_head(h)

    def trainable_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(p for p in layer.parameters() if p.requires_grad)
        params.extend(p for p in self.norm.parameters() if p.requires_grad)
        params.extend(p for p in self.lm_head.parameters() if p.requires_grad)
        return params

    def get_trainable_state_dict(self):
        state = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                state[name] = p.detach().cpu().clone()
        return state

    def load_trainable_state_dict(self, state):
        for name, p in self.named_parameters():
            if name in state:
                p.data.copy_(state[name].to(p.device))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), self.clip_grad)
        self.optimizer.step()
        return 0.0
