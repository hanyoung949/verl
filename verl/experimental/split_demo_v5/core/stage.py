"""Stage — 流水线中的通用 stage 容器。

每个 Stage 持有一组层，放在一张 GPU 上，和相邻 Stage 通过 StageTransport 通信。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .transport import StageTransport


class Stage(nn.Module):
    """流水线中的一个通用 stage。"""

    def __init__(self, stage_id, layers, device, trainable=False):
        super().__init__()
        self.stage_id = stage_id
        self.layers = nn.ModuleList(layers)
        self.device = torch.device(device)
        self.trainable = trainable
        self.prev_transport: Optional[StageTransport] = None
        self.next_transport: Optional[StageTransport] = None

    def forward(self, h, **kwargs):
        """前向计算。"""
        for layer in self.layers:
            h = self._call_layer(layer, h, **kwargs)
        return h

    def _call_layer(self, layer, hidden_states, **kwargs):
        try:
            output = layer(hidden_states, **kwargs)
        except TypeError:
            # 兼容不同 transformers 版本
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


class HeadStage(Stage):
    """Head stage: 包含 embed_tokens + front_layers + rotary_emb + optimizer。"""

    def __init__(self, embed_tokens, layers, rotary_emb, device, lr=1e-4, clip_grad=1.0):
        super().__init__("head", layers, device, trainable=True)
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        self.optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=lr)
        self.clip_grad = clip_grad

    def forward_input(self, input_ids, attention_mask):
        """从 input_ids 开始前向，返回 (h, position_ids, position_embeddings, causal_mask)。"""
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
            h = self._call_layer(layer, h,
                               position_ids=position_ids,
                               position_embeddings=position_embeddings,
                               attention_mask=causal_mask)

        return h, position_ids, position_embeddings, causal_mask

    def _build_position_ids(self, attention_mask):
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.clamp(min=0)

    def _build_causal_mask(self, attention_mask, dtype):
        if bool(torch.all(attention_mask == 1)):
            return None
        batch_size, seq_len = attention_mask.shape
        key_mask = attention_mask[:, None, None, :].bool()
        causal = torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
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


class Qwen35MTPDraftHead(nn.Module):
    """Qwen3.5 MTP draft head loaded from `mtp.*` checkpoint weights."""

    def __init__(self, embed_tokens, pre_fc_norm_embedding, pre_fc_norm_hidden, fc, layer, norm, lm_head):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.pre_fc_norm_embedding = pre_fc_norm_embedding
        self.pre_fc_norm_hidden = pre_fc_norm_hidden
        self.fc = fc
        self.layer = layer
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, input_ids, hidden_states, position_ids=None, position_embeddings=None,
                attention_mask=None, return_hidden=False):
        input_ids = input_ids.to(hidden_states.device)
        token_embeddings = self.embed_tokens(input_ids)
        if token_embeddings.size(1) != hidden_states.size(1):
            token_embeddings = token_embeddings[:, -hidden_states.size(1):]
            if position_ids is not None:
                position_ids = position_ids[:, -hidden_states.size(1):]

        h_emb = self.pre_fc_norm_embedding(token_embeddings)
        h_hidden = self.pre_fc_norm_hidden(hidden_states)
        h = self.fc(torch.cat([h_emb, h_hidden], dim=-1))
        h = self._call_layer(h, position_ids=position_ids,
                             position_embeddings=position_embeddings,
                             attention_mask=attention_mask)
        logits = self.lm_head(self.norm(h))
        if return_hidden:
            return logits, h
        return logits

    def _call_layer(self, hidden_states, **kwargs):
        try:
            output = self.layer(hidden_states, **kwargs)
        except TypeError:
            filtered = {k: v for k, v in kwargs.items() if k in ("attention_mask", "position_ids")}
            output = self.layer(hidden_states, **filtered)
        return output[0] if isinstance(output, tuple) else output


class TailStage(Stage):
    """Tail stage: 包含 tail_layers + norm + lm_head + optimizer。"""

    def __init__(self, layers, norm, lm_head, device, lr=1e-4, clip_grad=1.0, rotary_emb=None,
                 mtp_head=None, mtp_trainable=False):
        super().__init__("tail", layers, device, trainable=True)
        self.norm = norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb
        self.mtp_head = mtp_head
        self.mtp_trainable = mtp_trainable
        if self.mtp_head is not None:
            for name, p in self.mtp_head.named_parameters():
                if name.startswith(("embed_tokens.", "lm_head.")):
                    continue
                p.requires_grad = mtp_trainable
        self.optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=lr)
        self.clip_grad = clip_grad

    def forward_output(self, h, position_ids=None, position_embeddings=None, attention_mask=None,
                       return_hidden=False, **kwargs):
        """从 middle 输出继续前向，返回 logits；MTP rollout 需要 pre-norm hidden。"""
        h = h.to(self.device)
        if position_embeddings is None and position_ids is not None and self.rotary_emb is not None:
            try:
                position_embeddings = self.rotary_emb(h, position_ids)
            except TypeError:
                position_embeddings = self.rotary_emb(h, seq_len=h.size(1))
        for layer in self.layers:
            h = self._call_layer(layer, h,
                               position_ids=position_ids,
                               position_embeddings=position_embeddings,
                               attention_mask=attention_mask)
        raw_hidden = h
        logits = self.lm_head(self.norm(h))
        if return_hidden:
            return logits, raw_hidden
        return logits

    def draft_mtp_logits(self, input_ids, hidden_states, position_ids=None,
                         position_embeddings=None, attention_mask=None, return_hidden=False):
        if self.mtp_head is None:
            raise RuntimeError("MTP rollout is enabled, but TailStage has no MTP draft head.")
        return self.mtp_head(input_ids=input_ids, hidden_states=hidden_states,
                             position_ids=position_ids,
                             position_embeddings=position_embeddings,
                             attention_mask=attention_mask,
                             return_hidden=return_hidden)

    def trainable_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(p for p in layer.parameters() if p.requires_grad)
        params.extend(p for p in self.norm.parameters() if p.requires_grad)
        params.extend(p for p in self.lm_head.parameters() if p.requires_grad)
        if self.mtp_trainable and self.mtp_head is not None:
            params.extend(p for p in self.mtp_head.parameters() if p.requires_grad)
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
