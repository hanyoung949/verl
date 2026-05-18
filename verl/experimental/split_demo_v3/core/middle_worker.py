"""Middle Worker — rank1 侧的纯请求响应循环。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

from .middle_executor import FWD_ONLY, FWD_WITH_BWD, SHUTDOWN

_DTYPE_MAP = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}


class MiddleWorker:
    def __init__(self, middle_layers, rotary_emb, device, tensor_dtype):
        self.middle_layers = middle_layers
        self.rotary_emb = rotary_emb
        self.device = device
        self.tensor_dtype = tensor_dtype

    @classmethod
    def from_pretrained(cls, model_path, front_end, middle_end, lora_config, device, torch_dtype=torch.bfloat16):
        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        peft_model = get_peft_model(base_model, lora_config)
        causal_lm = _unwrap_causal_lm(peft_model)
        decoder = _find_decoder_stack(peft_model, causal_lm)
        layers = list(decoder.layers)
        middle_layers = nn.ModuleList(layers[front_end:middle_end])
        rotary_emb = getattr(decoder, "rotary_emb", None)
        for p in middle_layers.parameters():
            p.requires_grad = False
        if rotary_emb is not None:
            for p in rotary_emb.parameters():
                p.requires_grad = False
        middle_layers.to(device)
        if rotary_emb is not None:
            rotary_emb.to(device)
        return cls(middle_layers=middle_layers, rotary_emb=rotary_emb, device=device, tensor_dtype=torch_dtype)

    def _recv_tensor(self, shape, dtype):
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=0)
        return buf

    def _send_tensor(self, tensor):
        dist.send(tensor.contiguous(), dst=0)

    def _call_layer(self, layer, hidden_states, position_ids, position_embeddings, attention_mask):
        try:
            output = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                          position_embeddings=position_embeddings, use_cache=False)
        except TypeError:
            try:
                output = layer(hidden_states, attention_mask=attention_mask,
                              position_embeddings=position_embeddings, use_cache=False)
            except TypeError:
                output = layer(hidden_states, attention_mask=attention_mask,
                              position_ids=position_ids, use_cache=False)
        return output[0] if isinstance(output, tuple) else output

    def _run_layers(self, h, position_ids, position_embeddings, attention_mask):
        for layer in self.middle_layers:
            h = self._call_layer(layer, h, position_ids, position_embeddings, attention_mask)
        return h

    def run(self):
        while True:
            header = torch.zeros(6, dtype=torch.int64, device=self.device)
            dist.recv(header, src=0)
            flag, B, S, H, has_mask, dtype_code = header.tolist()
            if flag == SHUTDOWN:
                break
            tensor_dtype = _DTYPE_MAP[int(dtype_code)]
            h_front = self._recv_tensor((B, S, H), tensor_dtype)
            position_ids = self._recv_tensor((B, S), torch.int64)
            attention_mask = self._recv_tensor((B, 1, S, S), tensor_dtype) if has_mask else None
            cos_sin = None
            if self.rotary_emb is not None:
                try:
                    cos_sin = self.rotary_emb(h_front, position_ids)
                except TypeError:
                    cos_sin = self.rotary_emb(h_front, seq_len=S)
            if flag == FWD_ONLY:
                with torch.no_grad():
                    h_out = self._run_layers(h_front, position_ids, cos_sin, attention_mask)
                self._send_tensor(h_out)
            elif flag == FWD_WITH_BWD:
                h_in = h_front.detach().requires_grad_(True)
                h_out = self._run_layers(h_in, position_ids, cos_sin, attention_mask)
                self._send_tensor(h_out.detach())
                grad_out = self._recv_tensor(h_out.shape, h_out.dtype)
                torch.autograd.backward(h_out, grad_out)
                self._send_tensor(h_in.grad)


def _unwrap_causal_lm(model):
    queue = [model]
    visited = set()
    while queue:
        current = queue.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        if hasattr(current, "lm_head") and hasattr(current, "model"):
            return current
        for attr in ("base_model", "model"):
            if hasattr(current, attr):
                queue.append(getattr(current, attr))
    raise ValueError("Unable to unwrap CausalLM.")


def _find_decoder_stack(root_model, causal_lm):
    queue = [root_model, causal_lm, getattr(causal_lm, "model", None)]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if all(hasattr(current, n) for n in ("layers", "embed_tokens", "norm")):
            return current
        for attr in ("base_model", "model"):
            if hasattr(current, attr):
                queue.append(getattr(current, attr))
    raise ValueError("Unable to locate decoder stack.")
