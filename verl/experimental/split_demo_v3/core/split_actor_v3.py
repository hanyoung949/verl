"""SplitActorCore v3 — rank0 (edge) 侧模型核心。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

from .middle_executor import NCCLMiddleExecutor


class SplitActorCore(nn.Module):
    def __init__(self, model_path, front_end, middle_end, lora_config, device="cuda",
                 torch_dtype=torch.bfloat16, middle_executor=None):
        super().__init__()
        self.model_path = model_path
        self.front_end = front_end
        self.middle_end = middle_end
        resolved_device = torch.device(device)
        self.primary_device = (
            f"cuda:{resolved_device.index}" if resolved_device.index is not None
            else f"cuda:{torch.cuda.current_device()}"
        )
        self.torch_dtype = torch_dtype

        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        peft_model = get_peft_model(base_model, lora_config)

        causal_lm = self._unwrap_causal_lm(peft_model)
        decoder = self._find_decoder_stack(peft_model, causal_lm)

        layers = list(decoder.layers)
        if not (0 < front_end < middle_end < len(layers)):
            raise ValueError(f"Invalid split: front_end={front_end}, middle_end={middle_end}, total={len(layers)}")

        self.embed_tokens = decoder.embed_tokens
        self.front_layers = nn.ModuleList(layers[:front_end])
        self.tail_layers = nn.ModuleList(layers[middle_end:])
        self.norm = decoder.norm
        self.lm_head = causal_lm.lm_head
        self.decoder_rotary_emb = getattr(decoder, "rotary_emb", None)

        self.embed_tokens.to(self.primary_device)
        self.front_layers.to(self.primary_device)
        self.tail_layers.to(self.primary_device)
        self.norm.to(self.primary_device)
        self.lm_head.to(self.primary_device)
        if self.decoder_rotary_emb is not None:
            self.decoder_rotary_emb.to(self.primary_device)

        self.middle_executor = middle_executor or NCCLMiddleExecutor(middle_rank=1, device=self.primary_device)
        self._runtime_stats = {}

    def _unwrap_causal_lm(self, model):
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
        raise ValueError("Cannot unwrap CausalLM.")

    def _find_decoder_stack(self, root_model, causal_lm):
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
        raise ValueError("Cannot locate decoder stack.")

    def _get_rotary_emb(self):
        if self.decoder_rotary_emb is not None:
            return self.decoder_rotary_emb
        if hasattr(self.front_layers[0], "self_attn") and hasattr(self.front_layers[0].self_attn, "rotary_emb"):
            return self.front_layers[0].self_attn.rotary_emb
        raise AttributeError("Cannot find rotary_emb.")

    def _build_position_ids(self, attention_mask):
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.clamp(min=0)

    def reset_runtime_stats(self):
        self._runtime_stats = {
            "forward_calls": 0, "prefill_calls": 0, "decode_calls": 0,
            "return_to_primary_bytes": 0, "last_input_shape": None, "last_logits_shape": None,
        }
        if hasattr(self.middle_executor, "reset_stats"):
            self.middle_executor.reset_stats()

    def get_runtime_stats(self):
        stats = dict(self._runtime_stats)
        if hasattr(self.middle_executor, "get_stats"):
            stats["middle_executor"] = self.middle_executor.get_stats()
        return stats

    def _build_causal_mask(self, attention_mask, device, dtype):
        if bool(torch.all(attention_mask == 1)):
            return None
        batch_size, seq_len = attention_mask.shape
        key_mask = attention_mask[:, None, None, :].bool()
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
        full_mask = key_mask & causal
        additive_mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)
        additive_mask = additive_mask.masked_fill(~full_mask, torch.finfo(dtype).min)
        return additive_mask

    def _prepare_inputs(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.primary_device)
        attention_mask = attention_mask.to(self.primary_device)
        inputs_embeds = self.embed_tokens(input_ids)
        position_ids = self._build_position_ids(attention_mask).to(self.primary_device)
        rotary_emb = self._get_rotary_emb()
        try:
            position_embeddings = rotary_emb(inputs_embeds, position_ids)
        except TypeError:
            position_embeddings = rotary_emb(inputs_embeds, seq_len=input_ids.size(1))
        causal_mask = self._build_causal_mask(attention_mask, self.primary_device, inputs_embeds.dtype)
        return position_ids, position_embeddings, causal_mask, inputs_embeds

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

    def forward_front(self, input_ids, position_ids, position_embeddings, attention_mask, inputs_embeds=None):
        if inputs_embeds is None:
            h = self.embed_tokens(input_ids.to(self.primary_device))
        else:
            h = inputs_embeds
        for layer in self.front_layers:
            h = self._call_layer(layer, h, position_ids, position_embeddings, attention_mask)
        return h

    def forward_tail(self, hidden_states, position_ids, position_embeddings, attention_mask):
        h = hidden_states
        for layer in self.tail_layers:
            h = self._call_layer(layer, h, position_ids, position_embeddings, attention_mask)
        h = self.norm(h)
        return self.lm_head(h)

    def forward_full(self, input_ids, attention_mask):
        self._runtime_stats["forward_calls"] += 1
        self._runtime_stats["last_input_shape"] = tuple(input_ids.shape)
        position_ids, position_embeddings, causal_mask, inputs_embeds = self._prepare_inputs(input_ids, attention_mask)
        h_front = self.forward_front(input_ids, position_ids, position_embeddings, causal_mask, inputs_embeds)
        h_middle = self.middle_executor.execute(h_front, position_ids, position_embeddings, causal_mask)
        logits = self.forward_tail(h_middle, position_ids, position_embeddings, causal_mask)
        self._runtime_stats["last_logits_shape"] = tuple(logits.shape)
        return logits

    def trainable_parameters(self):
        params = []
        for module in (self.embed_tokens, self.front_layers, self.tail_layers, self.norm, self.lm_head):
            for param in module.parameters():
                if param.requires_grad:
                    params.append(param)
        return params

    def get_trainable_state_dict(self):
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().cpu().clone()
        return state

    def load_trainable_state_dict(self, state):
        for name, param in self.named_parameters():
            if name in state:
                param.data.copy_(state[name].to(param.device))

    def save_adapter(self, save_dir, global_step):
        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_trainable_state_dict(), target / "adapter_state.pt")
        meta = {"model_path": self.model_path, "front_end": self.front_end,
                "middle_end": self.middle_end, "global_step": global_step}
        (target / "adapter_meta.json").write_text(json.dumps(meta, indent=2))
        return target
