"""拆分 Actor 核心 — v2 NCCL 版。

与 v1 的核心差异：
- 不持有 middle_layers（它们在 rank1 独立进程中）
- middle_executor 使用 NCCLMiddleExecutor（NCCL P2P）替代 LocalMiddleExecutor（tensor.to）
- 无 middle_device 属性
"""

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
    """前-中-后分段执行的训练核心（NCCL 版）。

    rank0 持有：embed_tokens + front_layers + tail_layers + norm + lm_head + LoRA
    rank1 持有：middle_layers（冻结）
    通信方式：NCCL P2P（通过 middle_executor）
    """

    def __init__(
        self,
        model_path: str,
        front_end: int,
        middle_end: int,
        lora_config: LoraConfig,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        middle_executor: Optional[NCCLMiddleExecutor] = None,
    ) -> None:
        super().__init__()

        self.model_path = model_path
        self.front_end = front_end
        self.middle_end = middle_end
        # 规范化 device：确保 "cuda" 被解析为 "cuda:N" 形式
        resolved_device = torch.device(device)
        self.primary_device = f"cuda:{resolved_device.index}" if resolved_device.index is not None else f"cuda:{torch.cuda.current_device()}"
        self.torch_dtype = torch_dtype

        # 加载完整 HF CausalLM，统一施加 LoRA
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        )
        peft_model = get_peft_model(base_model, lora_config)

        causal_lm = self._unwrap_causal_lm(peft_model)
        decoder = self._find_decoder_stack(peft_model, causal_lm)

        layers = list(decoder.layers)
        if not (0 < front_end < middle_end < len(layers)):
            raise ValueError(
                f"Invalid split points: front_end={front_end}, middle_end={middle_end}, total_layers={len(layers)}"
            )

        # 只注册 rank0 需要的子模块（不注册 middle_layers）
        self.embed_tokens = decoder.embed_tokens
        self.front_layers = nn.ModuleList(layers[:front_end])
        self.tail_layers = nn.ModuleList(layers[middle_end:])
        self.norm = decoder.norm
        self.lm_head = causal_lm.lm_head
        self.decoder_rotary_emb = getattr(decoder, "rotary_emb", None)

        # 全部放到本进程设备
        self.embed_tokens.to(self.primary_device)
        self.front_layers.to(self.primary_device)
        self.tail_layers.to(self.primary_device)
        self.norm.to(self.primary_device)
        self.lm_head.to(self.primary_device)
        if self.decoder_rotary_emb is not None:
            self.decoder_rotary_emb.to(self.primary_device)

        # middle executor（默认 NCCL）
        self.middle_executor = middle_executor or NCCLMiddleExecutor(
            middle_rank=1,
            device=self.primary_device,
        )

        self._runtime_stats = {}

    # ── 与 v1 相同的辅助方法 ───────────────────────────────────

    def _unwrap_causal_lm(self, model: nn.Module) -> nn.Module:
        queue = [model]
        visited: set[int] = set()
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
        raise ValueError("Unable to unwrap a CausalLM object exposing both lm_head and model.")

    def _find_decoder_stack(self, root_model: nn.Module, causal_lm: nn.Module) -> nn.Module:
        queue = [root_model, causal_lm, getattr(causal_lm, "model", None)]
        visited: set[int] = set()
        while queue:
            current = queue.pop(0)
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))
            if all(hasattr(current, name) for name in ("layers", "embed_tokens", "norm")):
                return current
            for attr in ("base_model", "model"):
                if hasattr(current, attr):
                    queue.append(getattr(current, attr))
        raise ValueError("Unable to locate decoder stack with layers/embed_tokens/norm.")

    def _get_rotary_emb(self) -> nn.Module:
        if self.decoder_rotary_emb is not None:
            return self.decoder_rotary_emb
        if hasattr(self.front_layers[0], "self_attn") and hasattr(self.front_layers[0].self_attn, "rotary_emb"):
            return self.front_layers[0].self_attn.rotary_emb
        raise AttributeError("Unable to locate a usable rotary_emb module.")

    def _build_position_ids(self, attention_mask: Tensor) -> Tensor:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        return position_ids

    @staticmethod
    def _tensor_nbytes(tensor: Optional[Tensor]) -> int:
        if tensor is None:
            return 0
        return tensor.numel() * tensor.element_size()

    def reset_runtime_stats(self) -> None:
        self._runtime_stats = {
            "forward_calls": 0,
            "prefill_calls": 0,
            "decode_calls": 0,
            "return_to_primary_bytes": 0,
            "last_input_shape": None,
            "last_logits_shape": None,
        }
        if hasattr(self.middle_executor, "reset_stats"):
            self.middle_executor.reset_stats()

    def get_runtime_stats(self) -> dict:
        stats = dict(self._runtime_stats)
        if hasattr(self.middle_executor, "get_stats"):
            stats["middle_executor"] = self.middle_executor.get_stats()
        return stats

    def _build_causal_mask(self, attention_mask: Tensor, device: str, dtype: torch.dtype) -> Optional[Tensor]:
        if bool(torch.all(attention_mask == 1)):
            return None

        batch_size, seq_len = attention_mask.shape
        key_mask = attention_mask[:, None, None, :].bool()
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
        full_mask = key_mask & causal

        additive_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            device=device,
            dtype=dtype,
        )
        additive_mask = additive_mask.masked_fill(~full_mask, torch.finfo(dtype).min)
        return additive_mask

    def _prepare_inputs(
        self, input_ids: Tensor, attention_mask: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Optional[Tensor], Tensor]:
        """准备 position 信息和 mask。"""

        input_ids = input_ids.to(self.primary_device)
        attention_mask = attention_mask.to(self.primary_device)

        inputs_embeds = self.embed_tokens(input_ids)
        position_ids = self._build_position_ids(attention_mask).to(self.primary_device)

        rotary_emb = self._get_rotary_emb()
        try:
            position_embeddings = rotary_emb(inputs_embeds, position_ids)
        except TypeError:
            position_embeddings = rotary_emb(inputs_embeds, seq_len=input_ids.size(1))

        causal_mask = self._build_causal_mask(
            attention_mask=attention_mask,
            device=self.primary_device,
            dtype=inputs_embeds.dtype,
        )
        return position_ids, position_embeddings, causal_mask, inputs_embeds

    def _call_layer(
        self,
        layer: nn.Module,
        hidden_states: Tensor,
        position_ids: Tensor,
        position_embeddings: Optional[tuple[Tensor, Tensor]],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        try:
            output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        except TypeError:
            try:
                output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
            except TypeError:
                output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
        return output[0] if isinstance(output, tuple) else output

    # ── 分段执行 ───────────────────────────────────────────────

    def forward_front(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        if inputs_embeds is None:
            h = self.embed_tokens(input_ids.to(self.primary_device))
        else:
            h = inputs_embeds

        for layer in self.front_layers:
            h = self._call_layer(
                layer=layer,
                hidden_states=h,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        return h

    def forward_tail(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """执行尾段。NCCLMiddleExecutor 返回的 tensor 已在 rank0 设备上，无需 .to()。"""

        self._runtime_stats["return_to_primary_bytes"] += self._tensor_nbytes(hidden_states)

        h = hidden_states
        for layer in self.tail_layers:
            h = self._call_layer(
                layer=layer,
                hidden_states=h,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    def forward_full(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """完整前向。训练时计算 log_prob 用。"""

        self._runtime_stats["forward_calls"] += 1
        self._runtime_stats["last_input_shape"] = tuple(input_ids.shape)
        position_ids, position_embeddings, causal_mask, inputs_embeds = self._prepare_inputs(input_ids, attention_mask)

        h_front = self.forward_front(
            input_ids=input_ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            inputs_embeds=inputs_embeds,
        )
        h_middle = self.middle_executor.execute(
            hidden_states=h_front,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )
        logits = self.forward_tail(
            hidden_states=h_middle,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )
        self._runtime_stats["last_logits_shape"] = tuple(logits.shape)
        return logits

    def prefill(self, prompt_ids: Tensor, attention_mask: Tensor) -> Tensor:
        self._runtime_stats["prefill_calls"] += 1
        return self.forward_full(prompt_ids, attention_mask)

    def decode_step(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        self._runtime_stats["decode_calls"] += 1
        logits = self.forward_full(input_ids, attention_mask)
        return logits[:, -1, :]

    # ── 参数管理 ───────────────────────────────────────────────

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (self.embed_tokens, self.front_layers, self.tail_layers, self.norm, self.lm_head):
            for param in module.parameters():
                if param.requires_grad:
                    params.append(param)
        return params

    def get_trainable_state_dict(self) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().cpu().clone()
        return state

    def save_adapter(self, save_dir: str | Path, global_step: int) -> Path:
        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)

        torch.save(self.get_trainable_state_dict(), target / "adapter_state.pt")

        meta = {
            "model_path": self.model_path,
            "front_end": self.front_end,
            "middle_end": self.middle_end,
            "primary_device": self.primary_device,
            "torch_dtype": str(self.torch_dtype),
            "global_step": global_step,
        }
        (target / "adapter_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target
