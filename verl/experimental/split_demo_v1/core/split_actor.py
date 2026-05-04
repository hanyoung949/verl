"""拆分 Actor 核心实现。

这个类只承担“训练专用模型执行核心”的职责：
1. 完整前向 `forward_full`
2. rollout 所需的 `prefill / decode_step`
3. LoRA 参数的训练更新

它不负责：
- reward 计算
- rollout 策略
- 中段 transport 细节
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

from .middle_executor import LocalMiddleExecutor, MiddleExecutor


class SplitActorCore(nn.Module):
    """前-中-后分段执行的训练核心。

    设计目标：
    - front / tail 留在主卡（默认 cuda:0）
    - middle 放到第二张卡（默认 cuda:1）
    - optimizer 仅更新 front + tail 中的 LoRA 参数
    """

    def __init__(
        self,
        model_path: str,
        front_end: int,
        middle_end: int,
        lora_config: LoraConfig,
        primary_device: str = "cuda:0",
        middle_device: str = "cuda:1",
        torch_dtype: torch.dtype = torch.bfloat16,
        middle_executor: Optional[MiddleExecutor] = None,
    ) -> None:
        super().__init__()

        self.model_path = model_path
        self.front_end = front_end
        self.middle_end = middle_end
        self.primary_device = primary_device
        self.middle_device = middle_device
        self.torch_dtype = torch_dtype

        # 先加载完整 HF CausalLM，再统一施加 LoRA。
        # 这样拆出来的各层已经是带 adapter 的版本，后续只需要冻结 middle 即可。
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

        # 说明：
        # 1. 我们不把整个 peft_model 保存为 submodule，避免同一套层被重复注册；
        # 2. 直接把子模块摘出来挂到新的 SplitActorCore 上即可，底层参数对象不复制。
        self.embed_tokens = decoder.embed_tokens
        self.front_layers = nn.ModuleList(layers[:front_end])
        middle_layers = nn.ModuleList(layers[front_end:middle_end])
        self.tail_layers = nn.ModuleList(layers[middle_end:])
        self.norm = decoder.norm
        self.lm_head = causal_lm.lm_head
        self.decoder_rotary_emb = getattr(decoder, "rotary_emb", None)

        # middle 段完全冻结：包括 base 权重和其中可能存在的 LoRA adapter。
        for param in middle_layers.parameters():
            param.requires_grad = False

        # 设备放置：
        # - embedding/front/tail/norm/lm_head/rotary_emb 在主卡
        # - middle layers 交给 middle executor 管
        self.embed_tokens.to(self.primary_device)
        self.front_layers.to(self.primary_device)
        middle_layers.to(self.middle_device)
        self.tail_layers.to(self.primary_device)
        self.norm.to(self.primary_device)
        self.lm_head.to(self.primary_device)
        if self.decoder_rotary_emb is not None:
            self.decoder_rotary_emb.to(self.primary_device)

        self.middle_executor = middle_executor or LocalMiddleExecutor(
            middle_layers=middle_layers,
            device=self.middle_device,
        )
        self._runtime_stats = {}

    def _unwrap_causal_lm(self, model: nn.Module) -> nn.Module:
        """尽量从 wrapper 链里找到带 `lm_head` 的最外层 CausalLM。"""

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

        raise ValueError("Unable to unwrap a CausalLM object exposing both `lm_head` and `model`.")

    def _find_decoder_stack(self, root_model: nn.Module, causal_lm: nn.Module) -> nn.Module:
        """从整个 wrapper 链中找到真正持有 `layers/embed_tokens/norm` 的 decoder stack。

        这里不能只看 `causal_lm.model`，因为在某些 PEFT 包装路径下，
        你拿到的“最外层带 lm_head 的对象”未必就是最终那层 decoder stack 的直接父亲。
        """

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

        raise ValueError("Unable to locate decoder stack with layers/embed_tokens/norm in the PEFT/HF wrapper chain.")

    def _get_rotary_emb(self) -> nn.Module:
        """获取 canonical RoPE 模块。

        这里优先从 decoder/model 级别取 `rotary_emb`。
        原因是新版本 transformers（例如当前环境里的 Qwen2.5）已经把 RoPE
        提升到了 `model.rotary_emb`，不再挂在 `layer.self_attn.rotary_emb` 上。

        如果模型没有暴露 decoder 级 `rotary_emb`，再回退到 layer 级查找。
        """

        if self.decoder_rotary_emb is not None:
            return self.decoder_rotary_emb

        # 对某些旧版本/旧架构，rotary_emb 仍可能挂在 attention 层里。
        if hasattr(self.front_layers[0], "self_attn") and hasattr(self.front_layers[0].self_attn, "rotary_emb"):
            return self.front_layers[0].self_attn.rotary_emb

        raise AttributeError("Unable to locate a usable rotary_emb module on either decoder or first front layer.")

    def _build_position_ids(self, attention_mask: Tensor) -> Tensor:
        """根据 attention mask 构造 position ids。

        这里使用与 HF 常见实现一致的思路：
        - 有效 token 位置递增
        - padding 区域会先变成 -1，再 clamp 到 0
        """

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        return position_ids

    @staticmethod
    def _tensor_nbytes(tensor: Optional[Tensor]) -> int:
        if tensor is None:
            return 0
        return tensor.numel() * tensor.element_size()

    def reset_runtime_stats(self) -> None:
        """重置一轮运行中的统计信息。"""

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
        """获取当前运行统计。"""

        stats = dict(self._runtime_stats)
        if hasattr(self.middle_executor, "get_stats"):
            stats["middle_executor"] = self.middle_executor.get_stats()
        return stats

    def _build_causal_mask(self, attention_mask: Tensor, device: str, dtype: torch.dtype) -> Optional[Tensor]:
        """构造显式 4D causal mask。

        当前策略：
        - 如果 batch 是 padding-free（全部有效），直接返回 None，让 SDPA 自己处理 causal；
        - 若存在 padding，则生成一个加性 mask。
        """

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

    def _prepare_inputs(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor], Optional[Tensor], Tensor]:
        """统一准备 position 信息和 mask。

        返回：
        - position_ids
        - position_embeddings=(cos, sin)
        - causal_mask
        - inputs_embeds
        """

        input_ids = input_ids.to(self.primary_device)
        attention_mask = attention_mask.to(self.primary_device)

        inputs_embeds = self.embed_tokens(input_ids)
        position_ids = self._build_position_ids(attention_mask).to(self.primary_device)

        rotary_emb = self._get_rotary_emb()

        # 这里优先尝试标准的 rotary_emb(x, position_ids) 签名。
        # 若环境中的 transformers 版本稍有差异，则退回到更保守的调用方式。
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
        """兼容不同 transformers 版本的 layer 调用。"""

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

    def forward_front(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        """执行前段：embed + front layers。"""

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
        """执行尾段：tail layers + norm + lm_head。"""

        self._runtime_stats["return_to_primary_bytes"] += self._tensor_nbytes(hidden_states)
        h = hidden_states.to(self.primary_device)
        pos_ids = position_ids.to(self.primary_device)
        pos_emb = (
            position_embeddings[0].to(self.primary_device),
            position_embeddings[1].to(self.primary_device),
        )
        mask = attention_mask.to(self.primary_device) if attention_mask is not None else None

        for layer in self.tail_layers:
            h = self._call_layer(
                layer=layer,
                hidden_states=h,
                position_ids=pos_ids,
                position_embeddings=pos_emb,
                attention_mask=mask,
            )

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    def forward_full(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """完整前向。

        训练时：
        - 先用它算 old log prob
        - 再用它算 new log prob
        - 最终通过 log prob 计算策略损失
        """

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
        """prefill 接口。

        当前 naive 实现里，它就是完整前向。
        未来若加入 KV cache，这里会变成“初始化 cache + 返回最后位置 logits”。
        """

        self._runtime_stats["prefill_calls"] += 1
        return self.forward_full(prompt_ids, attention_mask)

    def decode_step(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """单步 decode 接口。

        当前 naive 版本直接重算完整序列，然后取最后一个位置的 logits。
        这虽然慢，但实现简单，适合先验证 split 架构和梯度链路。
        """

        self._runtime_stats["decode_calls"] += 1
        logits = self.forward_full(input_ids, attention_mask)
        return logits[:, -1, :]

    def trainable_parameters(self) -> list[nn.Parameter]:
        """返回真正需要更新的参数。

        这里依赖 PEFT 已经把 base model 参数冻结，只留下 LoRA 权重可训练；
        同时 middle 段又被我们额外整体冻结，所以最终只会留下 front/tail 的 LoRA 参数。
        """

        params: list[nn.Parameter] = []
        for module in (self.embed_tokens, self.front_layers, self.tail_layers, self.norm, self.lm_head):
            for param in module.parameters():
                if param.requires_grad:
                    params.append(param)
        return params

    def get_trainable_state_dict(self) -> dict[str, Tensor]:
        """导出当前所有可训练参数。

        当前 demo 不依赖完整的 PEFT `save_pretrained()` 流程，
        而是直接保存 `requires_grad=True` 的命名参数。
        这样最小、直接，足够支撑 demo 阶段的 adapter 保留需求。
        """

        state: dict[str, Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().cpu().clone()
        return state

    def save_adapter(self, save_dir: str | Path, global_step: int) -> Path:
        """保存最小 adapter 结果到指定目录。

        产物：
        - `adapter_state.pt`：当前可训练参数（主要是 LoRA）
        - `adapter_meta.json`：模型、切层和 step 元信息
        """

        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)

        torch.save(self.get_trainable_state_dict(), target / "adapter_state.pt")

        meta = {
            "model_path": self.model_path,
            "front_end": self.front_end,
            "middle_end": self.middle_end,
            "primary_device": self.primary_device,
            "middle_device": self.middle_device,
            "torch_dtype": str(self.torch_dtype),
            "global_step": global_step,
        }
        (target / "adapter_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target
