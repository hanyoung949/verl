"""Middle Worker — rank1 侧的纯请求响应 worker。

rank1 只运行 MiddleWorker.run()，是一个纯粹的请求响应循环：
1. 等待 rank0 的 header
2. 接收 hidden_states 等 tensor
3. 用本地 middle_layers 计算
4. 结果发回 rank0

不持有任何训练逻辑、optimizer、rollout、reward。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

from .middle_executor import FWD_ONLY, FWD_WITH_BWD, SHUTDOWN

# dtype 编码映射，与 rank0 的 _DTYPE_CODE 对应
_DTYPE_MAP = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}


class MiddleWorker:
    """rank1 中段 worker。"""

    def __init__(
        self,
        middle_layers: nn.ModuleList,
        rotary_emb: Optional[nn.Module],
        device: torch.device,
        tensor_dtype: torch.dtype,
    ) -> None:
        self.middle_layers = middle_layers
        self.rotary_emb = rotary_emb
        self.device = device
        self.tensor_dtype = tensor_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        front_end: int,
        middle_end: int,
        lora_config: LoraConfig,
        device: torch.device,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> MiddleWorker:
        """加载完整模型，提取 middle_layers + rotary_emb，冻结所有参数。"""

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        )
        peft_model = get_peft_model(base_model, lora_config)

        causal_lm = _unwrap_causal_lm(peft_model)
        decoder = _find_decoder_stack(peft_model, causal_lm)

        layers = list(decoder.layers)
        if not (0 < front_end < middle_end < len(layers)):
            raise ValueError(
                f"Invalid split points: front_end={front_end}, middle_end={middle_end}, total_layers={len(layers)}"
            )

        middle_layers = nn.ModuleList(layers[front_end:middle_end])
        rotary_emb = getattr(decoder, "rotary_emb", None)

        # 冻结所有参数
        for param in middle_layers.parameters():
            param.requires_grad = False
        if rotary_emb is not None:
            for param in rotary_emb.parameters():
                param.requires_grad = False

        # 移动到目标设备
        middle_layers.to(device)
        if rotary_emb is not None:
            rotary_emb.to(device)

        return cls(
            middle_layers=middle_layers,
            rotary_emb=rotary_emb,
            device=device,
            tensor_dtype=torch_dtype,
        )

    # ── 底层 NCCL 操作 ─────────────────────────────────────────

    def _recv_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        """从 rank0 接收一个 tensor。"""
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=0)
        return buf

    def _send_tensor(self, tensor: Tensor) -> None:
        """发送一个 contiguous tensor 到 rank0。"""
        dist.send(tensor.contiguous(), dst=0)

    # ── layer 调用 ─────────────────────────────────────────────

    def _call_layer(
        self,
        layer: nn.Module,
        hidden_states: Tensor,
        position_ids: Tensor,
        position_embeddings: Optional[tuple[Tensor, Tensor]],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """兼容不同 transformers 版本的 decoder layer 调用。"""

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

    def _run_layers(
        self,
        h: Tensor,
        position_ids: Tensor,
        position_embeddings: Optional[tuple[Tensor, Tensor]],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """依次执行所有 middle layers。"""
        for layer in self.middle_layers:
            h = self._call_layer(
                layer=layer,
                hidden_states=h,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        return h

    # ── 主循环 ─────────────────────────────────────────────────

    def run(self) -> None:
        """rank1 的主请求响应循环。"""

        while True:
            # 接收 header
            header = torch.zeros(6, dtype=torch.int64, device=self.device)
            dist.recv(header, src=0)
            flag, B, S, H, has_mask, dtype_code = header.tolist()

            if flag == SHUTDOWN:
                break

            tensor_dtype = _DTYPE_MAP[int(dtype_code)]

            # 接收数据 tensor
            h_front = self._recv_tensor((B, S, H), tensor_dtype)
            position_ids = self._recv_tensor((B, S), torch.int64)
            attention_mask = None
            if has_mask:
                # 接收 additive 4D mask [B, 1, S, S]
                attention_mask = self._recv_tensor((B, 1, S, S), tensor_dtype)

            # 本地重算 pos_emb（省去跨卡传输）
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
                # 保留计算图，支持梯度回传
                h_in = h_front.detach().requires_grad_(True)
                h_out = self._run_layers(h_in, position_ids, cos_sin, attention_mask)
                self._send_tensor(h_out.detach())

                # 等待 rank0 发来的 grad_h_middle
                grad_out = self._recv_tensor(h_out.shape, h_out.dtype)
                torch.autograd.backward(h_out, grad_out)
                self._send_tensor(h_in.grad)


# ── 辅助函数（从 v1 split_actor.py 提取，rank1 也需要） ─────────


def _unwrap_causal_lm(model: nn.Module) -> nn.Module:
    """从 wrapper 链中找到带 lm_head 的最外层 CausalLM。"""
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


def _find_decoder_stack(root_model: nn.Module, causal_lm: nn.Module) -> nn.Module:
    """从 wrapper 链中找到持有 layers/embed_tokens/norm 的 decoder stack。"""
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
