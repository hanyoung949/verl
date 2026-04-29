"""中段执行器。

这里的设计重点不是“把中段放到 cuda:1”本身，而是把“中段怎么执行、怎么传输”
收口成一个独立接口。这样未来如果从本机双卡切到 RPC / IPC，只改这一层。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn


class MiddleExecutor(ABC):
    """中段执行抽象接口。

    注意：这个接口不关心上层是训练还是生成，也不关心 transport 的具体实现。
    它只负责接收前段 hidden states，并返回经过中段后的 hidden states。
    """

    @abstractmethod
    def execute(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        position_embeddings: Optional[tuple[Tensor, Tensor]],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """执行中段层。"""


class LocalMiddleExecutor(MiddleExecutor):
    """本地双卡版中段执行器。

    当前 demo 中，它只是把张量搬到 `cuda:1` 然后执行 middle layers。
    未来若换成 RPC，这里可以改成：
    1. 序列化 hidden states
    2. 发送到远端
    3. 远端执行 middle layers
    4. 返回结果

    上层 `SplitActorCore` 不需要知道 transport 细节。
    """

    def __init__(self, middle_layers: nn.ModuleList, device: str = "cuda:1") -> None:
        self.middle_layers = middle_layers
        self.device = device
        self.reset_stats()

    @staticmethod
    def _tensor_nbytes(tensor: Optional[Tensor]) -> int:
        if tensor is None:
            return 0
        return tensor.numel() * tensor.element_size()

    def reset_stats(self) -> None:
        """重置中段执行与传输统计。

        这些统计只服务于 demo 阶段的可观察性：
        - 我们想知道每步大概搬了多少 hidden / pos / mask
        - 不追求和 NCCL / PCIe 实际底层字节完全一模一样
        - 只追求“数量级正确、便于比较”
        """

        self.stats = {
            "calls": 0,
            "forward_to_middle_bytes": 0,
            "position_ids_bytes": 0,
            "position_embeddings_bytes": 0,
            "attention_mask_bytes": 0,
            "last_hidden_shape": None,
            "last_position_ids_shape": None,
            "last_attention_mask_shape": None,
        }

    def get_stats(self) -> dict:
        """返回当前累计统计。"""

        return dict(self.stats)

    def _call_layer(
        self,
        layer: nn.Module,
        hidden_states: Tensor,
        position_ids: Tensor,
        position_embeddings: Optional[tuple[Tensor, Tensor]],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """尽量兼容不同 transformers 版本的 decoder layer 调用签名。

        不同版本的 Qwen/Llama layer 在 `position_embeddings`、`position_ids`
        等参数组合上会有些小差异，所以这里做一层兼容调用，减少上层逻辑复杂度。
        """

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

        # HF decoder layer 通常返回 tuple，第 0 位是新的 hidden states。
        return output[0] if isinstance(output, tuple) else output

    def execute(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        position_embeddings: Optional[tuple[Tensor, Tensor]],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """在本地第二张卡上执行中段层。

        关键点：
        1. `hidden_states.to(self.device)` 是可微的，梯度会自动回传；
        2. 当前 demo 的生成路径默认 padding-free，因此通常直接传 `attention_mask=None`；
        3. 若未来接入 padding batch，则由 `SplitActorCore` 统一构造 mask，
           然后这里仅负责搬运并透传，不自行生成 mask。
        """

        self.stats["calls"] += 1
        self.stats["last_hidden_shape"] = tuple(hidden_states.shape)
        self.stats["last_position_ids_shape"] = tuple(position_ids.shape)
        self.stats["last_attention_mask_shape"] = None if attention_mask is None else tuple(attention_mask.shape)
        self.stats["forward_to_middle_bytes"] += self._tensor_nbytes(hidden_states)
        self.stats["position_ids_bytes"] += self._tensor_nbytes(position_ids)
        if position_embeddings is not None:
            self.stats["position_embeddings_bytes"] += self._tensor_nbytes(position_embeddings[0])
            self.stats["position_embeddings_bytes"] += self._tensor_nbytes(position_embeddings[1])
        self.stats["attention_mask_bytes"] += self._tensor_nbytes(attention_mask)

        h = hidden_states.to(self.device)
        pos_ids = position_ids.to(self.device)
        mask = attention_mask.to(self.device) if attention_mask is not None else None

        if position_embeddings is None:
            pos_emb = None
        else:
            pos_emb = (
                position_embeddings[0].to(self.device),
                position_embeddings[1].to(self.device),
            )

        for layer in self.middle_layers:
            h = self._call_layer(
                layer=layer,
                hidden_states=h,
                position_ids=pos_ids,
                position_embeddings=pos_emb,
                attention_mask=mask,
            )

        return h
