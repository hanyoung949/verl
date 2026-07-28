"""Draft head used by the offline DVI feasibility experiment."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SplitDVIDraftHead(nn.Module):
    """Frozen vocabulary projection with a trainable low-rank delta."""

    def __init__(
        self,
        base_weight: torch.Tensor,
        rank: int,
        alpha: float,
        norm: str = "none",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if base_weight.ndim != 2:
            raise ValueError("base_weight must have shape [vocab_size, hidden_size]")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if norm not in {"none", "rmsnorm"}:
            raise ValueError("norm must be 'none' or 'rmsnorm'")

        weight = base_weight.detach().to(dtype=torch.float32, device="cpu")
        self.register_buffer("base_weight", weight.contiguous())
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.norm_type = norm

        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.lora_a = nn.Parameter(
            torch.randn(rank, weight.shape[1], generator=generator) * 0.02
        )
        self.lora_b = nn.Parameter(torch.zeros(weight.shape[0], rank))
        if norm == "rmsnorm":
            self.norm_weight = nn.Parameter(torch.ones(weight.shape[1]))
        else:
            self.register_parameter("norm_weight", None)

    @property
    def hidden_size(self) -> int:
        return self.base_weight.shape[1]

    @property
    def vocab_size(self) -> int:
        return self.base_weight.shape[0]

    def _normalize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm_weight is None:
            return hidden_states
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + 1e-6)
        return normalized * self.norm_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._normalize(hidden_states.float())
        base_logits = F.linear(hidden_states, self.base_weight)
        delta = F.linear(F.linear(hidden_states, self.lora_a), self.lora_b)
        return base_logits + self.scale * delta

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = {
            "lora_a": self.lora_a.detach().cpu().contiguous(),
            "lora_b": self.lora_b.detach().cpu().contiguous(),
        }
        if self.norm_weight is not None:
            state["norm_weight"] = self.norm_weight.detach().cpu().contiguous()
        return state
