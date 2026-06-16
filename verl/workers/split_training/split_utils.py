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

"""Shared split-training utilities for layer ranges and topology.

This module is the single source of truth for the 3-stage split math so that
engine, adapter merge, and tests do not re-implement the same formulas.
"""

from __future__ import annotations

import re
from typing import Literal

import torch

SplitLayerRanges = dict[str, tuple[int, int]]

# Local LoRA key produced by Stage0/Stage2.get_trainable_state_dict(), e.g.
#   layers.0.self_attn.q_proj.lora_A.default.weight
_LORA_LOCAL_KEY_RE = re.compile(
    r"^layers\.(\d+)\.self_attn\.([^.]+)\.lora_([AB])\.default\.weight$"
)


def compute_split_layer_ranges(
    num_hidden_layers: int,
    split_stage_0_size: int,
    split_stage_2_size: int,
) -> SplitLayerRanges:
    """Compute [start, end) layer ranges for each split stage.

    The math is kept identical to vLLM's ``_get_split_layer_range`` so that
    training and rollout use the same layer assignment.

    Args:
        num_hidden_layers: Total number of transformer layers.
        split_stage_0_size: Number of layers on stage_0 (Edge Head).
        split_stage_2_size: Number of layers on stage_2 (Edge Tail).

    Returns:
        Mapping from stage name to ``(start_layer, end_layer)``.
    """
    validate_split_layer_ranges(num_hidden_layers, split_stage_0_size, split_stage_2_size)

    stage_1_start = split_stage_0_size
    stage_1_end = num_hidden_layers - split_stage_2_size

    return {
        "stage_0": (0, split_stage_0_size),
        "stage_1": (stage_1_start, stage_1_end),
        "stage_2": (stage_1_end, num_hidden_layers),
    }


def validate_split_layer_ranges(
    num_hidden_layers: int,
    split_stage_0_size: int,
    split_stage_2_size: int,
) -> None:
    """Validate that the split sizes produce a non-empty stage_1.

    Raises:
        ValueError: If the sizes are invalid.
    """
    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
    if split_stage_0_size <= 0:
        raise ValueError(
            f"split_stage_0_size must be positive, got {split_stage_0_size}"
        )
    if split_stage_2_size <= 0:
        raise ValueError(
            f"split_stage_2_size must be positive, got {split_stage_2_size}"
        )

    stage_1_start = split_stage_0_size
    stage_1_end = num_hidden_layers - split_stage_2_size

    if not (0 < stage_1_start < stage_1_end < num_hidden_layers):
        raise ValueError(
            f"Invalid split layer range: stage_0=[0, {split_stage_0_size}), "
            f"stage_1=[{stage_1_start}, {stage_1_end}), "
            f"stage_2=[{stage_1_end}, {num_hidden_layers}), "
            f"with split_stage_0_size={split_stage_0_size}, "
            f"split_stage_2_size={split_stage_2_size}. "
            f"Ensure 0 < split_stage_0_size < "
            f"num_hidden_layers - split_stage_2_size < num_hidden_layers."
        )


def get_stage_for_layer(
    layer_idx: int,
    ranges: SplitLayerRanges,
) -> Literal["stage_0", "stage_1", "stage_2"]:
    """Return which stage owns the given global layer index.

    Raises:
        ValueError: If ``layer_idx`` is outside the total layer range.
    """
    for stage, (start, end) in ranges.items():
        if start <= layer_idx < end:
            return stage  # type: ignore[return-value]

    total_end = max(end for _, end in ranges.values())
    raise ValueError(
        f"layer_idx {layer_idx} is out of range [0, {total_end})"
    )


def merge_split_lora_state_dict(
    head_state: dict[str, torch.Tensor],
    tail_state: dict[str, torch.Tensor],
    ranges: SplitLayerRanges,
) -> dict[str, torch.Tensor]:
    """Merge head (stage_0) and tail (stage_2) local LoRA state dicts into a
    single PEFT-formatted state dict suitable for vLLM ``from_lora_tensors``.

    Middle-stage (stage_1) layers are intentionally omitted; the caller should
    remove any previously-loaded adapter before loading the merged dict so that
    stage_1 falls back to the base model.

    Args:
        head_state: Trainable state dict from stage_0. Keys are local, e.g.
            ``layers.0.self_attn.q_proj.lora_A.default.weight``.
        tail_state: Trainable state dict from stage_2 with the same local key
            convention.
        ranges: Layer ranges returned by :func:`compute_split_layer_ranges`.

    Returns:
        Merged state dict with PEFT global keys, e.g.
        ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight``.
    """
    merged: dict[str, torch.Tensor] = {}
    stage_0_start = ranges["stage_0"][0]
    stage_2_start = ranges["stage_2"][0]

    for src_state, stage in [(head_state, "stage_0"), (tail_state, "stage_2")]:
        stage_start = ranges[stage][0]
        for key, tensor in src_state.items():
            match = _LORA_LOCAL_KEY_RE.match(key)
            if not match:
                continue
            local_idx = int(match.group(1))
            proj = match.group(2)
            ab = match.group(3)
            global_idx = stage_start + local_idx
            global_key = (
                f"base_model.model.model.layers.{global_idx}.self_attn."
                f"{proj}.lora_{ab}.weight"
            )
            merged[global_key] = tensor

    return merged
