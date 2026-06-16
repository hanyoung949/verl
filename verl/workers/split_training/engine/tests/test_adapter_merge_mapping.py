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

import torch

from verl.workers.split_training.split_utils import (
    compute_split_layer_ranges,
    get_stage_for_layer,
)


def build_dummy_state(num_layers: int, prefix: str):
    state = {}
    for local_idx in range(num_layers):
        for proj in ("q_proj", "v_proj"):
            for ab in ("A", "B"):
                key = f"layers.{local_idx}.self_attn.{proj}.lora_{ab}.default.weight"
                state[key] = torch.randn(8, 8)
    return state


def resolve_adapter_source(layer_idx: int, ranges, head_state: dict, tail_state: dict):
    """Mirror the mapping logic used by SplitRayWorkerGroup.export_merged_adapter."""
    stage = get_stage_for_layer(layer_idx, ranges)
    if stage == "stage_1":
        return None

    local_idx = layer_idx - ranges[stage][0]
    src = head_state if stage == "stage_0" else tail_state
    src_key = f"layers.{local_idx}.self_attn.{{proj}}.lora_{{ab}}.default.weight"
    return stage, local_idx, src, src_key


def test_adapter_merge_mapping_24_layer_4_4():
    n_layers = 24
    ranges = compute_split_layer_ranges(n_layers, 4, 4)
    head_state = build_dummy_state(4, "head")
    tail_state = build_dummy_state(4, "tail")

    for proj in ("q_proj", "v_proj"):
        for ab in ("A", "B"):
            for layer_idx in range(n_layers):
                result = resolve_adapter_source(layer_idx, ranges, head_state, tail_state)
                stage = get_stage_for_layer(layer_idx, ranges)

                if stage == "stage_1":
                    assert result is None
                    continue

                _, local_idx, src, template = result
                src_key = template.format(proj=proj, ab=ab)
                assert src_key in src, f"missing {src_key} for layer {layer_idx}"

                if stage == "stage_0":
                    assert local_idx == layer_idx
                    assert src is head_state
                else:
                    assert local_idx == layer_idx - ranges["stage_2"][0]
                    assert src is tail_state


def test_adapter_merge_zeroes_middle_layers():
    n_layers = 24
    ranges = compute_split_layer_ranges(n_layers, 4, 4)

    middle_layers = list(range(ranges["stage_1"][0], ranges["stage_1"][1]))
    for layer_idx in middle_layers:
        assert get_stage_for_layer(layer_idx, ranges) == "stage_1"
