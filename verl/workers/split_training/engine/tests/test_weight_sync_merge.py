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
    merge_split_lora_state_dict,
)


def _make_tensor():
    return torch.randn(8, 8)


def test_merge_maps_local_to_global_keys():
    ranges = compute_split_layer_ranges(24, split_stage_0_size=4, split_stage_2_size=4)

    head_state = {
        "layers.0.self_attn.q_proj.lora_A.default.weight": _make_tensor(),
        "layers.3.self_attn.v_proj.lora_B.default.weight": _make_tensor(),
    }
    tail_state = {
        "layers.0.self_attn.q_proj.lora_A.default.weight": _make_tensor(),
        "layers.3.self_attn.v_proj.lora_B.default.weight": _make_tensor(),
    }

    merged = merge_split_lora_state_dict(head_state, tail_state, ranges)

    assert len(merged) == 4
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in merged
    assert "base_model.model.model.layers.3.self_attn.v_proj.lora_B.weight" in merged
    assert "base_model.model.model.layers.20.self_attn.q_proj.lora_A.weight" in merged
    assert "base_model.model.model.layers.23.self_attn.v_proj.lora_B.weight" in merged


def test_merge_skips_stage_1():
    ranges = compute_split_layer_ranges(24, split_stage_0_size=4, split_stage_2_size=4)

    head_state = {"layers.0.self_attn.q_proj.lora_A.default.weight": _make_tensor()}
    tail_state = {"layers.0.self_attn.q_proj.lora_A.default.weight": _make_tensor()}

    merged = merge_split_lora_state_dict(head_state, tail_state, ranges)

    for key in merged:
        layer_idx = int(key.split(".")[4])
        assert layer_idx < 4 or layer_idx >= 20


def test_merge_preserves_tensors():
    ranges = compute_split_layer_ranges(24, split_stage_0_size=4, split_stage_2_size=4)
    t = _make_tensor()
    merged = merge_split_lora_state_dict(
        {"layers.1.self_attn.q_proj.lora_A.default.weight": t},
        {},
        ranges,
    )
    assert merged["base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"] is t
