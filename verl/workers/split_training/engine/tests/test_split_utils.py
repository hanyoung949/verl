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

import pytest

from verl.workers.split_training.split_utils import (
    compute_split_layer_ranges,
    get_stage_for_layer,
    validate_split_layer_ranges,
)


def test_compute_ranges_24_layer_4_4():
    ranges = compute_split_layer_ranges(24, 4, 4)
    assert ranges == {
        "stage_0": (0, 4),
        "stage_1": (4, 20),
        "stage_2": (20, 24),
    }


def test_compute_ranges_36_layer_12_12():
    ranges = compute_split_layer_ranges(36, 12, 12)
    assert ranges == {
        "stage_0": (0, 12),
        "stage_1": (12, 24),
        "stage_2": (24, 36),
    }


def test_compute_ranges_36_layer_4_28_4():
    ranges = compute_split_layer_ranges(36, 4, 4)
    assert ranges == {
        "stage_0": (0, 4),
        "stage_1": (4, 32),
        "stage_2": (32, 36),
    }


def test_validate_invalid_zero_stage_0_size():
    with pytest.raises(ValueError, match="split_stage_0_size must be positive"):
        validate_split_layer_ranges(24, 0, 4)


def test_validate_invalid_zero_stage_2_size():
    with pytest.raises(ValueError, match="split_stage_2_size must be positive"):
        validate_split_layer_ranges(24, 4, 0)


def test_validate_invalid_overlap():
    with pytest.raises(ValueError, match="Invalid split layer range"):
        validate_split_layer_ranges(24, 20, 10)


def test_validate_invalid_exact_fit_no_middle():
    with pytest.raises(ValueError, match="Invalid split layer range"):
        validate_split_layer_ranges(24, 12, 12)


def test_get_stage_for_layer():
    ranges = compute_split_layer_ranges(24, 4, 4)
    assert get_stage_for_layer(0, ranges) == "stage_0"
    assert get_stage_for_layer(3, ranges) == "stage_0"
    assert get_stage_for_layer(4, ranges) == "stage_1"
    assert get_stage_for_layer(19, ranges) == "stage_1"
    assert get_stage_for_layer(20, ranges) == "stage_2"
    assert get_stage_for_layer(23, ranges) == "stage_2"


def test_get_stage_for_layer_out_of_range():
    ranges = compute_split_layer_ranges(24, 4, 4)
    with pytest.raises(ValueError, match="layer_idx .* is out of range"):
        get_stage_for_layer(24, ranges)
