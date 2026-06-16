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

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SplitStageWorkerConfig:
    """Minimal config for a split training stage worker (B0).

    All layer ranges follow the vLLM split convention:
        stage_0: layers[0:split_stage_0_size]
        stage_1: layers[split_stage_0_size:num_hidden_layers - split_stage_2_size]
        stage_2: layers[num_hidden_layers - split_stage_2_size:]
    """

    model_path: str = ""
    split_stage_0_size: int = 4
    split_stage_2_size: int = 4

    lr: float = 1e-4
    clip_grad: float = 1.0

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Placeholders to stay compatible with the BaseEngine signature in the future.
    model_config: Optional[Any] = None
    engine_config: Optional[Any] = None
    optimizer_config: Optional[Any] = None
    checkpoint_config: Optional[Any] = None
