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

"""In-memory weight sync from split training to a live vLLM rollout."""

from __future__ import annotations

import logging
from typing import Any

import ray
import torch
from transformers import AutoConfig

from .split_ray_worker_group import SplitRayWorkerGroup
from .split_utils import compute_split_layer_ranges, merge_split_lora_state_dict

logger = logging.getLogger(__name__)


class WeightSyncManager:
    """Push head/tail LoRA weights from split training into a live vLLM engine.

    The rollout handle can be any object that exposes ``update_lora_adapter``:
    ``vLLMHttpServer``, ``vLLMReplica``, or ``AgentLoopManager``.  This avoids
    recreating the rollout engine or relying on a shared filesystem path.
    """

    def __init__(self, rollout_handle: Any, config):
        self.rollout_handle = rollout_handle
        self.config = config

    def sync_from_worker_group(self, worker_group: SplitRayWorkerGroup) -> bool:
        """Fetch the merged LoRA state dict from ``worker_group`` and load it.

        Args:
            worker_group: A trained ``SplitRayWorkerGroup``.

        Returns:
            ``True`` if the rollout engine reports the adapter was loaded.
        """
        state_dict = worker_group.export_merged_lora_state_dict()
        peft_config = self._build_peft_config()
        return self._push(state_dict, peft_config)

    def sync_from_state_dict(
        self,
        head_state: dict[str, torch.Tensor],
        tail_state: dict[str, torch.Tensor],
        peft_config: dict | None = None,
    ) -> bool:
        """Merge a raw head/tail state dict and push it to the rollout engine.

        This is useful for tests or for callers that already have the state
        dicts in memory.
        """
        n_layers = AutoConfig.from_pretrained(self.config.model_path).num_hidden_layers
        ranges = compute_split_layer_ranges(
            n_layers, self.config.split_stage_0_size, self.config.split_stage_2_size
        )
        state_dict = merge_split_lora_state_dict(head_state, tail_state, ranges)
        if peft_config is None:
            peft_config = self._build_peft_config()
        return self._push(state_dict, peft_config)

    def _push(self, state_dict: dict[str, torch.Tensor], peft_config: dict) -> bool:
        logger.info(
            "Pushing merged LoRA adapter to rollout: %d tensors, peft_config=%s",
            len(state_dict),
            peft_config,
        )
        return ray.get(
            self.rollout_handle.update_lora_adapter.remote(state_dict, peft_config)
        )

    def _build_peft_config(self) -> dict:
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": list(self.config.lora_target_modules),
            "bias": "none",
        }
