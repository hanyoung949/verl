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

from __future__ import annotations

import logging
import os
import re
import socket
from typing import Any, Optional

import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoConfig, AutoModelForCausalLM

from .split_middle_worker import SplitMiddleWorker
from .split_stage_worker import SplitStageWorker
from .split_utils import (
    compute_split_layer_ranges,
    get_stage_for_layer,
    merge_split_lora_state_dict,
    SplitLayerRanges,
)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class SplitRayWorkerGroup:
    """A minimal worker group that places 3 split-training actors on 3 GPUs.

    This is intentionally lightweight for Phase B0/B1/B2: it does not reuse the
    generic ``RayWorkerGroup`` because head/tail/middle are heterogeneous.
    """

    def __init__(
        self,
        config,
        num_gpus: int = 3,
        name_prefix: str = "split",
        node_ip: Optional[str] = None,
    ) -> None:
        self.config = config
        self.name_prefix = name_prefix
        self.num_gpus = num_gpus
        assert num_gpus == 3, "Phase B0 only supports a 3-stage pipeline (head/middle/tail)."

        self.node_ip = node_ip or ray.util.get_node_ip_address()
        self.master_addr = self.node_ip
        self.master_port = _get_free_port()

        # One placement group with 3 GPU bundles, packed onto the same node.
        bundles = [{"CPU": 4, "GPU": 1} for _ in range(num_gpus)]
        if self.node_ip is not None:
            for b in bundles:
                b[f"node:{self.node_ip}"] = 0.001
        self.pg = placement_group(
            bundles=bundles,
            strategy="STRICT_PACK",
            name=f"{name_prefix}_pg",
        )
        ray.get(self.pg.ready())

        self.actors: list[Any] = []
        self._create_actor(rank=0, cls=SplitStageWorker)
        self._create_actor(rank=1, cls=SplitMiddleWorker)
        self._create_actor(rank=2, cls=SplitStageWorker)

    def _create_actor(self, rank: int, cls):
        env_vars = {
            "WORLD_SIZE": str(self.num_gpus),
            "RANK": str(rank),
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
        }
        actor = (
            ray.remote(cls)
            .options(
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg,
                    placement_group_bundle_index=rank,
                ),
                runtime_env={"env_vars": env_vars},
                name=f"{self.name_prefix}_{cls.__name__}_{rank}",
            )
            .remote(self.config)
        )
        self.actors.append(actor)

    def reset(self):
        """Initialize / re-initialize all stage workers."""
        refs = [a.reset.remote() for a in self.actors]
        return ray.get(refs)

    def train_micro_batch(self, data: dict) -> dict:
        """Run one forward/backward/optimizer step across the 3-stage pipeline."""
        head_ref = self.actors[0].train_micro_batch.remote(data)
        middle_ref = self.actors[1].train_micro_batch.remote()
        tail_ref = self.actors[2].train_micro_batch.remote(data)

        head_out, middle_out, tail_out = ray.get([head_ref, middle_ref, tail_ref])
        return {
            "head": head_out,
            "middle": middle_out,
            "tail": tail_out,
        }

    def get_trainable_state_dict(self):
        """Return trainable state dicts from head and tail."""
        return {
            "head": ray.get(self.actors[0].get_trainable_state_dict.remote()),
            "tail": ray.get(self.actors[2].get_trainable_state_dict.remote()),
        }

    def export_merged_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Return stage_0 + stage_2 LoRA weights as a single in-memory PEFT state dict.

        Middle-stage layers are omitted; the caller should remove any previously
        loaded adapter so that stage_1 falls back to the base model.
        """
        states = self.get_trainable_state_dict()
        n_layers = AutoConfig.from_pretrained(self.config.model_path).num_hidden_layers
        ranges = compute_split_layer_ranges(
            n_layers, self.config.split_stage_0_size, self.config.split_stage_2_size
        )
        return merge_split_lora_state_dict(states["head"], states["tail"], ranges)

    def export_merged_adapter(self, local_path: str, adapter_name: str = "split_adapter") -> str:
        """Export stage_0 + stage_2 LoRA weights as a single PEFT adapter.

        Middle-stage LoRA weights are zeroed out so the merged adapter only
        affects the trainable head/tail layers.
        """
        os.makedirs(local_path, exist_ok=True)
        states = self.get_trainable_state_dict()
        head_state = states["head"]
        tail_state = states["tail"]

        # Delay imports so that CPU-side merge does not require CUDA in the driver.
        from peft import LoraConfig, TaskType, get_peft_model

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
        )
        n_layers = base_model.config.num_hidden_layers

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        peft_model = get_peft_model(base_model, lora_config)

        ranges = compute_split_layer_ranges(
            n_layers, self.config.split_stage_0_size, self.config.split_stage_2_size
        )

        lora_pattern = re.compile(
            r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.default\.weight"
        )
        copied = 0
        zeroed = 0
        for name, param in peft_model.named_parameters():
            match = lora_pattern.match(name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            proj = match.group(2)
            ab = match.group(3)

            stage = get_stage_for_layer(layer_idx, ranges)
            stage_start = ranges[stage][0]
            local_idx = layer_idx - stage_start

            if stage == "stage_1":
                param.data.zero_()
                zeroed += 1
                continue

            src = head_state if stage == "stage_0" else tail_state
            src_key = f"layers.{local_idx}.self_attn.{proj}.lora_{ab}.default.weight"

            if src_key in src:
                param.data.copy_(src[src_key].to(param.device, dtype=param.dtype))
                copied += 1
            else:
                logger.warning(f"Missing LoRA weight {src_key} in stage state; leaving initialized value.")

        logger.info(
            f"Merged adapter: copied {copied} tensors, zeroed {zeroed} middle tensors into {local_path}"
        )
        peft_model.save_pretrained(local_path)
        return local_path

    def shutdown(self):
        # Shutdown actors, kill them, and release the placement group so GPUs
        # are not held after training finishes (important when rollout reuses
        # the same Ray cluster with non-uniform resource pools).
        refs = [a.shutdown.remote() for a in self.actors]
        result = ray.get(refs)
        for a in self.actors:
            ray.kill(a)
        self.actors = []
        if self.pg is not None:
            pg_id = self.pg.id
            remove_placement_group(self.pg)
            # Wait until the placement group is actually removed so the next
            # rollout placement group does not starve for GPUs.
            for _ in range(120):
                table = ray.util.placement_group_table()
                state = table.get(pg_id, {}).get("state")
                if state in ("REMOVED", None):
                    break
                import time
                time.sleep(0.5)
            self.pg = None
        return result
