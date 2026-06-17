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
    """A worker group that places split-training actors on GPUs.

    Supports both 3-rank (stage_1 TP=1) and N-rank (stage_1 TP>1) layouts.
    For 4 ranks: rank0=stage_0, rank1=stage_1_tp0, rank2=stage_1_tp1, rank3=stage_2.
    """

    def __init__(
        self,
        config,
        num_gpus: int = 3,
        name_prefix: str = "split",
        node_ip: Optional[str] = None,
        stage_1_tp: int = 1,
    ) -> None:
        self.config = config
        self.name_prefix = name_prefix
        self.num_gpus = num_gpus
        self.stage_1_tp = stage_1_tp
        expected_gpus = 2 + stage_1_tp  # stage_0 + stage_1_tp + stage_2
        assert num_gpus == expected_gpus, (
            f"Expected {expected_gpus} GPUs for stage_1_tp={stage_1_tp}, got {num_gpus}"
        )

        self.node_ip = node_ip or ray.util.get_node_ip_address()
        self.master_addr = self.node_ip
        self.master_port = _get_free_port()

        # Build topology: stage_0=[0], stage_1=[1..stage_1_tp], stage_2=[1+stage_1_tp]
        self.topology = {
            "stage_0": [0],
            "stage_1": list(range(1, 1 + stage_1_tp)),
            "stage_2": [1 + stage_1_tp],
        }

        bundles = [{"CPU": 4, "GPU": 1} for _ in range(num_gpus)]
        if self.node_ip is not None:
            for b in bundles:
                b[f"node:{self.node_ip}"] = 0.001
        self.pg = placement_group(
            bundles=bundles, strategy="STRICT_PACK", name=f"{name_prefix}_pg",
        )
        ray.get(self.pg.ready())

        self.actors: list[Any] = []
        self._create_actor(rank=0, cls=SplitStageWorker)
        for r in range(1, 1 + stage_1_tp):
            self._create_actor(rank=r, cls=SplitMiddleWorker)
        self._create_actor(rank=1 + stage_1_tp, cls=SplitStageWorker)

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

    @property
    def stage_0_actor(self):
        return self.actors[0]

    @property
    def stage_2_actor(self):
        return self.actors[-1]

    @property
    def stage_1_actors(self):
        return self.actors[1:-1]

    def reset(self):
        refs = [a.reset.remote() for a in self.actors]
        return ray.get(refs)

    def train_micro_batch(self, data: dict) -> dict:
        """Run one forward/backward/optimizer step across the pipeline."""
        refs = []
        for i, actor in enumerate(self.actors):
            if i == 0 or i == len(self.actors) - 1:
                refs.append(actor.train_micro_batch.remote(data))
            else:
                refs.append(actor.train_micro_batch.remote())
        results = ray.get(refs)
        out = {}
        for i, r in enumerate(results):
            stage = "stage_0" if i == 0 else ("stage_2" if i == len(self.actors) - 1 else f"stage_1_tp{i-1}")
            out[stage] = r
        out["head"] = out["stage_0"]
        out["tail"] = out["stage_2"]
        return out

    def infer_micro_batch(self, data: dict) -> dict:
        """Run one forward-only pass across the pipeline."""
        refs = []
        for i, actor in enumerate(self.actors):
            if i == 0 or i == len(self.actors) - 1:
                refs.append(actor.infer_micro_batch.remote(data))
            else:
                refs.append(actor.infer_micro_batch.remote())
        results = ray.get(refs)
        out = {}
        for i, r in enumerate(results):
            stage = "stage_0" if i == 0 else ("stage_2" if i == len(self.actors) - 1 else f"stage_1_tp{i-1}")
            out[stage] = r
        out["head"] = out["stage_0"]
        out["tail"] = out["stage_2"]
        return out

    def get_trainable_state_dict(self):
        stage_0_state = ray.get(self.actors[0].get_trainable_state_dict.remote())
        stage_2_state = ray.get(self.actors[-1].get_trainable_state_dict.remote())
        return {"stage_0": stage_0_state, "stage_2": stage_2_state, "head": stage_0_state, "tail": stage_2_state}

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
        return merge_split_lora_state_dict(states["stage_0"], states["stage_2"], ranges)

    def export_merged_adapter(self, local_path: str, adapter_name: str = "split_adapter") -> str:
        """Export stage_0 + stage_2 LoRA weights as a single PEFT adapter.

        Middle-stage LoRA weights are zeroed out so the merged adapter only
        affects the trainable stage_0/stage_2 layers.
        """
        os.makedirs(local_path, exist_ok=True)
        states = self.get_trainable_state_dict()
        stage_0_state = states["stage_0"]
        stage_2_state = states["stage_2"]

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

            src = stage_0_state if stage == "stage_0" else stage_2_state
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
        refs = [a.shutdown.remote() for a in self.actors]
        result = ray.get(refs)
        for a in self.actors:
            ray.kill(a)
        self.actors = []
        if self.pg is not None:
            pg_id = self.pg.id
            remove_placement_group(self.pg)
            for _ in range(120):
                table = ray.util.placement_group_table()
                state = table.get(pg_id, {}).get("state")
                if state in ("REMOVED", None):
                    break
                import time
                time.sleep(0.5)
            self.pg = None
        return result
