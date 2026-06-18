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

import os
from typing import Any

import torch
import torch.nn.functional as F

from verl.workers.split_training.engine import SplitTrainingEngine
from verl.single_controller.base import Worker
from verl.utils.distributed import initialize_global_process_group
from verl.utils.torch_functional import logprobs_from_logits_naive


class SplitStageWorker(Worker):
    """Ray worker for stage_0 (head) or stage_2 (tail) of split training.

    This worker runs in its own process, holds a SplitPipelineEngine configured
    for either stage_0 or stage_2, and participates in a 3-rank NCCL process group.
    """

    def __init__(self, config):
        import time, os
        t0 = time.time()
        rank_env = os.environ.get("RANK", "?")
        print(f"[rank{rank_env}] __init__ start", flush=True)
        Worker.__init__(self)
        print(f"[rank{rank_env}] Worker.__init__ done {time.time()-t0:.1f}s", flush=True)
        initialize_global_process_group(timeout_second=300)
        print(f"[rank{rank_env}] init_process_group done {time.time()-t0:.1f}s", flush=True)

        self.config = config
        stage_1_tp = getattr(config, 'stage_1_tp', 1)
        if stage_1_tp > 1:
            topology = {
                "stage_0": [0],
                "stage_1": list(range(1, 1 + stage_1_tp)),
                "stage_2": [1 + stage_1_tp],
            }
        else:
            topology = {"stage_0": [0], "stage_1": [1], "stage_2": [2]}

        self.engine = SplitTrainingEngine(
            model_config=config.model_config,
            engine_config=config.engine_config,
            optimizer_config=config.optimizer_config,
            checkpoint_config=config.checkpoint_config,
            topology=topology,
        )

        # Override defaults from the prototype before initialize() loads weights.
        self.engine.model_path = config.model_path
        self.engine.split_stage_0_size = config.split_stage_0_size
        self.engine.split_stage_2_size = config.split_stage_2_size
        self.engine.lr = config.lr
        self.engine.clip_grad = config.clip_grad
        self.engine.lora_r = config.lora_r
        self.engine.lora_alpha = config.lora_alpha
        self.engine.lora_dropout = config.lora_dropout
        self.engine.lora_target_modules = config.lora_target_modules

        self.engine.initialize()
        self.is_head = self.engine.is_head
        self.is_tail = self.engine.is_tail

    @staticmethod
    def _lm_loss(logits: torch.Tensor, data: dict) -> dict[str, Any]:
        """Simple left-shifted language modeling loss."""
        input_ids = data["input_ids"].to(logits.device)
        labels = data.get("labels", input_ids)
        if labels is not input_ids:
            labels = labels.to(logits.device)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return {"loss": loss, "metrics": {}}

    @staticmethod
    def _grpo_loss(logits: torch.Tensor, data: dict) -> dict[str, Any]:
        """Minimal GRPO policy-gradient loss on response tokens."""
        input_ids = data["input_ids"].to(logits.device)
        response_mask = data["response_mask"].to(logits.device).bool()
        old_log_probs = data["old_log_probs"].to(logits.device)
        advantages = data["advantages"].to(logits.device)
        eps = data.get("clip_range", 0.2)

        # Per-token log prob of the observed token at each position.
        new_log_probs = logprobs_from_logits_naive(logits, input_ids)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        pg_loss = -torch.sum(torch.min(surr1, surr2) * response_mask) / (
            response_mask.sum() + 1e-8
        )
        return {"loss": pg_loss, "metrics": {}}

    def _choose_loss_fn(self, data: dict):
        """Pick loss function based on the data fields available."""
        if "old_log_probs" in data:
            return self._grpo_loss
        return self._lm_loss

    def reset(self):
        """Re-initialize the engine (reload weights/optimizers).

        IMPORTANT: all ranks must call reset() concurrently (e.g. via
        ``ray.get([a.reset.remote() for a in actors])``).  The ping below
        is a collective — if called sequentially it will deadlock.
        """
        import time
        import torch
        import torch.distributed as dist
        t0 = time.time()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[rank{rank}] reset start, world_size={world_size}", flush=True)

        dummy = torch.zeros(1, device="cuda")
        print(f"[rank{rank}] cuda context ready {time.time()-t0:.1f}s", flush=True)

        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size
        if rank % 2 == 0:
            dist.send(dummy, dst=next_rank)
            dist.recv(dummy, src=next_rank)
        else:
            dist.recv(dummy, src=prev_rank)
            dist.send(dummy, dst=prev_rank)
        print(f"[rank{rank}] ring ping done {time.time()-t0:.1f}s", flush=True)

        dist.barrier()
        print(f"[rank{rank}] barrier done {time.time()-t0:.1f}s", flush=True)

        self.engine.initialize()
        print(f"[rank{rank}] engine.initialize done {time.time()-t0:.1f}s", flush=True)
        return {"rank": self._rank, "stage": "head" if self.is_head else "tail"}

    def train_micro_batch(self, data: dict) -> dict:
        """Run one forward/backward/optimizer step for stage_0 or stage_2.

        The middle worker must be invoked concurrently for the pipeline to make
        progress, because this method blocks on NCCL send/recv with rank 1.
        """
        loss_fn = self._choose_loss_fn(data)
        outputs = self.engine.train_batch(data, loss_fn)
        return outputs

    def infer_micro_batch(self, data: dict) -> dict:
        """Run one forward-only pass for stage_0 or stage_2."""
        loss_fn = self._choose_loss_fn(data)
        outputs = self.engine.infer_batch(data, loss_fn)
        return outputs

    def get_trainable_state_dict(self):
        """Return the trainable (LoRA) state dict for weight sync."""
        return self.engine.stage.get_trainable_state_dict()

    def shutdown(self):
        """Clean up the process group."""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
        return {"rank": self._rank, "shutdown": True}
