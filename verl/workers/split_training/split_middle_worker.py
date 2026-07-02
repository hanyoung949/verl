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

from verl.workers.split_training.engine import SplitTrainingEngine
from verl.single_controller.base import Worker

from verl.utils.distributed import initialize_global_process_group
from verl.utils.split_trace import get_split_trace_logger


class SplitMiddleWorker(Worker):
    """Ray worker for stage_1 of split training.

    Stage_1 is frozen and only forwards activations / backward gradients
    between stage_0 and stage_2. It exposes synchronous train/infer methods so
    the driver can invoke all stages concurrently.

    Supports multi-rank stage_1 (pipeline-split): when stage_1 has multiple
    ranks, each rank handles a subset of layers and relays to the next rank.
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

        # Determine stage from topology for trace logging
        if self._rank in topology["stage_0"]:
            os.environ["SPLIT_STAGE"] = "stage_0"
        elif self._rank in topology["stage_1"]:
            os.environ["SPLIT_STAGE"] = "stage_1"
        else:
            os.environ["SPLIT_STAGE"] = "stage_2"
        self._trace = get_split_trace_logger("train")
        self._train_micro_step = 0

        with self._trace.trace("worker_init"):
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

    def reset(self):
        """Re-initialize the engine (reload frozen middle weights).

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

        with self._trace.trace("worker_reset"):
            # Warm up CUDA context before barrier
            _ = torch.zeros(1, device="cuda")
            print(f"[rank{rank}] cuda context ready {time.time()-t0:.1f}s", flush=True)

            dist.barrier()
            print(f"[rank{rank}] barrier done {time.time()-t0:.1f}s", flush=True)

            self.engine.initialize()
            print(f"[rank{rank}] engine.initialize done {time.time()-t0:.1f}s", flush=True)
        return {"rank": self._rank, "stage": "stage_1"}

    def infer_micro_batch(self, data=None) -> dict:
        """Handle one forward-only pipeline iteration.

        ``data`` is ignored; rank 0 provides inputs through NCCL. This must be
        called concurrently with stage_0/stage_2 ``infer_micro_batch``.
        """
        self._train_micro_step += 1
        with self._trace.trace("infer_micro_batch", micro_step=self._train_micro_step):
            self.engine.stage.run_once(self.engine.topology)
        return {}

    def train_micro_batch(self, data=None) -> dict:
        """Handle one forward/backward pipeline iteration.

        ``data`` is ignored; the real inputs come from rank 0 via NCCL.  This
        method must be called concurrently with stage_0/stage_2 ``train_micro_batch``.
        """
        # run_once raises if it sees an unexpected flag, otherwise returns True.
        self._train_micro_step += 1
        with self._trace.trace("train_micro_batch", micro_step=self._train_micro_step):
            self.engine.stage.run_once(self.engine.topology)
        return {}

    def get_trainable_state_dict(self):
        """Stage_1 is frozen; return an empty state dict."""
        return {}

    def has_grad(self):
        """Return True if any stage_1 parameter has a grad (should be False)."""
        for p in self.engine.stage.layers.parameters():
            if p.grad is not None:
                return True
        if self.engine.stage.rotary_emb is not None:
            for p in self.engine.stage.rotary_emb.parameters():
                if p.grad is not None:
                    return True
        return False

    def shutdown(self):
        """Clean up the process group."""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
        return {"rank": self._rank, "shutdown": True}
