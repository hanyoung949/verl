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


class SplitMiddleWorker(Worker):
    """Ray worker for stage_1 of split training.

    Stage_1 is frozen and only forwards activations / backward gradients
    between stage_0 and stage_2. It exposes synchronous train/infer methods so
    the driver can invoke all stages concurrently.

    Supports multi-rank stage_1 (pipeline-split): when stage_1 has multiple
    ranks, each rank handles a subset of layers and relays to the next rank.
    """

    def __init__(self, config):
        Worker.__init__(self)
        initialize_global_process_group(timeout_second=300)
        # Force NCCL communicator initialization for all ranks.
        import torch.distributed as dist
        dist.barrier()

        self.config = config
        # Build topology from config or use default 3-rank layout.
        stage_1_tp = getattr(config, 'stage_1_tp', 1)
        self._stage_1_tp = stage_1_tp
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

    def reset(self):
        """Re-initialize the engine (reload frozen middle weights)."""
        # Create TP process group in reset() (collective: all ranks must participate).
        if self._stage_1_tp > 1:
            import torch.distributed as dist
            stage_1_ranks = list(range(1, 1 + self._stage_1_tp))
            self._tp_group = dist.new_group(ranks=stage_1_ranks)
            self.engine._tp_group = self._tp_group
        self.engine.initialize()
        return {"rank": self._rank, "stage": "stage_1"}

    def infer_micro_batch(self, data=None) -> dict:
        """Handle one forward-only pipeline iteration.

        ``data`` is ignored; rank 0 provides inputs through NCCL. This must be
        called concurrently with stage_0/stage_2 ``infer_micro_batch``.
        """
        self.engine.stage.run_once(self.engine.topology)
        return {}

    def train_micro_batch(self, data=None) -> dict:
        """Handle one forward/backward pipeline iteration.

        ``data`` is ignored; the real inputs come from rank 0 via NCCL.  This
        method must be called concurrently with stage_0/stage_2 ``train_micro_batch``.
        """
        # run_once raises if it sees an unexpected flag, otherwise returns True.
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
