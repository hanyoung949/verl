"""SplitEngine — 实现 verl BaseEngine 接口的层间拆分训练引擎。

内部通过 NCCL P2P 与 rank1（持有 middle_layers）通信。
启动方式：torchrun --nproc_per_node=2
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.device import get_device_name

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_verl_engine_base", "/root/workspace/dev/verl/verl/workers/engine/base.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BaseEngine = _mod.BaseEngine
EngineRegistry = _mod.EngineRegistry

from .middle_executor import NCCLMiddleExecutor
from .middle_worker import MiddleWorker


@EngineRegistry.register(model_type="llm", backend="split", device="cuda")
class SplitEngine(BaseEngine):
    def __init__(self, model_config=None, engine_config=None, optimizer_config=None, checkpoint_config=None):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.split_core: Optional[nn.Module] = None
        self.middle_worker: Optional[MiddleWorker] = None
        self.optimizer = None
        self.tokenizer = None
        self.mode = None
        self._parse_config()

    def _parse_config(self):
        if self.model_config is not None:
            self.model_path = getattr(self.model_config, "local_path", "~/share/Qwen2.5-3B-Instruct")
        else:
            self.model_path = "~/share/Qwen2.5-3B-Instruct"
        if self.engine_config is not None:
            self.front_end = int(getattr(self.engine_config, "front_end", 4))
            self.middle_end = int(getattr(self.engine_config, "middle_end", 32))
        else:
            self.front_end = 4
            self.middle_end = 32
        if self.optimizer_config is not None:
            self.lr = float(getattr(self.optimizer_config, "lr", 1e-4))
            self.clip_grad = float(getattr(self.optimizer_config, "clip_grad", 1.0))
        else:
            self.lr = 1e-4
            self.clip_grad = 1.0
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.0
        self.lora_target_modules = ["q_proj", "v_proj"]

    def _resolve_model_path(self, path):
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded) and Path(expanded).exists():
            return expanded
        if Path(expanded).exists():
            return str(Path(expanded).resolve())
        return path

    def _build_lora_config(self):
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=self.lora_r, lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout, target_modules=self.lora_target_modules,
        )

    def initialize(self):
        model_path = self._resolve_model_path(self.model_path)
        lora_config = self._build_lora_config()
        if self.rank == 1:
            device = torch.device("cuda")
            self.middle_worker = MiddleWorker.from_pretrained(
                model_path=model_path, front_end=self.front_end, middle_end=self.middle_end,
                lora_config=lora_config, device=device, torch_dtype=torch.bfloat16,
            )
            print(f"[SplitEngine rank1] initialized: middle_layers={len(self.middle_worker.middle_layers)}")
            return
        from .split_actor_v3 import SplitActorCore
        self.split_core = SplitActorCore(
            model_path=model_path, front_end=self.front_end, middle_end=self.middle_end,
            lora_config=lora_config, device="cuda", torch_dtype=torch.bfloat16,
        )
        self.split_core.reset_runtime_stats()
        self.optimizer = torch.optim.AdamW(self.split_core.trainable_parameters(), lr=self.lr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[SplitEngine rank0] initialized: trainable_params={len(self.split_core.trainable_parameters())}")

    @staticmethod
    def _normalize_batch(data):
        """将 DataProto 或 TensorDict 统一转换为 TensorDict。

        返回 (batch_tensor_dict, non_tensor_batch, meta_info)。
        """
        try:
            from verl.protocol import DataProto
            if isinstance(data, DataProto):
                return data.batch, data.non_tensor_batch, data.meta_info
        except ImportError:
            pass
        if isinstance(data, TensorDict):
            return data, {}, {}
        if isinstance(data, dict):
            return TensorDict(data, batch_size=[]), {}, {}
        raise TypeError(f"Unsupported batch type: {type(data)}")

    def forward_backward_batch(self, data, loss_function: Callable, forward_only=False):
        if self.rank == 1:
            return {}
        batch, non_tensor, meta = self._normalize_batch(data)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        ctx = torch.no_grad() if forward_only else nullcontext()
        with ctx:
            logits = self.split_core.forward_full(input_ids, attention_mask)
            if loss_function is not None:
                loss_output = loss_function(logits, batch)
                if isinstance(loss_output, dict):
                    loss = loss_output["loss"]
                    metrics = loss_output.get("metrics", {})
                else:
                    loss = loss_output
                    metrics = {}
            else:
                loss = None
                metrics = {}
            if not forward_only and loss is not None:
                loss.backward()
        result = {"logits": logits.detach(), "metrics": metrics}
        if loss is not None:
            result["loss"] = [loss.detach().item()]
        return result

    def train_batch(self, data, loss_function: Callable):
        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.optimizer_step()
        if "metrics" not in outputs:
            outputs["metrics"] = {}
        outputs["metrics"]["grad_norm"] = grad_norm
        return outputs

    def infer_batch(self, data, loss_function: Optional[Callable] = None):
        with torch.no_grad():
            return self.forward_backward_batch(data, loss_function, forward_only=True)

    def optimizer_zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        if self.optimizer is None or self.split_core is None:
            return 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(self.split_core.trainable_parameters(), max_norm=self.clip_grad)
        self.optimizer.step()
        if isinstance(grad_norm, torch.Tensor):
            return grad_norm.item()
        return float(grad_norm)

    def lr_scheduler_step(self):
        return self.lr

    @property
    def is_param_offload_enabled(self):
        return False

    @property
    def is_optimizer_offload_enabled(self):
        return False

    def train_mode(self, **kwargs):
        if self.split_core is not None:
            self.split_core.train()
        return nullcontext()

    def eval_mode(self, **kwargs):
        if self.split_core is not None:
            self.split_core.eval()
        return nullcontext()

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def get_data_parallel_group(self):
        return None

    def is_mp_src_rank_with_outputs(self):
        return self.rank == 0

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        pass

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        if self.rank != 0 or self.split_core is None:
            return
        target = Path(local_path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.split_core.get_trainable_state_dict(), target / "adapter_state.pt")
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), target / "optimizer.pt")
        meta = {
            "model_path": self.model_path, "front_end": self.front_end,
            "middle_end": self.middle_end, "global_step": global_step,
        }
        (target / "adapter_meta.json").write_text(json.dumps(meta, indent=2))

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **kwargs):
        if self.rank != 0 or self.split_core is None:
            return
        target = Path(local_path)
        state = torch.load(target / "adapter_state.pt", map_location="cuda")
        self.split_core.load_trainable_state_dict(state)
        if self.optimizer is not None and (target / "optimizer.pt").exists():
            self.optimizer.load_state_dict(torch.load(target / "optimizer.pt", map_location="cuda"))
