"""SplitPipelineEngine — 3-stage split pipeline 引擎。"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_verl_engine_base", "/root/workspace/dev/verl/verl/workers/engine/base.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BaseEngine = _mod.BaseEngine
EngineRegistry = _mod.EngineRegistry

from .stage import HeadStage, TailStage
from .middle_stage import MiddleStage
from .transport import StageTransport, FWD_ONLY, FWD_WITH_BWD, SHUTDOWN


@EngineRegistry.register(model_type="llm", backend="split_pipeline", device="cuda")
class SplitPipelineEngine(BaseEngine):
    """3-stage split pipeline 引擎。

    rank0: HeadStage (embed + front + LoRA)
    rank1: MiddleStage (middle, 冻结)
    rank2: TailStage (tail + norm + lm_head + LoRA + optimizer)
    """

    def __init__(self, model_config=None, engine_config=None, optimizer_config=None, checkpoint_config=None):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.stage = None
        self.is_head = self.rank == 0
        self.is_middle = self.rank == 1
        self.is_tail = self.rank == 2
        self._parse_config()

    def _parse_config(self):
        if self.model_config is not None:
            self.model_path = getattr(self.model_config, "local_path", "~/share/Qwen2.5-3B-Instruct")
        else:
            self.model_path = "~/share/Qwen2.5-3B-Instruct"
        self.front_end = 4
        self.middle_end = 32
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
        return path

    def _build_lora_config(self):
        return LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                         r=self.lora_r, lora_alpha=self.lora_alpha,
                         lora_dropout=self.lora_dropout, target_modules=self.lora_target_modules)

    def _load_model_parts(self):
        model_path = self._resolve_model_path(self.model_path)
        lora_config = self._build_lora_config()
        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        peft_model = get_peft_model(base_model, lora_config)
        causal_lm = _unwrap_causal_lm(peft_model)
        decoder = _find_decoder_stack(peft_model, causal_lm)
        layers = list(decoder.layers)
        embed_tokens = decoder.embed_tokens
        rotary_emb = getattr(decoder, "rotary_emb", None)
        norm = decoder.norm
        lm_head = causal_lm.lm_head
        return embed_tokens, layers, rotary_emb, norm, lm_head

    def initialize(self):
        embed_tokens, layers, rotary_emb, norm, lm_head = self._load_model_parts()

        if self.is_head:
            self.stage = HeadStage(embed_tokens, layers[:self.front_end], rotary_emb, "cuda")
            self.transport = StageTransport(0, 1, "cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(self._resolve_model_path(self.model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[rank{self.rank}] HeadStage: {len(self.stage.layers)} layers", flush=True)

        elif self.is_middle:
            self.stage = MiddleStage(layers[self.front_end:self.middle_end], rotary_emb, "cuda")
            print(f"[rank{self.rank}] MiddleStage: {len(self.stage.layers)} layers", flush=True)

        elif self.is_tail:
            self.stage = TailStage(layers[self.middle_end:], norm, lm_head, "cuda",
                                  lr=self.lr, clip_grad=self.clip_grad)
            self.tokenizer = AutoTokenizer.from_pretrained(self._resolve_model_path(self.model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[rank{self.rank}] TailStage: {len(self.stage.layers)} layers", flush=True)

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        if self.is_middle:
            return {}
        if self.is_head:
            return self._head_forward_backward(data, loss_function, forward_only)
        if self.is_tail:
            return self._tail_forward_backward(data, loss_function, forward_only)

    def _head_forward_backward(self, data, loss_function, forward_only):
        input_ids = data["input_ids"]
        attention_mask = data.get("attention_mask", torch.ones_like(input_ids))

        ctx = torch.no_grad() if forward_only else nullcontext()
        with ctx:
            h, pos_ids, pos_emb, mask = self.stage.forward_input(input_ids, attention_mask)

            # 发给 Middle: header + h + pos_ids + mask
            flag = FWD_ONLY if forward_only else FWD_WITH_BWD
            B, S, H = h.shape
            has_pos = 1
            has_mask = 0 if mask is None else 1
            dtype_code = 0  # bfloat16
            for val in [flag, B, S, H, has_pos, has_mask, dtype_code]:
                self.transport.send_int(val)
            self.transport.send(h)
            self.transport.send(pos_ids)
            if mask is not None:
                self.transport.send(mask)

            if not forward_only:
                # 等 Middle 的 grad 回来
                grad_h = self.transport.recv(h.shape, h.dtype)
                h.backward(grad_h)

        return {}

    def _tail_forward_backward(self, data, loss_function, forward_only):
        # 从 Middle 接收 h_middle
        B = self.transport.recv_int()
        S = self.transport.recv_int()
        H = self.transport.recv_int()
        dtype_code = self.transport.recv_int()
        tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
        h_middle = self.transport.recv((B, S, H), tensor_dtype)

        ctx = torch.no_grad() if forward_only else nullcontext()
        with ctx:
            h_in = h_middle if forward_only else h_middle.detach().requires_grad_(True)
            logits = self.stage.forward_output(h_in)

            loss = None
            metrics = {}
            if loss_function is not None:
                loss_output = loss_function(logits, data)
                if isinstance(loss_output, dict):
                    loss = loss_output["loss"]
                    metrics = loss_output.get("metrics", {})
                else:
                    loss = loss_output

            if not forward_only and loss is not None:
                loss.backward()
                # 把 grad 发给 Middle
                self.transport.send(h_in.grad)

        result = {"logits": logits.detach(), "metrics": metrics}
        if loss is not None:
            result["loss"] = [loss.detach().item()]
        return result

    def train_batch(self, data, loss_function):
        if self.is_middle:
            return {}
        if self.is_tail:
            self.stage.zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        if self.is_tail:
            grad_norm = self.stage.step()
            if "metrics" not in outputs:
                outputs["metrics"] = {}
            outputs["metrics"]["grad_norm"] = grad_norm
        return outputs

    def infer_batch(self, data, loss_function=None):
        return self.forward_backward_batch(data, loss_function, forward_only=True)

    def optimizer_zero_grad(self):
        if self.is_tail and self.stage is not None:
            self.stage.zero_grad()

    def optimizer_step(self):
        if self.is_tail and self.stage is not None:
            return self.stage.step()
        return 0.0

    def lr_scheduler_step(self):
        return self.lr

    def is_mp_src_rank_with_outputs(self):
        return self.is_tail

    @property
    def is_param_offload_enabled(self):
        return False

    @property
    def is_optimizer_offload_enabled(self):
        return False

    def train_mode(self, **kwargs):
        return nullcontext()

    def eval_mode(self, **kwargs):
        return nullcontext()

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def get_data_parallel_group(self):
        return None

    def to(self, device, model=True, optimizer=True, grad=True):
        pass

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        if not self.is_tail:
            return
        target = Path(local_path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.stage.get_trainable_state_dict(), target / "adapter_state.pt")
        torch.save(self.stage.optimizer.state_dict(), target / "optimizer.pt")
        meta = {"model_path": self.model_path, "front_end": self.front_end,
                "middle_end": self.middle_end, "global_step": global_step}
        (target / "adapter_meta.json").write_text(json.dumps(meta, indent=2))

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **kwargs):
        if not self.is_tail:
            return
        target = Path(local_path)
        state = torch.load(target / "adapter_state.pt", map_location="cuda")
        self.stage.load_trainable_state_dict(state)
        if (target / "optimizer.pt").exists():
            self.stage.optimizer.load_state_dict(torch.load(target / "optimizer.pt", map_location="cuda"))


def _unwrap_causal_lm(model):
    queue = [model]
    visited = set()
    while queue:
        current = queue.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        if hasattr(current, "lm_head") and hasattr(current, "model"):
            return current
        for attr in ("base_model", "model"):
            if hasattr(current, attr):
                queue.append(getattr(current, attr))
    raise ValueError("Cannot unwrap CausalLM.")


def _find_decoder_stack(root_model, causal_lm):
    queue = [root_model, causal_lm, getattr(causal_lm, "model", None)]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if all(hasattr(current, n) for n in ("layers", "embed_tokens", "norm")):
            return current
        for attr in ("base_model", "model"):
            if hasattr(current, attr):
                queue.append(getattr(current, attr))
    raise ValueError("Cannot locate decoder stack.")
