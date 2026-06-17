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

"""SplitTrainingEngine — production 3-stage split pipeline training engine."""

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

from verl.workers.engine.base import BaseEngine, EngineRegistry
from verl.utils.torch_functional import logprobs_from_logits

from ..split_utils import compute_split_layer_ranges
from .middle_stage import Stage1
from .stage import Stage0, Stage2
from .transport import StageTransport, FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE


@EngineRegistry.register(model_type="llm", backend="split_training", device="cuda")
class SplitTrainingEngine(BaseEngine):
    """3-stage split pipeline training engine aligned with vLLM split naming.

    Rank layout (single-rank stages for Phase C0):
        rank 0 -> stage_0 (Edge Head, trainable)
        rank 1 -> stage_1 (Cloud Body, frozen)
        rank 2 -> stage_2 (Edge Tail, trainable)
    """

    def __init__(
        self,
        model_config=None,
        engine_config=None,
        optimizer_config=None,
        checkpoint_config=None,
        topology=None,
    ):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.stage = None
        self.topology = topology or {"stage_0": [0], "stage_1": [1], "stage_2": [2]}
        self._resolve_stage()
        self._parse_config()

    def _resolve_stage(self):
        t = self.topology
        self.is_stage_0 = self.rank in t.get("stage_0", [0])
        self.is_stage_1 = self.rank in t.get("stage_1", [1])
        self.is_stage_2 = self.rank in t.get("stage_2", [2])
        # Convenience aliases for code that still thinks in head/middle/tail.
        self.is_head = self.is_stage_0
        self.is_middle = self.is_stage_1
        self.is_tail = self.is_stage_2

    @property
    def stage_0_rank(self):
        return self.topology["stage_0"][0]

    @property
    def stage_2_rank(self):
        return self.topology["stage_2"][0]

    @property
    def stage_1_first_rank(self):
        return self.topology["stage_1"][0]

    @property
    def stage_1_last_rank(self):
        return self.topology["stage_1"][-1]

    def _parse_config(self):
        if self.model_config is not None:
            self.model_path = getattr(self.model_config, "local_path", "")
        else:
            self.model_path = ""
        self.split_stage_0_size = 4
        self.split_stage_2_size = 4
        self.lr = 1e-4
        self.clip_grad = 1.0
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.0
        self.lora_target_modules = ["q_proj", "v_proj"]

    def _resolve_model_path(self, path: str) -> str:
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded) and Path(expanded).exists():
            return expanded
        return path

    def _build_lora_config(self):
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
        )

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
        num_layers = len(layers)
        ranges = compute_split_layer_ranges(
            num_layers, self.split_stage_0_size, self.split_stage_2_size
        )
        self.stage_0_end = ranges["stage_0"][1]
        self.stage_1_end = ranges["stage_1"][1]

        if self.is_stage_0:
            stage_0_layers = [l.to("cuda") for l in layers[:self.stage_0_end]]
            self.stage = Stage0(
                embed_tokens.to("cuda"),
                stage_0_layers,
                rotary_emb.to("cuda") if rotary_emb is not None else None,
                "cuda",
                lr=self.lr,
                clip_grad=self.clip_grad,
            )
            self.transport = StageTransport(self.rank, self.stage_1_first_rank, "cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(self._resolve_model_path(self.model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[rank{self.rank}] Stage0: {len(self.stage.layers)} layers", flush=True)

        elif self.is_stage_1:
            stage_1_layers_all = layers[self.stage_0_end : self.stage_1_end]
            n_stage1 = len(self.topology.get("stage_1", [1]))
            if n_stage1 > 1:
                # Multi-rank stage_1: split layers across ranks.
                rank_in_stage1 = self.topology["stage_1"].index(self.rank)
                per_rank = len(stage_1_layers_all) // n_stage1
                start = rank_in_stage1 * per_rank
                end = start + per_rank if rank_in_stage1 < n_stage1 - 1 else len(stage_1_layers_all)
                my_layers = stage_1_layers_all[start:end]
                src = self.stage_0_rank if rank_in_stage1 == 0 else self.topology["stage_1"][rank_in_stage1 - 1]
                dst = self.stage_2_rank if rank_in_stage1 == n_stage1 - 1 else self.topology["stage_1"][rank_in_stage1 + 1]
            else:
                my_layers = stage_1_layers_all
                src = None  # will use topology default
                dst = None
            self.stage = Stage1(
                [l.to("cuda") for l in my_layers],
                rotary_emb.to("cuda") if rotary_emb is not None else None,
                "cuda",
                src_rank=src,
                dst_rank=dst,
            )
            print(f"[rank{self.rank}] Stage1: {len(self.stage.layers)} layers (of {len(stage_1_layers_all)} total)", flush=True)

        elif self.is_stage_2:
            stage_2_layers = [l.to("cuda") for l in layers[self.stage_1_end :]]
            self.stage = Stage2(
                stage_2_layers,
                norm.to("cuda"),
                lm_head.to("cuda"),
                "cuda",
                lr=self.lr,
                clip_grad=self.clip_grad,
                rotary_emb=rotary_emb.to("cuda") if rotary_emb is not None else None,
            )
            self.transport = StageTransport(self.rank, self.stage_1_last_rank, "cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(self._resolve_model_path(self.model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[rank{self.rank}] Stage2: {len(self.stage.layers)} layers", flush=True)

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        if self.is_stage_1:
            return {}
        if self.is_stage_0:
            return self._stage_0_forward_backward(data, loss_function, forward_only)
        if self.is_stage_2:
            return self._stage_2_forward_backward(data, loss_function, forward_only)

    def _stage_0_forward_backward(self, data, loss_function, forward_only):
        input_ids = data["input_ids"]
        attention_mask = data.get("attention_mask", torch.ones_like(input_ids))

        ctx = torch.no_grad() if forward_only else nullcontext()
        with ctx:
            h, pos_ids, pos_emb, mask = self.stage.forward_input(input_ids, attention_mask)
            B, S, H = h.shape
            has_mask = 0 if mask is None else 1
            flag = FWD_ONLY if forward_only else FWD_WITH_BWD
            header = torch.tensor([flag, B, S, H, has_mask, 0], dtype=torch.int64, device="cuda")
            self.transport.send(header)
            self.transport.send(h)
            self.transport.send(pos_ids)
            if mask is not None:
                self.transport.send(mask)

            if not forward_only:
                grad_h = self.transport.recv(h.shape, h.dtype)
                h.backward(grad_h)
                return {}
            else:
                result_shape = torch.zeros(3, dtype=torch.int64, device="cuda")
                dist.recv(result_shape, src=self.stage_2_rank)
                logits = torch.empty(
                    result_shape[0].item(),
                    result_shape[1].item(),
                    result_shape[2].item(),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                dist.recv(logits, src=self.stage_2_rank)
                return {"logits": logits.detach()}

        return {}

    def _stage_2_forward_backward(self, data, loss_function, forward_only):
        header = torch.zeros(6, dtype=torch.int64, device="cuda")
        dist.recv(header, src=self.stage_1_last_rank)
        flag = int(header[0].item())
        if flag == PIPELINE_DONE:
            return {}
        _flag, B, S, H, _has_mask, dtype_code = [int(x) for x in header.tolist()]
        tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
        h_middle = self.transport.recv((B, S, H), tensor_dtype)
        pos_ids = self.transport.recv((B, S), torch.int64)

        pos_emb = None
        if self.stage.rotary_emb is not None:
            try:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), pos_ids.to("cuda"))
            except TypeError:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), seq_len=S)

        ctx = torch.no_grad() if forward_only else nullcontext()
        with ctx:
            h_in = h_middle if forward_only else h_middle.detach().requires_grad_(True)
            logits = self.stage.forward_output(
                h_in, position_ids=pos_ids, position_embeddings=pos_emb
            )

            loss = None
            metrics = {}
            if loss_function is not None:
                loss_output = loss_function(logits, data)
                if isinstance(loss_output, dict):
                    loss = loss_output["loss"]
                    metrics = loss_output.get("metrics", {})
                else:
                    loss = loss_output

            if forward_only:
                dist.send(
                    torch.tensor(
                        [logits.shape[0], logits.shape[1], logits.shape[2]],
                        dtype=torch.int64,
                        device="cuda",
                    ),
                    dst=self.stage_0_rank,
                )
                dist.send(logits, dst=self.stage_0_rank)
            elif loss is not None:
                loss.backward()
                self.transport.send(h_in.grad)

        result = {"logits": logits.detach(), "metrics": metrics}
        if loss is not None:
            result["loss"] = [loss.detach().item()]
        return result

    def train_batch(self, data, loss_function):
        if self.is_stage_1:
            return {}
        self.stage.zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.stage.step()
        if "metrics" not in outputs:
            outputs["metrics"] = {}
        outputs["metrics"]["grad_norm"] = grad_norm
        return outputs

    def infer_batch(self, data, loss_function=None):
        return self.forward_backward_batch(data, loss_function, forward_only=True)

    def sample_next_token(self, data, temperature=1.0, pad_token_id=0):
        """Stage_2 local sampling; only next_token + log_prob are sent back."""
        if self.is_stage_1:
            return {}
        if self.is_stage_0:
            return self._stage_0_sample(data, temperature, pad_token_id)
        if self.is_stage_2:
            return self._stage_2_sample(data, temperature, pad_token_id)

    def _stage_0_sample(self, data, temperature, pad_token_id):
        input_ids = data["input_ids"]
        attention_mask = data.get("attention_mask", torch.ones_like(input_ids))
        finished = data.get("finished", torch.zeros(input_ids.size(0), dtype=torch.bool, device="cuda"))

        with torch.no_grad():
            h, pos_ids, pos_emb, mask = self.stage.forward_input(input_ids, attention_mask)
            B, S, H = h.shape
            has_mask = 0 if mask is None else 1
            header = torch.tensor([FWD_ONLY, B, S, H, has_mask, 0], dtype=torch.int64, device="cuda")
            self.transport.send(header)
            self.transport.send(h)
            self.transport.send(pos_ids)
            if mask is not None:
                self.transport.send(mask)

            dist.send(torch.tensor([temperature], dtype=torch.float32, device="cuda"), dst=self.stage_2_rank)
            dist.send(finished.to(torch.int64), dst=self.stage_2_rank)
            dist.send(torch.tensor([pad_token_id], dtype=torch.int64, device="cuda"), dst=self.stage_2_rank)

            next_token = torch.empty(B, 1, dtype=torch.int64, device="cuda")
            dist.recv(next_token, src=self.stage_2_rank)
            log_prob = torch.empty(B, 1, dtype=torch.float32, device="cuda")
            dist.recv(log_prob, src=self.stage_2_rank)

            return {"next_token": next_token, "log_prob": log_prob}

    def _stage_2_sample(self, data, temperature, pad_token_id):
        header = torch.zeros(6, dtype=torch.int64, device="cuda")
        dist.recv(header, src=self.stage_1_last_rank)
        flag = int(header[0].item())
        if flag == PIPELINE_DONE:
            return {}
        _flag, B, S, H, _has_mask, dtype_code = [int(x) for x in header.tolist()]
        tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
        h_middle = self.transport.recv((B, S, H), tensor_dtype)
        pos_ids = self.transport.recv((B, S), torch.int64)

        temp_tensor = torch.zeros(1, dtype=torch.float32, device="cuda")
        dist.recv(temp_tensor, src=self.stage_0_rank)
        temperature = temp_tensor.item()

        finished_int = torch.zeros(B, dtype=torch.int64, device="cuda")
        dist.recv(finished_int, src=self.stage_0_rank)
        finished = finished_int.to(torch.bool)

        pad_id_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.recv(pad_id_tensor, src=self.stage_0_rank)
        pad_token_id = int(pad_id_tensor.item())

        pos_emb = None
        if self.stage.rotary_emb is not None:
            try:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), pos_ids.to("cuda"))
            except TypeError:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), seq_len=S)

        with torch.no_grad():
            logits = self.stage.forward_output(h_middle, position_ids=pos_ids, position_embeddings=pos_emb)
            last_logits = logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, pad_token_id),
                next_token,
            )

            log_prob = torch.log_softmax(last_logits, dim=-1)
            log_prob = torch.gather(log_prob, dim=-1, index=next_token).float()

        dist.send(next_token, dst=self.stage_0_rank)
        dist.send(log_prob, dst=self.stage_0_rank)
        return {"next_token": next_token, "log_prob": log_prob}

    def optimizer_zero_grad(self):
        if (self.is_stage_0 or self.is_stage_2) and self.stage is not None:
            self.stage.zero_grad()

    def optimizer_step(self):
        if (self.is_stage_0 or self.is_stage_2) and self.stage is not None:
            return self.stage.step()
        return 0.0

    def lr_scheduler_step(self):
        return self.lr

    def is_mp_src_rank_with_outputs(self):
        return self.is_stage_2

    @property
    def is_param_offload_enabled(self):
        return False

    @property
    def is_optimizer_offload_enabled(self):
        return False

    def train_mode(self, **kwargs):
        if self.stage is not None:
            self.stage.train()
        from verl.workers.engine.base import BaseEngineCtx

        return BaseEngineCtx(self, "train", **kwargs)

    def eval_mode(self, **kwargs):
        if self.stage is not None:
            self.stage.eval()
        from verl.workers.engine.base import BaseEngineCtx

        return BaseEngineCtx(self, "eval", **kwargs)

    def get_per_tensor_param(self, **kwargs):
        peft_config = None

        def _gen():
            if self.stage is not None:
                for name, param in self.stage.named_parameters():
                    yield name, param

        return _gen(), peft_config

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def get_data_parallel_group(self):
        return None

    def to(self, device, model=True, optimizer=True, grad=True):
        pass

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        target = Path(local_path)
        target.mkdir(parents=True, exist_ok=True)
        if self.is_stage_0:
            torch.save(self.stage.get_trainable_state_dict(), target / "stage_0_adapter_state.pt")
            torch.save(self.stage.optimizer.state_dict(), target / "stage_0_optimizer.pt")
        elif self.is_stage_2:
            torch.save(self.stage.get_trainable_state_dict(), target / "stage_2_adapter_state.pt")
            torch.save(self.stage.optimizer.state_dict(), target / "stage_2_optimizer.pt")
            meta = {
                "model_path": self.model_path,
                "split_stage_0_size": self.split_stage_0_size,
                "split_stage_2_size": self.split_stage_2_size,
                "global_step": global_step,
            }
            (target / "adapter_meta.json").write_text(json.dumps(meta, indent=2))

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **kwargs):
        target = Path(local_path)
        if self.is_stage_0:
            state = torch.load(target / "stage_0_adapter_state.pt", map_location="cuda")
            self.stage.load_trainable_state_dict(state)
            if (target / "stage_0_optimizer.pt").exists():
                self.stage.optimizer.load_state_dict(
                    torch.load(target / "stage_0_optimizer.pt", map_location="cuda")
                )
        elif self.is_stage_2:
            state = torch.load(target / "stage_2_adapter_state.pt", map_location="cuda")
            self.stage.load_trainable_state_dict(state)
            if (target / "stage_2_optimizer.pt").exists():
                self.stage.optimizer.load_state_dict(
                    torch.load(target / "stage_2_optimizer.pt", map_location="cuda")
                )


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
