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
_spec = _ilu.spec_from_file_location("_verl_engine_base", "/root/workspace/verl/verl/workers/engine/base.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BaseEngine = _mod.BaseEngine
EngineRegistry = _mod.EngineRegistry

from .stage import HeadStage, TailStage
from .middle_stage import MiddleStage
from .transport import StageTransport, FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE
from verl.utils.torch_functional import logprobs_from_logits


@EngineRegistry.register(model_type="llm", backend="split_pipeline", device="cuda")
class SplitPipelineEngine(BaseEngine):
    """3-stage split pipeline 引擎。

    rank0: HeadStage (embed + front + LoRA)
    rank1: MiddleStage (middle, 冻结)
    rank2: TailStage (tail + norm + lm_head + LoRA + optimizer)
    """

    def __init__(self, model_config=None, engine_config=None, optimizer_config=None, checkpoint_config=None, topology=None):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.stage = None
        self.topology = topology or {"head": [0], "middle": [1], "tail": [2]}
        self._resolve_stage()
        self._parse_config()

    def _resolve_stage(self):
        t = self.topology
        self.is_head = self.rank in t.get("head", [0])
        self.is_middle = self.rank in t.get("middle", [1])
        self.is_tail = self.rank in t.get("tail", [2])

    @property
    def head_rank(self):
        return self.topology["head"][0]

    @property
    def tail_rank(self):
        return self.topology["tail"][0]

    @property
    def middle_first_rank(self):
        return self.topology["middle"][0]

    @property
    def middle_last_rank(self):
        return self.topology["middle"][-1]

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
            head_layers = [l.to("cuda") for l in layers[:self.front_end]]
            self.stage = HeadStage(embed_tokens.to("cuda"), head_layers,
                                   rotary_emb.to("cuda") if rotary_emb is not None else None, "cuda",
                                   lr=self.lr, clip_grad=self.clip_grad)
            self.transport = StageTransport(self.rank, self.middle_first_rank, "cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(self._resolve_model_path(self.model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[rank{self.rank}] HeadStage: {len(self.stage.layers)} layers", flush=True)

        elif self.is_middle:
            mid_layers = [l.to("cuda") for l in layers[self.front_end:self.middle_end]]
            self.stage = MiddleStage(mid_layers,
                                     rotary_emb.to("cuda") if rotary_emb is not None else None, "cuda")
            print(f"[rank{self.rank}] MiddleStage: {len(self.stage.layers)} layers", flush=True)

        elif self.is_tail:
            tail_layers = [l.to("cuda") for l in layers[self.middle_end:]]
            self.stage = TailStage(tail_layers, norm.to("cuda"), lm_head.to("cuda"), "cuda",
                                  lr=self.lr, clip_grad=self.clip_grad,
                                  rotary_emb=rotary_emb.to("cuda") if rotary_emb is not None else None)
            self.transport = StageTransport(self.rank, self.middle_last_rank, "cuda")
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
                # FWD_WITH_BWD: Tail 算 loss.backward() → grad 经 Middle 回 Head
                # Head 收 grad_h 从 Middle
                grad_h = self.transport.recv(h.shape, h.dtype)
                h.backward(grad_h)
                return {}
            else:
                # FWD_ONLY: 收 Tail 返回的 logits（shape 包含 vocab size，避免硬编码）
                result_shape = torch.zeros(3, dtype=torch.int64, device="cuda")
                tail_rank = self.tail_rank
                dist.recv(result_shape, src=tail_rank)
                logits = torch.empty(result_shape[0].item(), result_shape[1].item(), result_shape[2].item(),
                                    dtype=torch.bfloat16, device="cuda")
                dist.recv(logits, src=tail_rank)
                return {"logits": logits.detach()}

        return {}

    def _tail_forward_backward(self, data, loss_function, forward_only):
        # 从 Middle 接收 header + activation + pos_ids
        header = torch.zeros(6, dtype=torch.int64, device="cuda")
        dist.recv(header, src=1)
        flag = int(header[0].item())
        if flag == PIPELINE_DONE:
            # Head 通知 pipeline 结束，直接返回空结果让调用方 break
            return {}
        _flag, B, S, H, _has_mask, dtype_code = [int(x) for x in header.tolist()]
        tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
        h_middle = self.transport.recv((B, S, H), tensor_dtype)
        pos_ids = self.transport.recv((B, S), torch.int64)

        # 用 Tail 自己的 rotary_emb 重算 position_embeddings
        pos_emb = None
        if self.stage.rotary_emb is not None:
            try:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), pos_ids.to("cuda"))
            except TypeError:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), seq_len=S)

        ctx = torch.no_grad() if forward_only else nullcontext()
        with ctx:
            h_in = h_middle if forward_only else h_middle.detach().requires_grad_(True)
            logits = self.stage.forward_output(h_in, position_ids=pos_ids, position_embeddings=pos_emb)

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
                # FWD_ONLY: 发送 logits 回 Head（直接 P2P，不经过 Middle）
                head_rank = self.head_rank
                dist.send(torch.tensor([logits.shape[0], logits.shape[1], logits.shape[2]],
                                       dtype=torch.int64, device="cuda"), dst=head_rank)
                dist.send(logits, dst=head_rank)
            elif loss is not None:
                loss.backward()
                # 把 grad 发给 Middle（Middle 再转发给 Head）
                self.transport.send(h_in.grad)

        result = {"logits": logits.detach(), "metrics": metrics}
        if loss is not None:
            result["loss"] = [loss.detach().item()]
        return result

    def train_batch(self, data, loss_function):
        if self.is_middle:
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
        """Tail 本地采样，只回传 next_token + log_prob。

        通信量从 O(B*S*vocab) 降到 O(B)。
        """
        if self.is_middle:
            return {}
        if self.is_head:
            return self._head_sample(data, temperature, pad_token_id)
        if self.is_tail:
            return self._tail_sample(data, temperature, pad_token_id)

    def _head_sample(self, data, temperature, pad_token_id):
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

            # 发送 temperature + finished_mask + pad_token_id（直接 P2P 到 Tail）
            tail_rank = self.tail_rank
            dist.send(torch.tensor([temperature], dtype=torch.float32, device="cuda"), dst=tail_rank)
            dist.send(finished.to(torch.int64), dst=tail_rank)
            dist.send(torch.tensor([pad_token_id], dtype=torch.int64, device="cuda"), dst=tail_rank)

            # 接收 next_token [B, 1] + log_prob [B, 1]
            next_token = torch.empty(B, 1, dtype=torch.int64, device="cuda")
            dist.recv(next_token, src=tail_rank)
            log_prob = torch.empty(B, 1, dtype=torch.float32, device="cuda")
            dist.recv(log_prob, src=tail_rank)

            return {"next_token": next_token, "log_prob": log_prob}

    def _tail_sample(self, data, temperature, pad_token_id):
        header = torch.zeros(6, dtype=torch.int64, device="cuda")
        dist.recv(header, src=1)
        flag = int(header[0].item())
        if flag == PIPELINE_DONE:
            return {}
        _flag, B, S, H, _has_mask, dtype_code = [int(x) for x in header.tolist()]
        tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
        h_middle = self.transport.recv((B, S, H), tensor_dtype)
        pos_ids = self.transport.recv((B, S), torch.int64)

        # 接收 temperature + finished_mask + pad_token_id（直接 P2P 从 Head）
        temp_tensor = torch.zeros(1, dtype=torch.float32, device="cuda")
        head_rank = self.head_rank
        dist.recv(temp_tensor, src=head_rank)
        temperature = temp_tensor.item()

        finished_int = torch.zeros(B, dtype=torch.int64, device="cuda")
        dist.recv(finished_int, src=head_rank)
        finished = finished_int.to(torch.bool)

        pad_id_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.recv(pad_id_tensor, src=head_rank)
        pad_token_id = int(pad_id_tensor.item())

        pos_emb = None
        if self.stage.rotary_emb is not None:
            try:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), pos_ids.to("cuda"))
            except TypeError:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), seq_len=S)

        with torch.no_grad():
            logits = self.stage.forward_output(h_middle, position_ids=pos_ids, position_embeddings=pos_emb)
            last_logits = logits[:, -1, :]  # [B, vocab]

            if temperature > 0:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

            # 对 finished 序列，强制设为 pad_token_id
            next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, pad_token_id), next_token)

            # 计算 log_prob（对 next_token）
            # 手动实现，避免依赖 flash-attn 在 Tail 端可能的问题
            log_prob = torch.log_softmax(last_logits, dim=-1)
            log_prob = torch.gather(log_prob, dim=-1, index=next_token).float()  # 转为 float32 匹配 Head 的 recv

        dist.send(next_token, dst=head_rank)
        dist.send(log_prob, dst=head_rank)
        return {"next_token": next_token, "log_prob": log_prob}

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
        if self.is_head:
            torch.save(self.stage.get_trainable_state_dict(), target / "head_adapter_state.pt")
            torch.save(self.stage.optimizer.state_dict(), target / "head_optimizer.pt")
        elif self.is_tail:
            torch.save(self.stage.get_trainable_state_dict(), target / "tail_adapter_state.pt")
            torch.save(self.stage.optimizer.state_dict(), target / "tail_optimizer.pt")
            meta = {"model_path": self.model_path, "front_end": self.front_end,
                    "middle_end": self.middle_end, "global_step": global_step}
            (target / "adapter_meta.json").write_text(json.dumps(meta, indent=2))

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **kwargs):
        target = Path(local_path)
        if self.is_head:
            state = torch.load(target / "head_adapter_state.pt", map_location="cuda")
            self.stage.load_trainable_state_dict(state)
            if (target / "head_optimizer.pt").exists():
                self.stage.optimizer.load_state_dict(torch.load(target / "head_optimizer.pt", map_location="cuda"))
        elif self.is_tail:
            state = torch.load(target / "tail_adapter_state.pt", map_location="cuda")
            self.stage.load_trainable_state_dict(state)
            if (target / "tail_optimizer.pt").exists():
                self.stage.optimizer.load_state_dict(torch.load(target / "tail_optimizer.pt", map_location="cuda"))


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
