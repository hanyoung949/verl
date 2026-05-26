"""SplitPipelineEngine — 3-stage split pipeline 引擎。"""

from __future__ import annotations

import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import nn
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_verl_engine_base", "/root/workspace/verl/verl/workers/engine/base.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BaseEngine = _mod.BaseEngine
EngineRegistry = _mod.EngineRegistry

from .stage import HeadStage, TailStage, Qwen35MTPDraftHead
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
        self.trust_remote_code = False
        self.mtp_enable = False
        self.mtp_enable_train = False
        self.mtp_enable_rollout = False
        self.mtp_num_speculative_tokens = 1
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
            self.model_path = getattr(self.model_config, "local_path", "Qwen/Qwen3.5-2B")
        else:
            self.model_path = "Qwen/Qwen3.5-2B"
        self.front_end = 4
        self.middle_end = 20
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
        base_model = _load_auto_model(model_path, trust_remote_code=self.trust_remote_code)
        base_causal_lm = _unwrap_causal_lm(base_model)
        base_decoder = _find_decoder_stack(base_model, base_causal_lm)
        mtp_head = None
        if self.mtp_enable:
            mtp_head = _build_qwen35_mtp_head(model_path, base_decoder, base_decoder.embed_tokens,
                                             base_causal_lm.lm_head, torch.bfloat16)
            if mtp_head is None and self.mtp_enable_rollout:
                raise ValueError(
                    "MTP rollout is enabled, but no `mtp.*` weights were found. "
                    "Use a Qwen3.5 checkpoint with MTP weights, for example Qwen/Qwen3.5-2B."
                )
        peft_model = get_peft_model(base_model, lora_config)
        causal_lm = _unwrap_causal_lm(peft_model)
        decoder = _find_decoder_stack(peft_model, causal_lm)
        layers = list(decoder.layers)
        embed_tokens = decoder.embed_tokens
        rotary_emb = getattr(decoder, "rotary_emb", None)
        norm = decoder.norm
        lm_head = causal_lm.lm_head
        return embed_tokens, layers, rotary_emb, norm, lm_head, mtp_head

    def initialize(self):
        embed_tokens, layers, rotary_emb, norm, lm_head, mtp_head = self._load_model_parts()

        if self.is_head:
            head_layers = [l.to("cuda") for l in layers[:self.front_end]]
            self.stage = HeadStage(embed_tokens.to("cuda"), head_layers,
                                   rotary_emb.to("cuda") if rotary_emb is not None else None, "cuda",
                                   lr=self.lr, clip_grad=self.clip_grad)
            self.transport = StageTransport(self.rank, self.middle_first_rank, "cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(self._resolve_model_path(self.model_path),
                                                           trust_remote_code=self.trust_remote_code)
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
                                  rotary_emb=rotary_emb.to("cuda") if rotary_emb is not None else None,
                                  mtp_head=mtp_head.to("cuda") if mtp_head is not None else None,
                                  mtp_trainable=self.mtp_enable_train)
            self.transport = StageTransport(self.rank, self.middle_last_rank, "cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(self._resolve_model_path(self.model_path),
                                                           trust_remote_code=self.trust_remote_code)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            mtp_msg = " + MTP rollout" if self.mtp_enable_rollout else ""
            print(f"[rank{self.rank}] TailStage: {len(self.stage.layers)} layers{mtp_msg}", flush=True)

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
        dist.recv(header, src=self.middle_last_rank)
        flag = int(header[0].item())
        if flag == PIPELINE_DONE:
            # Head 通知 pipeline 结束，直接返回空结果让调用方 break
            return {}
        _flag, B, S, H, has_mask, dtype_code = [int(x) for x in header.tolist()]
        tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
        h_middle = self.transport.recv((B, S, H), tensor_dtype)
        pos_ids = self.transport.recv((B, S), torch.int64)
        mask = self.transport.recv((B, 1, S, S), tensor_dtype) if has_mask else None

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
            logits = self.stage.forward_output(h_in, position_ids=pos_ids,
                                               position_embeddings=pos_emb,
                                               attention_mask=mask)

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
        """Tail 本地采样；MTP rollout 可一次回传多个 accepted tokens。"""
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
            seq_positions = torch.arange(S, dtype=torch.int64, device="cuda").unsqueeze(0).expand(B, S)
            last_valid_idx = torch.where(attention_mask.bool(), seq_positions, torch.full_like(seq_positions, -1))
            last_valid_idx = last_valid_idx.max(dim=-1).values.clamp_min(0)
            dist.send(last_valid_idx, dst=tail_rank)
            num_spec = self.mtp_num_speculative_tokens if self.mtp_enable_rollout else 0
            dist.send(torch.tensor([num_spec], dtype=torch.int64, device="cuda"), dst=tail_rank)
            if num_spec > 0:
                dist.send(input_ids.to(torch.int64).contiguous(), dst=tail_rank)
                draft_tokens = torch.empty(B, num_spec, dtype=torch.int64, device="cuda")
                dist.recv(draft_tokens, src=tail_rank)
                verify_ids = torch.cat([input_ids, draft_tokens], dim=-1)
                draft_mask = (~finished).long().unsqueeze(-1).expand(B, num_spec)
                verify_mask = torch.cat([attention_mask, draft_mask], dim=-1)
                vh, vpos_ids, _vpos_emb, vmask = self.stage.forward_input(verify_ids, verify_mask)
                vB, vS, vH = vh.shape
                v_has_mask = 0 if vmask is None else 1
                v_header = torch.tensor([FWD_ONLY, vB, vS, vH, v_has_mask, 0],
                                        dtype=torch.int64, device="cuda")
                self.transport.send(v_header)
                self.transport.send(vh)
                self.transport.send(vpos_ids)
                if vmask is not None:
                    self.transport.send(vmask)

            # MTP rollout may return multiple accepted tokens in one request.
            result_cols = torch.zeros(1, dtype=torch.int64, device="cuda")
            dist.recv(result_cols, src=tail_rank)
            L = int(result_cols.item())
            next_token = torch.empty(B, L, dtype=torch.int64, device="cuda")
            dist.recv(next_token, src=tail_rank)
            log_prob = torch.empty(B, L, dtype=torch.float32, device="cuda")
            dist.recv(log_prob, src=tail_rank)
            token_mask = torch.empty(B, L, dtype=torch.int64, device="cuda")
            dist.recv(token_mask, src=tail_rank)

            return {"next_token": next_token, "log_prob": log_prob, "token_mask": token_mask}

    def _tail_sample(self, data, temperature, pad_token_id):
        header = torch.zeros(6, dtype=torch.int64, device="cuda")
        dist.recv(header, src=self.middle_last_rank)
        flag = int(header[0].item())
        if flag == PIPELINE_DONE:
            return {}
        _flag, B, S, H, has_mask, dtype_code = [int(x) for x in header.tolist()]
        tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[dtype_code]
        h_middle = self.transport.recv((B, S, H), tensor_dtype)
        pos_ids = self.transport.recv((B, S), torch.int64)
        mask = self.transport.recv((B, 1, S, S), tensor_dtype) if has_mask else None

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

        last_valid_idx = torch.zeros(B, dtype=torch.int64, device="cuda")
        dist.recv(last_valid_idx, src=head_rank)

        num_spec_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.recv(num_spec_tensor, src=head_rank)
        num_spec = int(num_spec_tensor.item())
        mtp_input_ids = None
        if num_spec > 0:
            mtp_input_ids = torch.empty(B, S, dtype=torch.int64, device="cuda")
            dist.recv(mtp_input_ids, src=head_rank)

        pos_emb = None
        if self.stage.rotary_emb is not None:
            try:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), pos_ids.to("cuda"))
            except TypeError:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), seq_len=S)

        with torch.no_grad():
            if num_spec > 0:
                logits, raw_hidden = self.stage.forward_output(h_middle, position_ids=pos_ids,
                                                               position_embeddings=pos_emb,
                                                               attention_mask=mask,
                                                               return_hidden=True)
            else:
                logits = self.stage.forward_output(h_middle, position_ids=pos_ids,
                                                   position_embeddings=pos_emb,
                                                   attention_mask=mask)
                raw_hidden = None
            batch_idx = torch.arange(B, device="cuda")
            last_logits = logits[batch_idx, last_valid_idx, :]  # [B, vocab]

            if num_spec > 0:
                draft_tokens, draft_logits = self._draft_mtp_tokens(
                    mtp_input_ids, raw_hidden, pos_ids, pos_emb, mask,
                    last_valid_idx, num_spec, temperature, finished, pad_token_id)
                dist.send(draft_tokens, dst=head_rank)

                verify = self._recv_verify_activation(tensor_dtype)
                next_token, log_prob, token_mask = self._verify_mtp_draft(
                    verify, draft_tokens, draft_logits, last_valid_idx, S,
                    temperature, finished, pad_token_id)
            else:
                next_token = self._sample_from_logits(last_logits, temperature)
                next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, pad_token_id), next_token)
                log_prob = torch.log_softmax(last_logits, dim=-1)
                log_prob = torch.gather(log_prob, dim=-1, index=next_token).float()
                token_mask = (~finished).long().unsqueeze(-1)

        dist.send(torch.tensor([next_token.shape[1]], dtype=torch.int64, device="cuda"), dst=head_rank)
        dist.send(next_token, dst=head_rank)
        dist.send(log_prob, dst=head_rank)
        dist.send(token_mask, dst=head_rank)
        return {"next_token": next_token, "log_prob": log_prob}

    def _sample_from_logits(self, logits, temperature):
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _draft_mtp_tokens(self, input_ids, raw_hidden, pos_ids, pos_emb, mask,
                          last_valid_idx, num_spec, temperature, finished, pad_token_id):
        draft_tokens = []
        draft_logits = []
        logits, mtp_hidden = self.stage.draft_mtp_logits(
            input_ids, raw_hidden, position_ids=pos_ids, position_embeddings=pos_emb,
            attention_mask=mask, return_hidden=True)
        batch_idx = torch.arange(input_ids.size(0), device="cuda")
        step_logits = logits[batch_idx, last_valid_idx, :]
        hidden_cursor = mtp_hidden[batch_idx, last_valid_idx, :].unsqueeze(1)

        for step in range(num_spec):
            token = self._sample_from_logits(step_logits, temperature)
            token = torch.where(finished.unsqueeze(-1), torch.full_like(token, pad_token_id), token)
            draft_tokens.append(token)
            draft_logits.append(step_logits)

            if step == num_spec - 1:
                break

            next_pos = pos_ids[batch_idx, last_valid_idx].unsqueeze(1) + step + 1
            next_pos_emb = None
            if self.stage.rotary_emb is not None:
                try:
                    next_pos_emb = self.stage.rotary_emb(hidden_cursor, next_pos)
                except TypeError:
                    next_pos_emb = self.stage.rotary_emb(hidden_cursor, seq_len=hidden_cursor.size(1))
            logits, mtp_hidden = self.stage.draft_mtp_logits(
                token, hidden_cursor, position_ids=next_pos,
                position_embeddings=next_pos_emb, attention_mask=None,
                return_hidden=True)
            step_logits = logits[:, -1, :]
            hidden_cursor = mtp_hidden[:, -1:, :]

        return torch.cat(draft_tokens, dim=-1), torch.stack(draft_logits, dim=1)

    def _recv_verify_activation(self, tensor_dtype):
        header = torch.zeros(6, dtype=torch.int64, device="cuda")
        dist.recv(header, src=self.middle_last_rank)
        flag = int(header[0].item())
        if flag != FWD_ONLY:
            raise RuntimeError(f"[Tail] expected verify FWD_ONLY, got flag={flag}")
        _flag, B, S, H, has_mask, dtype_code = [int(x) for x in header.tolist()]
        tensor_dtype = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}.get(dtype_code, tensor_dtype)
        h_middle = self.transport.recv((B, S, H), tensor_dtype)
        pos_ids = self.transport.recv((B, S), torch.int64)
        mask = self.transport.recv((B, 1, S, S), tensor_dtype) if has_mask else None

        pos_emb = None
        if self.stage.rotary_emb is not None:
            try:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), pos_ids.to("cuda"))
            except TypeError:
                pos_emb = self.stage.rotary_emb(h_middle.to("cuda"), seq_len=S)
        return {"h": h_middle, "pos_ids": pos_ids, "pos_emb": pos_emb, "mask": mask}

    def _verify_mtp_draft(self, verify, draft_tokens, draft_logits, prefix_last_idx, prefix_physical_len,
                          temperature, finished, pad_token_id):
        verify_logits = self.stage.forward_output(
            verify["h"], position_ids=verify["pos_ids"],
            position_embeddings=verify["pos_emb"], attention_mask=verify["mask"])
        B, K = draft_tokens.shape
        verify_positions = torch.cat([
            prefix_last_idx.unsqueeze(1),
            torch.arange(prefix_physical_len, prefix_physical_len + K, dtype=torch.int64, device="cuda")
            .unsqueeze(0).expand(B, K),
        ], dim=1)
        batch_idx = torch.arange(B, device="cuda").unsqueeze(1)
        target_logits = verify_logits[batch_idx, verify_positions, :]
        max_out = K + 1
        out_tokens = torch.full((B, max_out), pad_token_id, dtype=torch.int64, device="cuda")
        out_log_probs = torch.zeros((B, max_out), dtype=torch.float32, device="cuda")
        out_mask = torch.zeros((B, max_out), dtype=torch.int64, device="cuda")
        eos_token_id = getattr(getattr(self, "tokenizer", None), "eos_token_id", None)

        for b in range(B):
            if bool(finished[b].item()):
                continue

            out_col = 0
            all_accepted = True
            for step in range(K):
                token = draft_tokens[b, step]
                chosen, accepted = self._verify_one_token(
                    target_logits[b, step], draft_logits[b, step], token, temperature)
                out_tokens[b, out_col] = chosen
                out_log_probs[b, out_col] = torch.log_softmax(target_logits[b, step], dim=-1)[chosen].float()
                out_mask[b, out_col] = 1
                out_col += 1
                if eos_token_id is not None and int(chosen.item()) == int(eos_token_id):
                    all_accepted = False
                    break
                if not accepted:
                    all_accepted = False
                    break

            if all_accepted and out_col < max_out:
                bonus = self._sample_from_logits(target_logits[b, K].unsqueeze(0), temperature).squeeze()
                out_tokens[b, out_col] = bonus
                out_log_probs[b, out_col] = torch.log_softmax(target_logits[b, K], dim=-1)[bonus].float()
                out_mask[b, out_col] = 1

        return out_tokens, out_log_probs, out_mask

    def _verify_one_token(self, target_logits, draft_logits, draft_token, temperature):
        if temperature <= 0:
            target_token = torch.argmax(target_logits)
            return target_token, bool(target_token.item() == draft_token.item())

        target_probs = torch.softmax(target_logits / temperature, dim=-1)
        draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
        p = target_probs[draft_token]
        q = draft_probs[draft_token].clamp_min(1e-12)
        accept_prob = torch.minimum(torch.ones_like(p), p / q)
        if bool((torch.rand((), device=target_logits.device) <= accept_prob).item()):
            return draft_token, True

        residual = (target_probs - draft_probs).clamp_min(0.0)
        residual_sum = residual.sum()
        residual = residual / residual_sum.clamp_min(1e-12) if residual_sum > 0 else target_probs
        fallback = torch.multinomial(residual, num_samples=1).squeeze(0)
        return fallback, False

    def optimizer_zero_grad(self):
        if (self.is_head or self.is_tail) and self.stage is not None:
            self.stage.zero_grad()

    def optimizer_step(self):
        if (self.is_head or self.is_tail) and self.stage is not None:
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
    queue = [root_model, causal_lm, getattr(causal_lm, "model", None), getattr(causal_lm, "language_model", None)]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if all(hasattr(current, n) for n in ("layers", "embed_tokens", "norm")):
            return current
        for attr in ("base_model", "model", "language_model", "module"):
            if hasattr(current, attr):
                queue.append(getattr(current, attr))
    raise ValueError("Cannot locate decoder stack.")


def _load_auto_model(model_path, trust_remote_code=False):
    kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": trust_remote_code}
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except (ValueError, OSError):
        pass

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise ValueError(
            "Qwen3.5 uses AutoModelForImageTextToText. Please install a transformers version "
            "that provides this auto class."
        ) from exc
    return AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)


def _build_qwen35_mtp_head(model_path, decoder, embed_tokens, lm_head, dtype):
    mtp_state = _load_mtp_state_dict(model_path)
    if not mtp_state:
        return None

    required = [
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ]
    missing = [name for name in required if name not in mtp_state]
    if missing:
        raise ValueError(f"MTP checkpoint is missing required tensors: {missing}")

    layer = copy.deepcopy(_select_mtp_decoder_layer(decoder))
    layer_state = {
        name.removeprefix("mtp.layers.0."): tensor.to(dtype=dtype)
        for name, tensor in mtp_state.items()
        if name.startswith("mtp.layers.0.")
    }
    missing_keys, unexpected_keys = layer.load_state_dict(layer_state, strict=False)
    if unexpected_keys:
        raise ValueError(f"Unexpected Qwen3.5 MTP layer keys: {unexpected_keys}")

    fc_weight = mtp_state["mtp.fc.weight"]
    fc = nn.Linear(fc_weight.shape[1], fc_weight.shape[0], bias=False, dtype=dtype)
    fc.weight.data.copy_(fc_weight.to(dtype=dtype))

    pre_fc_norm_embedding = copy.deepcopy(decoder.norm)
    pre_fc_norm_hidden = copy.deepcopy(decoder.norm)
    norm = copy.deepcopy(decoder.norm)
    pre_fc_norm_embedding.load_state_dict({"weight": mtp_state["mtp.pre_fc_norm_embedding.weight"].to(dtype=dtype)})
    pre_fc_norm_hidden.load_state_dict({"weight": mtp_state["mtp.pre_fc_norm_hidden.weight"].to(dtype=dtype)})
    norm.load_state_dict({"weight": mtp_state["mtp.norm.weight"].to(dtype=dtype)})

    mtp_head = Qwen35MTPDraftHead(embed_tokens=embed_tokens,
                                  pre_fc_norm_embedding=pre_fc_norm_embedding,
                                  pre_fc_norm_hidden=pre_fc_norm_hidden,
                                  fc=fc,
                                  layer=layer,
                                  norm=norm,
                                  lm_head=lm_head)
    if missing_keys:
        print(f"[split_demo_v5] MTP layer loaded with missing optional keys: {missing_keys}", flush=True)
    return mtp_head


def _select_mtp_decoder_layer(decoder):
    layer_types = getattr(getattr(decoder, "config", None), "layer_types", None)
    layers = list(decoder.layers)
    if layer_types:
        for layer, layer_type in zip(layers, layer_types, strict=False):
            if layer_type == "full_attention":
                return layer
    for layer in layers:
        if hasattr(layer, "self_attn"):
            return layer
    raise ValueError("Cannot find a decoder layer shape compatible with Qwen3.5 MTP.")


def _load_mtp_state_dict(model_path):
    local_dir = _resolve_local_checkpoint_dir(model_path)
    if local_dir is None:
        return {}

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ValueError("MTP weights are stored in safetensors; please install safetensors.") from exc

    index_path = local_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        files = sorted({filename for name, filename in weight_map.items() if name.startswith("mtp.")})
    else:
        files = sorted(path.name for path in local_dir.glob("*.safetensors"))

    state = {}
    for filename in files:
        path = local_dir / filename
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    state[key] = f.get_tensor(key)
    return state


def _resolve_local_checkpoint_dir(model_path):
    path = Path(os.path.expanduser(model_path))
    if path.exists():
        return path

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    local_path = snapshot_download(
        repo_id=model_path,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )
    return Path(local_path)
