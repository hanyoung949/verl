"""Tensor-Parallel Stage1 for split training.

Implements true TP (not pipeline-split) for stage_1 layers, matching vLLM's
inference-side TP sharding so that WeightSyncManager can directly sync LoRA
weights without repartitioning.

Current limitation: only supports tp_size=2 via manual send/recv all-reduce.
This avoids ``dist.new_group`` NCCL issues with mixed PP+TP topologies.

TP sharding convention (same as vLLM/Megatron):
  - Column-parallel (q_proj, k_proj, v_proj, gate_proj, up_proj):
    Split along output dimension.  Each rank holds (out_dim/tp, in_dim).
    Forward: local matmul, no communication.
  - Row-parallel (o_proj, down_proj):
    Split along input dimension.  Each rank holds (out_dim, in_dim/tp).
    Forward: local matmul + all-reduce.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn
import torch.nn.functional as F

from .transport import FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE, _DTYPE_MAP


# ---------------------------------------------------------------------------
# Manual 2-rank all-reduce (avoids dist.new_group NCCL issues)
# ---------------------------------------------------------------------------

def _tp_all_reduce(x: Tensor, tp_rank: int, tp_size: int) -> Tensor:
    """Manual 2-rank all-reduce using send/recv.

    Maps TP rank to global rank: TP ranks 0,1 → global ranks 1,2.
    Rank 0 sends partial sum to rank 1; rank 1 sums and sends back.
    """
    assert tp_size == 2, "Only tp_size=2 is supported for manual all-reduce"
    global_ranks = list(range(1, 1 + tp_size))
    peer = global_ranks[1 - tp_rank]

    if tp_rank == 0:
        dist.send(x.contiguous(), dst=peer)
        dist.recv(x, src=peer)
    else:
        buf = torch.empty_like(x)
        dist.recv(buf, src=peer)
        x = x + buf
        dist.send(x.contiguous(), dst=peer)
    return x


class _TPAllReduce(torch.autograd.Function):
    """All-reduce in forward, identity in backward."""

    @staticmethod
    def forward(ctx, x, tp_rank, tp_size):
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        return _tp_all_reduce(x, tp_rank, tp_size)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class _TPAllReduceGrad(torch.autograd.Function):
    """Identity in forward, all-reduce in backward."""

    @staticmethod
    def forward(ctx, x, tp_rank, tp_size):
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        return x

    @staticmethod
    def backward(ctx, grad):
        return _tp_all_reduce(grad, ctx.tp_rank, ctx.tp_size), None, None


# ---------------------------------------------------------------------------
# TP linear / LoRA primitives
# ---------------------------------------------------------------------------

class TPLinear(nn.Module):
    """Linear layer sharded across TP ranks."""

    def __init__(self, weight: Tensor, bias: Tensor | None, mode: str, tp_rank: int, tp_size: int):
        super().__init__()
        self.mode = mode
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        if mode == "column":
            chunks = weight.chunk(tp_size, dim=0)
            self.weight = nn.Parameter(chunks[tp_rank].contiguous())
            if bias is not None:
                self.bias = nn.Parameter(bias.chunk(tp_size, dim=0)[tp_rank].contiguous())
            else:
                self.bias = None
        elif mode == "row":
            chunks = weight.chunk(tp_size, dim=1)
            self.weight = nn.Parameter(chunks[tp_rank].contiguous())
            self.bias = nn.Parameter(bias.contiguous()) if bias is not None else None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.weight.dtype)
        out = F.linear(x, self.weight, self.bias if self.mode == "column" else None)
        if self.mode == "row":
            out = _TPAllReduce.apply(out, self.tp_rank, self.tp_size)
            if self.bias is not None:
                out = out + self.bias
        return out


class TPLora(nn.Module):
    """LoRA adapter sharded for TP.

    Column-parallel (q_proj, v_proj): lora_A full, lora_B sharded on output dim.
    Row-parallel (o_proj): lora_A sharded on input dim, lora_B full.
    """

    def __init__(self, lora_a: Tensor, lora_b: Tensor, mode: str, tp_rank: int, tp_size: int, scaling: float):
        super().__init__()
        self.scaling = scaling
        self.mode = mode
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        if mode == "column":
            self.lora_a = nn.Parameter(lora_a.contiguous())
            self.lora_b = nn.Parameter(lora_b.chunk(tp_size, dim=0)[tp_rank].contiguous())
        elif mode == "row":
            self.lora_a = nn.Parameter(lora_a.chunk(tp_size, dim=0)[tp_rank].contiguous())
            self.lora_b = nn.Parameter(lora_b.contiguous())

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.lora_a.dtype)
        out = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        if self.mode == "row":
            out = _TPAllReduce.apply(out, self.tp_rank, self.tp_size)
        return out


# ---------------------------------------------------------------------------
# Qwen2 TP layers
# ---------------------------------------------------------------------------

class TPQwen2Attention(nn.Module):
    """TP-aware Qwen2 self-attention with optional LoRA."""

    def __init__(self, orig_attn, tp_rank: int, tp_size: int):
        super().__init__()
        config = orig_attn.config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = orig_attn.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = TPLinear(orig_attn.q_proj.weight.data, getattr(orig_attn.q_proj, 'bias', None), "column", tp_rank, tp_size)
        self.k_proj = TPLinear(orig_attn.k_proj.weight.data, getattr(orig_attn.k_proj, 'bias', None), "column", tp_rank, tp_size)
        self.v_proj = TPLinear(orig_attn.v_proj.weight.data, getattr(orig_attn.v_proj, 'bias', None), "column", tp_rank, tp_size)
        self.o_proj = TPLinear(orig_attn.o_proj.weight.data, getattr(orig_attn.o_proj, 'bias', None), "row", tp_rank, tp_size)

        self.q_lora = self._make_lora(orig_attn.q_proj, "column", tp_rank, tp_size)
        self.v_lora = self._make_lora(orig_attn.v_proj, "column", tp_rank, tp_size)

        self.num_heads_local = self.num_heads // tp_size
        self.num_kv_heads_local = self.num_key_value_heads // tp_size

    @staticmethod
    def _make_lora(orig_proj, mode, tp_rank, tp_size):
        lora_a = getattr(orig_proj, 'lora_A', None)
        lora_b = getattr(orig_proj, 'lora_B', None)
        if lora_a is None or lora_b is None:
            return None
        a_weight = lora_a.default.weight.data if hasattr(lora_a, 'default') else lora_a.weight.data
        b_weight = lora_b.default.weight.data if hasattr(lora_b, 'default') else lora_b.weight.data
        scaling = getattr(orig_proj, 'scaling', {}).get('default', 1.0)
        if hasattr(orig_proj, 'scaling') and isinstance(orig_proj.scaling, (int, float)):
            scaling = orig_proj.scaling
        return TPLora(a_weight, b_weight, mode, tp_rank, tp_size, scaling)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.q_lora is not None:
            q = q + self.q_lora(hidden_states).to(q.dtype)
        if self.v_lora is not None:
            v = v + self.v_lora(hidden_states).to(v.dtype)

        q = q.view(bsz, seq_len, self.num_heads_local, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads_local, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads_local, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = self._apply_rotary(q, cos, sin)
            k = self._apply_rotary(k, cos, sin)

        if self.num_kv_heads_local < self.num_heads_local:
            repeat_factor = self.num_heads_local // self.num_kv_heads_local
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_out)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        d = x.shape[-1]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class TPQwen2MLP(nn.Module):
    """TP-aware Qwen2 MLP (SwiGLU)."""

    def __init__(self, orig_mlp, tp_rank: int, tp_size: int):
        super().__init__()
        self.gate_proj = TPLinear(orig_mlp.gate_proj.weight.data, None, "column", tp_rank, tp_size)
        self.up_proj = TPLinear(orig_mlp.up_proj.weight.data, None, "column", tp_rank, tp_size)
        self.down_proj = TPLinear(orig_mlp.down_proj.weight.data, None, "row", tp_rank, tp_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TPQwen2DecoderLayer(nn.Module):
    """A single Qwen2 decoder layer with TP."""

    def __init__(self, orig_layer, tp_rank: int, tp_size: int):
        super().__init__()
        self.self_attn = TPQwen2Attention(orig_layer.self_attn, tp_rank, tp_size)
        self.mlp = TPQwen2MLP(orig_layer.mlp, tp_rank, tp_size)
        self.input_layernorm = orig_layer.input_layernorm
        self.post_attention_layernorm = orig_layer.post_attention_layernorm

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# TPMiddleStage — drop-in replacement for Stage1
# ---------------------------------------------------------------------------

class TPMiddleStage:
    """Stage_1 with true Tensor Parallelism (tp_size=2 only).

    Each TP rank holds a shard of every layer's weights.  Forward/backward
    uses manual send/recv all-reduce for row-parallel layers.  Matches vLLM's
    inference TP so LoRA weights sync directly.
    """

    def __init__(self, layers, rotary_emb, device, tp_rank: int, tp_size: int):
        self.device = torch.device(device)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.rotary_emb = rotary_emb

        self.layers = nn.ModuleList([
            TPQwen2DecoderLayer(layer, tp_rank, tp_size) for layer in layers
        ])
        self.layers.to(self.device)
        if self.rotary_emb is not None:
            self.rotary_emb = self.rotary_emb.to(self.device)

        for p in self.layers.parameters():
            p.requires_grad = False

    def forward(self, h: Tensor, **kwargs):
        for layer in self.layers:
            h = layer(h, **kwargs)
        return h

    def _recv_tensor(self, shape, dtype, src):
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=src)
        return buf

    def _send_tensor(self, tensor: Tensor, dst):
        dist.send(tensor.contiguous(), dst=dst)

    def _get_pos_emb(self, h, pos_ids):
        if self.rotary_emb is not None:
            try:
                return self.rotary_emb(h, pos_ids)
            except TypeError:
                return self.rotary_emb(h, seq_len=h.shape[1])
        return None

    def _run_once_impl(self, topology, blocking_wait_for_header: bool):
        """Run one forward/backward iteration.

        tp_rank=0 is the "master": does inter-stage recv/send with stage_0/stage_2,
        and broadcasts input to tp_rank=1 before forward.
        tp_rank=1 is the "slave": receives input from tp_rank=0, participates in
        TP all-reduce during forward/backward.
        """
        t = topology or {"stage_0": [0], "stage_1": [1], "stage_2": [2]}
        stage_0_rank = t["stage_0"][0]
        stage_2_rank = t["stage_2"][0]
        peer = self._peer_global_rank()
        _VALID_FLAGS = {FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE}

        is_master = (self.tp_rank == 0)

        # --- Header exchange ---
        if is_master:
            header = torch.zeros(6, dtype=torch.int64, device=self.device)
            dist.recv(header, src=stage_0_rank)
            flag = int(header[0].item())
            if flag not in _VALID_FLAGS:
                raise RuntimeError(f"[TPStage1] illegal flag {flag}")
            # Forward flag to slave and stage_2
            dist.send(header.clone(), dst=peer)
            if flag != SHUTDOWN and flag != PIPELINE_DONE:
                dist.send(header.clone(), dst=stage_2_rank)
            elif flag == SHUTDOWN:
                return False
            elif flag == PIPELINE_DONE:
                dist.send(header.clone(), dst=stage_2_rank)
                return True
        else:
            header = torch.zeros(6, dtype=torch.int64, device=self.device)
            dist.recv(header, src=peer)
            flag = int(header[0].item())
            if flag == SHUTDOWN:
                return False
            if flag == PIPELINE_DONE:
                return True

        B, S, H, has_mask, dtype_code = [int(x) for x in header[1:].tolist()]
        tensor_dtype = _DTYPE_MAP[dtype_code]

        # --- Receive/broadcast input ---
        if is_master:
            h = self._recv_tensor((B, S, H), tensor_dtype, stage_0_rank)
            pos_ids = self._recv_tensor((B, S), torch.int64, stage_0_rank)
            mask = self._recv_tensor((B, 1, S, S), tensor_dtype, stage_0_rank) if has_mask else None
            # Broadcast to slave
            dist.send(h.contiguous(), dst=peer)
            dist.send(pos_ids.contiguous(), dst=peer)
            if mask is not None:
                dist.send(mask.contiguous(), dst=peer)
        else:
            h = self._recv_tensor((B, S, H), tensor_dtype, peer)
            pos_ids = self._recv_tensor((B, S), torch.int64, peer)
            mask = self._recv_tensor((B, 1, S, S), tensor_dtype, peer) if has_mask else None

        # --- Forward / backward ---
        if flag == FWD_ONLY:
            with torch.no_grad():
                pos_emb = self._get_pos_emb(h, pos_ids)
                h_out = self.forward(h, position_embeddings=pos_emb, attention_mask=mask)
            if is_master:
                self._send_tensor(h_out, stage_2_rank)
                self._send_tensor(pos_ids, stage_2_rank)

        elif flag == FWD_WITH_BWD:
            h_in = h.detach().requires_grad_(True)
            pos_emb = self._get_pos_emb(h_in, pos_ids)
            h_out = self.forward(h_in, position_embeddings=pos_emb, attention_mask=mask)

            if is_master:
                self._send_tensor(h_out.detach(), stage_2_rank)
                self._send_tensor(pos_ids, stage_2_rank)

                grad_out = self._recv_tensor(h_out.shape, h_out.dtype, stage_2_rank)
                # Forward grad to slave so both can backward
                dist.send(grad_out.contiguous(), dst=peer)
                torch.autograd.backward(h_out, grad_out)
                self._send_tensor(h_in.grad, stage_0_rank)
            else:
                grad_out = self._recv_tensor(h_out.shape, h_out.dtype, peer)
                torch.autograd.backward(h_out, grad_out)

        return True

    def _peer_global_rank(self):
        global_ranks = list(range(1, 1 + self.tp_size))
        return global_ranks[1 - self.tp_rank]

    def run_once(self, topology=None):
        return self._run_once_impl(topology, blocking_wait_for_header=True)

    def has_grad(self):
        for p in self.layers.parameters():
            if p.grad is not None:
                return True
        return False

    def run(self, topology=None):
        while True:
            if not self._run_once_impl(topology, blocking_wait_for_header=True):
                break
