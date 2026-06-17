"""Tensor-Parallel Stage1 for split training.

Implements true TP (not pipeline-split) for stage_1 layers, matching vLLM's
inference-side TP sharding so that WeightSyncManager can directly sync LoRA
weights without repartitioning.

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


class _AllReduce(torch.autograd.Function):
    """All-reduce in forward, identity in backward (gradient is already replicated)."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _AllReduceGrad(torch.autograd.Function):
    """Identity in forward, all-reduce in backward (for column-parallel gradient sync)."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


class TPLinear(nn.Module):
    """A linear layer sharded across TP ranks."""

    def __init__(self, weight: Tensor, bias: Tensor | None, mode: str, tp_rank: int, tp_size: int, tp_group=None):
        super().__init__()
        self.mode = mode
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group

        if mode == "column":
            # Split along output dimension (dim 0 of weight matrix)
            chunks = weight.chunk(tp_size, dim=0)
            self.weight = nn.Parameter(chunks[tp_rank].contiguous())
            if bias is not None:
                bias_chunks = bias.chunk(tp_size, dim=0)
                self.bias = nn.Parameter(bias_chunks[tp_rank].contiguous())
            else:
                self.bias = None
        elif mode == "row":
            # Split along input dimension (dim 1 of weight matrix)
            chunks = weight.chunk(tp_size, dim=1)
            self.weight = nn.Parameter(chunks[tp_rank].contiguous())
            self.bias = nn.Parameter(bias.contiguous()) if bias is not None else None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, self.bias if self.mode == "column" else None)
        if self.mode == "row":
            out = _AllReduce.apply(out, self.tp_group)
            if self.bias is not None:
                out = out + self.bias
        return out


class TPLora(nn.Module):
    """LoRA adapter sharded for TP."""

    def __init__(self, lora_a: Tensor, lora_b: Tensor, mode: str, tp_rank: int, tp_size: int, scaling: float, tp_group=None):
        super().__init__()
        self.scaling = scaling
        self.mode = mode
        self.tp_group = tp_group

        if mode == "column":
            # lora_A full, lora_B sharded along output dim
            self.lora_a = nn.Parameter(lora_a.contiguous())
            chunks_b = lora_b.chunk(tp_size, dim=0)
            self.lora_b = nn.Parameter(chunks_b[tp_rank].contiguous())
        elif mode == "row":
            # lora_A sharded along input dim, lora_B full
            chunks_a = lora_a.chunk(tp_size, dim=0)
            self.lora_a = nn.Parameter(chunks_a[tp_rank].contiguous())
            self.lora_b = nn.Parameter(lora_b.contiguous())

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        if self.mode == "row":
            out = _AllReduce.apply(out, self.tp_group)
        return out


class TPQwen2Attention(nn.Module):
    """TP-aware Qwen2 self-attention with optional LoRA."""

    def __init__(self, orig_attn, tp_rank: int, tp_size: int, tp_group=None):
        super().__init__()
        config = orig_attn.config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = orig_attn.head_dim
        self.hidden_size = config.hidden_size

        # TP shard q/k/v/o projections
        self.q_proj = TPLinear(orig_attn.q_proj.weight.data, getattr(orig_attn.q_proj, 'bias', None), "column", tp_rank, tp_size, tp_group)
        self.k_proj = TPLinear(orig_attn.k_proj.weight.data, getattr(orig_attn.k_proj, 'bias', None), "column", tp_rank, tp_size, tp_group)
        self.v_proj = TPLinear(orig_attn.v_proj.weight.data, getattr(orig_attn.v_proj, 'bias', None), "column", tp_rank, tp_size, tp_group)
        self.o_proj = TPLinear(orig_attn.o_proj.weight.data, getattr(orig_attn.o_proj, 'bias', None), "row", tp_rank, tp_size, tp_group)

        # LoRA for q_proj and v_proj (column-parallel)
        self.q_lora = self._make_lora(orig_attn.q_proj, "column", tp_rank, tp_size, tp_group)
        self.v_lora = self._make_lora(orig_attn.v_proj, "column", tp_rank, tp_size, tp_group)

        # Per-TP-rank head counts
        self.num_heads_local = self.num_heads // tp_size
        self.num_kv_heads_local = self.num_key_value_heads // tp_size

    @staticmethod
    def _make_lora(orig_proj, mode, tp_rank, tp_size, tp_group=None):
        """Extract LoRA weights from a PEFT-wrapped linear and create TPLora."""
        lora_a = getattr(orig_proj, 'lora_A', None)
        lora_b = getattr(orig_proj, 'lora_B', None)
        if lora_a is None or lora_b is None:
            return None
        a_weight = lora_a.default.weight.data if hasattr(lora_a, 'default') else lora_a.weight.data
        b_weight = lora_b.default.weight.data if hasattr(lora_b, 'default') else lora_b.weight.data
        scaling = getattr(orig_proj, 'scaling', {}).get('default', 1.0)
        if hasattr(orig_proj, 'scaling') and isinstance(orig_proj.scaling, (int, float)):
            scaling = orig_proj.scaling
        return TPLora(a_weight, b_weight, mode, tp_rank, tp_size, scaling, tp_group)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Apply LoRA (column-parallel: no all-reduce needed)
        if self.q_lora is not None:
            q = q + self.q_lora(hidden_states)
        if self.v_lora is not None:
            v = v + self.v_lora(hidden_states)

        # Reshape to (bsz, num_heads_local, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.num_heads_local, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads_local, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads_local, self.head_dim).transpose(1, 2)

        # Apply RoPE if position embeddings provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Apply rotary embedding (simplified — assumes standard RoPE)
            q = self._apply_rotary(q, cos, sin)
            k = self._apply_rotary(k, cos, sin)

        # GQA: repeat k/v heads if needed
        if self.num_kv_heads_local < self.num_heads_local:
            repeat_factor = self.num_heads_local // self.num_kv_heads_local
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_out)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        """Apply rotary position embedding."""
        # cos/sin shape: (1, 1, seq_len, head_dim) or (bsz, 1, seq_len, head_dim)
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class TPQwen2MLP(nn.Module):
    """TP-aware Qwen2 MLP (SwiGLU)."""

    def __init__(self, orig_mlp, tp_rank: int, tp_size: int, tp_group=None):
        super().__init__()
        self.gate_proj = TPLinear(orig_mlp.gate_proj.weight.data, None, "column", tp_rank, tp_size, tp_group)
        self.up_proj = TPLinear(orig_mlp.up_proj.weight.data, None, "column", tp_rank, tp_size, tp_group)
        self.down_proj = TPLinear(orig_mlp.down_proj.weight.data, None, "row", tp_rank, tp_size, tp_group)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TPQwen2DecoderLayer(nn.Module):
    """A single Qwen2 decoder layer with TP."""

    def __init__(self, orig_layer, tp_rank: int, tp_size: int, tp_group=None):
        super().__init__()
        self.self_attn = TPQwen2Attention(orig_layer.self_attn, tp_rank, tp_size, tp_group)
        self.mlp = TPQwen2MLP(orig_layer.mlp, tp_rank, tp_size, tp_group)
        self.input_layernorm = orig_layer.input_layernorm
        self.post_attention_layernorm = orig_layer.post_attention_layernorm

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        # Pre-norm + attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)
        hidden_states = residual + attn_out

        # Pre-norm + MLP + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out

        return hidden_states


class TPMiddleStage:
    """Stage_1 with true Tensor Parallelism.

    Each TP rank holds a shard of every layer's weights.  Forward/backward
    uses all-reduce for row-parallel layers.  This matches vLLM's inference
    TP so LoRA weights can be synced directly.
    """

    def __init__(self, layers, rotary_emb, device, tp_rank: int, tp_size: int):
        self.device = torch.device(device)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.rotary_emb = rotary_emb

        # Create a process group for just the TP ranks within stage_1.
        # The global process group includes all ranks (stage_0/1/2), but
        # TP all-reduce should only happen across stage_1 TP ranks.
        stage_1_ranks = list(range(1, 1 + tp_size))  # ranks 1,2 for tp_size=2
        self.tp_group = dist.new_group(ranks=stage_1_ranks)

        # Build TP layers from original layers
        self.layers = nn.ModuleList([
            TPQwen2DecoderLayer(layer, tp_rank, tp_size, self.tp_group) for layer in layers
        ])
        self.layers.to(self.device)
        if self.rotary_emb is not None:
            self.rotary_emb = self.rotary_emb.to(self.device)

        # Freeze all parameters
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
        t = topology or {"stage_0": [0], "stage_1": [1], "stage_2": [2]}
        recv_rank = t["stage_0"][0]
        send_rank = t["stage_2"][0]
        _VALID_FLAGS = {FWD_ONLY, FWD_WITH_BWD, SHUTDOWN, PIPELINE_DONE}

        header = torch.zeros(6, dtype=torch.int64, device=self.device)
        dist.recv(header, src=recv_rank)
        flag = int(header[0].item())

        if flag not in _VALID_FLAGS:
            raise RuntimeError(f"[TPStage1] illegal flag {flag} from rank {recv_rank}")

        if flag == SHUTDOWN:
            return False

        dist.send(header.clone(), dst=send_rank)

        if flag == PIPELINE_DONE:
            return True

        B, S, H, has_mask, dtype_code = [int(x) for x in header[1:].tolist()]
        tensor_dtype = _DTYPE_MAP[dtype_code]
        h = self._recv_tensor((B, S, H), tensor_dtype, recv_rank)
        pos_ids = self._recv_tensor((B, S), torch.int64, recv_rank)
        if has_mask:
            mask = self._recv_tensor((B, 1, S, S), tensor_dtype, recv_rank)
        else:
            mask = None

        if flag == FWD_ONLY:
            with torch.no_grad():
                pos_emb = self._get_pos_emb(h, pos_ids)
                h_out = self.forward(h, position_embeddings=pos_emb, attention_mask=mask)
            self._send_tensor(h_out, send_rank)
            self._send_tensor(pos_ids, send_rank)

        elif flag == FWD_WITH_BWD:
            h_in = h.detach().requires_grad_(True)
            pos_emb = self._get_pos_emb(h_in, pos_ids)
            h_out = self.forward(h_in, position_embeddings=pos_emb, attention_mask=mask)
            self._send_tensor(h_out.detach(), send_rank)
            self._send_tensor(pos_ids, send_rank)

            grad_out = self._recv_tensor(h_out.shape, h_out.dtype, send_rank)
            torch.autograd.backward(h_out, grad_out)
            self._send_tensor(h_in.grad, recv_rank)

        return True

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
