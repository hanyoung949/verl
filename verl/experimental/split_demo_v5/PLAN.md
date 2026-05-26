# Split Demo v5 — 3-Stage Pipeline

> 最后更新：2026-05-25
> 范围：`verl/experimental/split_demo_v5/`
> 目标：基于 split_demo_v4 创建 v5，在 3-stage (Head/Middle/Tail) 流水线上新增 Qwen3.5 MTP rollout。

---

## 0. 核心主旨

v5 回答一个问题：**在不改动 split_demo_v4、不引入多卡 per stage 的前提下，能不能给 3-stage split pipeline 加上 Qwen3.5 的 MTP 投机解码？**

```
v4 答案：能跑 3-stage pipeline，但没有 MTP。
v5 答案：复制 v4 后，只在 v5 中加载 `mtp.*` 权重并在 Tail rollout 中做 multi-token speculative verification。
```

v5 保持 v4 的单 rank / 单 CUDA device stage 布局：Head、Middle、Tail 仍分别运行在 3 个进程上。

---

## 1. 架构

### 1.1 3-stage 拓扑

```
torchrun --nproc_per_node=3

Stage 0 (Head):   GPU 0                Stage 1 (Middle): GPU 1              Stage 2 (Tail): GPU 2
embed_tokens                            middle_layers[4:20]                  tail_layers[20:24]
front_layers[0:4]                       rotary_emb (本地重算)                 norm
LoRA (可训练)                           冻结                                   lm_head
                                                                                 MTP draft head
                                                                                 LoRA (可训练)
                                                                                 optimizer
                                                                                 loss
                                                                                 checkpoint
```

### 1.2 通信流

```
Forward:
  Head → Middle:   send h_front [B, S, H] + position_ids [B, S] + mask
  Middle → Tail:   send h_middle [B, S, H] + position_ids [B, S] + mask

Backward:
  Tail → Middle:   send grad_h_middle [B, S, H]
  Middle → Head:   send grad_h_front [B, S, H]  (Middle 做本地 backward)
```

### 1.3 Stage 职责

| Stage | Rank | 层 | 可训练 | 持有 optimizer |
|---|---|---|---|---|
| Head | 0 | embed + front[0:4] | 是 (LoRA) | 否 |
| Middle | 1 | middle[4:20] | 否 (冻结) | 否 |
| Tail | 2 | tail[20:24] + norm + lm_head + MTP draft head | 是 (LoRA) | 是 |

Tail 是"主 stage"：持有 optimizer、计算 loss、保存 checkpoint，并在 rollout 时执行 MTP draft + target verification。

### 1.4 MTP 范围

- 默认模型：`Qwen/Qwen3.5-2B`
- 默认配置：`model.mtp.enable=true`、`enable_rollout=true`、`enable_train=false`
- 默认 `num_speculative_tokens=4`，一次请求最多返回 `num_speculative_tokens + 1` 个 token
- MTP 只用于 rollout，不加入训练辅助 loss
- 不修改 `split_demo_v4`

---

## 2. 核心模块

### 2.1 Stage

```python
class Stage:
    """流水线中的一个通用 stage。"""
    def __init__(self, stage_id, layers, device, trainable=False):
        self.stage_id = stage_id
        self.layers = nn.ModuleList(layers)
        self.device = device
        self.trainable = trainable
        self.prev_transport = None  # 和前一个 stage 的通信
        self.next_transport = None  # 和后一个 stage 的通信

    def forward(self, h_in, **kwargs):
        for layer in self.layers:
            h_in = self._call_layer(layer, h_in, **kwargs)
        return h_in

    def trainable_parameters(self):
        if not self.trainable:
            return []
        return [p for p in self.layers.parameters() if p.requires_grad]
```

### 2.2 HeadStage

```python
class HeadStage(Stage):
    """Head stage: 包含 embed_tokens。"""
    def __init__(self, embed_tokens, layers, rotary_emb, device):
        super().__init__("head", layers, device, trainable=True)
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb

    def forward(self, input_ids, attention_mask, **kwargs):
        h = self.embed_tokens(input_ids)
        position_ids = self._build_position_ids(attention_mask)
        position_embeddings = self.rotary_emb(h, position_ids)
        causal_mask = self._build_causal_mask(attention_mask)
        h = super().forward(h, position_ids=position_ids,
                          position_embeddings=position_embeddings,
                          attention_mask=causal_mask)
        return h, position_ids, position_embeddings, causal_mask
```

### 2.3 TailStage

```python
class TailStage(Stage):
    """Tail stage: 包含 norm + lm_head + optimizer。"""
    def __init__(self, layers, norm, lm_head, device, lr=1e-4, clip_grad=1.0):
        super().__init__("tail", layers, device, trainable=True)
        self.norm = norm
        self.lm_head = lm_head
        self.optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=lr)
        self.clip_grad = clip_grad

    def forward(self, h, **kwargs):
        h = super().forward(h, **kwargs)
        h = self.norm(h)
        return self.lm_head(h)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters(), max_norm=self.clip_grad)
        self.optimizer.step()
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
```

### 2.4 StageTransport

```python
class StageTransport:
    """两个相邻 stage 之间的 NCCL P2P 通信。"""
    def __init__(self, local_rank, remote_rank, device):
        self.local_rank = local_rank
        self.remote_rank = remote_rank
        self.device = device

    def send(self, tensor):
        dist.send(tensor.contiguous(), dst=self.remote_rank)

    def recv(self, shape, dtype):
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=self.remote_rank)
        return buf
```

### 2.5 SplitPipelineEngine

```python
@EngineRegistry.register(model_type="llm", backend="split_pipeline", device="cuda")
class SplitPipelineEngine(BaseEngine):
    """3-stage split pipeline 引擎。"""

    def __init__(self):
        self.stage = None          # 本进程的 stage
        self.is_head = False
        self.is_tail = False

    def initialize(self):
        # 根据 rank 创建对应的 stage
        # rank0 → HeadStage, rank1 → MiddleStage, rank2 → TailStage
        ...

    def forward_backward_batch(self, data, loss_fn, forward_only=False):
        if self.is_head:
            return self._head_forward_backward(data, loss_fn, forward_only)
        elif self.is_tail:
            return self._tail_forward_backward(data, loss_fn, forward_only)
        else:
            return self._middle_forward_backward(data, loss_fn, forward_only)

    def _head_forward_backward(self, data, loss_fn, forward_only):
        h, pos_ids, pos_emb, mask = self.stage.forward(data["input_ids"], data["attention_mask"])
        self.next_transport.send(h)          # 发给 Middle
        self.next_transport.send(pos_ids)
        self.next_transport.send(mask)
        if not forward_only:
            grad_h = self.next_transport.recv(h.shape, h.dtype)  # 从 Middle 收 grad
            h.backward(grad_h)
        return {}

    def _middle_forward_backward(self, data, loss_fn, forward_only):
        h = self.prev_transport.recv(...)     # 从 Head 收 h_front
        pos_ids = self.prev_transport.recv(...)
        mask = self.prev_transport.recv(...)
        h_out = self.stage.forward(h, ...)
        self.next_transport.send(h_out)       # 发给 Tail
        if not forward_only:
            grad_out = self.next_transport.recv(h_out.shape, h_out.dtype)
            h_in = h.detach().requires_grad_(True)
            h_out = self.stage.forward(h_in, ...)
            torch.autograd.backward(h_out, grad_out)
            self.prev_transport.send(h_in.grad)  # 发给 Head
        return {}

    def _tail_forward_backward(self, data, loss_fn, forward_only):
        h = self.prev_transport.recv(...)     # 从 Middle 收 h_middle
        logits = self.stage.forward(h, ...)
        loss = loss_fn(logits, data)
        if not forward_only:
            loss.backward()
            self.prev_transport.send(h.grad)  # 发给 Middle
        return {"logits": logits, "loss": [loss.item()]}
```

---

## 3. 实现计划

### Step 1: Stage + Transport + HeadStage + TailStage

创建核心模块，验证 import。

### Step 2: SplitPipelineEngine + main_split_v5.py

创建 3-stage 引擎和入口。

### Step 3: 跑 10 步 GRPO

验证训练正确性，对比 v3 结果。

### Step 4: 文档

写 STATUS.md 记录结果。

---

## 4. 与 v4 的对比

| 维度 | v4 | v5 |
|---|---|---|
| Stage 数量 | 3 (配置驱动) | 3 (配置驱动) |
| 默认模型 | Qwen2.5-3B-Instruct | Qwen/Qwen3.5-2B |
| Tail rollout | target logits 直接采样 | MTP 多 token draft + target verification |
| 每步返回 | `next_token` + `log_prob` | `next_token` + target `log_prob` + `token_mask` |
| MTP 权重 | 不加载 | 显式加载 `mtp.*` safetensors |
| 多 GPU per stage | 不支持 | 不支持 |

---

## 5. 后续

本轮只添加 MTP rollout。以下功能后续再做：

- N-stage (4+)
- Pipeline 1F1B 调度
- 多机 NCCL
- 多 GPU per stage
- Ray 编排
- vLLM + 权重同步
- MTP auxiliary training loss
