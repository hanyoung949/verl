# Split Demo v4 — 3-Stage Pipeline

> 最后更新：2026-05-18
> 范围：`verl/experimental/split_demo_v4/`
> 目标：Stage 抽象 + 3-stage (Head/Middle/Tail) 流水线，用配置驱动拓扑。

---

## 0. 核心主旨

v4 回答一个问题：**如果给我 3 张 GPU，能不能通过改配置把模型拆成 3 个 stage 跑起来？**

```
v3 答案：不能。rank0/rank1 硬编码，2 进程固定。
v4 答案：能。Stage 是通用容器，拓扑由配置决定。
```

v4 直接做 3-stage，不做 2-stage 复现。3-stage 更能体现 Stage 抽象的价值。

---

## 1. 架构

### 1.1 3-stage 拓扑

```
torchrun --nproc_per_node=3

Stage 0 (Head):   GPU 0                Stage 1 (Middle): GPU 1              Stage 2 (Tail): GPU 2
embed_tokens                            middle_layers[4:32]                  tail_layers[32:36]
front_layers[0:4]                       rotary_emb (本地重算)                 norm
LoRA (可训练)                           冻结                                   lm_head
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
| Middle | 1 | middle[4:32] | 否 (冻结) | 否 |
| Tail | 2 | tail[32:36] + norm + lm_head | 是 (LoRA) | 是 |

Tail 是"主 stage"：持有 optimizer、计算 loss、保存 checkpoint。

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

### Step 2: SplitPipelineEngine + main_split_v4.py

创建 3-stage 引擎和入口。

### Step 3: 跑 10 步 GRPO

验证训练正确性，对比 v3 结果。

### Step 4: 文档

写 STATUS.md 记录结果。

---

## 4. 与 v3 的对比

| 维度 | v3 | v4 |
|---|---|---|
| Stage 数量 | 2 (硬编码) | 3 (配置驱动) |
| 代码组织 | SplitActorCore (耦合) | Stage (解耦) |
| 通信 | NCCLMiddleExecutor (固定连 rank1) | StageTransport (通用) |
| 引擎 | SplitEngine | SplitPipelineEngine |
| 加 stage | 改代码 | 改配置 |
| 可扩展性 | 有限 | 可扩展到 N-stage |

---

## 5. v5 留给后续

v4 只做 3-stage。以下功能留给 v5：

- N-stage (4+)
- Pipeline 1F1B 调度
- 多机 NCCL
- 多 GPU per stage
- Ray 编排
- vLLM + 权重同步
- GTPO / Overlong / MTP
