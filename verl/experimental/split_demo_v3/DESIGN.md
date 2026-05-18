# Split Actor GRPO Demo v3 — verl 集成版设计文档

**目标**：把 split training 包装成 verl 的标准引擎接口（BaseEngine），复用 verl 的算法、数据协议和 checkpoint 体系。  
**验证平台**：单机 2×4090，`torchrun --nproc_per_node=2`。  
**范围**：`verl/experimental/split_demo_v3/`。  
**与 v2 的关系**：核心通信逻辑（NCCL P2P + autograd.Function）不变；变化在 Engine 接口层和数据协议层。

---

## 1. v2 的根本限制

v2 是一个独立 demo，有自己的训练循环、数据格式和 checkpoint 方案。它和 verl 框架完全隔离：

- 训练循环是手写的 `for step in range(total_steps)`
- 数据格式是原始 tensor + dict
- checkpoint 是手写的 `adapter_state.pt`
- 无法被 verl 的 `TrainingWorker` / `RayWorkerGroup` / `RayPPOTrainer` 发现和调用

v3 的目标是让 split training "说 verl 的语言"：

1. **引擎接口标准化**：实现 `SplitEngine(BaseEngine)`，所有 forward/backward/optimizer 操作通过统一接口
2. **数据协议兼容**：Engine 边界同时接受 `TensorDict` 和 `DataProto`
3. **算法复用**：GRPO advantage、loss 等计算复用 verl 的 `core_algos.py`
4. **checkpoint 规范**：通过 `engine.save_checkpoint()` / `load_checkpoint()` 接口

---

## 2. 进程分工

和 v2 相同，两个进程，一个训练一个服务：

```
torchrun --nproc_per_node=2 -m verl.experimental.split_demo_v3.main_split_v3

rank0 (edge / output rank)             rank1 (middle service rank)
─────────────────────────────          ──────────────────────────────
embed_tokens                           middle_layers[4:32]
front_layers[0:4] + LoRA (可训练)       rotary_emb（本地重算 pos_emb）
tail_layers[32:36] + LoRA (可训练)      （冻结，不做参数更新）
norm / lm_head
optimizer / rollout / reward
SplitEngine (BaseEngine 接口)
GRPO loss (grpo_loss_fn)
checkpoint save/load
```

rank1 的职责没有变化：仍然是纯粹的请求响应循环，不持有训练逻辑。

---

## 3. v3 核心架构

### 3.1 模块分层

```
main_split_v3.py (入口 + GRPO 训练循环)
    │
    ├── SplitEngine(BaseEngine)          ← v3 新增：verl 标准引擎接口
    │       ├── _normalize_batch()       ← DataProto / TensorDict 兼容
    │       ├── forward_backward_batch() ← 核心前向+反向
    │       ├── train_batch()            ← 训练一步
    │       ├── infer_batch()            ← 推理
    │       ├── save/load_checkpoint()   ← checkpoint
    │       └── SplitActorCore           ← 模型核心（同 v2）
    │               └── NCCLMiddleExecutor ← NCCL 通信（同 v2）
    │
    ├── SimpleSplitRollout               ← 自回归生成（同 v2）
    ├── FunctionReward                   ← 函数打分（同 v2）
    └── grpo_loss_fn()                   ← GRPO loss（v3 新增：可传给 Engine）
```

### 3.2 和 v2 的代码差异

| 文件 | v2 | v3 |
|---|---|---|
| `core/split_actor.py` | `SplitActorCore` | `SplitActorCore`（代码相同） |
| `core/middle_executor.py` | `NCCLMiddleExecutor` | `NCCLMiddleExecutor`（代码相同） |
| `core/middle_worker.py` | `MiddleWorker` | `MiddleWorker`（代码相同） |
| `split_trainer.py` | 手写训练循环 | **删除**，合并到 `main_split_v3.py` |
| `main_grpo_split.py` | 入口 | `main_split_v3.py`（通过 Engine 接口） |
| **新增** `core/split_engine.py` | 无 | `SplitEngine(BaseEngine)` |

v3 的变化集中在两个地方：
1. 新增 `split_engine.py`：BaseEngine 实现
2. 重构 `main_split_v3.py`：通过 Engine 接口调用，而不是直接操作 split_core

---

## 4. SplitEngine 设计

### 4.1 BaseEngine 接口实现

```python
@EngineRegistry.register(model_type="llm", backend="split", device="cuda")
class SplitEngine(BaseEngine):
    def initialize(self)                              # 加载模型
    def forward_backward_batch(data, loss_fn, ...)    # 前向+反向
    def train_batch(data, loss_fn)                    # 训练一步
    def infer_batch(data)                             # 推理
    def optimizer_zero_grad()                         # 清梯度
    def optimizer_step()                              # 更新参数
    def lr_scheduler_step()                           # 学习率调度
    def save_checkpoint(local_path, global_step, ...) # 保存
    def load_checkpoint(local_path, ...)              # 加载
    def get_data_parallel_size/rank/group()           # 数据并行
    def is_mp_src_rank_with_outputs()                 # 输出 rank
    def is_param_offload_enabled                      # offload 配置
    def is_optimizer_offload_enabled                  # offload 配置
    def train_mode() / eval_mode()                    # 上下文切换
    def to(device)                                    # 设备管理
```

共 15 个接口方法（含 2 个 property）。

### 4.2 _normalize_batch：DataProto 兼容

```python
@staticmethod
def _normalize_batch(data):
    if isinstance(data, DataProto):
        return data.batch, data.non_tensor_batch, data.meta_info
    if isinstance(data, TensorDict):
        return data, {}, {}
    raise TypeError(...)
```

Engine 内部统一使用 TensorDict 做计算，调用方可以传入 DataProto 或 TensorDict。

### 4.3 forward_backward_batch 数据流

```
输入: TensorDict / DataProto {input_ids, attention_mask, ...}
    ↓
_normalize_batch() → TensorDict
    ↓
split_core.forward_full(input_ids, attention_mask)
    ├─ embed + front_layers
    ├─ NCCLMiddleExecutor.execute() → rank1 middle_layers
    └─ tail_layers + norm + lm_head → logits
    ↓
loss_function(logits, batch) → {loss, metrics}
    ↓
if not forward_only: loss.backward()
    ↓
返回: {logits, loss, metrics}
```

---

## 5. 训练流程

### 5.1 Phase 1 规范路径

v3 的训练循环通过 Engine 接口调用，不再直接操作 split_core：

```python
# old_log_prob: 通过 infer_batch
infer_result = engine.infer_batch(TensorDict({input_ids, attention_mask}))
logits_old = infer_result["logits"]
old_log_prob = logprobs_from_logits(logits_old, response_ids)

# training: 通过 train_batch
train_result = engine.train_batch(
    TensorDict({input_ids, attention_mask, response_ids, response_mask,
                old_log_probs, advantages}),
    grpo_loss_fn
)
loss = train_result["loss"]
```

### 5.2 GRPO loss 函数

```python
def make_grpo_loss_fn(clip_low, clip_high, loss_agg_mode, loss_scale_factor):
    def grpo_loss_fn(logits, data):
        prompt_len = logits.shape[1] - data["response_ids"].shape[1]
        log_prob_new = logprobs_from_logits(logits[:, prompt_len-1:-1, :], data["response_ids"])
        
        neg_kl = clamp(log_prob_new - data["old_log_probs"], -20, 20)
        ratio = exp(neg_kl)
        adv = data["advantages"]
        
        pg_losses = maximum(-adv * ratio, -adv * ratio.clamp(1-clip_low, 1+clip_high))
        loss = agg_loss(pg_losses, data["response_mask"], loss_agg_mode, loss_scale_factor)
        
        return {"loss": loss, "metrics": {approx_kl, clipfrac}}
    return grpo_loss_fn
```

通过闭包捕获 clip 参数，loss 函数从 data 中动态读取 prompt_len。

---

## 6. 通信协议

和 v2 完全相同。一次 forward_full 的 NCCL 往返：

```
rank0                                     rank1
─────                                     ─────
h_front = embed + front(input)
send header [flag, B, S, H, mask, dtype]
send h_front [B, S, H]
send position_ids [B, S]
send attention_mask [B, 1, S, S] (可选)
                                          recv h_front, position_ids, mask
                                          pos_emb = rotary_emb(h_front, pos_ids)
                                          h_out = middle_layers(h_front)
send h_out [B, S, H] ◀─────────────────── send h_out
```

backward 时额外的梯度传递（FWD_WITH_BWD）：

```
loss.backward()
send grad_h_middle ──────────────────────▶ recv grad_out
                                            torch.autograd.backward(h_out, grad_out)
                                            grad_h_front = h_in.grad
recv grad_h_front ◀────────────────────── send h_in.grad
```

---

## 7. 算法配置

| 特性 | 配置值 | 说明 |
|---|---|---|
| GRPO advantage | `compute_grpo_outcome_advantage` | 组内相对 advantage，无需 critic |
| Dr.GRPO | `norm_adv_by_std_in_grpo: false` | 不按 std 归一化 |
| Dr.GRPO | `loss_agg_mode: seq-mean-token-sum-norm` | 序列级别 loss 聚合 |
| Dr.GRPO | `loss_scale_factor: 128` | 固定缩放因子 |
| DAPO clip-higher | `cliprange_low: 0.2, cliprange_high: 0.28` | 非对称裁剪 |
| DAPO dynamic sampling | `dynamic_sampling: true` | 过滤零方差 group |
| 去 KL / Reference | 不部署 ref model | 四篇论文一致证明可行 |

---

## 8. Checkpoint 格式

```
checkpoints/latest/
  adapter_state.pt      (3.2MB)   ← 32 个 LoRA 参数
  optimizer.pt          (6.5MB)   ← AdamW 状态
  adapter_meta.json     (108B)    ← model_path, front_end, middle_end, global_step
```

`adapter_meta.json` 内容：

```json
{
  "model_path": "~/share/Qwen2.5-3B-Instruct",
  "front_end": 4,
  "middle_end": 32,
  "global_step": 10
}
```

load 语义：rank0 恢复 LoRA 参数 + optimizer 状态，rank1 不恢复（冻结）。

---

## 9. 运行时指标

### 9.1 训练指标

```
[step 1]  reward=0.5000 valid_group_ratio=1.0000 loss=-0.0458 clipfrac=0.0469 approx_kl=-0.0049 gpu0_peak=2.62GB
[step 2]  reward=0.3750 valid_group_ratio=0.5000 loss=-0.0035 clipfrac=0.0469 approx_kl=-0.0003 gpu0_peak=2.62GB
[step 3]  reward=0.3333 valid_group_ratio=0.7500 loss=-0.2715 clipfrac=0.1667 approx_kl=0.0277 gpu0_peak=2.60GB
[step 4]  reward=0.5833 valid_group_ratio=0.7500 loss=0.2383 clipfrac=0.0938 approx_kl=0.0293 gpu0_peak=2.57GB
[step 5]  reward=0.5000 valid_group_ratio=0.7500 loss=-0.0145 clipfrac=0.1354 approx_kl=0.0352 gpu0_peak=2.56GB
[step 6]  reward=0.4167 valid_group_ratio=0.7500 loss=-0.1293 clipfrac=0.0729 approx_kl=0.0216 gpu0_peak=2.55GB
[step 7]  reward=0.5000 valid_group_ratio=0.5000 loss=-0.0083 clipfrac=0.0417 approx_kl=0.0055 gpu0_peak=2.60GB
[step 8]  reward=0.3750 valid_group_ratio=0.5000 loss=-0.0080 clipfrac=0.0312 approx_kl=0.0121 gpu0_peak=2.62GB
[step 9]  reward=0.5000 valid_group_ratio=0.7500 loss=-0.0080 clipfrac=0.1146 approx_kl=0.2070 gpu0_peak=2.62GB
[step 10] reward=0.4167 valid_group_ratio=0.7500 loss=0.1752 clipfrac=0.0625 approx_kl=0.0254 gpu0_peak=2.57GB
```

配置：`lr=5e-4, group_size=4, max_new_tokens=24, loss_scale_factor=24, update_epochs=2, mini_batch_size=8`

### 9.2 显存

| GPU | 分配 | 峰值 |
|---|---|---|
| GPU 0 (edge) | ~2.01-2.06 GB | ~2.55-2.62 GB |
| GPU 1 (center) | ~4 GB | ~4 GB |

### 9.3 与 v2 对比

| 指标 | v2 | v3 |
|---|---|---|
| reward 范围 | 0.33–0.75 | 0.33–0.58 |
| gpu0_peak | 2.53 GB | 2.55-2.62 GB |
| trainable params | 32 | 32 |
| 调用方式 | 直接 split_core | Engine 接口 |
| 数据格式 | tensor + dict | TensorDict / DataProto |

数值量级一致，v3 的 Engine 接口层没有引入性能开销。

---

## 10. v4 迁移边界

v3 设计为 v4 的基础。v4 不应重写 Engine 接口，只替换内部 computation flow：

| v3 组件 | v4 复用目标 |
|---|---|
| `SplitActorCore.forward_front()` | Head stage runner |
| `SplitActorCore.forward_tail()` | Tail stage runner |
| `MiddleWorker._run_layers()` | 通用 Middle stage runner |
| `MiddleWorker.run()` | Stage service loop + local backward |
| `NCCLMiddleExecutor` | 相邻 stage 的 transport/autograd bridge |
| `SplitEngine(BaseEngine)` | `SplitPipelineEngine` 接口参考 |
| `grpo_loss_fn` | v4 Tail stage loss 参考 |

### 保持稳定的接口

```
DataProto in
train_batch / infer_batch
metrics out
save_checkpoint / load_checkpoint
is_mp_src_rank_with_outputs
```

### 会变化的运行时

v3: `rank0 edge → rank1 middle → rank0 edge`
v4: `Head stage → Middle stage 1 → ... → Middle stage N → Tail stage`
