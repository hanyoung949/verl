# Split Actor GRPO Demo v3 — verl 集成技术方案

> **目标**：把 split transport 做成 verl Worker 的可插拔组件，复用 Ray/vLLM/FSDP/Checkpoint 等已有基础设施。
> **范围**：`verl/experimental/split_demo_v3/`
> **前置**：v2 NCCL 双进程版已验证通过（Phase 1-3）

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [verl 主线架构分析](#2-verl-主线架构分析)
3. [核心挑战](#3-核心挑战)
4. [整体架构设计](#4-整体架构设计)
5. [阶段 1：Ray 集成](#5-阶段-1ray-集成)
6. [阶段 2：SplitEngine](#6-阶段-2splitengine)
7. [阶段 3：vLLM 集成](#7-阶段-3vllm-集成)
8. [阶段 4：生产化](#8-阶段-4生产化)
9. [关键设计决策](#9-关键设计决策)
10. [风险与缓解](#10-风险与缓解)
11. [附录：verl 接口参考](#11-附录verl-接口参考)

---

## 1. 背景与动机

v2 (`split_demo_v2`) 用 `torchrun` 启动双进程，通过 NCCL P2P 实现跨进程前向和梯度传递，验证了"层间拆分 + 跨进程可微"的核心思路可行。

但 v2 有以下局限性：

- **进程管理**：`torchrun` 固定 2 进程，无容错、无弹性
- **训练引擎**：手写训练循环，没有复用 verl 的 Engine 抽象
- **推理引擎**：手写 NaiveSplitRollout，无 KV cache、无 vLLM
- **Checkpoint**：手写 adapter_state.pt，不兼容 verl 的 CheckpointEngine
- **数据协议**：原始 tensor + dict，不兼容 verl 的 DataProto

v3 的目标是把 split transport 集成进 verl 的框架中，复用其所有基础设施。

---

## 2. verl 主线架构分析

### 2.1 核心组件

```
main_ppo.py
    │
    └── RayPPOTrainer (Ray Actor, 单控制器)
            │
            ├── RayWorkerGroup
            │       ├── TrainingWorker (FSDP/Megatron)    ← 训练引擎
            │       │       └── BaseEngine (FSDPEngine/MegatronEngine)
            │       ├── RolloutWorker (vLLM/SGLang)       ← 推理引擎
            │       ├── RewardWorker                       ← reward model
            │       └── RefPolicyWorker                    ← reference model
            │
            ├── ResourcePoolManager                        ← GPU 资源管理
            └── DataProto                                  ← 统一数据协议
```

### 2.2 关键抽象

| 抽象 | 位置 | 作用 |
|---|---|---|
| `Worker` | `single_controller/base/worker.py` | Worker 基类，管理 rank/world_size/device |
| `TrainingWorker` | `workers/engine_workers.py` | 训练 Worker，持有 Engine |
| `BaseEngine` | `workers/engine/base.py` | 训练引擎接口，有 FSDP/Megatron 等实现 |
| `EngineRegistry` | `workers/engine/base.py` | 引擎注册表，按 (model_type, backend, device) 选择引擎 |
| `RayWorkerGroup` | `single_controller/ray/base.py` | Ray Actor 组管理，dispatch/collect |
| `ResourcePoolManager` | `single_controller/ray/base.py` | GPU 资源池管理 |
| `DataProto` | `protocol.py` | 统一数据协议，TensorDict 封装 |

### 2.3 BaseEngine 接口

```python
class BaseEngine:
    def initialize(self):                      # 加载模型/optimizer
    def forward_backward_batch(self, data, loss_fn, forward_only=False):  # 前向+反向
    def train_batch(self, data, loss_fn):      # 训练一步
    def infer_batch(self, data, loss_fn=None): # 推理
    def save_checkpoint(self, ...):            # 保存 checkpoint
    def load_checkpoint(self, ...):            # 加载 checkpoint
    def to(self, device, model=True, optimizer=True, grad=True):  # 设备管理
    def get_data_parallel_size/rank/group(self):  # 数据并行信息
    def is_mp_src_rank_with_outputs(self):     # 是否是模型并行输出 rank
```

### 2.4 EngineRegistry 注册机制

```python
@EngineRegistry.register(model_type="llm", backend="fsdp", device="cuda")
class FSDPEngine(BaseEngine):
    ...

# 使用时：
engine = EngineRegistry.new(model_type="llm", backend="fsdp", ...)
```

---

## 3. 核心挑战

### 3.1 范式冲突

| 维度 | verl 主线 | split 架构 | 冲突点 |
|---|---|---|---|
| 模型持有 | 每个 worker 完整模型 | 不同 worker 不同层 | FSDP 需要完整模型 wrap |
| 并行策略 | FSDP (数据并行, ZeRO 分片) | 层间拆分 (pipeline 式) | 两种通信模式不同 |
| 通信 | NCCL all-reduce (自动) | NCCL P2P send/recv (手动) | 不能混用 |
| 推理 | vLLM (完整模型) | 无完整模型 | vLLM 需要完整实例 |

### 3.2 关键问题

**Q1: FSDP 和层间拆分能共存吗？**

不能直接共存。FSDP wrap 需要完整模型，split 没有完整模型。但可以：
- 对 front/tail 的 LoRA 参数局部使用 FSDP（自定义 process_group）
- 或者干脆不用 FSDP，只用 LoRA（当前 demo 的做法，适合 <7B）

**Q2: verl 的 Worker 能持有"半个模型"吗？**

可以。Worker 只是一个 Ray Actor，内部逻辑完全自定义。关键是要实现 BaseEngine 接口。

**Q3: 梯度传递怎么接入 verl 的训练循环？**

verl 的 `forward_backward_batch` 返回 loss，内部调用 `loss.backward()`。split 架构的 backward 需要通过 NCCL 交换梯度。这需要在 `forward_backward_batch` 中嵌入 NCCL 梯度传递逻辑。

---

## 4. 整体架构设计

### 4.1 v3 架构图

```
RayPPOTrainer (单控制器)
    │
    ├── RayWorkerGroup
    │       │
    │       ├── EdgeWorker (Ray Actor, GPU 0)
    │       │       ├── SplitEngine
    │       │       │       ├── embed_tokens
    │       │       │       ├── front_layers + LoRA (训练)
    │       │       │       ├── tail_layers + LoRA (训练)
    │       │       │       ├── NCCLMiddleExecutor (P2P 通信)
    │       │       │       └── optimizer
    │       │       │
    │       │       └── vLLMRollout (推理用, 完整模型)
    │       │
    │       ├── CenterWorker (Ray Actor, GPU 1)
    │       │       └── MiddleExecutor (冻结 middle_layers)
    │       │
    │       └── (可选) RewardWorker
    │
    └── ResourcePoolManager
            Pool 0: [GPU 0]  ← EdgeWorker
            Pool 1: [GPU 1]  ← CenterWorker
```

### 4.2 与 v2 的关系

| 组件 | v2 | v3 |
|---|---|---|
| 进程管理 | torchrun | Ray Actor + WorkerGroup |
| rank0 逻辑 | SplitActorCore + Trainer | EdgeWorker + SplitEngine |
| rank1 逻辑 | MiddleWorker.run() | CenterWorker + MiddleExecutor |
| 通信 | 手写 NCCL P2P | 复用 verl 的 NCCL init |
| 训练循环 | 手写 SplitGRPOTrainer | 复用 verl 的 RayPPOTrainer |
| 数据 | 原始 dict | DataProto |
| 推理 | NaiveSplitRollout | vLLMRollout (后期) |
| Checkpoint | 手写 adapter_state.pt | CheckpointEngine (适配) |

---

## 5. 阶段 1：Ray 集成

**目标**：把 v2 的 torchrun 双进程改为 Ray 双 Actor，验证跨 Actor NCCL 通信。

### 5.1 需要创建的文件

```
split_demo_v3/
├── PLAN.md                          ← 本文档
├── core/
│   ├── __init__.py
│   ├── edge_worker.py               ← EdgeWorker (Ray Actor, 前段+尾段)
│   ├── center_worker.py             ← CenterWorker (Ray Actor, 中段)
│   ├── middle_executor.py           ← NCCLMiddleExecutor (从 v2 复用/改写)
│   └── split_engine.py              ← SplitEngine (BaseEngine 实现, 阶段 2)
├── rollout/
│   └── ...                          ← 从 v2 复用
├── reward/
│   └── ...                          ← 从 v2 复用
├── config/
│   └── split_demo_v3.yaml
├── main_split_v3.py                 ← 入口 (Ray init + WorkerGroup 创建)
└── split_trainer.py                 ← 适配 verl 接口的 trainer
```

### 5.2 EdgeWorker 设计

```python
from verl.single_controller.base.worker import Worker

class EdgeWorker(Worker):
    """rank0 Ray Actor: 持有 front + tail + LoRA, 通过 NCCL 与 CenterWorker 通信。"""

    def __init__(self, config):
        super().__init__()
        # 加载模型, 拆分, 只保留 front/tail
        # 创建 NCCLMiddleExecutor, 连接到 CenterWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, model_path, front_end, middle_end, lora_config):
        """初始化模型和 NCCL 连接。"""

    @register(dispatch_mode=...)
    def forward_full(self, data: TensorDict) -> TensorDict:
        """完整前向 (front → NCCL → tail)。"""

    @register(dispatch_mode=...)
    def train_step(self, data: TensorDict) -> TensorDict:
        """训练一步: forward → loss → backward (含 NCCL 梯度传递) → optimizer.step()。"""
```

### 5.3 CenterWorker 设计

```python
class CenterWorker(Worker):
    """rank1 Ray Actor: 持有冻结 middle_layers, 响应 NCCL 请求。"""

    def __init__(self, config):
        super().__init__()
        # 加载模型, 拆分, 只保留 middle_layers + rotary_emb
        # 冻结所有参数

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, model_path, front_end, middle_end, lora_config):
        """初始化 middle_layers。"""

    def run_loop(self):
        """请求响应循环 (从 v2 MiddleWorker.run() 复用)。"""
```

### 5.4 NCCL 连接建立

verl 已经支持 Ray 环境下的 NCCL 初始化。两个 Ray Actor 启动后，需要：

1. 获取对方的 MASTER_ADDR 和 MASTER_PORT
2. 调用 `dist.init_process_group(backend="nccl")`
3. 世界大小 = 2 (edge + center)

```python
# 在 main_split_v3.py 中
edge_worker = EdgeWorker.options(
    num_gpus=1,
    resources={"GPU": 1}
).remote(config)

center_worker = CenterWorker.options(
    num_gpus=1,
    resources={"GPU": 1}
).remote(config)

# 获取地址, 初始化 NCCL
ray.get([
    edge_worker.init_nccl.remote(center_addr, center_port),
    center_worker.init_nccl.remote(edge_addr, edge_port),
])
```

### 5.5 验证

```python
# Phase 1: 验证 Ray Actor 启动 + NCCL 通信
result = ray.get(edge_worker.forward_full.remote(test_data))
assert result.shape == expected_shape

# Phase 2: 验证梯度传递
result = ray.get(edge_worker.train_step.remote(test_data))
assert all params have .grad
```

---

## 6. 阶段 2：SplitEngine

**目标**：实现 `SplitEngine(BaseEngine)`，适配 verl 的训练引擎接口。

### 6.1 SplitEngine 接口

```python
@EngineRegistry.register(model_type="llm", backend="split", device="cuda")
class SplitEngine(BaseEngine):
    """层间拆分训练引擎。"""

    def initialize(self):
        # 加载模型 → 拆分 → 创建 optimizer (只包含 front/tail LoRA)
        # 建立与 CenterWorker 的 NCCL 连接

    def forward_backward_batch(self, data, loss_fn, forward_only=False):
        # 前向: front → NCCL → tail → logits
        # 如果 forward_only=False: loss.backward() (含 NCCL 梯度传递)
        # 返回 logits + loss

    def train_batch(self, data, loss_fn):
        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_fn, forward_only=False)
        grad_norm = self.optimizer_step()
        return outputs

    def save_checkpoint(self, local_path, ...):
        # 保存 front/tail 的 LoRA 参数 + optimizer state

    def load_checkpoint(self, local_path, ...):
        # 加载 LoRA 参数 + optimizer state

    def to(self, device, ...):
        # 设备管理 (front/tail 移动)

    def get_data_parallel_size/rank/group(self):
        # split 架构下没有数据并行, 返回 1/0/None
```

### 6.2 关键实现细节

**`forward_backward_batch` 的实现**:

```python
def forward_backward_batch(self, data, loss_fn, forward_only=False):
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]

    if forward_only:
        with torch.no_grad():
            logits = self.split_core.forward_full(input_ids, attention_mask)
            loss = loss_fn(logits, data)
        return {"loss": loss, "logits": logits}

    # 有梯度的路径
    logits = self.split_core.forward_full(input_ids, attention_mask)
    loss = loss_fn(logits, data)
    loss.backward()  # 梯度通过 NCCLMiddleExecutor 自动传递到 CenterWorker

    return {"loss": loss, "logits": logits}
```

**`is_mp_src_rank_with_outputs`**:

split 架构下, edge worker 是唯一的输出 rank (持有 lm_head), 返回 True。

### 6.3 与 TrainingWorker 的集成

verl 的 `TrainingWorker` 使用 `EngineRegistry.new()` 创建引擎。只需注册 SplitEngine：

```python
# split_demo_v3/core/split_engine.py
@EngineRegistry.register(model_type="llm", backend="split", device="cuda")
class SplitEngine(BaseEngine):
    ...
```

然后在 config 中指定 `engine_config.strategy: "split"` 即可。

---

## 7. 阶段 3：vLLM 集成

**目标**：用 vLLM 替代 NaiveSplitRollout，获得高性能推理。

### 7.1 核心问题

vLLM 需要完整模型实例。split 架构中没有一个进程持有完整模型。

### 7.2 解决方案

```
训练路径: SplitEngine (front+tail 在 edge, middle 在 center)
推理路径: vLLM (完整模型, 独立实例, 放在 edge worker 的剩余显存中)
权重同步: LoRA adapter → vLLM
```

### 7.3 架构

```
EdgeWorker (GPU 0)
    │
    ├── SplitEngine (训练用, ~2GB)
    │       ├── front_layers + LoRA
    │       └── tail_layers + LoRA
    │
    └── vLLM Rollout (推理用, ~6GB for 3B)
            └── 完整 Qwen2.5-3B (或只放 front+tail+moved_middle)

CenterWorker (GPU 1)
    └── middle_layers (冻结, ~4GB, 训练和推理共用)
```

### 7.4 权重同步

训练每 N 步后, 把 edge worker 的 LoRA adapter 同步到 vLLM:

```python
# 方案 A: LoRA adapter 同步 (推荐)
# edge worker 保存 LoRA state_dict → vLLM 加载 LoRA
vllm_worker.load_lora(lora_state_dict)

# 方案 B: 完整权重合并 (更重)
# edge worker 合并 LoRA → 完整模型权重 → vLLM 更新
model.merge_and_unload()  # PEFT 合并
vllm_worker.update_weights(full_state_dict)
```

方案 A 更轻量, 方案 B 在 vLLM 不支持 LoRA hot-swap 时使用。

### 7.5 注意事项

- vLLM 和 SplitEngine 共享 GPU 0 显存, 需要仔细分配
- 3B 模型: SplitEngine ~2GB + vLLM ~6GB ≈ 8GB < 24GB (4090), 可行
- 7B 模型: SplitEngine ~3GB + vLLM ~14GB ≈ 17GB < 24GB, 勉强可行
- 更大模型需要考虑 time-sharing (训练时卸载 vLLM, 推理时卸载 SplitEngine)

---

## 8. 阶段 4：生产化

### 8.1 Checkpoint 适配

实现 `SplitCheckpointEngine`:

```python
class SplitCheckpointEngine:
    def save(self, local_path, global_step, edge_worker, center_worker):
        # edge: 保存 LoRA state_dict + optimizer
        # center: 不需要保存 (middle 冻结)
        # 兼容 verl 的 checkpoint 目录结构

    def load(self, local_path, edge_worker, center_worker):
        # 加载 LoRA state_dict + optimizer
```

### 8.2 容错

- 利用 Ray 的 Actor 自动重启能力
- CenterWorker 崩溃 → Ray 重启 → 重新加载 middle_layers → 重新建立 NCCL
- EdgeWorker 崩溃 → Ray 重启 → 重新加载 front/tail → 从 checkpoint 恢复

### 8.3 多机扩展

```bash
# 机器 0: EdgeWorker + vLLM
ray start --head --port=6379

# 机器 1: CenterWorker
ray start --address=机器0_IP:6379

# NCCL init_method 从 env:// 改为 tcp://
```

### 8.4 监控集成

复用 verl 的 Metric 系统:

- GPU 显存: verl 的 `log_gpu_memory_usage`
- 训练指标: verl 的 `compute_data_metrics`
- NCCL 传输量: 自定义 metric

---

## 9. 关键设计决策

### 9.1 EdgeWorker 是否继承 Worker

**决定**: 是。

原因: verl 的 `Worker` 基类提供 rank/world_size 管理、`@register` 装饰器、dispatch/collect 机制。继承后可以直接被 `RayWorkerGroup` 管理。

### 9.2 SplitEngine 是否支持 FSDP

**决定**: 阶段 2 不支持, 阶段 4 再考虑。

原因: FSDP 和层间拆分范式冲突。对于 <7B 模型, LoRA 足够。更大模型需要考虑:
- 对 front/tail 的 LoRA 参数局部 FSDP (自定义 process_group)
- 或者引入 Megatron 的 pipeline parallel 替代手工 split

### 9.3 NCCL 连接方式

**决定**: 使用 verl 的 `initialize_global_process_group_ray`。

原因: verl 已经处理了 Ray 环境下的 NCCL 初始化, 包括获取 node IP、分配端口等。

### 9.4 数据协议

**决定**: 使用 DataProto, 但在 split 的 forward_backward_batch 中做内部转换。

原因: 与 verl 生态兼容。DataProto → 内部 tensor → NCCL 通信 → 内部 tensor → DataProto。

---

## 10. 风险与缓解

| 风险 | 严重性 | 缓解 |
|---|---|---|
| FSDP 和 split 冲突 | 高 | 阶段 2 不用 FSDP, 阶段 4 引入局部 FSDP |
| vLLM 和 split 共享显存不足 | 中 | time-sharing 或增大 GPU |
| NCCL 连接在 Ray 环境下失败 | 中 | 复用 verl 的 NCCL init, 有成熟的 fallback |
| BaseEngine 接口不完全适配 | 中 | 只实现必要方法, 其他抛 NotImplementedError |
| 梯度传递在 Ray 环境下异常 | 低 | v2 已验证 NCCL 梯度传递正确, Ray 不改变这一点 |

---

## 11. 附录：verl 接口参考

### Worker 基类

```python
# verl/single_controller/base/worker.py
class Worker(WorkerHelper):
    @property
    def rank(self) -> int
    @property
    def world_size(self) -> int

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def method(self, ...): ...

    _register_dispatch_collect_info(mesh_name, dp_rank, is_collect)
```

### BaseEngine 接口

```python
# verl/workers/engine/base.py
class BaseEngine:
    def initialize(self)
    def forward_backward_batch(self, data, loss_fn, forward_only=False)
    def train_batch(self, data, loss_fn)
    def infer_batch(self, data, loss_fn=None)
    def save_checkpoint(self, local_path, hdfs_path, global_step, max_ckpt_to_keep)
    def load_checkpoint(self, local_path, hdfs_path, del_local_after_load)
    def to(self, device, model=True, optimizer=True, grad=True)
    def get_data_parallel_size(self)
    def get_data_parallel_rank(self)
    def get_data_parallel_group(self)
    def is_mp_src_rank_with_outputs(self)
    def optimizer_zero_grad(self)
    def optimizer_step(self)
    def lr_scheduler_step(self)
```

### EngineRegistry

```python
# verl/workers/engine/base.py
class EngineRegistry:
    _engines = {}

    @classmethod
    def register(cls, model_type, backend, device="cuda"):
        """装饰器, 注册引擎。"""

    @classmethod
    def new(cls, model_type, backend, *args, **kwargs):
        """创建引擎实例。"""
```

### DataProto

```python
# verl/protocol.py
class DataProto:
    batch: TensorDict          # 张量数据
    non_tensor_batch: dict     # 非张量数据
    meta_info: dict            # 元信息

    @staticmethod
    def from_single_dict(data)
    def select_idxs(self, indices)
    def chunk(self, chunks)
```
