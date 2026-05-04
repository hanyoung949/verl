# Split Actor GRPO Demo v2 — 完整用户指南

> **适用范围**：`verl/experimental/split_demo_nccl/`
> **当前版本**：v2 NCCL 双进程版
> **验证平台**：单机 2×4090，`torchrun --nproc_per_node=2`
> **默认模型**：Qwen2.5-3B-Instruct，切法 4:28:4

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [文件结构与模块职责](#3-文件结构与模块职责)
4. [核心模块详解](#4-核心模块详解)
5. [通信协议详解](#5-通信协议详解)
6. [梯度正确性](#6-梯度正确性)
7. [训练流程与数据流](#7-训练流程与数据流)
8. [配置参考](#8-配置参考)
9. [运行指南](#9-运行指南)
10. [与 verl 主线的对比](#10-与-verl-主线的对比)
11. [工业级扩展路径](#11-工业级扩展路径)
12. [开发者指南](#12-开发者指南)

---

## 1. 项目概述

### 这是什么

split_demo_nccl 是一个**"边-云分离"的 LLM 强化学习训练框架原型**。核心思想是将一个大模型按层拆分到不同进程（甚至不同机器），让"边侧"只持有轻量的前段和尾段（训练部分），"中心"持有冻结的中段（推理部分），通过 NCCL P2P 通信实现跨进程前向和梯度回传。

### 要解决的问题

在真实部署场景中，边侧设备（如边缘服务器、终端设备）算力有限，无法独自容纳完整模型。但如果只让边侧持有 embedding + 前几层 + 最后几层 + LoRA，中段放到算力更强的中心节点，就能以极低的边侧显存完成训练。

### 两个版本

| 版本 | 目录 | 跨卡方式 | 进程模型 |
|---|---|---|---|
| v1 | `split_demo/` | `tensor.to('cuda:1')` | 单进程 |
| **v2** | **`split_demo_nccl/`** | **NCCL dist.send/recv** | **双进程** |

v2 是 v1 的演进：用真实进程边界替换 v1 的"借来的可微性"。v1 的 `tensor.to()` 之所以可微，是因为两张卡共享同一个进程的 autograd 图——换成真实跨进程/跨机器传输，这条隐式可微性就断了。v2 通过 `torch.autograd.Function` 显式实现了跨进程梯度传递。

### 当前验证结果

三阶段验证全部通过：

| 阶段 | 内容 | 结果 |
|---|---|---|
| Phase 1 | 双进程 1 step 冒烟测试 | ✅ 无死锁，输出 shape 正确 |
| Phase 2 | backward 后梯度正确性 | ✅ 32 个 LoRA 参数全部有 `.grad` |
| Phase 3 | 10 步完整 GRPO 训练 | ✅ reward 0.33–0.75，clipfrac 正常 |

---

## 2. 整体架构

### 2.1 双进程架构

```
torchrun --nproc_per_node=2 启动两个进程
    │
    ├── rank0 (GPU 0, cuda:0)          ─── rank1 (GPU 1, cuda:1)
    │   训练主进程                         纯 worker 进程
    │
    │   持有组件：                         持有组件：
    │   • embed_tokens                    • middle_layers[4:32]
    │   • front_layers[0:4]               • rotary_emb (本地重算 pos_emb)
    │   • tail_layers[32:36]
    │   • norm / lm_head
    │   • LoRA (32 个可训练参数)
    │   • optimizer (AdamW)
    │   • NaiveSplitRollout
    │   • FunctionReward
    │   • SplitGRPOTrainer
    │
    │   启动逻辑：                         启动逻辑：
    │   main() →                          main() →
    │   创建 SplitActorCore →              创建 MiddleWorker →
    │   创建 Trainer →                     worker.run()
    │   trainer.fit()                      (无限循环等请求)
    │
    │          ◀──── NCCL P2P ────▶
```

### 2.2 模型切分

以 Qwen2.5-3B 为例（36 层 decoder，hidden_size=2048，vocab=151936）：

```
┌────────────────── GPU 0 (rank0) ──────────────────┐  ┌────────── GPU 1 (rank1) ──────────┐
│                                                    │  │                                    │
│  input_ids [B, S]                                  │  │                                    │
│    ↓                                               │  │                                    │
│  embed_tokens → h [B, S, 2048]                     │  │                                    │
│    ↓                                               │  │                                    │
│  ┌────────────────────────────────┐                │  │                                    │
│  │ front_layers[0]                │                │  │                                    │
│  │ front_layers[1]                │  LoRA 可训练    │  │                                    │
│  │ front_layers[2]                │  ~2GB 峰值显存  │  │                                    │
│  │ front_layers[3]                │                │  │                                    │
│  └───────────────┬────────────────┘                │  │                                    │
│                  │ h_front [B, S, 2048]             │  │                                    │
│                  │                                  │  │                                    │
│    dist.send(h_front) ──────────────────────────────▶  ┌──────────────────────────────┐   │
│                  │                                  │  │ │ middle_layers[4]             │   │
│                  │                                  │  │ │ middle_layers[5]             │   │
│                  │                                  │  │ │    ...                       │   │
│                  │                                  │  │ │ middle_layers[31]            │   │
│                  │                                  │  │ │ (28 层, 全部冻结)             │   │
│                  │                                  │  │ │ ~4GB 峰值显存                │   │
│                  │                                  │  │ └──────────────┬───────────────┘   │
│                  │                                  │  │                │                   │
│    dist.recv(h_middle) ◀────────────────────────────────┘               │                   │
│                  │ h_middle [B, S, 2048]           │  │                                    │
│  ┌───────────────▼────────────────┐                │  │                                    │
│  │ tail_layers[32]                │                │  │                                    │
│  │ tail_layers[33]                │  LoRA 可训练    │  │                                    │
│  │ tail_layers[34]                │                │  │                                    │
│  │ tail_layers[35]                │                │  │                                    │
│  └───────────────┬────────────────┘                │  │                                    │
│    ↓                                               │  │                                    │
│  norm → lm_head → logits [B, S, 151936]            │  │                                    │
│                                                    │  │                                    │
└────────────────────────────────────────────────────┘  └────────────────────────────────────┘
```

### 2.3 为什么这么切

```
假设："边侧算力受限，中心算力更强"

配置：4:28:4
- front = 4 层（边侧，轻量）
- middle = 28 层（中心，冻结，不需要梯度）
- tail = 4 层（边侧，轻量）

边侧只需训练 LoRA 参数（32 个矩阵），显存约 2GB
中心只需做冻结推理，显存约 4GB
```

---

## 3. 文件结构与模块职责

### 3.1 目录树

```
split_demo_nccl/
├── __init__.py                  ← 包入口
├── config/
│   └── split_demo_nccl.yaml     ← Hydra 配置文件
├── core/
│   ├── __init__.py              ← 导出 NCCLMiddleExecutor, SplitActorCore
│   ├── middle_executor.py       ← ⭐ NCCL 通信核心（最关键技术点）
│   ├── middle_worker.py         ← ⭐ rank1 worker loop
│   └── split_actor.py           ← rank0 模型执行核心
├── rollout/
│   ├── __init__.py
│   ├── base.py                  ← RolloutOutput dataclass + SplitRolloutBackend ABC
│   └── naive.py                 ← 无 KV cache 的朴素自回归生成
├── reward/
│   ├── __init__.py
│   ├── base.py                  ← RewardAdapter ABC
│   └── function_reward.py       ← 规则函数打分（数学精确匹配）
├── split_trainer.py             ← GRPO 训练循环编排
├── main_grpo_split.py           ← 双进程入口（torchrun 启动）
└── checkpoints/                 ← 训练产物
    ├── global_step_10/
    └── latest/
```

### 3.2 哪些是新写的，哪些复用 v1

| 文件 | 来源 | 说明 |
|---|---|---|
| `core/middle_executor.py` | **新写** | NCCL 通信 + autograd.Function |
| `core/middle_worker.py` | **新写** | rank1 worker loop |
| `core/split_actor.py` | **从 v1 改写** | 去掉 middle_layers，改用 NCCLMiddleExecutor |
| `split_trainer.py` | **从 v1 改写** | 显存统计适配单卡 |
| `main_grpo_split.py` | **新写** | 双进程入口 |
| `config/` | **从 v1 改写** | save_dir 改为 nccl 版路径 |
| `rollout/` | **直接复用 v1** | 6 个文件一字未改 |
| `reward/` | **直接复用 v1** | 6 个文件一字未改 |

### 3.3 模块依赖关系

```
main_grpo_split.py
    ├── imports SplitActorCore      (core/split_actor.py)
    ├── imports MiddleWorker        (core/middle_worker.py)
    ├── imports NCCLMiddleExecutor  (core/middle_executor.py)
    ├── imports SplitGRPOTrainer    (split_trainer.py)
    ├── imports NaiveSplitRollout   (rollout/naive.py)
    ├── imports FunctionReward      (reward/function_reward.py)
    └── imports verl.core_algos     (复用 verl 主线的 GRPO 算法)

split_trainer.py
    └── imports from verl:
        ├── verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage
        ├── verl.trainer.ppo.core_algos.agg_loss
        └── verl.utils.torch_functional.logprobs_from_logits, masked_mean
```

---

## 4. 核心模块详解

### 4.1 `core/split_actor.py` — SplitActorCore

**职责**：rank0 侧的模型执行核心。对外暴露 `forward_full()`、`prefill()`、`decode_step()` 接口，内部协调 front 执行、NCCL 中段调用、tail 执行。

**构造流程**（`__init__`）：

```
1. AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=bfloat16)
2. get_peft_model(base_model, lora_config)          ← 先施加 LoRA
3. _unwrap_causal_lm() + _find_decoder_stack()      ← 从 wrapper 链中找到 CausalLM 和 decoder
4. 拆分 layers:
   self.front_layers = layers[0:4]      → cuda:0
   middle_layers     = layers[4:32]     → 不注册，交给 rank1
   self.tail_layers  = layers[32:36]    → cuda:0
5. self.embed_tokens, self.norm, self.lm_head → cuda:0
6. self.decoder_rotary_emb → cuda:0
7. self.middle_executor = NCCLMiddleExecutor(middle_rank=1, device="cuda:0")
```

**关键方法**：

| 方法 | 作用 |
|---|---|
| `forward_full(input_ids, attention_mask)` | 完整前向：front → NCCL → tail → logits |
| `forward_front(...)` | 执行 embed + front_layers |
| `forward_tail(...)` | 执行 tail_layers + norm + lm_head |
| `prefill(prompt_ids, mask)` | rollout 用，当前等于 forward_full |
| `decode_step(ids, mask)` | rollout 用，等于 forward_full()[, -1, :] |
| `trainable_parameters()` | 返回所有 `requires_grad=True` 的参数 |
| `save_adapter(save_dir, step)` | 保存 LoRA adapter + meta |

**与 v1 的差异**：

- 不持有 `middle_layers`（它们在 rank1）
- 没有 `middle_device` 属性
- `forward_tail` 中不需要 `h_middle.to(primary_device)`（NCCL 返回的 tensor 已在 rank0 设备上）
- 默认 `device="cuda"` 会自动解析为 `"cuda:0"`（rank0 的 LOCAL_RANK）

### 4.2 `core/middle_executor.py` — NCCLMiddleExecutor + _NCCLMiddleFunction

**这是整个项目最关键的技术文件。** 包含两个类：

#### NCCLMiddleExecutor

rank0 侧的中段执行器，替代 v1 的 `LocalMiddleExecutor`。

**核心方法**：

```python
def execute(hidden_states, position_ids, position_embeddings, attention_mask):
    """根据当前梯度上下文选择 FWD_ONLY 或 FWD_WITH_BWD"""
    if torch.is_grad_enabled():
        # 训练路径：需要 backward
        return _NCCLMiddleFunction.apply(hidden_states, position_ids, attention_mask, self)
    else:
        # 推理路径：只做前向
        return self._fwd_only(hidden_states, position_ids, attention_mask)
```

**底层 NCCL 操作**：

- `_send_tensor(tensor)`: `dist.send(tensor.contiguous(), dst=1)`
- `_recv_tensor(shape, dtype)`: 分配 buffer → `dist.recv(buf, src=1)`
- `_send_fwd_request(...)`: 发送 header + h_front + position_ids + optional mask
- `_exchange_backward(grad_h_middle)`: 发送 grad → 接收 grad_h_front
- `shutdown()`: 发送 SHUTDOWN header

**传输统计**（与 v1 接口对齐）：

```python
stats = {
    "calls": 0,                          # 执行次数
    "forward_to_middle_bytes": 0,        # 发往中段的 hidden 字节数
    "position_ids_bytes": 0,             # position_ids 字节数
    "position_embeddings_bytes": 0,      # v2 中恒为 0（不发送）
    "attention_mask_bytes": 0,           # mask 字节数
    "last_hidden_shape": None,           # 最近一次 hidden shape
}
```

#### _NCCLMiddleFunction

自定义 `torch.autograd.Function`，实现跨进程可微性。

```python
class _NCCLMiddleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_front, position_ids, attention_mask, executor):
        # 发送 FWD_WITH_BWD 请求给 rank1
        executor._send_fwd_request(h_front, position_ids, attention_mask, flag=FWD_WITH_BWD)
        # 接收 h_middle
        h_middle = executor._recv_tensor(h_front.shape, h_front.dtype)
        return h_middle

    @staticmethod
    def backward(ctx, grad_h_middle):
        # 发送 grad_h_middle 给 rank1
        # 接收 grad_h_front（rank1 本地 backward 的结果）
        grad_h_front = ctx.executor._exchange_backward(grad_h_middle)
        return grad_h_front, None, None, None
```

**为什么这是最关键技术点**：

- `forward()` 返回的 `h_middle` 会获得 autograd 的 `grad_fn`（指向 `_NCCLMiddleFunctionBackward`）
- 当下游 `loss.backward()` 触发时，autograd 会调用 `backward(ctx, grad_h_middle)`
- 这样梯度就自然地跨越了进程边界

### 4.3 `core/middle_worker.py` — MiddleWorker

**职责**：rank1 侧的纯请求响应 worker。不持有任何训练逻辑。

**构造流程**（`from_pretrained`）：

```
1. 加载完整 HF 模型
2. 施加 PEFT LoRA
3. 提取 middle_layers[front_end:middle_end] + rotary_emb
4. 冻结所有参数
5. 移动到 rank1 设备
```

**主循环**（`run()`）：

```python
while True:
    header = recv_header_from_rank0()       # [flag, B, S, H, has_mask, dtype_code]
    if flag == SHUTDOWN: break

    h_front = recv_tensor((B, S, H))
    position_ids = recv_tensor((B, S))
    mask = recv_tensor(...) if has_mask else None
    pos_emb = rotary_emb(h_front, position_ids)     # 本地重算，不接收

    if flag == FWD_ONLY:
        h_out = middle_layers(h_front, ...)          # no_grad
        send_tensor(h_out)

    elif flag == FWD_WITH_BWD:
        h_in = h_front.detach().requires_grad_(True)
        h_out = middle_layers(h_in, ...)
        send_tensor(h_out.detach())

        grad_out = recv_tensor(h_out.shape)           # 等待 rank0 的梯度
        torch.autograd.backward(h_out, grad_out)      # 本地 backward
        send_tensor(h_in.grad)                        # 返回 grad_h_front
```

**关键设计**：

- FWD_WITH_BWD 路径中，`h_in = h_front.detach().requires_grad_(True)` 创建了一个新的叶子节点
- `torch.autograd.backward(h_out, grad_out)` 会通过 middle layers 的计算图回传梯度
- `h_in.grad` 就是 `dL/dh_front`（正确的梯度）
- 这比直接把 `grad_out` 返回给 rank0 正确得多（见第 6 节）

### 4.4 `split_trainer.py` — SplitGRPOTrainer

**职责**：GRPO 训练循环的编排层。调用 SplitActorCore、NaiveSplitRollout、FunctionReward，自身不做跨卡操作。

**训练循环**（`fit()`，每个 step）：

```
Step 1  Generate          rollout.generate(prompt_ids, G, max_tokens)
                          → RolloutOutput {sequences, response_ids, ...}

Step 2  old_log_prob      model.forward_full(sequences) [torch.no_grad]
                          → logprobs_from_logits(logits, response_ids)

Step 3  Reward            reward.score(sequences, metadata)
                          → token_level_scores

Step 4  Dynamic Sampling  过滤 std(reward)==0 的 group

Step 5  Advantage         compute_grpo_outcome_advantage(...)
                          → advantages

Step 6  Policy Update     for mini_batch:
                            model.forward_full(mb_sequences)   ← 有梯度
                            → new_log_prob
                            → pg_loss = clip(ratio) × advantage
                            → loss.backward()
                            → optimizer.step()

Step 7  Logging           打印 reward / loss / clipfrac / approx_kl
```

**与 v1 的差异**：

- `_format_memory_stats()`: 只看 `model.primary_device`（rank0 只有一张卡）
- `_reset_peak_memory_stats()`: 同上

### 4.5 `rollout/` — NaiveSplitRollout

直接复用 v1。实现朴素自回归生成：

```
1. prompt → 复制 G 份（GRPO group sampling）
2. prefill(prompt_ids) → logits → 采样第一个 token
3. 逐 token 循环：
   decode_step(current_ids) → logits → 采样 → 追加
   直到全部序列遇到 EOS 或达到 max_new_tokens
4. 构造 RolloutOutput {sequences, response_ids, response_mask, prompt_len}
```

不做 KV cache，每步重算完整序列——速度慢但实现简单，适合架构验证。

### 4.6 `reward/` — FunctionReward

直接复用 v1。基于 Python 函数的打分：

```python
def math_exact_match(text, meta):
    return 1.0 if extract_last_number(text) == meta['answer'] else 0.0
```

只 decode response 部分，避免 prompt 内容干扰。

---

## 5. 通信协议详解

### 5.1 header 格式

每次请求开始，rank0 先发送一个 6 字段 `int64` header tensor：

```python
header = [flag, B, S, H, has_mask, dtype_code]
#         int   int int int int     int
```

| 字段 | 含义 | 取值 |
|---|---|---|
| `flag` | 请求类型 | 0=FWD_ONLY, 1=FWD_WITH_BWD, 2=SHUTDOWN |
| `B` | batch size | ≥1 |
| `S` | sequence length | ≥1 |
| `H` | hidden dimension | 2048 (Qwen2.5-3B) |
| `has_mask` | 是否携带 attention_mask | 0 或 1 |
| `dtype_code` | 张量 dtype 编码 | 0=bfloat16, 1=float16, 2=float32 |

### 5.2 三种 flag

| flag | 常量 | 触发场景 | rank1 行为 |
|---|---|---|---|
| 0 | `FWD_ONLY` | rollout / old_log_prob（`no_grad()` 下） | 不缓存，算完发回 |
| 1 | `FWD_WITH_BWD` | 训练 update loop（有梯度） | 缓存上下文，等 grad，本地 backward |
| 2 | `SHUTDOWN` | 训练结束 | 退出 worker loop |

### 5.3 FWD_ONLY 握手时序

```
rank0                                              rank1
─────                                              ─────
1. dist.send([0, B, S, H, has_mask, dtype]) ────▶  dist.recv(header)
2. dist.send(h_front [B,S,H], contiguous) ────────▶  dist.recv(h_front)
3. dist.send(position_ids [B,S]) ─────────────────▶  dist.recv(position_ids)
4. (可选) dist.send(mask [B,1,S,S]) ──────────────▶  dist.recv(mask)

                                                    5. pos_emb = rotary_emb(h_front, pos_ids)
                                                       h_out = middle_layers(h_front, ...)
                                                       (torch.no_grad)

6. dist.recv(h_out [B,S,H]) ◀────────────────────  dist.send(h_out, contiguous)
```

**不发送 position_embeddings**：rank1 本地有 rotary_emb 模块和 position_ids，可以自行重算 pos_emb。这比传输 (cos, sin) 两个 tensor 更省带宽。

### 5.4 FWD_WITH_BWD 握手时序

```
rank0                                              rank1
─────                                              ─────
1. dist.send([1, B, S, H, has_mask, dtype]) ────▶  dist.recv(header)
2. dist.send(h_front [B,S,H]) ───────────────────▶  dist.recv(h_front)
3. dist.send(position_ids [B,S]) ─────────────────▶  dist.recv(position_ids)
4. (可选) dist.send(mask [B,1,S,S]) ──────────────▶  dist.recv(mask)

                                                    5. pos_emb = rotary_emb(h_front, pos_ids)
                                                       h_in = h_front.detach().requires_grad_(True)
                                                       h_out = middle_layers(h_in, ...)

6. dist.recv(h_out [B,S,H]) ◀────────────────────  dist.send(h_out.detach())

7. (rank0 继续 tail forward → loss → loss.backward()
    产生 grad_h_middle)

8. dist.send(grad_h_middle [B,S,H]) ─────────────▶  dist.recv(grad_out)

                                                    9. torch.autograd.backward(h_out, grad_out)
                                                       grad_h_front = h_in.grad

10. dist.recv(grad_h_front [B,S,H]) ◀─────────────  dist.send(h_in.grad)

    grad 继续回传到 front_layers 的 LoRA 参数
```

**注意**：第 6 步 rank1 发送的是 `h_out.detach()`（断开计算图），因为 rank1 需要等 rank0 的 grad 回来再做 backward。如果发 `h_out` 本身，计算图会在跨进程时丢失。

### 5.5 所有 tensor 必须 contiguous

NCCL 要求发送的 tensor 是 contiguous 的。`NCCLMiddleExecutor._send_tensor()` 内部自动调用 `.contiguous()`：

```python
def _send_tensor(self, tensor):
    dist.send(tensor.contiguous(), dst=self.middle_rank)
```

如果忘记 contiguous，不一定会立即报错，而是可能导致：
- NCCL 参数不匹配
- shape/dtype 对不上
- `recv` 永久阻塞（看起来像死锁）

### 5.6 死锁条件

rank1 的 FWD_WITH_BWD 分支会等待 rank0 发来的 `grad_h_middle`。如果 rank0 在 `torch.no_grad()` 下调用了 `execute()`，会走 FWD_ONLY 分支；但如果不小心在有梯度的情况下调用，rank1 会等 grad 永远等不到。

**安全规则**：所有不做 backward 的路径（rollout、old_log_prob）必须包在 `torch.no_grad()` 里。`split_trainer.py` 中已正确处理：

```python
# Step 1: rollout（不回传梯度）
with torch.no_grad():
    rollout_output = self.rollout.generate(...)

# Step 2: old_log_prob（不回传梯度）
with torch.no_grad():
    logits_old = self.model.forward_full(...)

# Step 6: 训练 forward（需要 backward）
logits_new = self.model.forward_full(sequences[mb], mask[mb])  # 有梯度
loss.backward()  # 梯度通过 NCCL 回传
```

---

## 6. 梯度正确性

### 6.1 v1 为什么可微

v1 在单进程内，`tensor.to('cuda:1')` 是 PyTorch 内建的可微操作：

```python
h = hidden_states.to('cuda:1')     # PyTorch 内建可微
for layer in middle_layers:
    h = layer(h)[0]
return h.to('cuda:0')              # 也在 autograd 图内
```

整个 forward 在一个进程里，autograd 图连续，backward 自动传播。

### 6.2 v2 为什么可微

v2 中，`_NCCLMiddleFunction` 是一个 `torch.autograd.Function`，它告诉 autograd 如何计算梯度：

```python
class _NCCLMiddleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_front, ...):
        # 通过 NCCL 发送/接收
        h_middle = recv_from_rank1()
        return h_middle               # 返回的 tensor 有 grad_fn

    @staticmethod
    def backward(ctx, grad_h_middle):
        # 通过 NCCL 发送/接收
        grad_h_front = exchange_with_rank1(grad_h_middle)
        return grad_h_front, None, None, None
```

当 `loss.backward()` 触发时，autograd 会调用 `backward()` 方法，梯度自然跨越进程边界。

### 6.3 数学正确性

正确的梯度公式：

```
dL/dh_front = dL/dh_middle × dh_middle/dh_front
            = grad_h_middle × Jacobian(middle_layers)
```

rank1 的实现：

```python
h_in = h_front.detach().requires_grad_(True)   # 创建叶子节点
h_out = middle_layers(h_in, ...)                # 前向
# ... rank0 做 loss.backward() 产生 grad_h_middle ...
torch.autograd.backward(h_out, grad_out)        # 自动计算 Jacobian
grad_h_front = h_in.grad                        # 正确的 dL/dh_front
```

### 6.4 错误做法 vs 正确做法

```python
# ❌ 错误：把 grad_h_middle 直接当 grad_h_front 返回
# 等价于把 middle 当恒等映射 I，Jacobian(I) = I
# 梯度不经过 middle 的参数和结构，完全错误
grad_h_front = grad_h_middle

# ✅ 正确：在 rank1 做真实的本地 backward
torch.autograd.backward(h_out, grad_out)   # 经过 middle_layers 的计算图
grad_h_front = h_in.grad                    # 这才是 dL/dh_front
```

Phase 2 验证证明了正确性：所有 32 个 LoRA 可训练参数都收到了非 None 的 `.grad`。

---

## 7. 训练流程与数据流

### 7.1 单步训练的完整数据流

以 B=4, G=4, max_new_tokens=24 为例：

```
                 输入                      位置                    NCCL 通信
                 ────                      ────                    ──────────
Step 1 Rollout:  prompt [4, 256]  ──→  prefill × 1    (FWD_ONLY × 1)
                                       decode × 23    (FWD_ONLY × 23)
                                       共 24 次 NCCL 往返

Step 2 old_lp:   sequences [16, 280] ──→ forward_full (FWD_ONLY × 1)
                                         共 1 次 NCCL 往返

Step 6 Update:   sequences [mb, 280] ──→ forward_full (FWD_WITH_BWD × ~4)
                                         loss.backward() 触发 grad 回传
                                         共 ~4 次 NCCL 往返

总计约 29 次 NCCL 往返/step
```

### 7.2 单次 NCCL 往返的数据量

以 forward_full 一次调用为例（B=8, S=41, H=2048, bf16）：

```
发送: h_front      [8, 41, 2048]  bf16  ≈ 1.34 MB
发送: position_ids [8, 41]        int64 ≈ 2.6 KB
接收: h_middle     [8, 41, 2048]  bf16  ≈ 1.34 MB
─────────────────────────────────────────────────
单次往返 ≈ 2.7 MB
step 内 ≈ 29 次 × 2.7 MB ≈ 78 MB
```

日志中的 `to_middle=0.048GB` 是累计值（所有 forward 调用的 h_front 字节数之和）。

### 7.3 张量 shape 沿途变化

```
input_ids [B, S]                    原始 token ids
    ↓
embed_tokens → h [B, S, 2048]       embedding 后
    ↓
front_layers → h_front [B, S, 2048] 前 4 层输出
    ↓ NCCL
middle_layers → h_middle [B, S, 2048] 28 层输出
    ↓ NCCL
tail_layers → h [B, S, 2048]        后 4 层输出
    ↓
norm → h [B, S, 2048]               layer norm
    ↓
lm_head → logits [B, S, 151936]     最终输出（vocab size=151936）
```

---

## 8. 配置参考

### 8.1 完整配置（`config/split_demo_nccl.yaml`）

```yaml
model:
  path: "~/share/Qwen2.5-3B-Instruct"    # 模型路径，优先本地

data:
  train_file: "verl/experimental/split_demo/sample_data/train.jsonl"
  max_prompt_length: 256                   # prompt 最大 token 数

split:
  front_end: 4                             # layers[0:4] 在 GPU 0
  middle_end: 32                           # layers[4:32] 在 GPU 1

lora:
  r: 16                                    # LoRA rank
  lora_alpha: 32                           # LoRA alpha
  lora_dropout: 0.0                        # LoRA dropout
  target_modules: ["q_proj", "v_proj"]     # LoRA 目标模块

rollout:
  group_size: 4                            # 每个 prompt 生成 G 条 response
  max_new_tokens: 128                      # 最大生成 token 数
  temperature: 1.0                         # 采样温度
  top_p: 1.0                               # top-p 采样

algorithm:
  norm_adv_by_std_in_grpo: false           # Dr.GRPO: 不按 std 归一化
  cliprange_low: 0.2                       # PPO clip 下界
  cliprange_high: 0.28                     # PPO clip 上界 (DAPO clip-higher)
  loss_agg_mode: "seq-mean-token-sum-norm" # loss 聚合模式
  loss_scale_factor: 128                   # Dr.GRPO: 固定常数 == max_new_tokens
  dynamic_sampling: true                   # DAPO: 过滤同质 group
  update_epochs: 1                         # 每 step 的 epoch 数
  mini_batch_size: 8                       # mini-batch 大小

trainer:
  train_batch_size: 4                      # 每 step 的 prompt batch 大小
  total_steps: 10                          # 总训练步数
  lr: 1.0e-4                               # 学习率
  max_grad_norm: 1.0                       # 梯度裁剪
  log_freq: 1                              # 每隔多少步打印一次
  log_gpu_memory: true                     # 打印显存统计
  log_transport: true                      # 打印 NCCL 传输统计
  save_dir: "verl/experimental/split_demo_nccl/checkpoints"
  save_latest: true                        # 保存 latest 快捷方式
  seed: 42                                 # 随机种子
```

### 8.2 最常改的配置项

| 配置项 | 改法 | 场景 |
|---|---|---|
| `model.path` | 换模型路径 | 换模型 |
| `split.front_end / middle_end` | 调整切层比例 | 调整边-云资源分配 |
| `rollout.group_size / max_new_tokens` | 增大/减小 | 控制生成质量和速度 |
| `algorithm.loss_scale_factor` | 必须 == max_new_tokens | Dr.GRPO 要求 |
| `trainer.total_steps / lr` | 增大步数/调整学习率 | 正式训练 |

### 8.3 命令行覆盖示例

```bash
torchrun --nproc_per_node=2 \
  -m verl.experimental.split_demo_nccl.main_grpo_split \
  model.path=/root/share/Qwen2.5-7B-Instruct \
  split.front_end=4 \
  split.middle_end=24 \
  trainer.total_steps=200 \
  trainer.lr=5e-4
```

---

## 9. 运行指南

### 9.1 环境要求

```bash
source /root/workspace/miniconda3/etc/profile.d/conda.sh
conda activate verl
```

验证：
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 期望输出: True 2
```

### 9.2 启动命令

```bash
torchrun --nproc_per_node=2 \
  -m verl.experimental.split_demo_nccl.main_grpo_split
```

### 9.3 Phase 1: 冒烟测试

```bash
torchrun --nproc_per_node=2 \
  -m verl.experimental.split_demo_nccl.main_grpo_split \
  trainer.total_steps=1 \
  algorithm.dynamic_sampling=false
```

检查：无 crash、输出 shape 正确、两个进程正常退出。

### 9.4 Phase 3: 完整训练

```bash
torchrun --nproc_per_node=2 \
  -m verl.experimental.split_demo_nccl.main_grpo_split \
  trainer.total_steps=10 \
  trainer.log_freq=1 \
  trainer.lr=5e-4 \
  rollout.group_size=4 \
  rollout.max_new_tokens=24 \
  algorithm.loss_scale_factor=24 \
  algorithm.update_epochs=2 \
  algorithm.mini_batch_size=8
```

### 9.5 常见问题

| 现象 | 可能原因 | 解决方法 |
|---|---|---|
| 卡住不动（死锁） | `torch.no_grad()` 遗漏 | 检查 rollout 和 old_log_prob 是否包在 `no_grad` 里 |
| `gpu_mem=unknown` | `primary_device` 不是 `"cuda:N"` 格式 | 已修复（自动解析） |
| `_runtime_stats` KeyError | 未调用 `reset_runtime_stats()` | trainer.fit() 会自动调用 |
| rank1 报错后 rank0 不退出 | NCCL timeout 默认 60s | 等 60s 后 rank0 自动超时退出 |

---

## 10. 与 verl 主线的对比

### 10.1 verl 主线架构概览

verl 是字节跳动开源的 LLM 强化学习训练框架，基于 Ray 做分布式调度：

```
main_ppo.py
    │
    └── RayPPOTrainer (Ray actor)
            │
            ├── RayWorkerGroup
            │       ├── TrainingWorker (FSDP/Megatron)    ← actor 训练
            │       ├── RolloutWorker (vLLM/SGLang)       ← rollout 生成
            │       ├── RewardWorker                       ← reward model
            │       └── RefPolicyWorker                    ← reference model
            │
            └── ResourcePoolManager                        ← GPU 资源管理
```

核心特性：
- **Ray 分布式**：WorkerGroup 管理多个 Ray Worker，自动调度
- **FSDP / Megatron**：模型并行训练，支持 ZeRO-1/2/3
- **vLLM / SGLang**：高性能推理引擎，支持 continuous batching
- **3D-HybridEngine**：训练和推理之间零冗余权重切换
- **DataProto**：统一的数据传递协议
- **单控制器**：`single_controller/` 提供 Ray WorkerGroup 的抽象

### 10.2 逐项对比

| 维度 | verl 主线 | split_demo_nccl |
|---|---|---|
| **进程管理** | Ray（Actor、PlacementGroup） | torchrun（`dist.init_process_group`） |
| **进程数量** | 弹性：2-数百个 worker | 固定 2 个（rank0 + rank1） |
| **通信方式** | NCCL all-reduce / Ray object store | NCCL P2P（send/recv） |
| **模型并行** | FSDP（数据并行）/ Megatron（张量并行） | **层间拆分**（pipeline 式，但无 pipeline parallel） |
| **训练引擎** | FSDP + torch.optim / Megatron | 原生 PyTorch + 手动 `.backward()` |
| **推理引擎** | vLLM / SGLang / HF Transformers | 自写 NaiveSplitRollout（无 KV cache） |
| **模型同步** | 权重拷贝（训练→推理） | 同一模型，rank0 训练 front/tail，rank1 推理 middle |
| **RL 算法** | PPO / GRPO / DAPO / REINFORCE++ / ... | GRPO + DAPO 子集 |
| **Reward** | 函数 / 模型 / 自定义 Manager | 函数（FunctionReward） |
| **数据协议** | DataProto（TensorDict 封装） | 原始 tensor + dict |
| **数据加载** | StatefulDataLoader + Sampler | 手写随机采样 |
| **Checkpoint** | CheckpointEngine（支持 HuggingFace / Megatron） | 手写 adapter_state.pt + optimizer.pt |
| **配置管理** | Hydra + OmegaConf dataclass | Hydra + DictConfig |
| **规模** | 已验证 671B / 数百 GPU | 3B / 2 GPU |
| **代码量** | ~50000 行（主线） | ~1500 行 |
| **LoRA** | 支持（FSDP / Megatron LoRA） | 支持（PEFT LoRA） |
| **容错** | Ray 自动重启 worker | 无容错（任一进程挂 = 全挂） |
| **多机** | Ray 自动调度多机 | 需手动配置 `dist.init_process_group` 的 `init_method` |

### 10.3 核心差异分析

**1. 进程管理：Ray vs torchrun**

verl 使用 Ray 做进程管理，能自动调度、容错、弹性伸缩。split_demo_nccl 使用 `torchrun`，固定 2 个进程，无容错。

**2. 模型并行：FSDP vs 层间拆分**

verl 的 FSDP 是数据并行（每个 worker 持有完整模型，数据分片），split_demo_nccl 是层间拆分（不同 worker 持有不同层，数据不分片）。两者完全不同。

**3. 推理引擎：vLLM vs 手写**

verl 集成 vLLM/SGLang，支持 KV cache、continuous batching、PagedAttention。split_demo_nccl 的 NaiveSplitRollout 每步重算完整序列，非常慢。

**4. 数据协议：DataProto vs 原始 tensor**

verl 有 `DataProto`（基于 TensorDict），统一管理 batch 数据、metadata、padding。split_demo_nccl 用原始 dict 和 tensor。

---

## 11. 工业级扩展路径

### 11.1 当前 demo 的局限性

| 局限 | 说明 |
|---|---|
| 只有 2 个进程 | 无法利用多机多卡集群 |
| 无 KV cache | 推理每步重算，非常慢 |
| 无 vLLM | 无法利用 PagedAttention、continuous batching |
| 无容错 | rank0 或 rank1 挂了，整个训练崩 |
| 无 FSDP | 无法用 ZeRO 优化显存 |
| 无 Megatron | 无法做张量并行 |
| 手写训练循环 | 没有复用 verl 的 RayPPOTrainer |
| 固定切层 | 不支持动态调整 front/middle/tail 比例 |
| 单机 | 不支持跨机器 NCCL（需替换通信层） |

### 11.2 扩展方向及难度

#### 可直接在 demo 上做的扩展（不改 trainer）

| 扩展 | 改哪里 | 难度 | 说明 |
|---|---|---|---|
| 更大模型（7B） | 配置 + 调整切层 | 低 | 需要更多显存，可能需要多卡 |
| 更复杂 reward | 换 FunctionReward 的 fn | 低 | 换个函数即可 |
| KV cache 推理 | 新 CachedSplitRollout | 中 | SplitActorCore 加 init_kv_cache |
| Reward Model | 新 ModelReward(RewardAdapter) | 中 | 独立 RM 模型放 rank1 空余显存 |

#### 需要较大改动的扩展

| 扩展 | 改哪里 | 难度 | 说明 |
|---|---|---|---|
| vLLM rollout | 重构 rollout 层 + 权重同步 | **高** | vLLM 需要完整模型实例，不能与 SplitActorCore 共享，需加 sync 协议 |
| FSDP 训练 | 重构 SplitActorCore 设备管理 | **高** | FSDP 和手动 `.to(device)` 冲突 |
| Ray 集成 | 引入 WorkerGroup + ResourcePool | **高** | 需要重新设计进程管理 |
| 多机 NCCL | 替换 `dist.init_process_group` | 中 | 改 init_method 即可，主体代码不动 |
| RPC 中间层 | 新 RPCMiddleExecutor | 中 | 替换 middle_executor，其余不动 |

### 11.3 关键差距

从 demo 到工业级，最大的三个差距是：

**1. Ray 集成**（最难）

verl 用 Ray 管理所有 worker，支持自动调度、容错、弹性伸缩。当前 demo 用 `torchrun` 启动固定进程，没有任何调度能力。

要接入 Ray：
- 需要把 SplitActorCore、MiddleWorker 包装成 Ray Actor
- 需要 ResourcePoolManager 管理 GPU 资源
- 需要处理 Ray 的序列化（模型权重序列化开销大）

**2. vLLM 集成**（次难）

vLLM 需要完整模型实例（不能只持有部分层）。正确的扩展方式不是让 vLLM 理解 split 模型，而是：

```
训练路径：SplitActorCore（拆分的，用于训练前向）
推理路径：vLLM（完整的，用于生成）
两者通过权重同步协议解耦
```

这正是 verl 主线 `actor_rollout_ref` 架构已解决的问题。

**3. FSDP / Megatron**（最核心）

当前 demo 用原生 PyTorch + 手动 `.backward()`。要支持大模型（>7B），必须引入 FSDP（ZeRO-2/3）或 Megatron（张量并行）来分摊显存。

但 FSDP 和当前的"手动 `.to(device)` + NCCL P2P"架构冲突——FSDP 会自动管理参数分片和通信，和手动搬运不兼容。

### 11.4 建议路径

**不要在 demo 上打补丁。** demo 的价值是验证"层间拆分 + 跨进程梯度传递"这个核心思路可行。要走向生产，应该向 verl 主线靠拢：

```
当前：demo 验证 split 架构可行
         ↓
下一步：在 verl 主线的 Worker 框架下实现 split 策略
         ↓
最终：融入 verl 的 RayPPOTrainer，复用 FSDP / vLLM / checkpoint 等基础设施
```

具体来说：
1. 把 `MiddleExecutor` 接口做成 verl Worker 的一个可插拔 transport 层
2. 复用 verl 的 `RayWorkerGroup` 和 `ResourcePoolManager` 做进程管理
3. 复用 verl 的 `DataProto` 做数据传递
4. 复用 verl 的 `CheckpointEngine` 做 checkpoint
5. 复用 vLLM / SGLang 做推理

---

## 12. 开发者指南

### 12.1 如何修改切层

编辑 `config/split_demo_nccl.yaml`：

```yaml
split:
  front_end: 4       # layers[0:4] 在 GPU 0
  middle_end: 32     # layers[4:32] 在 GPU 1，layers[32:36] 在 GPU 0
```

约束：`0 < front_end < middle_end < total_layers`

### 12.2 如何替换 MiddleExecutor

实现 `MiddleExecutor` 接口（`execute()` 方法），注入到 `SplitActorCore`：

```python
# 自定义实现
class RPCMiddleExecutor:
    def execute(self, hidden_states, position_ids, position_embeddings, attention_mask):
        # 序列化 → gRPC 发送 → 远端执行 → 返回结果
        ...

# 注入
model = SplitActorCore(..., middle_executor=RPCMiddleExecutor(...))
```

trainer、rollout、reward 全部不需要修改。

### 12.3 如何扩展到多机

当前 NCCL 使用 `env://` 作为 init_method，由 torchrun 自动设置 `RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT`。

多机时，只需在每台机器上正确设置这些环境变量，或者改用 TCP/共享文件系统 init_method：

```bash
# 机器 0（master）
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.1.1 --master_port=29500 ...

# 机器 1
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
  --master_addr=192.168.1.1 --master_port=29500 ...
```

主体代码不需要任何修改。

### 12.4 如何换模型

将 `model.path` 指向新模型路径，同时调整 `split` 配置匹配新模型的层数：

```bash
# Qwen2.5-7B (28 层)
torchrun --nproc_per_node=2 \
  -m verl.experimental.split_demo_nccl.main_grpo_split \
  model.path=/root/share/Qwen2.5-7B-Instruct \
  split.front_end=4 \
  split.middle_end=24
```

约束：新模型必须是标准 HF CausalLM（有 `model.layers`、`embed_tokens`、`norm`、`lm_head`），且 transformers 版本支持 `position_embeddings` 参数传递。

---

## 附录 A：日志格式说明

一行典型日志：

```
[step 1] reward=0.5000 valid_group_ratio=1.0000 loss=-0.0455 clipfrac=0.0417 approx_kl=-0.0067
         gpu0_alloc=2.06GB gpu0_reserv=5.15GB gpu0_peak=2.53GB
         fwd_calls=29 prefill_calls=1 decode_calls=23
         to_middle=0.0493GB pos_ids=0.0001GB pos_emb=0.0000GB mask=0.0008GB to_primary=0.0493GB
         last_hidden=(8, 41, 2048) last_logits=(8, 41, 151936)
```

| 字段 | 含义 |
|---|---|
| `reward` | 有效样本的平均 reward |
| `valid_group_ratio` | 通过 dynamic sampling 的 group 比例 |
| `loss` | 策略梯度损失 |
| `clipfrac` | 被 clip 的 token 比例 |
| `approx_kl` | 近似 KL 散度 |
| `gpu0_alloc` | GPU 0 当前已分配显存 |
| `gpu0_peak` | GPU 0 本 step 峰值显存 |
| `to_middle` | 累计发往中段的 hidden 字节数 |
| `to_primary` | 累计从 rank1 返回的 hidden 字节数 |
| `pos_emb` | v2 中为 0（pos_emb 不再跨卡发送） |
| `last_logits` | 最终 logits shape `(B, S, vocab)` |

## 附录 B：术语表

| 术语 | 含义 |
|---|---|
| rank | 进程编号（0 或 1） |
| local_rank | 本机 GPU 编号 |
| front / middle / tail | 模型的三个分段 |
| NCCL | NVIDIA Collective Communications Library |
| P2P | Point-to-Point（点对点通信） |
| FWD_ONLY | 只做前向，不做 backward |
| FWD_WITH_BWD | 做前向，并等待 backward |
| GRPO | Group Relative Policy Optimization |
| DAPO | Decoupled Clip and Dynamic Sampling Policy Optimization |
| Dr.GRPO | Dr. GRPO（固定 loss_scale_factor 的 GRPO 变体） |
| LoRA | Low-Rank Adaptation |
| PEFT | Parameter-Efficient Fine-Tuning |
| autograd.Function | PyTorch 自定义前向/反向传播 |
