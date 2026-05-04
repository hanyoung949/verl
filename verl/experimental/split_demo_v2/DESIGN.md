# Split Actor GRPO Demo v2 — 双进程 NCCL 版设计文档

**目标**：在 v1 单进程基线基础上，引入真实进程边界，用 NCCL P2P 通信替换 `tensor.to('cuda:1')`。  
**验证平台**：单机 2×4090，`torchrun --nproc_per_node=2`。  
**范围**：`verl/experimental/split_demo_nccl/`，不修改主线代码，不修改 v1。  
**与 v1 的关系**：rollout / reward / trainer 逻辑不变；核心差异在 `core/` 和入口。

---

## 1. v1 的根本限制

v1 的跨卡"传输"依赖：

```python
h = hidden_states.to('cuda:1')
```

这在 PyTorch 里是可微的，因为两张卡共享同一个进程的 autograd 图。  
换成任何真实的跨进程 / 跨机器传输，这条隐式可微性立刻断掉。

v2 的目标就是把这条"借来的可微性"替换成：

1. **真实进程边界**：rank0 ↔ rank1，各自持有自己的 CUDA context
2. **显式 NCCL P2P 通信**：`dist.send` / `dist.recv`
3. **正确的跨进程反向传播**：rank1 本地缓存 forward 上下文，收到 `grad_h_middle` 后本地做 backward，把 `grad_h_front` 发回 rank0

---

## 2. 进程分工

```
torchrun --nproc_per_node=2

rank0  (local_rank=0, cuda:0)        rank1  (local_rank=1, cuda:1)
─────────────────────────────        ──────────────────────────────
embed_tokens                         middle_layers[front_end:middle_end]
front_layers[0:front_end]            rotary_emb（本地重算 pos_emb）
tail_layers[middle_end:]
norm / lm_head
LoRA A/B（可训练）
optimizer / rollout / reward
logging / checkpoint
```

rank1 **只运行 `MiddleWorker.run()`**，是一个纯粹的请求响应 worker，不持有任何训练逻辑。

---

## 3. 设备设置

```python
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device("cuda")   # 当前进程的"默认卡"
```

- rank0 的 `device` 指向物理 GPU0
- rank1 的 `device` 指向物理 GPU1
- 代码里统一用 `"cuda"`，不硬编码 `cuda:0` / `cuda:1`

---

## 4. 通信协议

### 4.1 header 格式

每次 forward 请求开始时，rank0 先发一个固定长度的 `int64` header tensor：

```
[flag, B, S, H, has_mask, dtype_code]
```

| 字段 | 含义 |
|---|---|
| `flag` | 0=FWD_ONLY, 1=FWD_WITH_BWD, 2=SHUTDOWN |
| `B` | batch size |
| `S` | sequence length |
| `H` | hidden dim |
| `has_mask` | 0 或 1，是否跟随 attention_mask |
| `dtype_code` | 张量 dtype 编码（0=bfloat16, 1=float16, 2=float32）|

`dtype_code` 的目的：让 rank1 的 `_recv_tensor` 不依赖硬编码 dtype，只从 header 里解析。这样切换 `torch_dtype`（fp16 / bf16 / fp32）时，通信层自动跟着配置走，不需要改代码。

> **协议约束（单飞请求）**：当前协议假设任意时刻最多只有一个未完成的 `FWD_WITH_BWD` 请求。header 里没有 `request_id`，rank1 的缓存也是单槽的。这在当前串行 trainer 下成立（每次 `forward_full` 完成后紧接着 `loss.backward()`，不存在并发中段请求）。若未来引入 pipeline overlap 或 gradient checkpointing，需要在 header 里加 `request_id` 并把缓存改为 `dict`。

### 4.2 三种 flag

| flag | 触发场景 | rank1 行为 |
|---|---|---|
| `FWD_ONLY` | rollout / old_log_prob（在 `no_grad` 下） | 不缓存，直接算完发回 |
| `FWD_WITH_BWD` | 训练 update loop（有梯度） | 缓存 `(h_in, h_out)`，发回 `h_middle`，再等 backward |
| `SHUTDOWN` | 训练结束 | 退出 worker loop |

### 4.3 每次 FWD_ONLY 的完整握手

```
rank0                           rank1
  ──── header [FWD_ONLY, ...] ──▶
  ──── h_front [B, S, H]      ──▶
  ──── position_ids [B, S]    ──▶
  ──── attention_mask（可选）  ──▶
                                  本地重算 pos_emb
                                  torch.no_grad() 跑 middle layers
  ◀─── h_middle [B, S, H]    ────
```

### 4.4 每次 FWD_WITH_BWD 的完整握手

```
rank0                           rank1
  ──── header [FWD_WITH_BWD,...] ▶
  ──── h_front [B, S, H]      ──▶
  ──── position_ids [B, S]    ──▶
  ──── attention_mask（可选）  ──▶
                                  本地重算 pos_emb
                                  h_in = h_front.detach().requires_grad_(True)
                                  h_out = middle(h_in, ...)
                                  缓存 (h_in, h_out)
  ◀─── h_middle [B, S, H]    ────

  （rank0 继续 tail forward → loss → loss.backward()）

  ──── grad_h_middle [B, S, H] ─▶
                                  torch.autograd.backward(h_out, grad_h_middle)
                                  grad_h_front = h_in.grad
                                  清空缓存
  ◀─── grad_h_front [B, S, H] ───
```

### 4.5 为什么 pos_emb 不发送

`pos_emb = rotary_emb(h_front, position_ids)` 完全由 `position_ids` 和 `rotary_emb` 模块决定，两者 rank1 都有：

- `position_ids` 通过 header 之后随 h_front 一起发送
- `rotary_emb` 模块在 rank1 加载模型时已经存在

重算成本极低（比传输 (cos, sin) 张量更优），减少跨进程传输量。

### 4.6 协议隐含的结构约束

> **当前 v2 仅支持"中段前后 hidden shape 和 dtype 不变"的纯 decoder block。**

`_NCCLMiddleFunction.forward` 直接按 `h_front.shape / dtype` 接收 `h_middle`，rank1 的反向也假设 `grad_h_front` 和 `grad_h_middle` 同型。

这在标准 decoder-only transformer 中天然成立。若未来扩展到带线性投影、维度压缩或 MoE side output 的结构，需要在 header 里单独编码 `h_middle` 的 shape，并在 `_NCCLMiddleFunction.backward` 里分别分配两种形状的 buffer。

> **所有 `dist.send/recv` 的 tensor 都必须是 contiguous，且 rank0 / rank1 对 shape、dtype 的理解完全一致。**
>
> - header tensor：固定 `int64`、固定长度、发送前 contiguous
> - `h_front / h_middle / grad_h_middle / grad_h_front`：发送前 `.contiguous()`
> - `position_ids / attention_mask`：同样建议发送前 `.contiguous()`
>
> 否则实现期最常见的问题不是“立刻报错”，而是：
>
> - NCCL 参数不匹配
> - shape/dtype 对不上
> - `recv` 永久阻塞，看起来像死锁

> **`attention_mask` 的协议语义沿用 v1：默认传的是显式 4D additive mask（float/bfloat16，padding 位置为 `-inf`），而不是 bool mask。**
>
> 原因：
>
> - v1 的 `_build_causal_mask()` 已经返回 additive mask
> - rank1 当前伪码按 `tensor_dtype` 接收 mask，也与 additive mask 更自然
>
> 因此 v2 实现时不要临时把 mask 改成 bool 语义，否则 rank0 / rank1 很容易在协议层对不上。

---

## 5. 可微性正确性

### v1 为什么能可微

```python
h_middle = hidden_states.to('cuda:1')   # PyTorch 内建可微
# 中段层计算...
return h_middle                          # 仍在 autograd 图内
```

整个 forward 在一个进程里，autograd 图连续。

### v2 为什么能可微（正确实现）

rank0 侧用 `_NCCLMiddleFunction(torch.autograd.Function)`：

```python
class _NCCLMiddleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_front, position_ids, attention_mask, executor):
        # 发送请求给 rank1，接收 h_middle
        # h_middle 是一个新 tensor，autograd 不知道它的来历
        # 但 Function 的 forward/backward 接口告诉 autograd 怎么回传梯度
        ...
        return h_middle

    @staticmethod
    def backward(ctx, grad_h_middle):
        # 把 grad_h_middle 发给 rank1
        # rank1 在它自己的进程里对缓存的 h_out 做 backward，算出 grad_h_front
        # 把 grad_h_front 发回来
        # autograd 继续向 h_front（front layers）回传
        return grad_h_front, None, None, None
```

rank1 侧的 backward 计算是：

```python
# 正确的梯度公式：
# dL/dh_front = dL/dh_middle · dh_middle/dh_front
#             = grad_h_middle · Jacobian(middle)

torch.autograd.backward(h_out, grad_h_middle)
grad_h_front = h_in.grad   # ← 这才是正确的 dL/dh_front
```

**错误做法**（直接把 grad_h_middle 当 grad_h_front 回传）：

```python
# 等价于把中段当恒等映射，梯度计算错误
return grad_h_middle, None, None, None  # ← 不能这样做
```

---

## 6. 文件结构

```
verl/experimental/split_demo_nccl/
├── DESIGN.md
├── __init__.py
├── config/
│   └── split_demo_nccl.yaml
├── core/
│   ├── __init__.py
│   ├── middle_executor.py    ← NCCLMiddleExecutor + _NCCLMiddleFunction
│   ├── middle_worker.py      ← MiddleWorker（rank1 入口）
│   └── split_actor.py        ← SplitActorCore（rank0 only，无 middle_layers）
├── rollout/
│   ├── __init__.py
│   ├── base.py               ← 与 v1 相同
│   └── naive.py              ← 与 v1 相同
├── reward/
│   ├── __init__.py
│   ├── base.py               ← 与 v1 相同
│   └── function_reward.py    ← 与 v1 相同
├── split_trainer.py          ← 与 v1 基本相同，rank0 独占
└── main_grpo_split.py        ← 双进程入口（dist init + rank 分支）
```

与 v1 的核心差异：

| 文件 | v1 | v2 |
|---|---|---|
| `core/middle_executor.py` | `LocalMiddleExecutor`（`.to()`) | `NCCLMiddleExecutor`（dist.send/recv） |
| `core/middle_worker.py` | 不存在 | 新增，rank1 worker loop |
| `core/split_actor.py` | 持有 middle_layers（cuda:1） | 不持有 middle_layers（rank1 独占） |
| `main_grpo_split.py` | 单进程，直接启动 | 双进程，dist init + rank 分支 |

---

## 7. 各模块设计

### 7.1 `core/middle_executor.py`

```python
FWD_ONLY     = 0
FWD_WITH_BWD = 1
SHUTDOWN     = 2

class _NCCLMiddleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_front, position_ids, attention_mask, executor):
        ctx.executor = executor
        executor._send_fwd_request(h_front, position_ids, attention_mask, flag=FWD_WITH_BWD)
        return executor._recv_tensor(h_front.shape, h_front.dtype)

    @staticmethod
    def backward(ctx, grad_h_middle):
        grad_h_front = ctx.executor._exchange_backward(grad_h_middle)
        return grad_h_front, None, None, None

class NCCLMiddleExecutor:
    def __init__(self, middle_rank=1, device="cuda"):
        self.middle_rank = middle_rank
        self.device = torch.device(device)
        self.stats = {...}   # 传输量统计，与 v1 MiddleExecutor.stats 对齐

    def execute(self, hidden_states, position_ids, position_embeddings, attention_mask):
        # position_embeddings 忽略，rank1 本地重算
        #
        # 前提：调用方有责任在所有"只做前向、不需要回传梯度"的路径上显式使用
        # `torch.no_grad()`（rollout / old_log_prob 均满足此条件）。
        # 若调用方在没有 no_grad() 的情况下调用 execute()，即使本意不做 backward，
        # 这里也会走 FWD_WITH_BWD 分支，rank1 会缓存上下文并一直等待 grad tensor
        # 导致双端死锁。因此：改 trainer 时必须保证此不变量。
        if torch.is_grad_enabled():
            return _NCCLMiddleFunction.apply(
                hidden_states, position_ids, attention_mask, self)
        else:
            return self._fwd_only(hidden_states, position_ids, attention_mask)

    def shutdown(self):
        # 必须和协议保持一致：始终发 6 字段 header
        header = torch.tensor([SHUTDOWN, 0, 0, 0, 0, 0], dtype=torch.int64, device=self.device)
        dist.send(header, dst=self.middle_rank)
```

### 7.2 `core/middle_worker.py`

```python
class MiddleWorker:
    @classmethod
    def from_pretrained(cls, model_path, front_end, middle_end, lora_config, device, torch_dtype):
        # 加载完整模型 → LoRA → BFS unwrap → 取 middle_layers + rotary_emb
        # 冻结所有参数，放到 device
        # torch_dtype 保存为 self.tensor_dtype，供 _recv_tensor 使用
        #
        # 注：当前实现完整加载整个模型再丢弃 front/tail，仅用于验证便利。
        # 若进入生产化阶段，可改为按分段加载（只初始化 middle 权重），
        # 以消除启动时的完整模型显存峰值。
        ...

    def run(self):
        # dtype_code → torch.dtype 映射（与 rank0 header 编码对应）
        _DTYPE_MAP = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}

        while True:
            header = torch.zeros(6, dtype=torch.int64, device=self.device)
            dist.recv(header, src=0)
            flag, B, S, H, has_mask, dtype_code = header.tolist()
            tensor_dtype = _DTYPE_MAP[int(dtype_code)]

            if flag == SHUTDOWN:
                break

            h_front      = self._recv_tensor((B, S, H),    tensor_dtype)
            position_ids = self._recv_tensor((B, S),        torch.int64)
            mask         = self._recv_tensor((B, 1, S, S), tensor_dtype) if has_mask else None

            # 本地重算 pos_emb
            cos, sin = self.rotary_emb(h_front, position_ids)

            if flag == FWD_ONLY:
                with torch.no_grad():
                    h = self._run_layers(h_front, position_ids, (cos, sin), mask)
                dist.send(h.contiguous(), dst=0)

            elif flag == FWD_WITH_BWD:
                h_in = h_front.detach().requires_grad_(True)
                h_out = self._run_layers(h_in, position_ids, (cos, sin), mask)
                dist.send(h_out.detach().contiguous(), dst=0)

                # 等 backward
                grad_out = self._recv_tensor(h_out.shape, h_out.dtype)
                torch.autograd.backward(h_out, grad_out)
                dist.send(h_in.grad.contiguous(), dst=0)
```

### 7.3 `core/split_actor.py`

与 v1 最大差异：

- `__init__` 加载完整模型，只把 front/tail/embed/norm/lm_head 放到 `device`，middle_layers **不注册、不放 GPU**
- `middle_executor` 默认是 `NCCLMiddleExecutor(middle_rank=1, device=device)`
- `forward_tail` 不需要 `.to(primary_device)`，因为 `NCCLMiddleExecutor` 返回的 tensor 已在 rank0 的 device 上
- 传输统计方式改变：`NCCLMiddleExecutor` 自己维护 stats

### 7.4 `split_trainer.py`

核心训练循环与 v1 **基本相同**：rollout → old_log_prob → reward → advantage → update epoch → log → save。

差异点：

- v2 的 `model.forward_full()` 内部调用 `NCCLMiddleExecutor`，会触发跨进程 NCCL 通信，但对 trainer 完全透明
- rank1 在 worker loop 里响应请求，trainer 不感知中段的存在
- 若未来 v2 需要 resume / multi-node checkpoint，`save_training_artifacts` 可能需要协调 rank1 侧的 state，届时会有差异；当前阶段 trainer 无需修改

### 7.5 `main_grpo_split.py`

```python
import os
from datetime import timedelta
import torch.distributed as dist

@hydra.main(config_path="config", config_name="split_demo_nccl", version_base=None)
def main(config):
    # timeout 决定 dist.recv 在对端进程已消失时最多等多久才抛 RuntimeError。
    # 60 秒对单机 2×4090 足够；多机 / WAN 场景可按需调大。
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    rank      = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 这几个变量 rank0 / rank1 都要用，必须在分支之前统一解析。
    # rank1 需要它们来决定加载哪些层；不能在 if rank == 0 分支里才解析。
    model_path  = resolve_model_path(str(config.model.path))
    front_end   = int(config.split.front_end)
    middle_end  = int(config.split.middle_end)
    lora_config = build_lora_config(config.lora)   # 返回 peft.LoraConfig

    if rank == 1:
        worker = MiddleWorker.from_pretrained(
            model_path, front_end, middle_end, lora_config, device=torch.device("cuda"), ...
        )
        try:
            worker.run()
        except Exception:
            # 调试阶段：记录完整 traceback，然后 re-raise。
            # 不能 `pass`——那会把真实 bug（协议错误、shape 不匹配等）全部吞掉。
            # re-raise 后 finally 仍会执行（Python 语义保证），进程以非零退出码退出，
            # torchrun 会打印错误并终止所有 worker。
            import logging, traceback
            logging.error("rank1 MiddleWorker crashed:\n%s", traceback.format_exc())
            raise
        finally:
            # 无论正常退出（收到 SHUTDOWN）还是异常退出，都需要销毁 process group。
            # 若 rank0 已先调用 destroy_process_group，这里再调用通常无害（幂等）。
            dist.destroy_process_group()
        return

    # rank 0
    set_seed(...)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    samples   = load_jsonl_samples(config.data.train_file)
    model     = SplitActorCore(model_path, ...)
    rollout   = NaiveSplitRollout(model, tokenizer, ...)
    reward    = FunctionReward(math_exact_match, tokenizer)
    trainer   = SplitGRPOTrainer(model, rollout, reward, tokenizer, config)

    try:
        trainer.fit(samples)
        model.middle_executor.shutdown()   # 正常结束：发 6 字段 SHUTDOWN header 给 rank1
    finally:
        # 无论正常还是异常，都销毁 process group。
        # rank0 退出/crash 会触发 NCCL 超时或 group 销毁，
        # 使 rank1 阻塞中的任何 dist.recv（无论是在等 header 还是在等 grad）
        # 抛出异常，由 rank1 的 except 接住，进入 finally 清理。
        # 注意：异常路径不再单独发 SHUTDOWN——此时 rank1 可能正在等 grad tensor，
        # 而非 header，发 header 并不能解救它。
        dist.destroy_process_group()
```

运行命令：

```bash
torchrun --nproc_per_node=2 \
  -m verl.experimental.split_demo_nccl.main_grpo_split
```

---

## 8. 验证计划（分阶段）

### Phase 1：最小端到端冒烟测试

目标：验证双进程能正常启动、NCCL 通信不崩、正常退出。

`trainer.total_steps=1` 会走完整 trainer 流程（rollout → old_log_prob → reward → advantage → update），因此同时覆盖：

- **FWD_ONLY**：rollout 阶段 + old_log_prob 阶段（两者都在 `no_grad()` 下）
- **FWD_WITH_BWD**：update loop 里的训练 forward + backward

> 注：Phase 1 **不是** "只验证 FWD_ONLY"，而是"整条链路跑通、不崩"。
> 若要单独隔离验证 FWD_ONLY，需要在 trainer 外单独写 smoke 脚本，只做 rollout + old_log_prob，不进入 update loop。当前暂不要求这一步。

```bash
torchrun --nproc_per_node=2 \
  -m verl.experimental.split_demo_nccl.main_grpo_split \
  trainer.total_steps=1 \
  algorithm.dynamic_sampling=false
```

观察点：无 crash，`sequences` shape 正确，rank1 正常退出（不死锁）。

### Phase 2：梯度正确性验证

在 Phase 1 通过后，专门核查一次 `loss.backward()` 后的梯度：

- rank0 front LoRA 参数有 `.grad`（非 None，非全零）
- middle_layers（rank1 侧）的参数没有 `.grad`（它们在独立进程，rank0 侧根本不存在其引用）

可在 `split_trainer.py` 里临时插一条断言：

```python
for name, p in self.model.lora_parameters():
    assert p.grad is not None, f"LoRA param {name} has no grad"
```

### Phase 3：完整 GRPO 训练

接回完整配置，跑 10 步，观察：

- `reward / valid_group_ratio / clipfrac / approx_kl` 的走势
- 对比 v1 RUNTIME.md 基线，数值量级应当一致（允许因随机种子不同有差异）

---

## 9. 与未来落地的对应关系

| v2 中 | 对应真实场景 |
|---|---|
| rank0 | 边侧设备（前段 + 尾段 + LoRA） |
| rank1 | 中心服务（中段，冻结） |
| `dist.send/recv` (NCCL) | 同机 NVLink / InfiniBand / 未来 WAN |
| `MiddleWorker.run()` | 中心服务的请求响应 loop |
| `NCCLMiddleExecutor` | 边侧的 transport adapter |

从 v2 切到真实多机，只需替换 `dist.init_process_group` 的 `init_method`（从 `env://` 改为实际地址），主体代码不动。

切到 WAN / 自定义协议时，只需替换 `NCCLMiddleExecutor` 的 `_send_fwd_request` / `_recv_tensor` / `_exchange_backward` 三个方法，其余不动。
