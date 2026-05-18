# Split Demo v3 — veRL 兼容的 Split Actor 基线

> 最后更新：2026-05-18
> 范围：`verl/experimental/split_demo_v3/`
> 目标：构建一个干净的 2-rank split actor 基线，使其符合 veRL 的数据流和 Engine 抽象，同时保留未来 v4 多 stage pipeline 所需的核心逻辑。

---

## 0. 摘要

v3 不是最终的多机 A800 架构。v3 是一个最小但有价值的系统：它应该先“说 veRL 的语言”。

v3 的目标是：

```text
v3 = 2-rank torchrun split actor
   + BaseEngine-compatible SplitEngine
   + DataProto-compatible train/infer inputs
   + veRL-style old_log_prob/update_actor/checkpoint semantics
   + reusable front/middle/tail boundaries for v4
```

v3 的目标不是：

```text
v3 != Ray 双 Actor 生产运行时
v3 != 多 stage pipeline
v3 != stage 内 tensor parallel runtime
v3 != 不惜代价完整接入 RayPPOTrainer
```

长期设计仍然是：

```text
veRL control flow
  rollout -> reward -> old_log_prob -> advantage -> actor update -> weight sync

custom split computation flow
  Head stage -> Middle stage(s) -> Tail stage
  activation/grad 通过 NCCL/RDMA 传输，不走 Ray object store
```

v3 应该先把上层 veRL 接口做正确，再由 v4 扩展下层 computation flow。

---

## 1. 设计定位

### 1.1 为什么需要 v3

v1 和 v2 已经证明了 split learning 的核心可行性：

- v1 证明了同进程 split training 可以通过可微 device transfer 跑通。
- v2 证明了跨进程 NCCL send/recv 和 middle layer 本地 backward 是正确的。
- v3 要证明：同一个 split actor 可以被整理成 veRL 风格的训练后端。

v3 最重要的产物不是“能跑的 demo”，而是一个干净的 Engine 边界：

```text
外部视角：
  一个 veRL-compatible actor engine

内部视角：
  rank0 edge process + rank1 frozen middle process
```

### 1.2 v3 应该为 v4 保留什么

| v3 组件 | v4 迁移目标 |
|---|---|
| `SplitEngine(BaseEngine)` | `SplitPipelineEngine(BaseEngine)` |
| `SplitActorCore.forward_front()` | Head stage runner |
| `SplitActorCore.forward_tail()` | Tail stage runner |
| `MiddleWorker._run_layers()` | 通用 Middle stage runner |
| `MiddleWorker.run()` | Stage service loop 和 local backward 协议 |
| `NCCLMiddleExecutor` | 相邻 stage 的 transport/autograd bridge |
| `main_split_v3.py` GRPO loop | Phase-0 数值基线 |
| checkpoint save/load | per-stage checkpoint 设计 |
| runtime stats | stage-level compute/communication metrics |

### 1.3 v3 应该避免什么

- 不以 Ray 双 Actor backward 为优化目标。
- 不通过 Ray RPC 传 activation 或 gradient tensor。
- 不在 v3 引入多 stage pipeline 代码。
- 不在 v3 引入 stage 内 tensor parallel。
- 不把 split 语义藏进无法映射到 veRL `TrainingWorker` 路径的临时训练循环里。

---

## 2. veRL 数据流与 v3 映射

对当前工作最有用的 veRL 抽象是两级数据流：

```text
Control flow:
  RLTrainer 调度 rollout、reward、logprob、advantage、update、checkpoint。

Computation flow:
  Model engine 执行 forward、backward、optimizer、inference、checkpoint。
```

即使 v3 仍然使用自定义入口，也应该让 control-flow 语义尽量贴近 veRL。

### 2.1 Rollout 生成

veRL 形态：

```text
RLTrainer
  -> ActorRolloutRefWorker.generate_sequences()
  -> Rollout engine 生成 prompt + response
  -> 返回 DataProto
```

v3 形态：

```text
main_split_v3.py
  -> SimpleSplitRollout / NaiveSplitRollout
  -> SplitActorCore.forward_full() under torch.no_grad()
  -> 返回 sequences、attention_mask、response_ids、response_mask
```

v3 要求：

- rollout 输出 key 要兼容 veRL 风格的 `DataProto`。
- v3 可以继续使用 naive rollout。
- vLLM 不是 v3 完成条件。

### 2.2 Reward 计算

veRL 形态：

```text
RLTrainer
  -> RewardWorker 或 reward function
  -> token_level_rewards / scores
```

v3 形态：

```text
FunctionReward
  -> math_exact_match
  -> token_level_scores
```

v3 要求：

- reward 输出布局应能自然放入 `DataProto.batch["token_level_rewards"]`。
- 保留 `uid` 或 group index，用于 GRPO 的 group-wise advantage 计算。

### 2.3 Old Log Probability

veRL 形态：

```text
Actor engine infer_batch()
  -> old_log_probs
```

v3 形态：

```text
with torch.no_grad():
  logits_old = split_core.forward_full(sequences, attention_mask)
  old_log_prob = logprobs_from_logits(...)
```

v3 要求：

- 将这类语义迁移到 `SplitEngine.infer_batch()` 或一个等价的轻量 wrapper 后面。
- 输出 key 应使用 veRL 习惯命名：`old_log_probs`，或明确记录 v3 alias。

### 2.4 Reference Log Probability

v3 计划采用去 KL / 去 Reference Model 的算法路径。

v3 要求：

- baseline 不要求 ref model。
- 除非未来实验需要，否则 `compute_ref_log_prob` 不在 v3 范围内。
- config 中应明确这一点。

### 2.5 Advantage 计算

veRL 形态：

```text
RLTrainer 在本地根据 rewards/logprobs/group ids 计算 advantage。
```

v3 形态：

```text
compute_grpo_outcome_advantage(...)
```

v3 要求：

- 继续复用 veRL 的 `compute_grpo_outcome_advantage`。
- group id 放在 `non_tensor_batch`，或使用文档明确的 tensor-compatible 编码。

### 2.6 Actor Update

veRL 形态：

```text
ActorRolloutRefWorker.update_actor(mini_batch_data)
  -> TrainingWorker.train_mini_batch()
  -> engine.train_batch()
```

v3 形态：

```text
for minibatch:
  logits_new = split_core.forward_full(...)
  loss = GRPO/DAPO/Dr.GRPO loss
  loss.backward()
  optimizer.step()
```

v3 要求：

- 训练路径必须能通过 `SplitEngine.train_batch(data, loss_fn)` 调用。
- 自定义 loop 可以保留用于 debug，但 Engine 路径是规范接口。
- rank0 返回 metrics；rank1 只负责 NCCL service。

### 2.7 Weight Sync

veRL 形态：

```text
actor weights -> rollout engine
```

v3 形态：

```text
naive rollout 和 training 使用同一个 SplitActorCore
```

v3 要求：

- v3 完成不要求 vLLM 权重同步。
- checkpoint 格式应为未来 adapter export to vLLM 做准备。

---

## 3. v3 运行时架构

### 3.1 主线进程拓扑

v3 主线使用 `torchrun` 启动两个稳定 OS 进程：

```text
torchrun --nproc_per_node=2 -m verl.experimental.split_demo_v3.main_split_v3

rank0: edge / output rank
  embed_tokens
  front_layers + LoRA
  tail_layers + LoRA
  norm + lm_head
  GRPO loss
  optimizer
  checkpoint

rank1: middle service rank
  frozen middle_layers
  recv activation
  local forward
  recv grad_out
  local backward
  send grad_in
```

### 3.2 Forward 协议

```text
rank0:
  h_front = embed + front(input_ids)
  send h_front, position_ids, attention_mask

rank1:
  h_in = recv h_front
  h_out = middle_layers(h_in)
  send h_out

rank0:
  logits = tail + norm + lm_head(h_out)
```

### 3.3 Backward 协议

```text
rank0:
  loss.backward()
  _NCCLMiddleFunction.backward sends grad_h_middle

rank1:
  h_in = h_front.detach().requires_grad_(True)
  h_out = middle_layers(h_in)
  recv grad_h_middle
  torch.autograd.backward(h_out, grad_h_middle)
  send h_in.grad

rank0:
  receives grad_h_front
  autograd continues through front layers
```

关键不变量：

```python
grad_h_front != grad_h_middle
grad_h_front = grad_h_middle @ Jacobian(middle_layers)
```

### 3.4 Rank 职责

| Rank | Role | 返回 logits/loss | 更新参数 | 持有 optimizer |
|---|---|---:|---:|---:|
| rank0 | edge/output | yes | yes | yes |
| rank1 | middle/service | no | no | no |

`is_mp_src_rank_with_outputs()` 只应在 rank0 返回 true。

---

## 4. SplitEngine 设计

`SplitEngine` 是 v3 接入 veRL Model Engine 抽象的桥。

### 4.1 必须实现的接口

```python
initialize()
forward_backward_batch(data, loss_function, forward_only=False)
train_batch(data, loss_function)
infer_batch(data, loss_function=None)
optimizer_zero_grad()
optimizer_step()
lr_scheduler_step()
save_checkpoint(...)
load_checkpoint(...)
train_mode()
eval_mode()
to(...)
get_data_parallel_size()
get_data_parallel_rank()
get_data_parallel_group()
is_mp_src_rank_with_outputs()
```

### 4.2 数据归一化

Engine 应同时接受内部 TensorDict 和 veRL DataProto：

```python
def normalize_batch(data):
    if isinstance(data, DataProto):
        return data.batch, data.non_tensor_batch, data.meta_info
    if isinstance(data, TensorDict):
        return data, {}, {}
    raise TypeError(...)
```

v3 应保证模型计算路径与输入容器无关。训练数学不应依赖调用方使用 TensorDict 还是 DataProto。

### 4.3 输入与输出 key

最小输入 key：

```text
input_ids
attention_mask
responses
response_mask
old_log_probs
advantages
```

可选 key：

```text
token_level_rewards
uid / group_id
position_ids
loss_mask
```

最小输出 metrics：

```text
loss
grad_norm
clipfrac
approx_kl
entropy, if cheap
```

### 4.4 Train Batch 语义

除非显式配置，否则 `train_batch` 应只执行一次 optimizer update：

```text
zero_grad
forward_backward_batch(forward_only=False)
clip_grad_norm
optimizer.step
return metrics
```

如果 PPO/GRPO 的多 epoch 训练仍在 Engine 外部执行，则调用方负责：

- mini-batch slicing
- epoch loop
- dynamic sampling
- advantage computation

Engine 负责：

- forward
- backward
- 与 middle rank 交换梯度
- optimizer step
- model state

### 4.5 Infer Batch 语义

`infer_batch` 应满足：

- 在 `torch.no_grad()` 下执行。
- 使用同一条 split forward 路径。
- 根据调用方需要返回 logits 或 log probabilities。
- 不修改 optimizer 或 gradients。

DataProto key 稳定后，old-log-prob 计算应优先放在这里。

---

## 5. 算法基线

v3 算法基线是 GRPO，并保留 DAPO / Dr.GRPO 风格配置。

### 5.1 默认选择

| Feature | v3 setting |
|---|---|
| Critic | disabled |
| Reference model / KL | disabled by default |
| Advantage | GRPO group-relative outcome advantage |
| Std normalization | configurable, Dr.GRPO default is no std normalization |
| Dynamic sampling | filter groups with zero reward variance |
| Clip higher | asymmetric clip, e.g. low 0.2, high 0.28 |
| Loss aggregation | sequence/token aggregation compatible with Dr.GRPO experiments |

### 5.2 为什么 v3 默认不需要 Reference Model

目标边云架构受益于移除 Reference Model：

- 模型角色更少。
- 边侧资源压力更低。
- 不需要额外 ref-log-prob split forward。
- veRL worker mapping 更简单。

这是算法选择，应保留可配置性；但 v3 baseline 不应依赖 ref model。

### 5.3 v3 暂不实现的算法优化

- GTPO gradient correction
- entropy-based filtering beyond simple metrics
- MTP / draft model
- overlong reward shaping beyond current reward hooks

这些属于后续优化层。v3 首先要保证 split actor engine 正确。

---

## 6. Checkpoint 与状态

### 6.1 v3 Checkpoint 格式

当前目标：

```text
checkpoints/latest/
  adapter_state.pt
  optimizer.pt
  adapter_meta.json
```

`adapter_meta.json` 应包含：

```text
model_path
front_end
middle_end
global_step
lora_r
lora_alpha
lora_target_modules
trainable_param_names
```

### 6.2 Load 语义

`load_checkpoint` 应恢复：

- rank0 上的 trainable LoRA / front / tail state。
- rank0 上的 optimizer state。
- rank1 不恢复参数状态，因为 middle 冻结且从 base model 加载。

### 6.3 v4 兼容性

v4 需要 per-stage checkpoint：

```text
checkpoints/latest/
  topology.json
  stages/
    stage_000_head/
      adapter_state.pt
      optimizer.pt
    stage_005_tail/
      adapter_state.pt
      optimizer.pt
```

v3 不需要实现这个格式，但 v3 metadata 应包含足够信息，用于未来推导 head/tail split。

---

## 7. Metrics 与可观测性

v3 应报告对 veRL 集成和 v4 性能对比都有价值的指标。

### 7.1 训练指标

```text
reward_mean
loss
clipfrac
approx_kl
grad_norm
valid_group_ratio
```

### 7.2 运行时指标

```text
rank0_peak_memory
rank0_allocated_memory
middle_executor_calls
forward_to_middle_bytes
attention_mask_bytes
last_hidden_shape
```

### 7.3 面向 v4 的指标

这些指标在 v3 中可以先由 rank0 估计：

```text
activation_bytes_per_forward
gradient_bytes_per_backward
num_split_round_trips
```

v4 会将它们扩展为 per-stage compute/communication timings。

---

## 8. Ray 与 vLLM 定位

### 8.1 Ray 在 v3 中的定位

Ray 双 Actor backward 不是 v3 完成条件。

已知问题不应简单描述为“Ray actor 是线程”。Ray Actor 是进程，但当前实验混合了：

- persistent NCCL rank semantics
- Ray RPC method lifetime
- CUDA tensor crossing driver/actor boundaries
- PyTorch autograd C++ worker threads
- NCCL/CUDA context initialization

这使 center backward 很脆弱，并已经产生过 deadlock。

v3 决策：

```text
main_split_v3.py = mainline
main_ray_v3.py = experiment / issue reproduction
edge_worker.py, center_worker.py = experiment
center_process.py = possible future control-plane reference
```

未来 Ray 的角色：

```text
Ray = control plane
  resource placement
  process launch
  health check
  log collection
  restart from checkpoint

NCCL/RDMA = data plane
  activation transfer
  gradient transfer
```

### 8.2 vLLM 在 v3 中的定位

vLLM 很有价值，但不是 v3 完成条件。

v3 可以继续使用 naive rollout，直到 Engine / DataProto / checkpoint 接口稳定。未来 vLLM 集成应采用：

```text
training path: SplitEngine
rollout path: vLLM full-model instance or future split-aware rollout
weight sync: LoRA adapter export/import
```

---

## 9. 分阶段工作计划

每个阶段都必须可以独立验证。

### Phase 0：干净基线

目标：当前 v3 torchrun 路径是数值基线。

命令：

```bash
torchrun --nproc_per_node=2 -m verl.experimental.split_demo_v3.main_split_v3
```

通过标准：

- 完成 10 个 GRPO step。
- checkpoint 保存成功。
- rank0 LoRA 参数收到梯度。
- rank1 middle 参数保持冻结。
- metrics 记录到 `STATUS.md`。

### Phase 1：SplitEngine 规范路径

目标：Engine 方法成为训练/推理的规范边界。

工作：

- 至少有一条 smoke path 使用 `train_batch`。
- `infer_batch` 能在 no-grad 下计算 logits/log_probs。
- rank1 永远不作为 output rank 返回 logits/metrics。

通过标准：

- 直接 loop 和 `SplitEngine.train_batch` 的 loss 量级一致。
- `is_mp_src_rank_with_outputs()` 只在 rank0 为 true。

### Phase 2：DataProto 兼容

目标：匹配 veRL 数据容器预期。

工作：

- 在 Engine 边界增加 DataProto normalization。
- 文档化支持的 key。
- 保留 TensorDict 路径用于 debug。

通过标准：

- 同一个 batch 下，TensorDict 和 DataProto 输入得到相同 logits/loss。
- GRPO advantage / dynamic sampling 所需输入可通过 DataProto 承载。

### Phase 3：Checkpoint Load 验证

目标：checkpoint 不只是能保存，还能恢复。

工作：

- 训练后保存。
- 重新创建 engine。
- 加载 checkpoint。
- 继续训练 1 step。

通过标准：

- 加载后的 adapter state 与保存的 trainable state 匹配。
- optimizer state 成功加载。
- 恢复后的 step 能完成。

### Phase 4：文档化 veRL Worker 映射

目标：说明 v3 如何位于 veRL worker 体系下，而不强行要求完整 RayPPOTrainer 集成。

工作：

- 文档化 `TrainingWorker -> SplitEngine`。
- 文档化 rank0 collect 行为。
- 文档化 rank1 service 行为。
- 文档化 no-ref / no-critic 假设。

通过标准：

- `PLAN.md` 和 `STATUS.md` 对主线文件与非主线文件的描述一致。

### Phase 5：冻结 v4 复用边界

目标：v4 可以从稳定迁移图开始。

工作：

- 冻结 front/middle/tail role responsibilities。
- 冻结 transport responsibilities。
- 冻结 checkpoint metadata fields。

通过标准：

- v4 `PLAN.md` 可以引用 v3 组件，而不依赖 Ray 实验路径。

---

## 10. v4 迁移契约

v4 不应该重写 veRL-facing interface。v4 应替换内部 computation flow。

### 10.1 应保持稳定的接口

```text
DataProto in
train_batch / infer_batch
metrics out
save_checkpoint / load_checkpoint
rank with outputs
```

### 10.2 会变化的运行时

v3：

```text
rank0 edge -> rank1 middle -> rank0 edge
```

v4：

```text
Head stage -> Middle stage 1 -> ... -> Middle stage N -> Tail stage
```

### 10.3 未来 v4 硬件目标

示例：

```text
Machine 1: 4 x A800
  GPU0-1: Head stage
  GPU2-3: Tail stage

Machine 2: 8 x A800
  GPU0-1: Middle stage 1
  GPU2-3: Middle stage 2
  GPU4-5: Middle stage 3
  GPU6-7: Middle stage 4
```

v4 plan 应按以下顺序推进：

1. 2-stage topology abstraction，复现 v3。
2. 3-stage Head/Middle/Tail。
3. N-stage single-GPU pipeline。
4. multi-node single-GPU-per-stage。
5. two-GPU-per-stage。
6. Ray control-plane orchestration。
7. vLLM rollout and adapter sync。

---

## 11. 风险与缓解

| 风险 | 严重性 | 缓解 |
|---|---:|---|
| Ray backward deadlock 分散 v3 注意力 | high | Ray 保留为非主线实验 |
| DataProto mismatch 引入隐蔽错误 | high | 做 TensorDict/DataProto 等价测试 |
| Engine 路径与手写 loop 分叉 | high | 让 `train_batch` smoke path 成为规范路径 |
| Middle backward 返回错误梯度 | high | 保留 v2 梯度检查并比较 trainable grads |
| checkpoint 只保存但不能恢复 | medium | 增加 load-and-continue 验证 |
| v3 过度绑定 2-rank 名字 | medium | 文档化 stage migration map，避免新增硬编码 rank 名 |
| 过早集成 vLLM 分散重点 | medium | 等 split engine 接口稳定后再做 |

---

## 12. v3 完成标准

满足以下所有条件后，v3 才算完成：

- `torchrun --nproc_per_node=2 -m verl.experimental.split_demo_v3.main_split_v3` 完成 10 个 GRPO step。
- `SplitEngine.train_batch` 和 `SplitEngine.infer_batch` 可作为 public engine methods 使用。
- Engine 边界同时支持 DataProto 和 TensorDict 输入。
- checkpoint save/load/resume 已验证。
- rank0 是唯一 output rank。
- rank1 是 frozen middle layers 的 service rank。
- Ray Actor backward 被文档化为非主线问题。
- v4 migration map 已文档化，且不依赖 Ray 双 Actor runtime。

完成这些后，v4 工程可以在不推倒 v3 的前提下开始。
