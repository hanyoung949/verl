# Split Demo v3 — 状态总览

> 最后更新：2026-05-18
> 详见 [DESIGN.md](DESIGN.md) 了解完整设计。

---

## Phase 验证结果

| Phase | 内容 | 状态 | 验证方式 |
|---|---|---|---|
| Phase 0 | 干净基线：10 步 GRPO 训练 | ✅ | `torchrun --nproc_per_node=2` |
| Phase 1 | SplitEngine 规范路径：train_batch/infer_batch | ✅ | 10 步 GRPO 通过 Engine 接口 |
| Phase 2 | DataProto 兼容：_normalize_batch | ✅ | Engine 同时接受 TensorDict 和 DataProto |
| Phase 3 | Checkpoint save/load | ✅ | adapter state 匹配，loss 量级一致 |
| Phase 4-5 | 文档化 + v4 迁移边界 | ✅ | DESIGN.md + STATUS.md |

---

## 训练记录

### 配置

```yaml
model: Qwen2.5-3B-Instruct (36 层, hidden=2048, vocab=151936)
split: 4:28:4 (front=4, middle=28, tail=4)
lora: r=16, alpha=32, target=[q_proj, v_proj]
train: lr=5e-4, group_size=4, max_new_tokens=24, loss_scale_factor=24
update: epochs=2, mini_batch_size=8
dynamic_sampling: true
clip: low=0.2, high=0.28
```

### 10 步 GRPO 训练

```
[step 1]  reward=0.5000 valid=1.00 loss=-0.0458 clipfrac=0.0469 kl=-0.0049 gpu_peak=2.62GB
[step 2]  reward=0.3750 valid=0.50 loss=-0.0035 clipfrac=0.0469 kl=-0.0003 gpu_peak=2.62GB
[step 3]  reward=0.3333 valid=0.75 loss=-0.2715 clipfrac=0.1667 kl=0.0277  gpu_peak=2.60GB
[step 4]  reward=0.5833 valid=0.75 loss=0.2383  clipfrac=0.0938 kl=0.0293  gpu_peak=2.57GB
[step 5]  reward=0.5000 valid=0.75 loss=-0.0145 clipfrac=0.1354 kl=0.0352  gpu_peak=2.56GB
[step 6]  reward=0.4167 valid=0.75 loss=-0.1293 clipfrac=0.0729 kl=0.0216  gpu_peak=2.55GB
[step 7]  reward=0.5000 valid=0.50 loss=-0.0083 clipfrac=0.0417 kl=0.0055  gpu_peak=2.60GB
[step 8]  reward=0.3750 valid=0.50 loss=-0.0080 clipfrac=0.0312 kl=0.0121  gpu_peak=2.62GB
[step 9]  reward=0.5000 valid=0.75 loss=-0.0080 clipfrac=0.1146 kl=0.2070  gpu_peak=2.62GB
[step 10] reward=0.4167 valid=0.75 loss=0.1752  clipfrac=0.0625 kl=0.0254  gpu_peak=2.57GB
```

### 指标分析

| 指标 | 值 | 说明 |
|---|---|---|
| reward 范围 | 0.33–0.58 | 10 步太短，无明显趋势 |
| valid_group_ratio | 0.50–1.00 | dynamic sampling 过滤了部分 group |
| clipfrac | 0.03–0.17 | 在合理范围，没有爆炸 |
| approx_kl | -0.005–0.21 | 第 9 步偏高但未失控 |
| gpu0_peak | 2.55–2.62 GB | 比 v2 (2.53GB) 略高，Engine 接口层有微小开销 |
| trainable params | 32 | 与 v1/v2 相同 |

### Checkpoint

```
checkpoints/latest/
  adapter_state.pt      3287694 bytes (32 LoRA 参数)
  optimizer.pt          6580734 bytes (AdamW 状态)
  adapter_meta.json     108 bytes (模型路径 + 切层配置 + step)
```

load 验证：adapter state 匹配 ✅，loss 量级一致（diff=0.012，BF16 精度范围） ✅

---

## 文件清单

```
split_demo_v3/
├── __init__.py
├── PLAN.md                        ← 技术方案（813 行）
├── DESIGN.md                      ← 设计文档
├── STATUS.md                      ← 本文档
├── config/
│   └── split_demo_v3.yaml
├── core/
│   ├── __init__.py
│   ├── split_engine.py            ← SplitEngine(BaseEngine)
│   ├── split_actor_v3.py          ← SplitActorCore
│   ├── middle_executor.py         ← NCCLMiddleExecutor
│   └── middle_worker.py           ← MiddleWorker
├── rollout/                       ← 复用 v2
├── reward/                        ← 复用 v2
├── main_split_v3.py               ← 入口 + GRPO 循环
└── checkpoints/latest/            ← 训练产物
```

---

## 启动命令

```bash
source /root/workspace/miniconda3/etc/profile.d/conda.sh && conda activate verl

torchrun --nproc_per_node=2 -m verl.experimental.split_demo_v3.main_split_v3 \
  trainer.total_steps=10 \
  trainer.log_freq=1 \
  trainer.lr=5e-4 \
  rollout.group_size=4 \
  rollout.max_new_tokens=24 \
  algorithm.loss_scale_factor=24
```

---

## 与 v2 的对比

| 维度 | v2 | v3 |
|---|---|---|
| 引擎接口 | 自定义 | SplitEngine(BaseEngine) |
| 数据格式 | tensor + dict | TensorDict / DataProto |
| GRPO 算法 | 自写 advantage + loss | 复用 verl core_algos |
| Checkpoint | 手写 adapter_state.pt | engine.save_checkpoint() |
| 注册机制 | 无 | @EngineRegistry.register(backend="split") |
| 调用方式 | 直接 split_core | engine.train_batch() / infer_batch() |
| 数值结果 | reward 0.33–0.75 | reward 0.33–0.58 |
| 显存 (gpu0) | 2.53 GB | 2.55-2.62 GB |

---

## v4 迁移边界

详见 [DESIGN.md §10](DESIGN.md#10-v4-迁移边界)。

### 保持稳定的接口

```
DataProto in
train_batch / infer_batch
metrics out
save_checkpoint / load_checkpoint
is_mp_src_rank_with_outputs
```

### 可迁移的组件

```
SplitActorCore.forward_front()  → Head stage runner
SplitActorCore.forward_tail()   → Tail stage runner
MiddleWorker._run_layers()      → Middle stage runner
MiddleWorker.run()              → Stage service loop
NCCLMiddleExecutor              → transport/autograd bridge
SplitEngine(BaseEngine)         → SplitPipelineEngine
grpo_loss_fn                    → v4 loss 参考
```
