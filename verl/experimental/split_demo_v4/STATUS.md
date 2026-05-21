# Split Demo v4 — 状态总览

> 最后更新：2026-05-21
> 详见 [DESIGN.md](DESIGN.md) 了解完整设计。

---

## Phase 验证结果

| Phase | 内容 | 状态 | 验证方式 |
|---|---|---|---|
| Phase A | 协议稳定：Tail 本地采样 + 保存 old_log_prob + 动态 vocab | ✅ | 10 步 GPU 通过 |
| Phase B | 代码抽象：SplitTrainer + BaseRolloutBackend + flag 校验 | ✅ | 10 步 GPU 通过 |
| Phase C | 可配置化：topology + BaseEngine 对齐 + Checkpoint 完整 | ✅ | 10 步 GPU 通过 |

---

## 训练记录

### 配置

```
model: Qwen2.5-3B-Instruct (36 层, hidden=2048, vocab=151936)
split: 4:28:4 (front=4, middle=28, tail=4)
topology: head=[0], middle=[1], tail=[2]
lora: r=16, alpha=32, target=[q_proj, v_proj]
train: lr=1e-4, group_size=4, max_new_tokens=128, loss_scale_factor=128
update: epochs=1, mini_batch_size=8
dynamic_sampling: true
clip: low=0.2, high=0.28
```

### 50 步 GRPO 训练

| step | reward | valid | loss | approx_kl | clipfrac |
|------|--------|-------|------|-----------|----------|
| 1 | 0.1250 | 2/4 | 0.0007 | 0.0052 | 0.0186 |
| 2 | 0.2500 | 3/4 | 0.0624 | 0.0037 | 0.0117 |
| 3 | 0.1250 | 2/4 | 0.0006 | 0.0017 | 0.0078 |
| 4 | 0.1250 | 2/4 | 0.0002 | 0.0013 | 0.0039 |
| 5 | 0.3125 | 4/4 | -0.0622 | 0.0026 | 0.0098 |
| 6 | 0.2500 | 3/4 | -0.1870 | 0.0010 | 0.0078 |
| 7 | 0.1250 | 2/4 | 0.0004 | 0.0007 | 0.0156 |
| 8 | 0.3750 | 4/4 | -0.0636 | 0.0013 | 0.0010 |
| 9 | 0.4375 | 4/4 | -0.3123 | 0.0027 | 0.0088 |
| 10 | 0.1875 | 3/4 | 0.0013 | -0.0003 | 0.0039 |
| 11 | 0.3750 | 3/4 | -0.0001 | 0.0041 | 0.0059 |
| 12 | 0.2500 | 2/4 | 0.0006 | 0.0014 | 0.0049 |
| 13 | 0.1875 | 2/4 | -0.0004 | -0.0010 | 0.0020 |
| 14 | 0.2500 | 3/4 | 0.1255 | -0.0004 | 0.0078 |
| 15 | 0.1875 | 2/4 | 0.0005 | 0.0013 | 0.0156 |
| 16 | 0.3750 | 4/4 | -0.0310 | 0.0033 | 0.0088 |
| 17 | 0.3125 | 3/4 | -0.0646 | -0.0028 | 0.0020 |
| 18 | 0.3125 | 2/4 | -0.0002 | 0.0018 | 0.0088 |
| 19 | 0.3750 | 4/4 | -0.1257 | 0.0007 | 0.0049 |
| 20 | 0.2500 | 3/4 | 0.0636 | 0.0027 | 0.0117 |
| 21 | 0.3750 | 3/4 | 0.2495 | 0.0009 | 0.0059 |
| 22 | 0.5000 | 4/4 | -0.0298 | 0.0039 | 0.0117 |
| 23 | 0.4375 | 4/4 | -0.0304 | 0.0012 | 0.0146 |
| 24 | 0.1875 | 2/4 | -0.0001 | 0.0020 | 0.0078 |
| 25 | 0.3750 | 2/4 | 0.0007 | 0.0006 | 0.0107 |
| 26 | 0.1250 | 2/4 | 0.0002 | 0.0015 | 0.0088 |
| 27 | 0.2500 | 2/4 | -0.0014 | -0.0003 | 0.0107 |
| 28 | 0.0625 | 1/4 | -0.0002 | -0.0006 | 0.0117 |
| 29 | 0.1250 | 1/4 | 0.0001 | -0.0004 | 0.0020 |
| 30 | 0.0000 | 0/4 | — | — | — |
| 31 | 0.1250 | 2/4 | 0.0003 | -0.0010 | 0.0107 |
| 32 | 0.1250 | 1/4 | 0.0008 | 0.0013 | 0.0000 |
| 33 | 0.0625 | 1/4 | -0.0003 | 0.0021 | 0.0098 |
| 34 | 0.3125 | 3/4 | 0.1262 | -0.0009 | 0.0137 |
| 35 | 0.1250 | 1/4 | 0.0001 | -0.0027 | 0.0137 |
| 36 | 0.3750 | 4/4 | 0.2817 | -0.0010 | 0.0068 |
| 37 | 0.4375 | 2/4 | 0.0007 | 0.0023 | 0.0176 |
| 38 | 0.3125 | 4/4 | 0.1878 | 0.0023 | 0.0156 |
| 39 | 0.1250 | 2/4 | -0.0000 | 0.0000 | 0.0020 |
| 40 | 0.1250 | 2/4 | 0.0006 | 0.0020 | 0.0059 |
| 41 | 0.1875 | 3/4 | 0.0010 | 0.0017 | 0.0059 |
| 42 | 0.0625 | 1/4 | 0.0009 | 0.0036 | 0.0117 |
| 43 | 0.2500 | 4/4 | 0.0007 | 0.0031 | 0.0068 |
| 44 | 0.1875 | 2/4 | -0.0005 | 0.0028 | 0.0098 |
| 45 | 0.2500 | 2/4 | 0.0011 | 0.0062 | 0.0117 |
| 46 | 0.3125 | 3/4 | 0.0001 | 0.0011 | 0.0020 |
| 47 | 0.1250 | 2/4 | 0.0002 | 0.0021 | 0.0137 |
| 48 | 0.4375 | 4/4 | -0.1568 | 0.0028 | 0.0156 |
| 49 | 0.0625 | 1/4 | -0.0006 | -0.0000 | 0.0098 |
| 50 | 0.1250 | 2/4 | 0.0008 | -0.0000 | 0.0107 |

### 指标分析

| 指标 | 值 | 说明 |
|---|---|---|
| reward 范围 | 0.00–0.50 | 50 步中无明显持续上升，波动较大 |
| valid_group_ratio | 0.25–1.00 | dynamic_sampling 过滤了约 37% 的 group |
| clipfrac | 0.00–0.019 | 非常健康，无 clip 爆炸 |
| approx_kl | -0.003–0.006 | 极低且稳定，说明 policy 变化很小 |
| loss | -0.31–0.28 | 在合理范围内波动 |
| steps skipped (all filtered) | 1 (step 30) | dynamic_sampling 全过滤时正确跳过 |
| gpu_peak (Head/rank0) | 1.76 GB | embed + front 4 层 + LoRA |
| gpu_peak (Tail/rank2) | 2.98 GB | tail 4 层 + norm + lm_head + LoRA + optimizer |
| gpu_peak (Middle/rank1) | 7.22 GB | middle 28 层（冻结，无梯度/optimizer） |
| overlong penalty | 支持 | `algorithm.overlong_penalty.{enable, buffer_len, penalty_factor}` |
| response length shaping | 不支持 | 需接入 verl 主仓库能力 |

### Checkpoint

```
checkpoints/latest/
  head_adapter_state.pt   4923454 bytes (Head LoRA 参数)
  head_optimizer.pt       9852926 bytes (Head AdamW 状态)
  tail_adapter_state.pt   4923454 bytes (Tail LoRA 参数)
  tail_optimizer.pt       9852926 bytes (Tail AdamW 状态)
  adapter_meta.json       108 bytes (模型路径 + 切层配置 + step)
```

save/load 验证：Head + Tail adapter state 各自匹配 ✅

---

## 启动命令

```bash
source /root/workspace/miniconda3/etc/profile.d/conda.sh && conda activate verl

# 默认 10 步 smoke test
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 \
  -m verl.experimental.split_demo_v4.main_split_v4

# 50 步完整记录
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 \
  -m verl.experimental.split_demo_v4.main_split_v4 \
  trainer.total_steps=50
```

---

## 与 v3 的对比

| 维度 | v3 | v4 |
|---|---|---|
| 进程数 | 2 (Head+Tail) | 3 (Head+Middle+Tail) |
| 引擎接口 | SplitEngine(BaseEngine) | SplitPipelineEngine(BaseEngine) |
| Rollout | 回传 full logits | Tail 本地采样 (O(B) 通信) |
| old_log_prob | infer_batch 重算 | rollout 时保存 |
| 训练循环 | main 内嵌 | SplitTrainer + BaseRolloutBackend |
| 代码抽象 | engine 层 | trainer + rollout + engine 三层 |
| topology | 硬编码 | 配置化 |
| checkpoint | 只有 Tail | Head + Tail 各自保存 |
| reward (50步 mean) | — | 0.235 |
| clipfrac (50步 mean) | — | 0.009 |
| approx_kl (50步 mean) | — | 0.0014 |
| gpu_peak (Head) | 2.55–2.62 GB | 1.76 GB |
| gpu_peak (Tail) | 2.55–2.62 GB | 2.65 GB |

> v3 是 2-stage（Head+Middle 绑在一起），v4 把 Middle 独立出去，Head 只保留 embed+front，所以 Head 显存从 ~2.6GB 降到 1.76GB。Tail 显存和 v3 基本一致。
