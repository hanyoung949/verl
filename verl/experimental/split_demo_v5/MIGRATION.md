# split_demo_v5 状态与计划

## 一、当前完成状态

### 已完成（v3 对齐）
| 步骤 | 实现方式 | 状态 |
|------|---------|------|
| Rollout | 3-stage pipeline（Head→Middle→Tail），Tail→Head 直接回传 logits | ✓ |
| old_log_prob | Rollout 时保存 logits，直接计算（消除重前向） | ✓ |
| Reward | Head 本地 FunctionReward.score() | ✓ |
| token_level_scores | 与 v3 一致 | ✓ |
| dynamic_sampling | 与 v3 一致 | ✓ |
| advantage | compute_grpo_outcome_advantage | ✓ |
| update_epochs × mini_batch | 与 v3 一致 | ✓ |
| backward | FWD_WITH_BWD: Head→Middle→Tail，Tail.backward→Middle.backward→Head.backward | ✓（单独测试通过） |
| Head optimizer.step | AdamW | ✓ |
| Tail optimizer.step | AdamW | ✓ |
| multi-step + 干净退出 | SHUTDOWN 协议 | ✓ |

### 验证结果
- 2 steps × 2 epochs × mini_batch=4: 跑通（infer_batch 路径）
- 3 steps × 2 epochs × mini_batch=8: 跑通（infer_batch 路径）
- backward + optimizer step: 单独测试通过

---

## 二、当前阻塞问题 — 已修复

### 问题描述（已解决）
将 `infer_batch` 替换为 `train_batch` 后，训练循环崩溃。

### 根因（已解决）
Head 同时通过两条路径给 Tail 发东西：
- dist.send: 训练数据（Head→Tail 直连）
- transport.send: FWD_WITH_BWD header（Head→Middle→Tail）

Tail 先收 Middle header，再收 Head 数据，导致消息错位。

### 修复方案（已实施）
添加 TRAIN_MB / TRAIN_DONE 控制消息，统一接收顺序：

```
Head 端（每个 mini-batch）:
  1. dist.send(TRAIN_MB header) → Tail
  2. dist.send(训练数据 tensors) → Tail
  3. engine.train_batch() → transport.send(FWD_WITH_BWD) → Middle

Tail 端（训练循环）:
  while True:
    1. dist.recv(ctrl_header) from Head → TRAIN_MB or TRAIN_DONE
    2. if TRAIN_MB: dist.recv(训练数据) from Head
    3. dist.recv(header) from Middle → FWD_WITH_BWD
    4. recv activation from Middle
    5. GRPO loss + backward + optimizer step
    6. send grad → Middle
  if TRAIN_DONE: break
```

关键改动：
- Tail 先收 Head 的控制消息，再收 Middle 的 header
- 用 TRAIN_MB / TRAIN_DONE 明确区分训练状态
- 删除了多余的 PIPELINE_DONE 通知

### 验证结果
- 3 steps × 2 epochs × mini_batch=8: 跑通 ✓
- multi-step: 跑通 ✓
- 干净退出: ✓

---

## 三、第二轮修复 — 通信协议死锁与状态机清理

### 问题描述
在代码审查和实际运行中发现以下严重问题：

1. **Head/Tail old_log_prob 阶段必死锁**
   - Head 发送 rollout results 后先 `recv(tail_ready)`，再调用 `infer_batch`
   - Tail 先进入 old_log_prob forward loop，完成后才发送 ready
   - 结果：Head 等 Tail ready，Tail 等 Head 的 `infer_batch` → 互等死锁

2. **`dynamic_sampling` 全过滤时破坏协议**
   - Head 直接 `continue` 跳过当前 step，但 Tail 已完成 old_log_prob loop 并等待训练数据
   - 导致 Tail 把下一轮 rollout 消息误读为训练数据，协议错位

3. **GRPO 配置重复发送**
   - Head 在 Phase 2e 中发送了两遍 GRPO 配置，Tail 只接收一遍

4. **Tail 接收 rollout results 时 `range(4)` 与 Head 实际发送 2 个 tensor 不匹配**
   - 原代码 Head 发送 `response_ids + response_mask`，Tail 却 `for _ in range(4)`

5. **`loss_agg_mode` 在 Tail 被硬编码为 `"token-mean"`**
   - 与 Head 读取的 `config.algorithm.loss_agg_mode` 不一致

6. **Tail 前向逻辑复制三份，维护困难**
   - `SimplePipelineRollout._pipeline_forward`、Phase 1 rollout loop、Phase 2 old_log_prob loop 代码重复

### 修复方案（已实施）

1. **解除 old_log_prob 死锁**
   - Tail：接收 rollout results 后立即发送 ready，再进入 old_log_prob loop
   - Head：发送 rollout results 后 `recv(tail_ready)`，再调用 `infer_batch`

2. **引入 `STEP_SKIP` 控制消息**
   - `transport.py` 新增 `STEP_SKIP = 6` 和 `TRAIN_START = 7`
   - Head 在 `dynamic_sampling` 全过滤时发送 `STEP_SKIP`，Tail 收到后直接 `continue`
   - Head 在正常训练前发送 `TRAIN_START`，Tail 再进入数据接收

3. **删除重复 GRPO 配置发送**
   - 删除 Head Phase 2e 中第二遍 `grpo_cfg` 发送

4. **修复 `range(4)` → `range(2)`**
   - 与 Head 实际发送的 tensor 数量对齐

5. **统一 `loss_agg_mode`**
   - Tail 直接使用 `str(config.algorithm.loss_agg_mode)`

6. **抽取 `_tail_forward_logits` helper**
   - 将 Tail 从 Middle 接收 activation → rotary_emb → forward_output → 发送 logits 的重复逻辑抽取为独立函数

7. **统一 `engine.train_batch()` 调用语义**
   - `core/stage.py`：为 `HeadStage` / `TailStage` 补充 `.zero_grad()` 和 `.step()`（含 grad clipping）
   - `core/pipeline_engine.py`：`train_batch()` 在 Head 和 Tail 都统一调用 `zero_grad()` + `step()`
   - Head 和 Tail 都调用 `engine.train_batch()`，Tail 将 loss/metrics 回传 Head

### 验证结果
- 10 steps × 1 epoch × mini_batch=8（完整配置）: 跑通 ✓
- dynamic_sampling skip 路径: 跑通 ✓（step 10 全过滤正常 skip）
- multi-step + 干净退出: ✓

---

## 四、算法能力声明（与 v3 对齐）

v4 的算法实现与 v1/v2/v3 完全一致，支持以下配置组合（即 v3 文档所称的 "GRPO + DAPO/Dr.GRPO 风格"）：

| 特性 | 配置项 | 说明 |
|------|--------|------|
| GRPO advantage | `compute_grpo_outcome_advantage` | 组内相对 advantage，无需 critic |
| Dr.GRPO | `norm_adv_by_std_in_grpo: false` | 不按 std 归一化 |
| Dr.GRPO | `loss_agg_mode: seq-mean-token-sum-norm` | 序列级别 loss 聚合 |
| Dr.GRPO | `loss_scale_factor: 128` | 固定缩放因子 |
| DAPO clip-higher | `cliprange_low: 0.2, cliprange_high: 0.28` | 非对称裁剪 |
| DAPO dynamic sampling | `dynamic_sampling: true` | 过滤零方差 group |

**注意**：当前实现为标准 GRPO + 上述配置项。若需严谨宣称"完整 DAPO/Dr.GRPO"，还需逐条对照论文实现专有逻辑（如 overlong filtering/penalty、特定 token-level loss 细节等）。

---

## 五、待优化项（暂不实施）

1. **old_log_prob 重新前向**：rollout 时保存 sampled token 的 old_log_prob，消除 `infer_batch` 重前向
2. **Tail 本地 sample**：rollout 时 Tail 不再回传 full logits `[B,S,vocab]`，只回传 `next_token` + `done_mask`
3. **Reward / Advantage 下沉到 Tail**：减少训练数据同步
4. **硬编码 vocab size**：当前写死 151936，应从 `lm_head.out_features` 获取

---

## 五、文件清单

| 文件 | 修改内容 |
|------|---------|
| `config/split_demo_v5.yaml` | 模型路径修正 |
| `core/pipeline_engine.py` | 设备管理、Tail transport/rotary_emb、Head optimizer |
| `core/stage.py` | HeadStage optimizer、TailStage rotary_emb + position_embeddings |
| `core/transport.py` | PIPELINE_DONE 常量 |
| `core/middle_stage.py` | Head→Tail 转发（FWD_ONLY + FWD_WITH_BWD） |
| `main_split_v5.py` | 完整训练循环（rollout→reward→advantage→train）；删除 `_pipeline_forward` Tail 死分支；添加 `STEP_SKIP`/`TRAIN_START` 校验 |
