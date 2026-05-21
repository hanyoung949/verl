# split_demo_v4 工作记录

> 记录从 v4 baseline 到 Phase A 完成的全部修改、问题修复和优化。

---

## 核心协议修复

### 解除 Head/Tail old_log_prob 死锁
- Head 在 infer_batch 后等待 Tail 返回 logits，但 Tail 在某些路径下未发送 logits
- 统一协议：Head 发送 FWD_ONLY header → Middle 转发 → Tail 计算 logits → Tail 发送 logits 回 Head
- 删除 `_pipeline_forward` 中 Tail 侧的死分支

### STEP_SKIP / TRAIN_START 控制消息
- dynamic_sampling 全过滤时，Head 跳过训练但 Tail 仍在等待训练数据，导致协议错位
- 引入 `STEP_SKIP`：Head 通知 Tail 本 step 跳过训练
- 引入 `TRAIN_START`：Head 通知 Tail 开始接收训练数据
- Tail 侧增加 `assert ctrl_flag == TRAIN_START` 校验

### range(4)→range(2) 不匹配
- `update_epochs` 循环中 `range(4)` 与 config 不匹配
- 改为 `range(update_epochs)`

### 删除重复 GRPO 配置发送
- Head 在每个 mini-batch 都发送 GRPO 配置给 Tail
- 改为一次性发送完整训练数据 + GRPO 配置，mini-batch 只发 indices

### 统一 loss_agg_mode
- Head 和 Tail 的 loss_agg_mode 可能不一致
- Tail 从 config 读取，不再从 Head 接收

---

## 代码清理

- **删除 `_pipeline_forward` Tail 死分支**：Tail 永远不会走到 `_pipeline_forward`
- **抽取 `_tail_forward_logits()` helper**：减少协议修改时遗漏的风险
- **补充 HeadStage/TailStage `zero_grad()` / `step()`**：`train_batch()` 内调用，确保训练循环完整

---

## A1. vocab size 去硬编码

- `main_split_v4.py` Head 接收 logits：shape tensor 从 `torch.zeros(2)` 改为 `torch.zeros(3)`
- `main_split_v4.py` `_tail_forward_logits()` 发送 shape：增加 `logits.shape[2]`
- `pipeline_engine.py` `_head_forward_backward()`：同步改为 3 维 shape + 动态 vocab size

---

## 修复设计混乱：统一 Tail 端 infer_batch/train_batch 路径

**问题**：infer_batch 时 Tail 走手动 loop（`_tail_forward_logits`），train_batch 时 Tail 走 engine API（`_tail_forward_backward`）。Tail 端维护两套接收逻辑。

**修复**：
1. `pipeline_engine.py` `_tail_forward_backward`：
   - 导入 `PIPELINE_DONE`
   - 开头处理 `PIPELINE_DONE` header
   - `forward_only=True` 时发送 logits 回 Head（直接 P2P）
2. `main_split_v4.py` Phase 1/2 Tail loop：
   - Phase 1：`engine.infer_batch()` 替代手动 loop
   - Phase 2：`engine.forward_backward_batch(..., forward_only=True)` 替代手动 loop
3. `main_split_v4.py` `_pipeline_forward`：
   - Head 分支改为调用 `self.engine.infer_batch(data)`

---

## A2. Tail 本地采样

**目标**：通信量从 `O(B·S·vocab)` 降到 `O(B)`

**新增 `sample_next_token(data, temperature=1.0, pad_token_id=0)`**：
- **Head (`_head_sample`)**：发送 input + temperature/finished/pad_id → 接收 next_token + log_prob
- **Tail (`_tail_sample`)**：接收 activation + 采样参数 → `multinomial` 采样 → 计算 log_prob → 发送回 Head

**NCCL 死锁修复**：
- `finished_mask` 类型不匹配：Head 发 `int64`，Tail 原代码收 `bool` → Tail 收 `int64` 再转 `bool`
- `log_prob` dtype 不匹配：Tail 原计算返回 `bfloat16`，Head 收 `float32` → Tail 显式 `.float()`

---

## A3. Rollout 保存 old_log_prob

**目标**：每个 step 省一次完整前向计算

- `generate()` 每步保存 `step_log_prob`，rollout 结束 `out.old_log_probs` 直接可用
- Phase 2 Head 删除 `infer_batch` 重算，直接使用 `rollout_output.old_log_probs`

---

## 验证记录

- GPU 5,6,7 运行 10 steps 通过，无 NCCL timeout
- loss 正常波动（0.0002 → 0.2505 → -0.1260）
- reward 从 0.1250 涨到 0.3125
- dynamic_sampling 正常生效（valid 1~4/4）
