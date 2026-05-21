# Split Demo v4 — 3-Stage Pipeline 设计文档

**目标**：在 3 张 GPU 上验证 Head–Middle–Tail 三层拆分的 GRPO 训练通路，通信协议稳定，支持 Tail 本地采样。  
**范围**：`verl/experimental/split_demo_v4/`。  
**验证平台**：单机 3×GPU，`torchrun --nproc_per_node=3`。  
**与 v3 的关系**：v3 是 2-stage（rank0/rank1 硬编码），v4 是 3-stage，引入独立的 MiddleStage 和 StageTransport 抽象。

---

## 1. v3 的根本限制

v3 的 2-stage 设计把模型拆成"前段+尾段"，中间段（middle layers）和尾段绑在同一个 rank 上：

- rank0：embed + front + tail + norm + lm_head（ all in one ）
- rank1：middle layers（冻结，纯转发）

这导致 rank0 的显存压力依然很大，且无法独立扩展中间段。

v4 的目标是把 middle layers 独立成一个 **MiddleStage**，形成真正的 3-stage 流水线：

```
Head (rank0) → Middle (rank1) → Tail (rank2)
```

---

## 2. 进程分工

```
torchrun --nproc_per_node=3 -m verl.experimental.split_demo_v4.main_split_v4

Stage 0 (Head):   GPU 0                Stage 1 (Middle): GPU 1              Stage 2 (Tail): GPU 2
─────────────────────────────────     ──────────────────────────────       ──────────────────────────────
embed_tokens                            middle_layers[4:32]                  tail_layers[32:36]
front_layers[0:4]                       rotary_emb (本地重算)                 norm
LoRA (可训练)                           冻结                                 lm_head
                                                                             LoRA (可训练)
                                                                             optimizer
                                                                             loss / checkpoint
```

| Stage | Rank | 层 | 可训练 | 持有 optimizer |
|---|---|---|---|---|
| Head | 0 | embed + front[0:4] | 是 (LoRA) | 否 |
| Middle | 1 | middle[4:32] | 否 (冻结) | 否 |
| Tail | 2 | tail[32:36] + norm + lm_head | 是 (LoRA) | 是 |

Tail 是"主 stage"：持有 optimizer、计算 loss、保存 checkpoint、本地采样。

---

## 3. 核心架构

### 3.1 模块分层

```
main_split_v4.py (入口 + GRPO 训练循环)
    │
    ├── SplitPipelineEngine(BaseEngine)
    │       ├── initialize()             # 按 rank 创建对应 stage
    │       ├── forward_backward_batch() # Head/Middle/Tail 分发
    │       ├── train_batch()            # Tail: zero_grad → fwd/bwd → step
    │       ├── infer_batch()            # Head→Middle→Tail, Tail→Head logits
    │       ├── sample_next_token()      # A2: Tail 本地采样，回传 token+log_prob
    │       └── save/load_checkpoint()   # Tail 保存 adapter + optimizer
    │
    ├── SimplePipelineRollout            # 自回归生成（调用 engine.sample_next_token）
    ├── FunctionReward                   # 函数打分（math_exact_match）
    └── make_grpo_loss_fn()              # GRPO loss（clip / loss_agg_mode）
```

### 3.2 Stage 抽象

```python
class Stage(nn.Module):
    def __init__(self, stage_id, layers, device, trainable=False)
    def forward(self, h, **kwargs)
    def trainable_parameters(self)

class HeadStage(Stage):
    def __init__(self, embed_tokens, layers, rotary_emb, device, ...)
    def forward_input(self, input_ids, attention_mask)  # 返回 (h, pos_ids, pos_emb, mask)

class MiddleStage(Stage):
    def run(self)  # 无限循环：recv header → 转发 → fwd → send

class TailStage(Stage):
    def __init__(self, layers, norm, lm_head, device, lr, clip_grad, ...)
    def forward_output(self, h, position_ids, position_embeddings)  # 返回 logits
    def zero_grad(self) / step(self)
```

---

## 4. 通信协议

### 4.1 Forward 路径

```
Head → Middle:  header(6-int) + h_front + position_ids + [mask]
Middle → Tail:  header(6-int) + h_middle + position_ids
Tail → Head:    next_token [B,1] + log_prob [B,1]   (A2: 直接 P2P，不回传 full logits)
```

### 4.2 Backward 路径（训练时）

```
Tail → Middle:  grad_h_middle
Middle → Head:  grad_h_front  (Middle 本地做 backward，不保存 grad)
```

### 4.3 协议常量

```python
FWD_ONLY      = 0   # 推理模式（rollout / old_log_prob）
FWD_WITH_BWD  = 1   # 训练模式
SHUTDOWN      = 2   # Middle run() 退出
PIPELINE_DONE = 3   # 本次 pipeline 结束
TRAIN_MB      = 4   # 开始一个 mini-batch
TRAIN_DONE    = 5   # 训练结束
STEP_SKIP     = 6   # 本 step 跳过（dynamic_sampling 全过滤）
TRAIN_START   = 7   # 开始接收训练数据
```

---

## 5. 关键优化（Phase A）

### A1. vocab size 去硬编码
- Head 接收 logits shape 从 `torch.zeros(2)` 改为 `torch.zeros(3)`
- vocab size 从硬编码 `151936` 改为 `result_shape[2].item()`
- 换模型时不会炸

### A2. Tail 本地采样
- 新增 `engine.sample_next_token(data, temperature, pad_token_id)`
- Tail 计算 `last_logits = logits[:, -1, :]`，本地 `multinomial` 采样
- 只回传 `next_token [B,1]` + `log_prob [B,1]`
- 通信量从 `O(B·S·vocab)`（~400MB/步）降到 `O(B)`（~64 bytes/步）

### A3. Rollout 保存 old_log_prob
- `generate()` 每步保存 `step_log_prob`
- rollout 结束后 `out.old_log_probs` 直接可用
- Phase 2 删除 `infer_batch` 重算，每个 step 省一次完整前向

---

## 6. 训练循环

```
for step in range(total_steps):
    # Phase 1: Rollout
    Head: rollout_output = rollout.generate(prompt_ids, ...)
    Head: send PIPELINE_DONE
    Tail: while True: engine.sample_next_token() → break on PIPELINE_DONE

    # Phase 2: Reward + old_log_prob + Advantage + Train
    Head: sync rollout_output (response_ids, response_mask, prompt_len) to Tail
    Head: old_log_prob = rollout_output.old_log_probs      # A3: 无需重算
    Head: send PIPELINE_DONE
    Tail: while True: engine.forward_backward_batch() → break on PIPELINE_DONE

    Head: rewards = reward_fn.score(sequences, ...)
    Head: advantages = compute_grpo_outcome_advantage(...)
    Head: dynamic_sampling → valid_seq

    Head: send TRAIN_START → send (resp_ids, resp_mask, old_lp, advantages) → send GRPO cfg
    Tail: recv train data → make_grpo_loss_fn

    for epoch in range(update_epochs):
        for mb in minibatches:
            Head: send TRAIN_MB + mb indices
            Head: engine.train_batch(mb_data, grpo_loss_fn)
            Tail: engine.train_batch() → _tail_forward_backward → backward → step
            Tail: send loss/metrics back to Head

    Head: send TRAIN_DONE
```

---

## 7. 已知问题与限制

1. **Head 不持有 optimizer**：当前只有 Tail 保存 checkpoint，Head 的 LoRA 状态在 checkpoint 中缺失（C3 修复）。
2. **rank 硬编码**：`rank==0=Head`, `rank==1=Middle`, `rank==2=Tail`（C1 修复）。
3. **Middle 无非法 flag 校验**：收到未知 flag 时静默转发，可能 hang（B3 修复）。
4. **NCCL P2P 无超时**：当前依赖默认 60s timeout，未配置化（B 阶段考虑）。

---

## 8. 验证方式

```bash
# 3 GPU 运行
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 \
  -m verl.experimental.split_demo_v4.main_split_v4
```

smoke test 通过标准：10 steps 无 NCCL timeout，loss 正常波动，reward 有变化。
