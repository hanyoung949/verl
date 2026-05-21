# split_demo_v4 路线图

> Phase A 已完成（协议稳定 + 通信优化）。B/C/D 为后续规划。

---

## Phase B：近期做（1 周内）

**目标**：把训练循环和通信协议解耦，抽出可替换的边界。

### B1. 抽取 SplitTrainer ✅

**现状**：`main_split_v4.py` 500+ 行，既管协议又管算法。

**修复**：把训练循环抽到 `SplitTrainer` 类，main 只负责组件组装。

```python
class SplitTrainer:
    def __init__(self, engine, rollout_backend, reward_fn, config): ...
    def fit(self, samples): ...
```

**收益**：后续换 backend（vLLM/FSDP）时，main 文件不动，只换组件。

### B2. 引入 BaseRolloutBackend 接口 ✅

**现状**：直接实例化 `SimplePipelineRollout`，无抽象层。

**修复**：

```python
class BaseRolloutBackend(ABC):
    @abstractmethod
    def generate(self, prompt_ids, attention_mask, group_size, max_new_tokens): ...
    @abstractmethod
    def run_tail_loop(self): ...
```

当前实现为 `SimplePipelineRollout`。后续 vLLM 接入时实现 `VLLMSplitRolloutBackend`。

**配置预留**：`rollout.backend: hf | vllm`

**收益**：Rollout 后端可插拔。

### B3. 控制消息状态机校验 ✅

**现状**：Tail 侧已加 `TRAIN_START` assert，Middle 转发层无 flag 校验。

**修复**：Middle 转发层增加非法 flag 拒绝（如收到未知 flag 时 raise）。

**收益**：协议错位时快速失败，不 hang。

---

## Phase C：之后做（2 周内）

**目标**：把 rank 硬编码改成可配置的 placement/topology。

### C1. placement / topology 配置（2 天）

**现状**：`rank==0` = Head, `rank==1` = Middle, `rank==2` = Tail，硬编码。

**修复**：

```yaml
split:
  topology:
    head:
      ranks: [0]
    middle:
      ranks: [1]   # 后续可扩展为 [1, 2, 3] 做 PP
    tail:
      ranks: [2]
```

用 `topology` 替代 `stage_map`，支持同 stage 多 rank、跨机、DP/PP 混合。

**收益**：为后续多机多卡扩展预留概念空间。

### C2. SplitPipelineEngine 对齐 verl BaseEngine（1 天）

**现状**：已有 `train_batch()` / `infer_batch()` / `save_checkpoint()` / `load_checkpoint()`，但缺少部分生命周期接口。

**修复**：补充 `train_mode()` / `eval_mode()` / `lr_scheduler_step()` / `get_data_parallel_size()` 等。

**收益**：后续可直接注册到 `EngineRegistry`，被 verl 主 trainer 复用。

### C3. Checkpoint 完整性（半天）

**现状**：`save_checkpoint()` 只存 Tail 的 trainable state + optimizer。

**修复**：补充 Head 的 trainable state + optimizer state 保存。

**收益**：Head LoRA 和 Tail LoRA 都能恢复。

---

## Phase D：远期再看（不实现，只规划）

| # | 能力 | 预留接口 | 当前占位 |
|---|------|----------|----------|
| D1 | vLLM Rollout | `BaseRolloutBackend` + `rollout.backend=vllm` | ✅ B2 已留接口 |
| D2 | FSDP2 训练 | `BaseEngine` + `engine.backend=fsdp` | C2 已对齐接口 |
| D3 | 多机 Middle PP | `topology` 支持多 rank | C1 已留配置格式 |
| D4 | 1F1B Schedule | `engine.train_batch()` 内部注入 microbatch | 接口已预留 |
| D5 | DAPO overlong penalty | `reward_fn` 可替换为 `DAPORewardManager` | B1 抽出 Trainer 后，换 reward_fn 即可 |

---

## DAPO / Dr.GRPO 能力声明

| 特性 | 当前状态 | 是否缺失 |
|------|----------|----------|
| 非对称 clip (`cliprange_low≠cliprange_high`) | ✅ | 否 |
| dynamic_sampling | ✅ | 否 |
| 固定 loss_scale_factor | ✅ | 否 |
| 不按 std 归一化 | ✅ | 否 |
| **Overlong penalty** | ❌ | **是** |
| **Response length shaping** | ❌ | **是** |

**Dr.GRPO 已完整支持**。

**DAPO 的表层配置已支持**，专有算法项（overlong/length shaping）需接入 verl 主仓库的 `DAPORewardManager` 才能补齐。作为远期增强项，不跟稳定性修复抢优先级。
