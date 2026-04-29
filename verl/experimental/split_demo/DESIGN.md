# Split Actor GRPO Demo — 设计文档

**目标**：在单机 2×4090 上验证"前段–中段–尾段"三层拆分 Actor 的 GRPO 训练通路。  
**范围**：`verl/experimental/split_demo/`，不修改任何主线代码。  
**支持模型**：仅限标准 HF Qwen2.5 / Llama3，不使用 verl 的 Ulysses 注意力补丁。  
**设计原则**：核心执行层可复用，rollout / reward / transport 均为可替换策略层。

### 运行约定

- **模型优先从 `~/share` 选择**：如果 `~/share` 下已有可用模型，优先使用本地路径，不主动重新下载。
- **禁止删除 `~/share` 内容**：`~/share` 被视为共享资源目录，demo 相关操作只读不删。
- **每次关键边界变化都同步更新文档**：例如依赖链收缩、模型来源策略、rollout / reward / transport 的职责变化，都要及时写回本设计文档。
- **入口统一解析本地模型路径**：`main_grpo_split.py` 会先对 `model.path` 做 `expanduser + to_absolute_path`。若本地路径存在，则按本地目录加载；仅在本地不存在时，才把它当作 Hugging Face repo id 处理。

### `smoke test` 是什么

这里的 `smoke test` 指的是**最小可运行验证**，目的不是看训练效果，而是尽快确认最基础的链路没断：

1. 模型能否成功加载
2. tokenizer / 配置 / LoRA 能否正常初始化
3. `SplitActorCore.forward_full()` 能否在双卡上跑通
4. `loss.backward()` 后 front / tail 的可训练参数是否真的有梯度
5. rollout / reward / GRPO advantage 的 tensor shape 是否对齐

也就是说，`smoke test` 主要回答的是：

> “这套 demo 在当前环境里，最基本的 forward / backward / update 能不能活着跑起来？”

它**不**回答这些问题：

- 训练效果好不好
- reward 曲线漂不漂亮
- 速度够不够快
- 是否已经适合切到 7B / vLLM / speculative

---

### 首选本地模型

结合当前 `~/share` 里的现成模型，demo 的**首个 smoke test 默认模型**定为：

```text
~/share/Qwen2.5-0.5B-Instruct
```

选择它的原因：

- 已经存在于本地 `~/share`
- Qwen2.5 架构和我们当前实现目标一致
- 体量小，适合快速暴露 split / RoPE / PEFT / device / shape 的真实问题
- 比 1.5B / 3B / 7B 更适合做第一轮链路验证

建议的使用顺序：

1. **第一轮 smoke test**：`~/share/Qwen2.5-0.5B-Instruct`
2. **第二轮更接近目标**：`~/share/qwen2.5-1.5B-Instruct`
3. **确认显存和逻辑稳定后**：`~/share/Qwen2.5-3B-Instruct` 或 `~/share/Qwen2.5-7B-Instruct`

### 当前目标切层场景

当前默认目标场景已经调整为：

```text
model = ~/share/Qwen2.5-3B-Instruct
```

业务假设是：

> **边侧算力受限，因此前段和尾段都应尽量小，中段尽量大。**

需要注意的一点：

- `Qwen2.5-3B` 的 `num_hidden_layers = 36`
- 因此它**不能精确做** `4:24:4`
- 在这个模型上，最接近该意图的实际切法是：

```text
4 : 28 : 4
```

即：

- front = 4 层
- middle = 28 层
- tail = 4 层

所以默认配置已经按这个切法设置，而不是继续沿用之前偏居中的 `8:8:余下` 风格。

---

## 1. 整体架构

```
GPU 0                          GPU 1
┌──────────────────────┐       ┌─────────────────────┐
│  Embedding           │       │                     │
│  Front layers 0..N-1 │──H0──▶│  Middle layers N..M │
│  + LoRA A (训练)     │       │  (全部参数冻结)      │
│                      │◀─grad─│                     │
└──────────────────────┘       └──────────┬──────────┘
         ▲                               │ H1
         │ grad                          ▼
┌──────────────────────┐       ┌─────────────────────┐
│  Tail layers M+1..L  │◀─H1───┘  (经 MiddleExecutor) │
│  + LoRA B (训练)     │
│  LM Head             │
└──────────────────────┘

优化器：AdamW，只包含 LoRA A + LoRA B 参数
```

### 层次划分

```
SplitGRPOTrainer              ← 训练编排，不直接做跨卡操作
    │
    ├── SplitActorCore        ← 模型执行核心（长期资产，接口稳定）
    │       └── MiddleExecutor    ← 跨段传输与中段执行（可替换）
    │
    ├── SplitRolloutBackend   ← 生成策略（可替换）
    │       NaiveSplitRollout / CachedSplitRollout / SpeculativeSplitRollout
    │
    └── RewardAdapter         ← 打分策略（可替换）
            FunctionReward / ModelReward
```

---

## 2. 文件结构

```
verl/experimental/split_demo/
├── DESIGN.md
├── __init__.py
├── config/
│   └── split_demo.yaml
├── core/
│   ├── __init__.py
│   ├── split_actor.py        ← SplitActorCore
│   └── middle_executor.py    ← MiddleExecutor 及 LocalMiddleExecutor
├── rollout/
│   ├── __init__.py
│   ├── base.py               ← SplitRolloutBackend（抽象基类）
│   └── naive.py              ← NaiveSplitRollout
├── reward/
│   ├── __init__.py
│   ├── base.py               ← RewardAdapter（抽象基类）
│   └── function_reward.py    ← FunctionReward
├── sample_data/
│   └── train.jsonl           ← 仓库内最小样本，供入口级 smoke test 直接使用
├── split_trainer.py          ← SplitGRPOTrainer
└── main_grpo_split.py        ← 入口
```

---

## 3. 各模块设计

### 3.1 `core/middle_executor.py` — `MiddleExecutor`

这是 demo 里**最重要的可替换接口**，隔离"中段执行在哪里、怎么传"：

```python
class MiddleExecutor(ABC):
    @abstractmethod
    def execute(
        self,
        hidden_states:       Tensor,                      # from front, on cuda:0
        position_ids:        Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask:      Optional[Tensor],
    ) -> Tensor:
        """执行中段层，返回 hidden_states（设备由实现决定，SplitActorCore 负责搬回 cuda:0）"""
```

**当前实现 `LocalMiddleExecutor`**：

```python
class LocalMiddleExecutor(MiddleExecutor):
    def __init__(self, middle_layers: nn.ModuleList, device: str = 'cuda:1'):
        self.middle_layers = middle_layers
        self.device = device

    def execute(self, hidden_states, position_ids, position_embeddings, attention_mask):
        h = hidden_states.to(self.device)                    # 保留计算图，grad 可回传
        pos_emb = (position_embeddings[0].to(self.device),
                   position_embeddings[1].to(self.device))
        # attention_mask 的约定：
        # 1. 当前 demo 的生成路径是 padding-free，调用方直接传 None，
        #    让 HF/SDPA 走 is_causal=True 的默认因果掩码逻辑。
        # 2. 如果未来接入带 padding 的 batch，则由 SplitActorCore 统一构造
        #    好 attention_mask（通常是显式 4D causal mask）后再透传进来。
        # 3. MiddleExecutor 本身不负责生成 mask，只负责“搬运 + 执行中段”。
        for layer in self.middle_layers:
            h = layer(h, attention_mask=attention_mask,
                      position_embeddings=pos_emb)[0]
        return h                                              # 仍在 self.device
```

**未来可替换为**：

| 实现 | 场景 |
|------|------|
| `LocalMiddleExecutor` | 单机双卡，当前 demo |
| `RPCMiddleExecutor` | 云端中间层，gRPC 发送/接收 hidden states |
| `QueueMiddleExecutor` | 进程间 IPC |

> **梯度流**：`tensor.to(device)` 是可微操作，backward 自动从 cuda:1 回传到 cuda:0。Middle 层 `requires_grad=False` 只阻断参数梯度，不截断激活梯度。

---

### 3.2 `core/split_actor.py` — `SplitActorCore`

模型执行核心，对 rollout 和 trainer 暴露稳定接口，内部实现可演化（加 KV cache、改推理路径）。

#### 构造流程

```
AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=bfloat16)
    │
    ├── 对完整模型应用 PEFT LoRA（target_modules = q_proj, v_proj）
    │   └── middle 层也会被加上 LoRA adapter
    │
    ├── 保存 canonical rotary_emb：
    │   优先取 decoder/model 级别的 rotary_emb
    │   （当前 transformers 5.6 下的 Qwen2.5 是 model.rotary_emb）
    │   若不存在，再回退到 layer[0].self_attn.rotary_emb
    │
    ├── 拆分 layers：
    │   self.front_layers = model.model.layers[0:front_end]    → cuda:0
    │   middle_layers     = model.model.layers[front_end:middle_end]  （交给 MiddleExecutor）
    │   self.tail_layers  = model.model.layers[middle_end:]    → cuda:0
    │   self.embed_tokens, self.norm, self.lm_head             → cuda:0
    │
    ├── 冻结 middle（base + LoRA adapter 一起冻结）
    │   for p in middle_layers.parameters(): p.requires_grad = False
    │
    └── self.middle_executor = LocalMiddleExecutor(middle_layers, device='cuda:1')
```

> **为什么先 LoRA 再拆**：PEFT `get_peft_model` 需要完整 HF 模型，不能作用于子 ModuleList。先应用再拆分，middle 的 LoRA 参数随 base 权重一起冻结。

#### 接口

```python
class SplitActorCore(nn.Module):

    # ── 训练接口 ───────────────────────────────────────────────────────

    def forward_full(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """完整前向，返回 logits [B, S, vocab] on cuda:0。训练时计算 log_prob 用。"""

    # ── 生成接口（rollout 层调用） ─────────────────────────────────────

    def prefill(self, prompt_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        处理完整 prompt，返回 logits [B, S, vocab]。
        Naive 模式 = forward_full。
        KV Cache 模式在此初始化缓存，返回最后位置的 logits。
        """

    def decode_step(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        单步 decode，返回 next-token logits [B, vocab]。
        Naive 模式 = forward_full(...)[:, -1, :]（重算完整序列）。
        KV Cache 模式只处理新 token，利用已缓存的 K/V。
        """

    # ── 内部辅助 ──────────────────────────────────────────────────────

    def _prepare_inputs(self, input_ids, attention_mask):
        """计算 position_ids 和 position_embeddings=(cos,sin)，全在 cuda:0。"""

    def trainable_parameters(self) -> list[nn.Parameter]:
        """返回 front + tail 中 requires_grad=True 的参数（LoRA A/B）。"""
```

#### HF 原生 Decoder Layer 调用约定

```python
# position_embeddings 预计算一次，各段复用
# 注意：rotary_emb 的挂载位置依 transformers 版本而变，
# 当前实现会优先取 decoder/model 级 rotary_emb。
pos_ids = torch.arange(seq_len).unsqueeze(0).expand(B, -1).to('cuda:0')
dummy_h = torch.zeros(B, seq_len, hidden_size, device='cuda:0', dtype=model.dtype)
pos_emb = self._get_rotary_emb()(dummy_h, pos_ids)     # (cos, sin) on cuda:0

# 标准 HF decoder layer 调用（无 verl Ulysses 补丁）
h = layer(h,
          attention_mask=None,           # padding-free → SDPA is_causal=True 自动处理
          position_embeddings=pos_emb,   # 显式传入，跳过层内 self.rotary_emb 调用
          use_cache=False)[0]

# Middle 段通过 MiddleExecutor.execute() 统一处理 .to(device)
h_mid = self.middle_executor.execute(h_front, pos_ids, pos_emb, attention_mask=None)
h_tail = h_mid.to('cuda:0')             # 搬回后交给 tail
```

---

### 3.3 `rollout/` — `SplitRolloutBackend`

生成策略层，与模型执行解耦。

#### 抽象基类（`rollout/base.py`）

```python
@dataclass
class RolloutOutput:
    sequences:      Tensor          # [B*G, prompt_len+response_len]
    attention_mask: Tensor          # [B*G, prompt_len+response_len]
    response_ids:   Tensor          # [B*G, response_len]
    response_mask:  Tensor          # [B*G, response_len]
    prompt_len:     int
    # 以下字段供未来扩展，当前实现填 None
    draft_acceptance_rate: Optional[float] = None
    kv_cache:              Optional[Any]   = None

class SplitRolloutBackend(ABC):
    @abstractmethod
    def generate(
        self,
        prompt_ids:     Tensor,
        group_size:     int,
        max_new_tokens: int,
    ) -> RolloutOutput: ...
```

> **为什么用 dataclass 而非 dict**：字段名访问比字符串 key 更安全，新增字段加 `Optional` 不破坏旧调用方。

#### 当前实现 `NaiveSplitRollout`（`rollout/naive.py`）

调用 `SplitActorCore.prefill / decode_step`，不使用 KV cache：

```
1. 将 prompt 复制 G 份 → input_ids [B*G, prompt_len]
2. prefill(input_ids, attn_mask) → 取 logits[:, -1, :] 采样第一个 token
3. 逐 token 循环（≤ max_new_tokens）：
   a. decode_step(current_ids, attn_mask) → next_logits [B*G, vocab]
   b. temperature + top_p 采样 → next_token [B*G, 1]
   c. 追加到序列，更新 attn_mask
   d. EOS 检测，全批次停止时退出
4. 构造 response_mask（含 EOS，之后为 0）
5. 返回 RolloutOutput
```

**未来替换路径**：

| 实现 | 接入方式 |
|------|---------|
| `NaiveSplitRollout` | 当前，无 KV cache |
| `CachedSplitRollout` | `SplitActorCore` 加 `init_kv_cache` 后替换，Trainer 不感知 |
| `SpeculativeSplitRollout` | 内部持有 Draft Model，用 `decode_step` 做 verification |

---

### 3.4 `reward/` — `RewardAdapter`

打分策略层，Trainer 只依赖抽象基类。

#### 抽象基类（`reward/base.py`）

```python
class RewardAdapter(ABC):
    @abstractmethod
    def score(
        self,
        sequences:      Tensor,     # [B*G, seq_len]
        attention_mask: Tensor,     # [B*G, seq_len]
        metadata:       dict,       # 附加信息（如 answer, prompt_len）
    ) -> Tensor: ...                # [B*G]，每条序列一个标量分数
```

#### 当前实现 `FunctionReward`（`reward/function_reward.py`）

```python
class FunctionReward(RewardAdapter):
    def __init__(self, fn: Callable, tokenizer): ...

    def score(self, sequences, attention_mask, metadata):
        prompt_len = metadata['prompt_len']
        # 只 decode response 部分，避免 prompt 内容干扰 reward 逻辑。
        response_ids = sequences[:, prompt_len:]
        texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        return torch.tensor([self.fn(text, metadata) for text in texts],
                            dtype=torch.float32, device=sequences.device)

# 示例规则函数：只看 response 文本，不看 prompt
def math_exact_match(text: str, meta: dict) -> float:
    return 1.0 if extract_last_number(text) == meta['answer'] else 0.0
```

**未来替换路径**：

| 实现 | 接入方式 |
|------|---------|
| `FunctionReward` | 当前 |
| `ModelReward` | 持有独立 RM 模型（可放 GPU 1 空余显存），`score()` 内部调用模型 forward |

---

### 3.5 `split_trainer.py` — `SplitGRPOTrainer`

训练编排层，只依赖上述四个抽象接口，不直接做跨卡操作。

#### 核心依赖（直接 import）

| 功能 | 来源 |
|------|------|
| 数据容器 | `verl.DataProto` |
| GRPO advantage | `verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage` |
| Loss 聚合 | `verl.trainer.ppo.core_algos.agg_loss` |
| Log probs | `verl.utils.torch_functional.logprobs_from_logits` |

> **为什么不用 `compute_policy_loss`**：该函数不接受 `loss_scale_factor`，在 `seq-mean-token-sum-norm` 模式下会 fallback 到用当前 batch 的 `loss_mask.shape[-1]` 做归一化，偏离 Dr.GRPO 的固定常数要求。改为手算 `pg_losses` 矩阵后直接调 `agg_loss(..., loss_scale_factor=...)`。
>
> **为什么不再从 `ray_trainer` 导入 `compute_advantage`**：虽然 `ray_trainer.compute_advantage` 表面上只是一个小辅助函数，但导入它会把 checkpoint / engine / fsdp 等主线重依赖整串拉进来，污染 demo 的故障域。当前 demo 已改为直接调用 `core_algos.compute_grpo_outcome_advantage`，只保留自己真正需要的 GRPO 逻辑。

#### Tensor 维度约定（贯穿训练循环）

| 张量 | shape | 说明 |
|------|-------|------|
| `sequences` | `[B*G, prompt_len+response_len]` | 全序列 |
| `attention_mask` | `[B*G, prompt_len+response_len]` | 全序列有效 token |
| `response_ids` | `[B*G, response_len]` | 仅 response token ids |
| `response_mask` | `[B*G, response_len]` | 仅 response 有效 token |
| `token_level_scores` | `[B*G, response_len]` | reward 仅在 EOS 位置，其余 0 |
| `old_log_prob` | `[B*G, response_len]` | response 每个位置的 log prob |
| `advantages` | `[B*G, response_len]` | compute_advantage 输出 |

**所有 response-only 张量在 response 维度对齐（不可混用全序列 mask）。**

#### 训练循环（单次迭代）

```
Step 1  Generate
        out: RolloutOutput = rollout.generate(prompt_ids, G, max_new_tokens)

Step 2  Compute old_log_prob  (torch.no_grad)
        logits_old = model.forward_full(out.sequences, out.attention_mask)
        old_log_prob = logprobs_from_logits(
            logits_old[:, prompt_len-1:-1, :],    # [B*G, response_len, vocab]
            out.response_ids                       # [B*G, response_len]
        )   # → [B*G, response_len]

Step 3  Reward
        rewards = reward.score(out.sequences, out.attention_mask, metadata)   # [B*G]
        token_level_scores = torch.zeros_like(out.response_mask, dtype=float)
        last_valid = out.response_mask.sum(dim=1) - 1
        token_level_scores[range(B*G), last_valid] = rewards

Step 4  Dynamic Sampling（DAPO）
        rewards_grouped = rewards.view(B, G)
        valid_groups = rewards_grouped.std(dim=1) > 1e-6
        if valid_groups.sum() == 0: continue
        valid_seq = valid_groups.unsqueeze(1).expand(B, G).reshape(-1)
        # 过滤所有 response-level 张量（保持对齐）
        sequences_valid     = out.sequences[valid_seq]
        attn_mask_valid     = out.attention_mask[valid_seq]
        response_ids_valid  = out.response_ids[valid_seq]
        old_lp_valid        = old_log_prob[valid_seq]
        token_scores_valid  = token_level_scores[valid_seq]
        resp_mask_valid     = out.response_mask[valid_seq]
        uid_valid           = uid_array[valid_seq]

Step 5  Advantage（复用 verl）
        advantages_valid, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_scores_valid,
            response_mask=resp_mask_valid,
            index=uid_valid,
            norm_adv_by_std_in_grpo=False,   # Dr.GRPO
        )

Step 6  Policy Update（mini-batch）
        for mb in chunk(range(N), mini_batch_size):
            logits_new = model.forward_full(sequences_valid[mb], attn_mask_valid[mb])
            log_prob_new = logprobs_from_logits(
                logits_new[:, prompt_len-1:-1, :], response_ids_valid[mb]
            )   # [mb, response_len]

            neg_kl    = torch.clamp(log_prob_new - old_lp_valid[mb], -20, 20)
            ratio     = neg_kl.exp()
            approx_kl = masked_mean(-neg_kl, resp_mask_valid[mb])
            clipped   = (ratio.detach() < 1 - 0.2) | (ratio.detach() > 1 + 0.28)
            clipfrac  = masked_mean(clipped.float(), resp_mask_valid[mb])

            adv = advantages_valid[mb]
            pg_losses = torch.maximum(-adv * ratio,
                                      -adv * ratio.clamp(1 - 0.2, 1 + 0.28))

            loss = agg_loss(
                loss_mat=pg_losses,
                loss_mask=resp_mask_valid[mb],
                loss_agg_mode='seq-mean-token-sum-norm',
                loss_scale_factor=config.algorithm.loss_scale_factor,   # 固定常数
            )
            loss.backward()
            clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            optimizer.step(); optimizer.zero_grad()

Step 7  Logging
        log(loss, rewards[valid_seq].mean(), clipfrac, approx_kl,
            valid_groups.float().mean())
```

---

### 3.6 `config/split_demo.yaml`

```yaml
model:
  path: "~/share/Qwen2.5-3B-Instruct"

data:
  train_file: "verl/experimental/split_demo/sample_data/train.jsonl"
  max_prompt_length: 256

split:
  # Qwen2.5-3B: 36 layers total → 4:28:4（边侧轻、中心重）
  front_end: 4   # layers[0:4]   on cuda:0
  middle_end: 32  # layers[4:32]  on cuda:1（冻结）

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.0
  target_modules: ["q_proj", "v_proj"]

rollout:
  group_size: 4
  max_new_tokens: 128
  temperature: 1.0
  top_p: 1.0

algorithm:
  norm_adv_by_std_in_grpo: false
  cliprange_low: 0.2
  cliprange_high: 0.28
  loss_agg_mode: "seq-mean-token-sum-norm"
  loss_scale_factor: 128   # 固定 == max_new_tokens，Dr.GRPO 要求
  dynamic_sampling: true
  update_epochs: 1
  mini_batch_size: 8

trainer:
  train_batch_size: 4
  total_steps: 200
  lr: 1.0e-4
  max_grad_norm: 1.0
  log_freq: 10
  log_gpu_memory: true
  log_transport: true
  save_dir: "verl/experimental/split_demo/checkpoints"
  save_latest: true
  seed: 42
```

---

### 3.7 `main_grpo_split.py`

```python
@hydra.main(config_path="config", config_name="split_demo", version_base=None)
def main(config):
    set_seed(int(config.trainer.seed))

    model_path = resolve_model_path(str(config.model.path))   # expanduser + 本地优先
    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    samples    = load_jsonl_samples(config.data.train_file)   # to_absolute_path 内部处理

    lora_config = LoraConfig(r=config.lora.r, lora_alpha=config.lora.lora_alpha, ...)
    model   = SplitActorCore(model_path, front_end=config.split.front_end,
                             middle_end=config.split.middle_end, lora_config=lora_config)
    rollout = NaiveSplitRollout(model, tokenizer, temperature=config.rollout.temperature, ...)
    reward  = FunctionReward(fn=math_exact_match, tokenizer=tokenizer)
    trainer = SplitGRPOTrainer(model, rollout, reward, tokenizer, config)
    trainer.fit(samples)
```

当前代码中，默认配置已经指向：

```text
verl/experimental/split_demo/sample_data/train.jsonl
```

因此在不额外准备外部数据的情况下，也可以直接做入口级 smoke test。

---

## 4. 扩展路径

### 无需修改 Trainer 的扩展

| 能力 | 改哪里 | 改法 |
|------|--------|------|
| 更复杂规则 reward | `FunctionReward` 的 `fn` | 换函数 |
| Reward Model | 新增 `ModelReward(RewardAdapter)` | 实现 `score()` |
| KV Cache 生成 | `SplitActorCore` + 新 `CachedSplitRollout` | `SplitRolloutBackend` 接口不变 |
| Speculative Decoding | 新增 `SpeculativeSplitRollout` | 内部持有 Draft Model，Trainer 不感知 |
| 云端中间层 | 新增 `RPCMiddleExecutor(MiddleExecutor)` | `SplitActorCore` 构造时注入 |

### 需要修改 Trainer 的扩展

| 能力 | 需要改什么 | 原因 |
|------|-----------|------|
| vLLM rollout | 重构 rollout 层；加权重同步机制 | vLLM 需要独立完整模型实例，不能与 `SplitActorCore` 共享；须加 sync 协议 |
| FSDP 训练 | 重构 `SplitActorCore` 设备管理 | FSDP 和手动 `.to(device)` 冲突 |
| 多机中间层 | 加断线重连、序列化 | `RPCMiddleExecutor` 需要错误处理 |

> **vLLM 的正确扩展思路**：不是让 vLLM "理解" split 模型，而是让 vLLM 负责生成（完整模型），`SplitActorCore` 只负责训练前向，两者通过权重同步协议解耦。这是 verl 主线 `actor_rollout_ref` 架构已解决的问题，届时应当向主线靠拢，而不是在此 demo 上继续打补丁。

---

## 5. 显存分析

### 当前基线：Qwen2.5-3B（36 层，hidden=2048，vocab=151936）

配置：B=4，G=4，max_new_tokens=24（推荐观测命令），bf16，切法 4:28:4

| 组件 | GPU | 实测 alloc | 实测 peak |
|------|-----|-----------|---------|
| embed + front(0:4) + tail(32:36) + lm_head + LoRA | cuda:0 | ~2.0 GB | ~3.5–3.8 GB |
| middle(4:32) 冻结（LocalMiddleExecutor） | cuda:1 | ~4.1 GB | ~8.2 GB |

数据来源：RUNTIME.md 第 5 节实测日志。

### 参考：训练前向传输量（单 step，推荐观测配置）

| 张量 | 实测大小 |
|------|---------|
| `to_middle`（h_front） | ~0.048 GB |
| `to_primary`（h_middle 回传） | ~0.048 GB |
| `pos_emb` (cos+sin) | ~0.006 GB |
| `pos_ids` / `mask` | 可忽略 |

主要传输开销是 hidden states，pos_emb 约为其 1/8。

### 7B 参考值

GPU 0 峰值约 17 GB（逼近 24 GB 边界的主要项：logits 张量和 backward autograd 图）。实际落地时建议先用 `~/share/Qwen2.5-0.5B-Instruct` 做第一轮 smoke test，链路稳定后再切到 1.5B / 3B / 7B。

### 运行期可观察性

当前 demo 已内建两类运行期统计：

1. **GPU 显存**
   - `gpu0_alloc / gpu0_reserv / gpu0_peak`
   - `gpu1_alloc / gpu1_reserv / gpu1_peak`

2. **split 传输量**
   - `to_middle`：前段 hidden 送往中段的累计字节量
   - `pos_ids`
   - `pos_emb`
   - `mask`
   - `to_primary`：中段输出搬回主卡的累计字节量
   - `last_hidden`
   - `last_logits`
   - `forward_calls / prefill_calls / decode_calls`

这些统计的目标不是做精确 profiler，而是让 demo 阶段能快速判断：

- 双卡是否都在工作
- 主要传输开销是不是集中在 hidden states
- 随模型尺寸从 0.5B 切到 1.5B / 3B 时，显存和传输量是否符合直觉

---

## 6. 三步验证顺序

### Step 1：前向通路

```python
model  = SplitActorCore(config)
ids    = torch.randint(0, 1000, (2, 16)).cuda(0)
mask   = torch.ones(2, 16, dtype=torch.long).cuda(0)
logits = model.forward_full(ids, mask)
assert logits.shape == (2, 16, vocab_size) and logits.device.index == 0
```

### Step 2：反向梯度流

```python
loss = logits[:, -1, :].mean()
loss.backward()
for name, p in model.named_parameters():
    if p.requires_grad:
        assert p.grad is not None, f"missing grad: {name}"
for name, p in model.middle_executor.middle_layers.named_parameters():
    assert p.grad is None, f"middle leaked grad: {name}"
```

### Step 3：完整 GRPO 循环

跑 20 步，观察：
- `mean_reward` 有上升趋势
- `valid_group_ratio` 稳定（dynamic sampling 不全丢）
- `clipfrac` 在 0.05–0.3 范围内
- `approx_kl` 不爆炸

> 实际运行状态、已通过的 smoke test、命令示例、环境信息与已知问题，统一记录在：
>
> [RUNTIME.md](/root/workspace/dev/verl/verl/experimental/split_demo/RUNTIME.md)

---

## 7. 当前 Demo 范围之外

| 不做 | 说明 |
|------|------|
| KV Cache | 接口已预留（`prefill` / `decode_step`），实现留给 `CachedSplitRollout` |
| Speculative Decoding | 接口已预留，实现留给 `SpeculativeSplitRollout` |
| vLLM | 需要权重同步机制，不是自然延伸，见第 4 节 |
| 云端 RPC | 接口已预留（`MiddleExecutor`），实现留给 `RPCMiddleExecutor` |
| Ray / FSDP | 单进程验证，不引入分布式开销 |
