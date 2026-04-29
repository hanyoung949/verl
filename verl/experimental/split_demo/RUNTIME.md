# Split Actor GRPO Demo — 运行文档

这个文档只保留**当前基线 demo**的信息，不再记录已经过时的 0.5B / 1.5B 试验细节。

当前基线：

- 模型：`/root/share/Qwen2.5-3B-Instruct`
- 切层：`4 : 28 : 4`
- 目标场景：**边侧算力受限，中心算力更强**

---

## 1. 环境

### conda 环境

- Miniconda：`/root/workspace/miniconda3`
- 环境名：`verl`

激活：

```bash
source /root/workspace/miniconda3/etc/profile.d/conda.sh
conda activate verl
```

环境创建脚本：

- [scripts/create_verl_dev_conda_env.sh](/root/workspace/dev/verl/scripts/create_verl_dev_conda_env.sh)

### GPU

已确认：

- `nvidia-smi` 能看到两张 4090
- 在 `verl` 环境里：

```text
torch.cuda.is_available() == True
torch.cuda.device_count() == 2
```

---

## 2. 当前默认配置

配置文件：

- [split_demo.yaml](/root/workspace/dev/verl/verl/experimental/split_demo/config/split_demo.yaml)

关键项：

```yaml
model:
  path: "~/share/Qwen2.5-3B-Instruct"

split:
  front_end: 4
  middle_end: 32
```

对应 36 层模型的切法：

```text
front  = layers[0:4]
middle = layers[4:32]
tail   = layers[32:36]
```

即：

```text
4 : 28 : 4
```

---

## 3. 数据

默认样本文件：

- [train.jsonl](/root/workspace/dev/verl/verl/experimental/split_demo/sample_data/train.jsonl)

当前样本数：

- 200 条

生成脚本：

- [scripts/generate_split_demo_math_dataset.py](/root/workspace/dev/verl/scripts/generate_split_demo_math_dataset.py)

---

## 4. 固定运行命令

### 4.1 默认基线 demo

```bash
source /root/workspace/miniconda3/etc/profile.d/conda.sh
conda activate verl

python -m verl.experimental.split_demo.main_grpo_split
```

### 4.2 当前推荐观测命令

这是当前最有信息量的一条命令：

```bash
source /root/workspace/miniconda3/etc/profile.d/conda.sh
conda activate verl

python -m verl.experimental.split_demo.main_grpo_split \
  trainer.total_steps=10 \
  trainer.train_batch_size=4 \
  trainer.log_freq=1 \
  trainer.lr=5e-4 \
  rollout.group_size=4 \
  rollout.max_new_tokens=24 \
  algorithm.loss_scale_factor=24 \
  algorithm.update_epochs=2 \
  algorithm.mini_batch_size=8
```

它的用途是：

- 看真实 reward 下 `valid_group_ratio / reward / clipfrac / approx_kl`
- 看双卡显存
- 看 split 传输量

---

## 5. 当前基线实验结果（3B + 4:28:4）

命令就是上面的“当前推荐观测命令”。

代表性日志（step 1）：

```text
[step 1] reward=0.4375 valid_group_ratio=0.7500 loss=-0.2044 clipfrac=0.1250 approx_kl=-0.0557 gpu0_alloc=2.01GB gpu0_reserv=5.89GB gpu0_peak=3.54GB gpu1_alloc=4.13GB gpu1_reserv=9.04GB gpu1_peak=8.20GB fwd_calls=29 prefill_calls=1 decode_calls=23 to_middle=0.0480GB pos_ids=0.0001GB pos_emb=0.0060GB mask=0.0007GB to_primary=0.0480GB last_hidden=(4, 41, 2048) last_logits=(4, 41, 151936)
```

代表性日志（step 10）：

```text
[step 10] reward=0.6250 valid_group_ratio=1.0000 loss=-0.1393 clipfrac=0.0885 approx_kl=-0.0259 gpu0_alloc=2.06GB gpu0_reserv=6.28GB gpu0_peak=3.84GB gpu1_alloc=4.13GB gpu1_reserv=9.08GB gpu1_peak=8.23GB fwd_calls=29 prefill_calls=1 decode_calls=23 to_middle=0.0493GB pos_ids=0.0001GB pos_emb=0.0062GB mask=0.0008GB to_primary=0.0493GB last_hidden=(8, 41, 2048) last_logits=(8, 41, 151936)
```

### 结论

1. **切层符合场景**
   - 边侧卡（gpu0）更轻：
     - `gpu0_alloc ≈ 2.0GB`
     - `gpu0_peak ≈ 3.5 ~ 3.8GB`
   - 中段卡（gpu1）更重：
     - `gpu1_alloc ≈ 4.13GB`
     - `gpu1_peak ≈ 8.20 ~ 8.23GB`

2. **主要传输开销符合预期**
   - `to_middle ≈ 0.048 ~ 0.049GB`
   - `to_primary ≈ 0.048 ~ 0.049GB`
   - 大头确实是 hidden states，不是 `pos_ids` / `mask`

3. **训练信号已经可见**
   - `clipfrac` 不再是 0，约 `0.03 ~ 0.20`
   - `approx_kl` 有明显波动，约 `-0.05 ~ 0.19`
   - `valid_group_ratio` 在 `0.5 / 0.75 / 1.0` 间变化

当前判断：

> 这个 demo 已经足够证明：  
> `Qwen2.5-3B + 4:28:4` 的 split 架构在单机双卡上是成立的，  
> 而且比之前更均匀的切法更符合“边侧轻、中心重”的目标。

---

## 6. 当前保存位置

训练结束后会自动保存到固定基目录：

- [checkpoints](/root/workspace/dev/verl/verl/experimental/split_demo/checkpoints)

会同时写两份：

1. 固定最新结果：
   - [latest](/root/workspace/dev/verl/verl/experimental/split_demo/checkpoints/latest)
2. 按 step 保存：
   - [global_step_10](/root/workspace/dev/verl/verl/experimental/split_demo/checkpoints/global_step_10)

每个目录下当前有：

- `adapter_state.pt`
- `adapter_meta.json`
- `optimizer.pt`
- `trainer_state.json`

### `adapter_state.pt` 里是什么

当前保存的是：

> 所有 `requires_grad=True` 的命名参数

也就是这个 demo 里前段 / 尾段上的 LoRA 相关可训练参数。

已验证：

- `adapter_state.pt` 中有 32 个 trainable tensor
- 键名示例：

```text
front_layers.0.self_attn.q_proj.lora_A.default.weight
```

所以当前保存的不是完整 base model，而是：

> **最小 adapter + optimizer + trainer state**

---

## 7. 这个 demo 怎么运行 / 怎么改配置

入口：

- [main_grpo_split.py](/root/workspace/dev/verl/verl/experimental/split_demo/main_grpo_split.py)

执行链路：

1. 读取 Hydra 配置
2. 解析本地模型路径（优先 `~/share`）
3. 读取 `train.jsonl`
4. 构建：
   - `SplitActorCore`
   - `NaiveSplitRollout`
   - `FunctionReward`
   - `SplitGRPOTrainer`
5. 调 `trainer.fit(samples)`

### 最常改的配置

1. `model.path`
2. `split.front_end / split.middle_end`
3. `rollout.group_size / max_new_tokens / temperature / top_p`
4. `algorithm.loss_scale_factor / update_epochs / mini_batch_size`
5. `trainer.total_steps / train_batch_size / lr / save_dir`

命令行覆盖示例：

```bash
python -m verl.experimental.split_demo.main_grpo_split \
  model.path=/root/share/Qwen2.5-7B-Instruct \
  split.front_end=4 \
  split.middle_end=24 \
  trainer.total_steps=20
```

---

## 8. DAPO / Dr.GRPO 现在哪些是真的生效

### Dr.GRPO

当前这几个点是**真实生效**的：

- `norm_adv_by_std_in_grpo = false`
- `loss_agg_mode = seq-mean-token-sum-norm`
- `loss_scale_factor = 固定常数`

所以可以明确说：

> **Dr.GRPO 的核心训练形式已生效。**

### DAPO

当前这两个关键机制是**真实在跑**的：

1. `dynamic_sampling`
2. `clip-higher`（asymmetric clipping）

所以更准确的说法是：

> **当前 demo 已实现 DAPO 的 dynamic sampling 和 clip-higher。**

### 还没有完整实现的部分

还没有完整 recipe 化的内容包括：

- 更复杂 reward shaping
- 更完整 DAPO 数据策略
- Reward Model
- vLLM rollout
- KV cache / speculative

因此当前 demo 的定位是：

> **架构验证版 + 关键训练机制已接通版**

---

## 9. 近期修复

### 9.1 rollout 显式关闭梯度

修复前：

- `split_trainer.py` 在调用 `self.rollout.generate(...)` 时没有包 `torch.no_grad()`
- rollout 阶段每次 prefill / decode_step 都会白白构建 autograd 图
- 对 LoRA 训练来说，这些图不会被反向使用，只会额外吃显存

修复后：

- rollout 已显式放进 `with torch.no_grad():`

影响：

- 更符合 rollout 本来就“不回传梯度”的语义
- 显存行为更干净

### 9.2 `algorithm.dynamic_sampling` 配置项真正生效

修复前：

- 代码里无条件执行：

```python
valid_groups = rewards_grouped.std(dim=1) > 1e-6
```

- 即使把 yaml 里的 `dynamic_sampling: false` 改掉，也不会生效

修复后：

- 现在会真实读取：

```yaml
algorithm:
  dynamic_sampling: true/false
```

语义：

- `true`：按 DAPO 逻辑过滤同质 group
- `false`：不过滤，所有 group 都参与更新

### 9.3 RoPE 设备放置更稳

修复前：

- `decoder_rotary_emb` 没有显式搬到 `primary_device`
- 当前 Qwen2.5 因为内部 `.to(x.device)` 机制还能跑
- 但对别的模型这是潜在脆弱点

修复后：

- 若存在 decoder 级 `rotary_emb`，初始化时会显式 `.to(primary_device)`

### 9.4 causal mask 更贴近 HF 标准

修复前：

- `_build_causal_mask()` 用了：

```python
key_mask & query_mask & causal
```

- 这会让 padding query 整行变成 `-inf`

修复后：

- 改为：

```python
key_mask & causal
```

意义：

- 更贴近 HF 常见实现
- 少一类中间激活里无意义的 NaN 干扰

### 9.5 死代码清理

已清理：

- `split_trainer.py` 中未使用的 `AdvantageEstimator` 导入
- `split_actor.py` 中未使用的 `hidden_size` 成员

### 9.6 trainer 侧冗余路径清理

已清理三处小问题：

1. **去掉了无意义的 `DataProto -> 立即读回` 包装**
   - 之前 `compute_grpo_outcome_advantage(...)` 的结果会先塞进 `DataProto`
   - 随后又立刻从里面读回 `advantages`
   - 这一层包装不参与任何后续逻辑，现已删除

2. **不再保留未使用的 `returns_valid`**
   - 现在直接写成：

   ```python
   advantages_valid, _ = compute_grpo_outcome_advantage(...)
   ```

   - 更符合当前 trainer 只消费 `advantages` 的事实

3. **日志中的 `reward` 现在只统计真正参与更新的样本**
   - 之前是 `rewards.mean()`
   - 现在改成 `rewards[valid_seq].mean()`
   - 这样日志均值和实际参与梯度更新的数据集一致
