# v4 迁移指南 — 新机器环境配置与验证

> 适用于将 v4 (3-stage pipeline) 从当前 2 卡机器迁移到 3+ 卡机器。

---

## 1. 环境配置

### 1.1 基础环境

```bash
# 创建 conda 环境
conda create -n verl python=3.12 -y
conda activate verl

# PyTorch (CUDA 12.1)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 核心依赖
pip install transformers==5.6.2 peft==0.19.1 hydra-core==1.3.2 omegaconf==2.3.0
pip install tensordict==0.10.0 numpy==1.26.4

# 安装 verl (源码)
cd /path/to/verl
pip install -e .

# Ray (v5 后期需要，v4 暂不需要)
pip install ray==2.55.1
```

### 1.2 模型下载

```bash
# 下载 Qwen2.5-3B-Instruct 到本地
mkdir -p ~/share
# 从 HuggingFace 下载，或从已有目录拷贝
# 需要包含: config.json, model.safetensors, tokenizer.json, ...
ls ~/share/Qwen2.5-3B-Instruct/
# config.json  model.safetensors  tokenizer.json  ...
```

### 1.3 验证环境

```bash
# 检查 CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
# 期望输出: CUDA: True GPUs: 3  (或更多)

# 检查 NCCL
python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"

# 检查模型路径
ls ~/share/Qwen2.5-3B-Instruct/config.json
```

---

## 2. v4 验证步骤

### Step 1: Import 验证

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate verl

python -c "
from verl.experimental.split_demo_v4.core.stage import Stage, HeadStage, TailStage
from verl.experimental.split_demo_v4.core.middle_stage import MiddleStage
from verl.experimental.split_demo_v4.core.transport import StageTransport
from verl.experimental.split_demo_v4.core.pipeline_engine import SplitPipelineEngine
print('v4 imports OK')
"
```

### Step 2: 3-stage Smoke Test

```bash
torchrun --nproc_per_node=3 \
  -m verl.experimental.split_demo_v4.main_split_v4
```

检查：
- 3 个进程都启动成功
- `[rank0] HeadStage: 4 layers`
- `[rank1] MiddleStage: 28 layers`
- `[rank2] TailStage: 4 layers`
- 无 crash、无死锁

### Step 3: 10 步 GRPO 训练

```bash
torchrun --nproc_per_node=3 \
  -m verl.experimental.split_demo_v4.main_split_v4 \
  trainer.total_steps=10 \
  trainer.log_freq=1
```

检查：
- 10 步训练跑通
- reward 有波动
- checkpoint 保存成功

### Step 4: 数值对比 v3

对比 v4 和 v3 的训练指标，确认量级一致：
- reward 范围
- gpu0_peak 显存
- clipfrac / approx_kl

---

## 3. 多机扩展（后续）

当单机 3 卡验证通过后，扩展到多机：

```bash
# 机器 0 (Head + Tail)
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 \
  --master_addr=机器0_IP --master_port=29500 \
  -m verl.experimental.split_demo_v4.main_split_v4

# 机器 1 (Middle)
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
  --master_addr=机器0_IP --master_port=29500 \
  -m verl.experimental.split_demo_v4.main_split_v4
```

注意：多机时 `main_split_v4.py` 需要调整 rank 到 stage 的映射逻辑。

---

## 4. 目录结构参考

```
verl/experimental/
├── split_demo_v1/    ✅ 单进程 split (2 卡验证通过)
├── split_demo_v2/    ✅ NCCL 双进程 (2 卡验证通过)
├── split_demo_v3/    ✅ verl BaseEngine 集成 (2 卡验证通过)
└── split_demo_v4/    ⚠️ 3-stage Stage 抽象 (需 3 卡验证)
    ├── PLAN.md
    ├── MIGRATION.md     ← 本文档
    ├── core/
    │   ├── stage.py           Stage / HeadStage / TailStage
    │   ├── transport.py       StageTransport
    │   ├── middle_stage.py    MiddleStage (冻结, 请求响应循环)
    │   └── pipeline_engine.py SplitPipelineEngine(BaseEngine)
    ├── config/
    ├── reward/
    ├── rollout/
    └── main_split_v4.py       3-stage 入口
```

---

## 5. 已知限制

| 限制 | 说明 | 后续计划 |
|---|---|---|
| 需要 3 张 GPU | 当前机器只有 2 张 | 迁移到 3+ GPU 机器 |
| Rollout 生成慢 | 每个 token 需要 Head→Middle→Tail 往返 | v5 引入 vLLM |
| 无 micro-batch 流水 | 同步执行，GPU 利用率低 | v5 引入 1F1B |
| 无 Ray 编排 | torchrun 固定 3 进程 | v5 引入 Ray |
