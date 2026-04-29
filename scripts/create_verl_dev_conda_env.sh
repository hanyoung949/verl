#!/usr/bin/env bash
set -euo pipefail

# 这个脚本的目标很简单：
# 1. 使用已经安装好的 Miniconda；
# 2. 创建专用于当前仓库的 conda 环境 `verl`；
# 3. 安装 split_demo 和一般 verl 开发所需的核心依赖；
# 4. 将当前仓库以 editable 模式安装进去，便于边改边跑。

CONDA_ROOT="${CONDA_ROOT:-/root/workspace/miniconda3}"
ENV_NAME="${ENV_NAME:-verl}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "未找到 conda 初始化脚本: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

echo "[1/5] 创建 conda 环境 ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.12 pip

echo "[2/5] 激活环境并升级基础打包工具"
conda activate "${ENV_NAME}"
pip install --upgrade pip setuptools wheel

echo "[3/5] 安装 PyTorch（CUDA 12.1 wheel）"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[4/5] 安装 verl 常用核心依赖"
pip install \
  accelerate \
  codetiming \
  datasets \
  dill \
  hydra-core \
  "numpy<2.0.0" \
  pandas \
  peft \
  "pyarrow>=19.0.0" \
  pybind11 \
  pylatexenc \
  pre-commit \
  "ray[default]" \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
  torchdata \
  transformers \
  wandb \
  "packaging>=20.0" \
  uvicorn \
  fastapi \
  latex2sympy2_extended \
  math_verify \
  tensorboard

echo "[5/5] 以 editable 模式安装当前仓库"
pip install -e "${REPO_ROOT}"

echo
echo "环境创建完成。后续进入仓库后可执行："
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
