"""split demo 主入口。"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

from .core import SplitActorCore
from .reward import FunctionReward
from .reward.function_reward import math_exact_match
from .rollout import NaiveSplitRollout
from .split_trainer import SplitGRPOTrainer


def set_seed(seed: int) -> None:
    """统一设置随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl_samples(path: str) -> list[dict]:
    """读取 jsonl 数据。

    期望每行至少有：
    - prompt
    - answer
    """

    # Hydra 运行时可能切换工作目录，因此这里统一先转绝对路径。
    file_path = Path(to_absolute_path(os.path.expanduser(path)))
    if not file_path.exists():
        raise FileNotFoundError(f"Training file not found: {file_path}")

    samples = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "prompt" not in item or "answer" not in item:
                raise ValueError(f"Line {line_no} is missing required keys 'prompt'/'answer'.")
            samples.append(item)

    if not samples:
        raise ValueError(f"No valid samples found in {file_path}")
    return samples


def resolve_model_path(path: str) -> str:
    """解析模型路径。

    规则：
    1. 若它是本地存在的路径（包括 `~/share/...` 这种写法），返回其绝对路径；
    2. 若本地不存在，则原样返回，交给 transformers 走 Hugging Face repo 逻辑。

    这样可以兼容：
    - 本地共享模型目录
    - 远端 repo id
    """

    expanded = os.path.expanduser(path)
    absolute = to_absolute_path(expanded)
    if Path(absolute).exists():
        return absolute
    return path


@hydra.main(config_path="config", config_name="split_demo", version_base=None)
def main(config: DictConfig) -> None:
    """入口函数。"""

    set_seed(int(config.trainer.seed))

    model_path = resolve_model_path(str(config.model.path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = load_jsonl_samples(config.data.train_file)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(config.lora.r),
        lora_alpha=int(config.lora.lora_alpha),
        lora_dropout=float(config.lora.lora_dropout),
        target_modules=list(config.lora.target_modules),
    )

    model = SplitActorCore(
        model_path=model_path,
        front_end=int(config.split.front_end),
        middle_end=int(config.split.middle_end),
        lora_config=lora_config,
    )

    rollout = NaiveSplitRollout(
        model=model,
        tokenizer=tokenizer,
        temperature=float(config.rollout.temperature),
        top_p=float(config.rollout.top_p),
    )

    reward = FunctionReward(
        fn=math_exact_match,
        tokenizer=tokenizer,
    )

    trainer = SplitGRPOTrainer(
        model=model,
        rollout=rollout,
        reward=reward,
        tokenizer=tokenizer,
        config=config,
    )
    trainer.fit(samples)


if __name__ == "__main__":
    main()
