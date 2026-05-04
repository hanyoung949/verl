"""split demo NCCL 版主入口。

通过 torchrun 启动双进程：
- rank0: 训练逻辑（front + tail + trainer）
- rank1: middle worker（纯请求响应）

运行命令：
    torchrun --nproc_per_node=2 -m verl.experimental.split_demo_nccl.main_grpo_split
"""

from __future__ import annotations

import json
import logging
import os
import random
import traceback
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

from .core.middle_worker import MiddleWorker
from .core.split_actor import SplitActorCore
from .reward import FunctionReward
from .reward.function_reward import math_exact_match
from .rollout import NaiveSplitRollout
from .split_trainer import SplitGRPOTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl_samples(path: str) -> list[dict]:
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
    expanded = os.path.expanduser(path)
    absolute = to_absolute_path(expanded)
    if Path(absolute).exists():
        return absolute
    return path


def build_lora_config(lora_cfg) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.lora_alpha),
        lora_dropout=float(lora_cfg.lora_dropout),
        target_modules=list(lora_cfg.target_modules),
    )


@hydra.main(config_path="config", config_name="split_demo_v2", version_base=None)
def main(config: DictConfig) -> None:
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 共享配置解析（rank0 和 rank1 都需要）
    model_path = resolve_model_path(str(config.model.path))
    front_end = int(config.split.front_end)
    middle_end = int(config.split.middle_end)
    lora_config = build_lora_config(config.lora)

    # ── rank1: middle worker ───────────────────────────────────

    if rank == 1:
        device = torch.device("cuda")
        worker = MiddleWorker.from_pretrained(
            model_path=model_path,
            front_end=front_end,
            middle_end=middle_end,
            lora_config=lora_config,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        try:
            worker.run()
        except Exception:
            logging.error("rank1 MiddleWorker crashed:\n%s", traceback.format_exc())
            raise
        finally:
            dist.destroy_process_group()
        return

    # ── rank0: training logic ──────────────────────────────────

    set_seed(int(config.trainer.seed))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = load_jsonl_samples(config.data.train_file)

    model = SplitActorCore(
        model_path=model_path,
        front_end=front_end,
        middle_end=middle_end,
        lora_config=lora_config,
        device="cuda",
        torch_dtype=torch.bfloat16,
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

    try:
        trainer.fit(samples)
        model.middle_executor.shutdown()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
