"""split demo v4 主入口 — 3-stage pipeline。

运行命令：
    torchrun --nproc_per_node=3 -m verl.experimental.split_demo_v4.main_split_v4
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .core.pipeline_engine import SplitPipelineEngine
from .reward import FunctionReward
from .reward.function_reward import math_exact_match
from .rollout import SimplePipelineRollout
from .trainer.split_trainer import SplitTrainer, set_seed


def load_jsonl_samples(path):
    file_path = Path(to_absolute_path(os.path.expanduser(path)))
    if not file_path.exists():
        raise FileNotFoundError(f"Training file not found: {file_path}")
    samples = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "prompt" not in item or "answer" not in item:
                raise ValueError("Missing required keys 'prompt'/'answer'.")
            samples.append(item)
    if not samples:
        raise ValueError(f"No valid samples found in {file_path}")
    return samples


@hydra.main(config_path="config", config_name="split_demo_v4", version_base=None)
def main(config: DictConfig):
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    topology = {
        "head": list(config.split.topology.head),
        "middle": list(config.split.topology.middle),
        "tail": list(config.split.topology.tail),
    }
    engine = SplitPipelineEngine(topology=topology)
    engine.model_path = str(config.model.path)
    engine.front_end = int(config.split.front_end)
    engine.middle_end = int(config.split.middle_end)
    engine.lr = float(config.trainer.lr)
    engine.clip_grad = float(config.trainer.max_grad_norm)
    engine.initialize()

    # Middle: 进入请求响应循环
    if engine.is_middle:
        try:
            engine.stage.run(topology=topology)
        finally:
            dist.destroy_process_group()
        return

    # rank0 (Head) + rank2 (Tail): 训练逻辑
    set_seed(int(config.trainer.seed))
    tokenizer = engine.tokenizer
    samples = load_jsonl_samples(config.data.train_file)

    rollout_backend = SimplePipelineRollout(engine, tokenizer)
    reward_fn = FunctionReward(fn=math_exact_match, tokenizer=tokenizer)

    trainer = SplitTrainer(engine, rollout_backend, reward_fn, config)
    trainer.fit(samples)


if __name__ == "__main__":
    main()
