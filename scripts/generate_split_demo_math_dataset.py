#!/usr/bin/env python3
"""生成 split_demo 用的简单双算术题数据集。

设计目标：
1. 生成固定 200 条、可复现的样本；
2. 保持题目足够简单，reward 仍然可以用当前的数字精确匹配；
3. 比最初那几条 toy 数据更有分布感，便于观察真实 reward / dynamic sampling。
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def build_samples(num_samples: int = 200, seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    samples: list[dict[str, str]] = []

    templates = [
        "What is {a} + {b}? Answer with only the final number.",
        "Compute {a} + {b}. Only output the final number.",
        "What is {a} - {b}? Answer with only the final number.",
        "Compute {a} - {b}. Only output the final number.",
        "What is {a} * {b}? Answer with only the final number.",
        "Compute {a} * {b}. Only output the final number.",
    ]

    while len(samples) < num_samples:
        op_type = len(samples) % len(templates)

        if op_type in (0, 1):  # 加法
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)
            answer = a + b
        elif op_type in (2, 3):  # 减法，避免太多负数
            a = rng.randint(0, 99)
            b = rng.randint(0, a)
            answer = a - b
        else:  # 乘法，控制答案规模
            a = rng.randint(0, 20)
            b = rng.randint(0, 20)
            answer = a * b

        template = templates[op_type]
        prompt = template.format(a=a, b=b)
        samples.append({"prompt": prompt, "answer": str(answer)})

    return samples


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "verl" / "experimental" / "split_demo" / "sample_data" / "train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = build_samples(num_samples=200, seed=42)

    with output_path.open("w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"wrote {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
