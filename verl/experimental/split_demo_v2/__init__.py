"""split_demo_nccl — Split Actor GRPO Demo v2 (双进程 NCCL 版)。

在 v1 单进程基线基础上，引入真实进程边界，
用 NCCL P2P 通信替换 tensor.to('cuda:1')。

通过 torchrun 启动：
    torchrun --nproc_per_node=2 -m verl.experimental.split_demo_nccl.main_grpo_split
"""
