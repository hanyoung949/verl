"""split_demo 包入口。

这个实验目录的目标很明确：
1. 先在单机双卡上验证“前段-中段-尾段”的拆分训练是否跑通；
2. 保持 rollout / reward / middle transport 的替换边界，
   方便未来逐步接入 KV cache、投机解码、Reward Model 甚至 RPC 中间层。
"""

