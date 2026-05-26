"""Rollout backend 抽象基类。

任何 rollout 实现（pipeline、DP、TP、张量并行等）都必须继承 BaseRolloutBackend，
从而保证 SplitTrainer 可以以统一方式调用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRolloutBackend(ABC):
    """Rollout 后端抽象基类。

    职责：
      - Head 端：把 prompt 扩展成 response（generate）。
      - Tail 端：进入服务循环，接收并处理 Head 的采样请求（run_tail_loop）。
    """

    @abstractmethod
    def generate(self, prompt_ids, group_size, max_new_tokens, attention_mask=None):
        """Head 端调用：执行完整 rollout，返回结果对象。

        返回对象必须至少包含以下属性：
          - sequences:       [B*G, prompt_len + response_len]
          - attention_mask:  [B*G, prompt_len + response_len]
          - response_ids:    [B*G, response_len]
          - response_mask:   [B*G, response_len]
          - old_log_probs:   [B*G, response_len]
          - prompt_len:      int
        """
        pass

    @abstractmethod
    def run_tail_loop(self):
        """Tail 端调用：阻塞式服务循环，直到 rollout 阶段结束。

        当 Head 端 generate() 完成后，Tail 端应收到结束信号并退出循环。
        """
        pass
