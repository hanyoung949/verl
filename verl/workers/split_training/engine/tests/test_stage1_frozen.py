# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from verl.workers.split_training.engine.middle_stage import Stage1


class DummyLayer(nn.Module):
    """A dummy transformer layer that ignores extra kwargs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states, **kwargs):
        return hidden_states * self.scale


def test_stage1_initializes_frozen():
    layers = [DummyLayer(8) for _ in range(3)]
    stage = Stage1(layers, rotary_emb=None, device="cpu")

    for p in stage.layers.parameters():
        assert not p.requires_grad
    assert not stage.has_grad()


def test_stage1_forward_backward_no_grad():
    layers = [DummyLayer(8) for _ in range(3)]
    stage = Stage1(layers, rotary_emb=None, device="cpu")

    h = torch.randn(2, 4, 8, requires_grad=True)
    h_out = stage.forward(h)
    assert h_out.shape == h.shape

    grad_out = torch.ones_like(h_out)
    h_out.backward(grad_out)

    # The input grad is populated, but stage_1 parameters must stay grad-free.
    assert h.grad is not None
    assert not stage.has_grad()
