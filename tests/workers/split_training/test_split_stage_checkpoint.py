"""CPU tests for split-stage checkpoint forwarding."""

from __future__ import annotations

from verl.workers.split_training.split_stage_worker import SplitStageWorker


class _FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    def save_checkpoint(self, **kwargs) -> None:
        self.calls.append(("save", kwargs))

    def load_checkpoint(self, **kwargs) -> None:
        self.calls.append(("load", kwargs))


def test_checkpoint_methods_forward_adapter_and_optimizer_state() -> None:
    worker = object.__new__(SplitStageWorker)
    worker.engine = _FakeEngine()
    worker.is_head = True

    saved = worker.save_checkpoint("/tmp/policy", global_step=3)
    loaded = worker.load_checkpoint("/tmp/policy")

    assert saved == {"stage": "stage_0", "global_step": 3}
    assert loaded == {"stage": "stage_0", "loaded": True}
    assert worker.engine.calls == [
        (
            "save",
            {"local_path": "/tmp/policy", "global_step": 3},
        ),
        (
            "load",
            {
                "local_path": "/tmp/policy",
                "del_local_after_load": False,
            },
        ),
    ]
