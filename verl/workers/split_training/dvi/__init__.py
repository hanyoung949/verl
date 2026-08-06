"""Offline training utilities for SplitRL DVI experiments."""

from .draft_head import SplitDVIDraftHead
from .offline import (
    DVITrainConfig,
    DVITrainResult,
    build_offline_report,
    evaluate_offline_draft_head,
    load_base_projection,
    split_request_ids,
    train_offline_draft_head,
)
from .policy_version import compute_policy_version
from .v2_replay import (
    ReplayMetrics,
    V2ReplayReport,
    evaluate_v2_artifact,
)
from .telemetry_merge import (
    DVIPartialSpoolMerger,
    MergeReport,
)


def replay_greedy_draft_head(*args, **kwargs):
    """Lazily invoke the replay evaluator to keep ``python -m`` clean."""
    from .replay import replay_greedy_draft_head as _replay

    return _replay(*args, **kwargs)


def replay_stochastic_bound_draft_head(*args, **kwargs):
    """Lazily invoke the stochastic bound replay evaluator."""
    from .stochastic_replay import (
        replay_stochastic_bound_draft_head as _replay,
    )

    return _replay(*args, **kwargs)


__all__ = [
    "DVIPartialSpoolMerger",
    "DVITrainConfig",
    "DVITrainResult",
    "evaluate_offline_draft_head",
    "MergeReport",
    "SplitDVIDraftHead",
    "build_offline_report",
    "compute_policy_version",
    "load_base_projection",
    "replay_greedy_draft_head",
    "replay_stochastic_bound_draft_head",
    "ReplayMetrics",
    "V2ReplayReport",
    "evaluate_v2_artifact",
    "split_request_ids",
    "train_offline_draft_head",
]
