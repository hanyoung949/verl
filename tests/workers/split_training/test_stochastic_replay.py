"""Focused tests for stochastic bound replay semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.split_dvi.stochastic_sampler import (
    SparseDraftDistribution,
    make_sparse_draft_distribution,
)

from verl.workers.split_training.dvi import stochastic_replay as replay
from verl.workers.split_training.dvi.stochastic_replay import (
    MINIMUM_SEEDS,
    _PositionData,
    _acceptance_probability_bounds,
    _filter_grouped_positions,
    _mean_ci95,
    _normalize_excluded_positions,
    _normalize_seeds,
    _simulate_path,
    _teacher_probability_bounds,
)


def _record() -> SimpleNamespace:
    return SimpleNamespace(
        verifier_topk_ids=[1],
        verifier_topk_logprobs=[float(torch.log(torch.tensor(0.2)))],
        verifier_residual_mass=0.8,
    )


def _position(position: int) -> _PositionData:
    record = _record()
    distribution = SparseDraftDistribution(
        token_ids=torch.tensor([0], dtype=torch.int64),
        probabilities=torch.tensor([1.0]),
    )
    return _PositionData(
        "request-0",
        position,
        record,
        distribution,
        coverage=0.0,
        overlap_lower=0.0,
        overlap_upper=0.8,
    )


def test_teacher_probability_and_acceptance_bounds_use_residual_mass() -> None:
    record = _record()

    assert _teacher_probability_bounds(record, 1) == pytest.approx((0.2, 0.2))
    assert _teacher_probability_bounds(record, 0) == pytest.approx((0.0, 0.8))
    assert _acceptance_probability_bounds(record, 1, 0.4) == pytest.approx(
        (0.5, 0.5)
    )
    assert _acceptance_probability_bounds(record, 0, 1.0) == pytest.approx(
        (0.0, 0.8)
    )


def test_q_contract_applies_temperature_topk_and_renormalization() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    distribution = make_sparse_draft_distribution(
        logits, temperature=0.7, top_k=2
    )

    assert distribution.token_ids.tolist() == [2, 3]
    assert distribution.probabilities.sum().item() == pytest.approx(1.0)
    expected = torch.softmax(torch.tensor([2.0, 3.0]) / 0.7, dim=0)
    assert torch.allclose(distribution.probabilities, expected)


def test_bound_paths_stop_at_first_reject_and_count_bonus_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = SimpleNamespace(
        requests=[
            SimpleNamespace(
                request_id="request-0",
                response_token_ids=[0, 1, 2],
            )
        ]
    )
    positions = {position: _position(position) for position in range(3)}
    # The proposal stream is irrelevant for a one-token Q. The acceptance
    # stream is fixed between the lower (0.0) and upper (0.8) bounds.
    monkeypatch.setattr(replay, "stateless_uniform", lambda *args: 0.5)

    lower = _simulate_path(
        reader,
        {"request-0": positions},
        draft_length=4,
        seed=0,
        lower_bound=True,
        num_proposals=3,
        bonus_token=True,
    )
    upper = _simulate_path(
        reader,
        {"request-0": positions},
        draft_length=4,
        seed=0,
        lower_bound=False,
        num_proposals=3,
        bonus_token=True,
    )

    assert lower["prefix_hist"] == {0: 3}
    assert lower["mean_advancement"] == pytest.approx(1.0)
    assert upper["prefix_hist"] == {3: 1}
    assert upper["mean_advancement"] == pytest.approx(4.0)
    assert upper["mean_advancement"] > lower["mean_advancement"]


def test_ci_requires_eight_seeds_and_reports_student_t_interval() -> None:
    with pytest.raises(ValueError, match="at least 8 seeds"):
        _normalize_seeds(None, num_seeds=MINIMUM_SEEDS - 1, seed=0)

    summary = _mean_ci95([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert summary["n"] == MINIMUM_SEEDS
    assert summary["ci95"][0] < summary["mean"] < summary["ci95"][1]


def test_population_filter_is_explicit_and_position_scoped() -> None:
    grouped = {
        "request-0": {0: _position(0), 1: _position(1)},
        "request-1": {0: _position(0), 2: _position(2)},
    }

    assert _normalize_excluded_positions([0, 2, 0]) == [0, 2]
    assert _filter_grouped_positions(grouped, [0]) == 2
    assert set(grouped["request-0"]) == {1}
    assert set(grouped["request-1"]) == {2}

    with pytest.raises(ValueError, match="non-negative"):
        _normalize_excluded_positions([-1])


def _exact_position(position: int) -> _PositionData:
    record = SimpleNamespace(
        verifier_topk_ids=[0],
        verifier_topk_logprobs=[float(torch.log(torch.tensor(0.8)))],
        verifier_residual_mass=0.2,
    )
    distribution = SparseDraftDistribution(
        token_ids=torch.tensor([0], dtype=torch.int64),
        probabilities=torch.tensor([1.0]),
    )
    return _PositionData(
        "request-0",
        position,
        record,
        distribution,
        coverage=1.0,
        overlap_lower=0.8,
        overlap_upper=0.8,
    )


def test_runtime_bonus_caps_max_advancement_at_block_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = SimpleNamespace(
        requests=[
            SimpleNamespace(
                request_id="request-0",
                response_token_ids=[0, 1, 2, 3],
            )
        ]
    )
    positions = {position: _exact_position(position) for position in range(4)}
    monkeypatch.setattr(replay, "stateless_uniform", lambda *args: 0.5)

    result = _simulate_path(
        reader,
        {"request-0": positions},
        draft_length=4,
        num_proposals=3,
        bonus_token=True,
        seed=0,
        lower_bound=False,
    )

    assert result["prefix_hist"] == {3: 1}
    assert result["advancement_hist"] == {4: 1}
    assert result["max_advancement"] == 4
    assert 5 not in result["advancement_hist"]


def test_runtime_reject_advancement_is_prefix_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for reject_position in range(3):
        reader = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    request_id="request-0",
                    response_token_ids=list(range(reject_position + 1)),
                )
            ]
        )
        positions = {
            position: _exact_position(position)
            for position in range(reject_position + 1)
        }

        def acceptance_uniform(
            _seed: int,
            _generation: int,
            position: int,
            _kind: object,
        ) -> float:
            return 0.9 if position == reject_position else 0.5

        monkeypatch.setattr(replay, "stateless_uniform", acceptance_uniform)
        result = _simulate_path(
            reader,
            {"request-0": positions},
            draft_length=4,
            num_proposals=3,
            bonus_token=True,
            seed=0,
            lower_bound=False,
        )

        assert result["prefix_hist"] == {reject_position: 1}
        assert result["advancement_hist"] == {reject_position + 1: 1}


def test_draft_logit_semantics_resolve_runtime_temperature() -> None:
    processed = SimpleNamespace(
        draft_logit_semantics="processed_at_temperature",
        calibration_temperature=0.7,
    )
    assert replay._resolve_draft_distribution_temperature(processed, 0.7) == (
        1.0,
        "processed_at_temperature",
    )
    raw = SimpleNamespace(draft_logit_semantics="raw_target_logits")
    assert replay._resolve_draft_distribution_temperature(raw, 0.7) == (
        0.7,
        "raw_target_logits",
    )
    with pytest.raises(ValueError, match="does not match"):
        replay._resolve_draft_distribution_temperature(
            processed, 0.8
        )
