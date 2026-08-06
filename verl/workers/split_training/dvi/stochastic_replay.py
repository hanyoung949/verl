"""Stochastic bound replay for captured Stage-DVI trajectories.

Artifact v1 stores exact target probabilities only on a top-k support. This
module therefore reports coupled lower/upper accept/reject paths and does not
sample correction tokens; correction is always one committed token and is
only needed for token-for-token runtime comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from vllm.v1.worker.gpu.split_dvi.artifact import DVIArtifactReader, DVICaptureRecord
from vllm.v1.worker.gpu.split_dvi.stochastic_sampler import (
    DVIDrawKind,
    SparseDraftDistribution,
    make_sparse_draft_distribution,
    sample_sparse_draft_token,
    stateless_uniform,
)

from .offline import (
    _sha256_file,
    load_base_projection,
    load_train_result,
)
from .replay import _group_capture_records, _load_replay_model, _sampling_is_stochastic


STOCHASTIC_REPLAY_SCHEMA_VERSION = "dvi-stochastic-replay-v1"
DEFAULT_DRAFT_TEMPERATURE = 0.7
DEFAULT_DRAFT_TOP_K = 16
MINIMUM_SEEDS = 8
_T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


class _PositionData:
    def __init__(
        self,
        request_id: str,
        position: int,
        record: DVICaptureRecord,
        draft_distribution: SparseDraftDistribution,
        coverage: float,
        overlap_lower: float,
        overlap_upper: float,
    ) -> None:
        self.request_id = request_id
        self.position = position
        self.record = record
        self.draft_distribution = draft_distribution
        self.coverage = coverage
        self.overlap_lower = overlap_lower
        self.overlap_upper = overlap_upper


def _describe(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "quantiles": {key: 0.0 for key in ("p10", "p25", "p50", "p75", "p90")},
        }
    ordered = sorted(float(value) for value in values)

    def quantile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower, upper = math.floor(index), math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": quantile(0.5),
        "min": ordered[0],
        "max": ordered[-1],
        "quantiles": {
            "p10": quantile(0.1),
            "p25": quantile(0.25),
            "p50": quantile(0.5),
            "p75": quantile(0.75),
            "p90": quantile(0.9),
        },
    }


def _mean_ci95(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"n": 0, "mean": 0.0, "stddev": 0.0, "ci95": [0.0, 0.0]}
    mean = sum(values) / len(values)
    if len(values) < 2:
        stddev = half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        stddev = math.sqrt(variance)
        critical = _T_CRITICAL_95.get(len(values) - 1, 1.96)
        half_width = critical * stddev / math.sqrt(len(values))
    interval = [mean - half_width, mean + half_width]
    if 0.0 <= mean <= 1.0:
        interval = [max(0.0, interval[0]), min(1.0, interval[1])]
    return {"n": len(values), "mean": mean, "stddev": stddev, "ci95": interval}


def _derive_request_seed(seed: int, request_id: str) -> int:
    payload = f"{seed}\0{request_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=True)


def _resolve_runtime_draft_config(
    sampling_config: dict[str, Any],
    draft_temperature: float | None,
    draft_top_k: int | None,
) -> tuple[float, int]:
    configured_temperature = sampling_config.get("draft_temperature")
    if configured_temperature is None:
        configured_temperature = sampling_config.get("rollout_temperature")
    if configured_temperature is None:
        configured_temperature = sampling_config.get("temperature")
    temperature = (
        float(draft_temperature)
        if draft_temperature is not None
        else float(configured_temperature or DEFAULT_DRAFT_TEMPERATURE)
    )
    configured_top_k = sampling_config.get("draft_top_k")
    top_k = int(draft_top_k if draft_top_k is not None else
               (configured_top_k or DEFAULT_DRAFT_TOP_K))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "stochastic replay requires a positive finite draft temperature"
        )
    if top_k <= 0:
        raise ValueError("stochastic replay draft_top_k must be positive")
    return temperature, top_k


def _resolve_draft_distribution_temperature(
    train_result: Any,
    runtime_temperature: float,
) -> tuple[float, str]:
    semantics = getattr(train_result, "draft_logit_semantics", "raw_target_logits")
    if semantics == "raw_target_logits":
        return runtime_temperature, semantics
    if semantics != "processed_at_temperature":
        raise ValueError(f"Unsupported draft logit semantics {semantics!r}")
    calibration_temperature = getattr(train_result, "calibration_temperature", None)
    if calibration_temperature is None or not math.isfinite(float(calibration_temperature)) or float(calibration_temperature) <= 0.0:
        raise ValueError("draft calibration_temperature must be positive and finite")
    if not math.isclose(float(calibration_temperature), runtime_temperature, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Draft calibration temperature does not match runtime temperature")
    return 1.0, semantics


def _resolve_stochastic_block_config(
    sampling_config: dict[str, Any],
    draft_length: int,
    num_proposals: int | None,
    bonus_token: bool | None,
) -> tuple[int, bool]:
    configured_proposals = sampling_config.get("num_proposals")
    proposal_count = int(
        num_proposals
        if num_proposals is not None
        else (
            configured_proposals
            if configured_proposals is not None
            else draft_length - 1
        )
    )
    configured_bonus = sampling_config.get("bonus_token")
    has_bonus = bool(
        bonus_token
        if bonus_token is not None
        else (configured_bonus if configured_bonus is not None else True)
    )
    if proposal_count <= 0:
        raise ValueError("stochastic replay num_proposals must be positive")
    if proposal_count > draft_length:
        raise ValueError(
            "stochastic replay num_proposals cannot exceed draft_length"
        )
    if proposal_count + int(has_bonus) > draft_length:
        raise ValueError(
            "stochastic replay proposals plus bonus exceed draft_length"
        )
    return proposal_count, has_bonus


def _normalize_seeds(
    seeds: Iterable[int] | None,
    *,
    num_seeds: int,
    seed: int,
) -> list[int]:
    if seeds is None:
        if num_seeds < MINIMUM_SEEDS:
            raise ValueError(
                f"stochastic replay requires at least {MINIMUM_SEEDS} seeds"
            )
        return [seed + offset for offset in range(num_seeds)]
    normalized = [int(value) for value in seeds]
    if len(normalized) < MINIMUM_SEEDS:
        raise ValueError(
            f"stochastic replay requires at least {MINIMUM_SEEDS} seeds"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("stochastic replay seeds must be unique")
    return normalized


def _normalize_excluded_positions(
    exclude_positions: Iterable[int] | None,
) -> list[int]:
    """Normalize an explicit response-position population filter."""
    if exclude_positions is None:
        return []
    normalized = sorted({int(position) for position in exclude_positions})
    if any(position < 0 for position in normalized):
        raise ValueError("excluded response positions must be non-negative")
    return normalized


def _filter_grouped_positions(
    grouped: dict[str, dict[int, DVICaptureRecord]],
    exclude_positions: Sequence[int],
) -> int:
    """Remove only the requested non-terminal positions from replay scope."""
    excluded = set(exclude_positions)
    removed = 0
    if not excluded:
        return removed
    for positions in grouped.values():
        for position in tuple(positions):
            if position in excluded:
                del positions[position]
                removed += 1
    return removed


def _teacher_probability_bounds(
    record: DVICaptureRecord,
    token_id: int,
) -> tuple[float, float]:
    topk = dict(zip(record.verifier_topk_ids, record.verifier_topk_logprobs))
    if token_id in topk:
        probability = math.exp(float(topk[token_id]))
        return probability, probability
    return 0.0, float(record.verifier_residual_mass)


def _acceptance_probability_bounds(
    record: DVICaptureRecord,
    token_id: int,
    draft_probability: float,
) -> tuple[float, float]:
    if draft_probability <= 0.0 or not math.isfinite(draft_probability):
        raise ValueError("sampled draft probability must be positive and finite")
    lower, upper = _teacher_probability_bounds(record, token_id)
    return min(1.0, lower / draft_probability), min(1.0, upper / draft_probability)


def _build_position_data(
    reader: DVIArtifactReader,
    model: torch.nn.Module,
    grouped: dict[str, dict[int, DVICaptureRecord]],
    *,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> tuple[dict[str, dict[int, _PositionData]], list[dict[str, object]]]:
    by_request: dict[str, dict[int, _PositionData]] = {
        request.request_id: {} for request in reader.requests
    }
    coverage_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for request in reader.requests:
            positions = grouped[request.request_id]
            ordered_positions = sorted(positions)
            if not ordered_positions:
                continue
            hidden = torch.stack([
                positions[position].stage_0_hidden.float()
                for position in ordered_positions
            ]).to(device)
            # Keep the full-vocabulary projection and top-k processing on the
            # selected device. Only the bounded Q support is needed by the
            # replay loop; moving every position's full logits to CPU makes a
            # large artifact needlessly memory- and bandwidth-bound.
            logits = model(hidden).float()
            for row, position in enumerate(ordered_positions):
                record = positions[position]
                # This is the runtime Q contract: raw logits, temperature,
                # top-k support, then support renormalization.
                device_distribution = make_sparse_draft_distribution(
                    logits[row], temperature=temperature, top_k=top_k
                )
                distribution = SparseDraftDistribution(
                    token_ids=device_distribution.token_ids.cpu(),
                    probabilities=device_distribution.probabilities.cpu(),
                )
                teacher_support = set(record.verifier_topk_ids)
                coverage = sum(
                    probability
                    for token_id, probability in zip(
                        distribution.token_ids.tolist(),
                        distribution.probabilities.tolist(),
                    )
                    if token_id in teacher_support
                )
                overlap_lower = overlap_upper = 0.0
                for token_id, q_probability in zip(
                    distribution.token_ids.tolist(),
                    distribution.probabilities.tolist(),
                ):
                    p_lower, p_upper = _teacher_probability_bounds(record, token_id)
                    overlap_lower += min(p_lower, q_probability)
                    overlap_upper += min(p_upper, q_probability)
                by_request[request.request_id][position] = _PositionData(
                    request.request_id, position, record, distribution, coverage,
                    overlap_lower, overlap_upper
                )
                coverage_rows.append({
                    "request_id": request.request_id,
                    "position": position,
                    "mass": coverage,
                    "acceptance_overlap_lower": overlap_lower,
                    "acceptance_overlap_upper": overlap_upper,
                })
    return by_request, coverage_rows


def _contiguous_runs(positions: dict[int, _PositionData]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for position in sorted(positions):
        if current and position != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(position)
    if current:
        runs.append(current)
    return runs


def _simulate_path(
    reader: DVIArtifactReader,
    by_request: dict[str, dict[int, _PositionData]],
    *,
    draft_length: int,
    seed: int,
    lower_bound: bool,
    num_proposals: int | None = None,
    bonus_token: bool = False,
) -> dict[str, object]:
    accepted_tokens = proposed_tokens = sampled_tokens = cycles = 0
    expected_acceptance_sum = 0.0
    incomplete_windows = 0
    prefix_hist: Counter[int] = Counter()
    advancement_hist: Counter[int] = Counter()
    proposal_limit = draft_length if num_proposals is None else num_proposals
    if proposal_limit <= 0 or proposal_limit > draft_length:
        raise ValueError("num_proposals must be in [1, draft_length]")
    if proposal_limit + int(bonus_token) > draft_length:
        raise ValueError("num_proposals plus bonus exceed draft_length")

    for request in reader.requests:
        positions = by_request[request.request_id]
        if not positions:
            continue
        request_seed = _derive_request_seed(seed, request.request_id)
        for run in _contiguous_runs(positions):
            cursor, run_end = run[0], run[-1] + 1
            while cursor < run_end:
                window_positions = range(
                    cursor,
                    min(len(request.response_token_ids), cursor + proposal_limit),
                )
                window = [positions.get(position) for position in window_positions]
                if not window or any(item is None for item in window):
                    if cursor + len(window) < len(request.response_token_ids):
                        incomplete_windows += 1
                    break
                window_data = [item for item in window if item is not None]
                accepted = 0
                for item in window_data:
                    proposal = sample_sparse_draft_token(
                        item.draft_distribution,
                        request_seed=request_seed,
                        generation_id=0,
                        absolute_position=item.position,
                    )
                    q_probability = item.draft_distribution.probability(proposal)
                    lower, upper = _acceptance_probability_bounds(
                        item.record, proposal, q_probability
                    )
                    acceptance_probability = lower if lower_bound else upper
                    expected_acceptance_sum += acceptance_probability
                    proposed_tokens += 1
                    uniform = stateless_uniform(
                        request_seed, 0, item.position, DVIDrawKind.ACCEPTANCE
                    )
                    if uniform < acceptance_probability:
                        accepted += 1
                        accepted_tokens += 1
                    else:
                        break
                all_proposals_accepted = accepted == len(window_data)
                if all_proposals_accepted:
                    advancement = len(window_data) + int(bonus_token)
                else:
                    advancement = accepted + 1
                prefix_hist[accepted] += 1
                advancement_hist[advancement] += 1
                sampled_tokens += advancement
                cycles += 1
                cursor += advancement
                if cursor >= run_end or cursor not in positions:
                    break

    return {
        "seed": seed,
        "accepted_tokens": accepted_tokens,
        "proposed_tokens": proposed_tokens,
        "sampled_tokens": sampled_tokens,
        "cycles": cycles,
        "expected_acceptance": expected_acceptance_sum / max(proposed_tokens, 1),
        "observed_acceptance": accepted_tokens / max(proposed_tokens, 1),
        "mean_advancement": sampled_tokens / max(cycles, 1),
        "prefix_hist": dict(sorted(prefix_hist.items())),
        "advancement_hist": dict(sorted(advancement_hist.items())),
        "incomplete_windows": incomplete_windows,
        "draft_length": draft_length,
        "lower_bound_path": lower_bound,
        "num_proposals": proposal_limit,
        "bonus_token": bonus_token,
        "max_advancement": proposal_limit + int(bonus_token),
    }


def _path_summary(seed_results: Sequence[dict[str, object]]) -> dict[str, object]:
    def values(name: str) -> list[float]:
        return [float(result[name]) for result in seed_results]

    max_prefix = max(
        (max((int(key) for key in result["prefix_hist"]), default=0)
         for result in seed_results),
        default=0,
    )
    prefix_distribution: dict[int, float] = {}
    prefix_ci95: dict[int, list[float]] = {}
    for prefix in range(max_prefix + 1):
        per_seed = [
            float(result["prefix_hist"].get(prefix, 0)) / max(int(result["cycles"]), 1)
            for result in seed_results
        ]
        summary = _mean_ci95(per_seed)
        prefix_distribution[prefix] = float(summary["mean"])
        prefix_ci95[prefix] = list(summary["ci95"])
    return {
        "expected_acceptance": _mean_ci95(values("expected_acceptance")),
        "observed_acceptance": _mean_ci95(values("observed_acceptance")),
        "mean_advancement": _mean_ci95(values("mean_advancement")),
        "prefix_distribution": prefix_distribution,
        "prefix_distribution_ci95": prefix_ci95,
        "seed_results": list(seed_results),
    }


def replay_stochastic_bound_draft_head(
    result_dir: str | Path,
    artifact_dir: str | Path,
    base_projection: torch.Tensor | None = None,
    *,
    draft_length: int | None = None,
    draft_temperature: float | None = None,
    draft_top_k: int | None = None,
    num_proposals: int | None = None,
    bonus_token: bool | None = None,
    seeds: Iterable[int] | None = None,
    exclude_positions: Iterable[int] | None = None,
    num_seeds: int = MINIMUM_SEEDS,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, object]:
    """Replay stochastic DVI with target-probability lower/upper bounds."""
    result_path, artifact_path = Path(result_dir), Path(artifact_dir)
    reader = DVIArtifactReader(artifact_path)
    result = load_train_result(result_path)
    if reader.manifest.base_checkpoint_hash != result.source_base_checkpoint_hash:
        raise ValueError(
            "Replay artifact base checkpoint does not match draft training"
        )
    if not _sampling_is_stochastic(reader.manifest.sampling_config):
        raise ValueError(
            "stochastic bound replay requires a stochastic capture artifact"
        )

    temperature, top_k = _resolve_runtime_draft_config(
        reader.manifest.sampling_config, draft_temperature, draft_top_k
    )
    distribution_temperature, draft_logit_semantics = (
        _resolve_draft_distribution_temperature(result, temperature)
    )
    replay_seeds = _normalize_seeds(seeds, num_seeds=num_seeds, seed=seed)
    excluded_position_values = _normalize_excluded_positions(exclude_positions)
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must select cpu or cuda")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA replay requested but CUDA is unavailable")

    model = _load_replay_model(result_path, base_projection, target_device)
    k = draft_length if draft_length is not None else reader.manifest.draft_length
    if k <= 0:
        raise ValueError("draft_length must be positive")
    if k > reader.manifest.draft_length:
        raise ValueError(
            f"draft_length {k} exceeds artifact draft_length "
            f"{reader.manifest.draft_length}"
        )

    proposal_count, has_bonus = _resolve_stochastic_block_config(
        reader.manifest.sampling_config,
        k,
        num_proposals,
        bonus_token,
    )

    grouped, capture_records, terminal_records = _group_capture_records(reader)
    artifact_nonterminal_capture_records = capture_records
    artifact_capture_records = capture_records + terminal_records
    excluded_records = _filter_grouped_positions(
        grouped, excluded_position_values
    )
    capture_records -= excluded_records
    by_request, coverage_rows = _build_position_data(

        reader,
        model,
        grouped,
        temperature=distribution_temperature,
        top_k=top_k,
        device=target_device,

    )
    lower_results = [
        _simulate_path(
            reader,
            by_request,
            draft_length=k,
            seed=replay_seed,
            lower_bound=True,
            num_proposals=proposal_count,
            bonus_token=has_bonus,
        )
        for replay_seed in replay_seeds
    ]
    upper_results = [
        _simulate_path(
            reader,
            by_request,
            draft_length=k,
            seed=replay_seed,
            lower_bound=False,
            num_proposals=proposal_count,
            bonus_token=has_bonus,
        )
        for replay_seed in replay_seeds
    ]
    lower_summary = _path_summary(lower_results)
    upper_summary = _path_summary(upper_results)

    coverage_values = [float(row["mass"]) for row in coverage_rows]
    overlap_lower = [float(row["acceptance_overlap_lower"]) for row in coverage_rows]
    overlap_upper = [float(row["acceptance_overlap_upper"]) for row in coverage_rows]
    response_tokens = sum(
        len(request.response_token_ids) for request in reader.requests
    )
    covered_response_tokens = sum(len(positions) for positions in grouped.values())
    lower_acceptance = lower_summary["expected_acceptance"]
    upper_acceptance = upper_summary["expected_acceptance"]
    lower_advancement = lower_summary["mean_advancement"]
    upper_advancement = upper_summary["mean_advancement"]

    return {
        "schema_version": STOCHASTIC_REPLAY_SCHEMA_VERSION,
        "mode": "stochastic_bound",
        "result_dir": str(result_path.resolve()),
        "artifact_dir": str(artifact_path.resolve()),
        "artifact_manifest_sha256": _sha256_file(artifact_path / "manifest.json"),
        "num_requests": len(reader.requests),
        "evaluation_scope": (
            "runtime_metric_population"
            if excluded_position_values
            else "all_nonterminal_capture_records"
        ),
        "artifact_capture_records": artifact_capture_records,
        "artifact_nonterminal_capture_records": artifact_nonterminal_capture_records,
        "capture_records": capture_records,
        "included_records": capture_records,
        "excluded_records": excluded_records,
        "excluded_position_values": excluded_position_values,
        "terminal_records_excluded": terminal_records,
        "response_tokens": response_tokens,
        "covered_response_tokens": covered_response_tokens,
        "capture_coverage": covered_response_tokens / max(response_tokens, 1),
        "replay_scope": "contiguous_capture_windows",
        "bound_scope": "topk_exact_residual_mass_outside_support",
        "stochastic_replay_exact": False,
        "stochastic_replay_reason": (
            "artifact v1 lacks teacher probability for arbitrary draft support; "
            "outside top-k uses P(x)=0/residual"
        ),
        "draft_temperature": temperature,
        "draft_distribution_temperature": distribution_temperature,
        "draft_top_k": top_k,
        "draft_logit_semantics": draft_logit_semantics,
        "draft_calibration_temperature": result.calibration_temperature,
        "num_proposals": proposal_count,
        "bonus_token": has_bonus,
        "max_advancement": proposal_count + int(has_bonus),
        "draft_q_contract": (
            "raw draft logits -> runtime temperature -> top-k -> renormalize"
            if draft_logit_semantics == "raw_target_logits"
            else "processed draft logits at calibration temperature -> top-k -> renormalize"
        ),
        "draft_mass_on_teacher_support": {
            "summary": _describe(coverage_values),
            "per_position": coverage_rows,
        },
        "expected_acceptance_overlap_bounds": {
            "lower": _describe(overlap_lower),
            "upper": _describe(overlap_upper),
        },
        "metric_semantics": {
            "distribution_overlap": (
                "Use expected_acceptance_overlap_bounds for draft ranking; "
                "these are distribution-level sum-min bounds.",
            ),
            "replay_path": (
                "Use bound_paths expected_acceptance for advancement estimation; "
                "it is sampled along lower and upper paths.",
            ),
        },
        "tv_proxy_bounds": {
            "lower": _describe([1.0 - value for value in overlap_upper]),
            "upper": _describe([1.0 - value for value in overlap_lower]),
        },
        "expected_acceptance_lower": float(lower_acceptance["mean"]),
        "expected_acceptance_upper": float(upper_acceptance["mean"]),
        "expected_acceptance_lower_ci95": lower_acceptance["ci95"],
        "expected_acceptance_upper_ci95": upper_acceptance["ci95"],
        "mean_advancement_lower": float(lower_advancement["mean"]),
        "mean_advancement_upper": float(upper_advancement["mean"]),
        "mean_advancement_lower_ci95": lower_advancement["ci95"],
        "mean_advancement_upper_ci95": upper_advancement["ci95"],
        "prefix_distribution_lower": lower_summary["prefix_distribution"],
        "prefix_distribution_upper": upper_summary["prefix_distribution"],
        "prefix_distribution_lower_ci95": lower_summary["prefix_distribution_ci95"],
        "prefix_distribution_upper_ci95": upper_summary["prefix_distribution_ci95"],
        "bound_paths": {"lower": lower_summary, "upper": upper_summary},
        "uncertainty": {
            "num_seeds": len(replay_seeds),
            "seeds": replay_seeds,
            "confidence_level": 0.95,
            "ci_method": "two-sided Student-t over per-seed means",
            "single_seed_numbers_are_not_reported": True,
        },
        "device": str(target_device),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-projection-key")
    parser.add_argument("--draft-length", type=int)
    parser.add_argument("--draft-temperature", type=float)
    parser.add_argument("--draft-top-k", type=int)
    parser.add_argument("--num-proposals", type=int)
    parser.add_argument(
        "--exclude-position",
        action="append",
        type=int,
        default=None,
        help="Exclude a response position from the replay population; repeatable.",
    )
    parser.add_argument(
        "--bonus-token",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=MINIMUM_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    base_projection = None
    if args.base_checkpoint is not None:
        base_projection = load_base_projection(
            args.base_checkpoint, args.base_projection_key
        )
    report = replay_stochastic_bound_draft_head(
        args.result_dir, args.artifact_dir, base_projection,
        draft_length=args.draft_length,
        draft_temperature=args.draft_temperature,
        draft_top_k=args.draft_top_k,
        num_proposals=args.num_proposals,
        bonus_token=args.bonus_token,
        exclude_positions=args.exclude_position,
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=args.device,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
