"""Fail-closed replay evaluation for dense schema-v2 DVI artifacts.

The evaluator deliberately groups rows by cycle identity.  It never rebuilds
the population from contiguous response positions, which is invalid for the
sparse capture produced by the runtime.
"""

from __future__ import annotations

import json
import math
import struct
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from vllm.v1.worker.gpu.split_dvi.artifact import (
    DVIArtifactError,
    DVIArtifactReader,
    DVIEvaluationRecord,
)

_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1


def _philox_uint32(seed: int, offset: int) -> int:
    c0, c1 = offset & _MASK32, (offset >> 32) & _MASK32
    c2 = c3 = 0
    k0, k1 = seed & _MASK32, (seed >> 32) & _MASK32
    for _ in range(10):
        next_c0 = (((0xCD9E8D57 * c2) >> 32) ^ c1 ^ k0) & _MASK32
        next_c2 = (((0xD2511F53 * c0) >> 32) ^ c3 ^ k1) & _MASK32
        c1 = (0xCD9E8D57 * c2) & _MASK32
        c3 = (0xD2511F53 * c0) & _MASK32
        c0, c2 = next_c0, next_c2
        k0 = (k0 + 0x9E3779B9) & _MASK32
        k1 = (k1 + 0xBB67AE85) & _MASK32
    return c0


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _triton_uniform(seed: int, offset: int) -> float:
    value = _philox_uint32(seed, offset)
    signed = value if value < (1 << 31) else value - (1 << 32)
    positive = (~signed) & _MASK32 if signed < 0 else signed
    scale = _float32(4.6566127342e-10)
    value = _float32(_float32(float(positive)) * scale)
    return max(value, scale)


def _proposal_seed(request_seed: int, generation_id: int) -> int:
    value = (
        (request_seed & _MASK64)
        ^ 0xD1B54A32D192ED03
        ^ ((generation_id * 0x9E3779B97F4A7C15) & _MASK64)
    )
    return value - (1 << 64) if value >= (1 << 63) else value


def _runtime_proposal(row: DVIEvaluationRecord, q_dist: dict[int, float]) -> int:
    if row.request_seed is None or row.rng_position is None:
        raise DVIArtifactError(
            "v2 resimulation requires request_seed and rng_position"
        )
    nested_seed = _philox_uint32(
        _proposal_seed(row.request_seed, row.generation_id), row.rng_position
    )
    logits_by_token = dict(
        zip(row.draft_support_token_ids, row.draft_support_logits)
    )
    best_token: int | None = None
    best_value = -math.inf
    for support_index, token in enumerate(q_dist):
        uniform = _triton_uniform(nested_seed, support_index)
        log_one_minus = _float32(math.log1p(-uniform))
        gumbel = _float32(-_float32(math.log(_float32(-log_one_minus))))
        value = _float32(_float32(float(logits_by_token[token])) + gumbel)
        if value > best_value:
            best_token, best_value = token, value
    if best_token is None:
        raise DVIArtifactError("v2 resimulation has an empty draft distribution")
    return best_token


@dataclass(frozen=True)
class ReplayMetrics:
    cycles: int
    expected_first_acceptance: float
    realized_first_acceptance: float
    proposal_acceptance: float
    prefix_histogram: dict[str, int]
    mean_advancement: float
    coverage: float


@dataclass(frozen=True)
class V2ReplayReport:
    schema_version: str
    exact_reproduction: bool
    exact_cycles: int
    total_cycles: int
    mismatches: tuple[str, ...]
    proposal_reproduction: bool
    reproduced_proposals: int
    total_proposals: int
    proposal_mismatches: tuple[str, ...]
    all_cycles: ReplayMetrics
    without_first_block: ReplayMetrics
    resimulation: ReplayMetrics
    resimulation_without_first_block: ReplayMetrics

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mismatches"] = list(self.mismatches)
        data["proposal_mismatches"] = list(self.proposal_mismatches)
        return data

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _q_distribution(row: DVIEvaluationRecord, *, temperature: float, top_k: int) -> dict[int, float]:
    """Return Q over the captured support after runtime renormalization."""
    if row.draft_token_id is None:
        return {}
    if not row.draft_support_token_ids:
        raise DVIArtifactError("v2 replay requires draft support logits")
    if len(row.draft_support_token_ids) != len(row.draft_support_logits):
        raise DVIArtifactError("draft support ids/logits length mismatch")
    if not math.isfinite(temperature) or temperature <= 0:
        raise DVIArtifactError("sampling temperature must be positive and finite")
    entries = list(zip(row.draft_support_token_ids, row.draft_support_logits))
    entries.sort(key=lambda item: item[1], reverse=True)
    if top_k > 0:
        entries = entries[:top_k]
    scaled = [float(logit) / temperature for _, logit in entries]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    denom = sum(weights)
    if not math.isfinite(denom) or denom <= 0:
        raise DVIArtifactError("invalid draft support logits")
    return {token: weight / denom for (token, _), weight in zip(entries, weights)}


def _teacher_probability(row: DVIEvaluationRecord) -> float:
    if not row.teacher_probs_on_draft_support:
        raise DVIArtifactError("v2 replay requires exact teacher support probabilities")
    try:
        index = row.draft_support_token_ids.index(row.draft_token_id)  # type: ignore[arg-type]
    except ValueError:
        return 0.0
    return float(row.teacher_probs_on_draft_support[index])


def _group_cycles(records: Iterable[DVIEvaluationRecord]) -> list[list[DVIEvaluationRecord]]:
    groups: dict[tuple[str, int], list[DVIEvaluationRecord]] = defaultdict(list)
    for row in records:
        groups[(row.request_id, row.cycle_id)].append(row)
    if not groups:
        raise DVIArtifactError("v2 artifact contains no evaluation cycles")
    result = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row.row_index)
        count = rows[0].num_proposals
        expected = list(range(count + 1))
        actual = [row.row_index for row in rows]
        if actual != expected:
            raise DVIArtifactError(
                f"evaluation cycle rows are not dense: cycle={key}, expected={expected}, actual={actual}"
            )
        for index, row in enumerate(rows):
            expected_kind = "proposal" if index < count else "bonus"
            if row.num_proposals != count or row.row_kind != expected_kind:
                raise DVIArtifactError(f"evaluation cycle row kind/count mismatch: cycle={key}")
        result.append(rows)
    return result


def _proposal_reproduction(
    cycles: list[list[DVIEvaluationRecord]], *, temperature: float, top_k: int
) -> tuple[int, int, tuple[str, ...]]:
    """Reproduce every recorded proposal with the production RNG contract."""
    reproduced = 0
    total = 0
    mismatches: list[str] = []
    for rows in cycles:
        for row in rows[:-1]:
            total += 1
            sampled = _runtime_proposal(
                row,
                _q_distribution(row, temperature=temperature, top_k=top_k),
            )
            if sampled == row.draft_token_id:
                reproduced += 1
            else:
                mismatches.append(
                    f"{row.request_id}/{row.cycle_id}/{row.row_index}: "
                    f"recorded={row.draft_token_id}, reproduced={sampled}"
                )
    return reproduced, total, tuple(mismatches)


def _metrics(cycles: list[list[DVIEvaluationRecord]], *, temperature: float, top_k: int,
             simulated: bool = False, seed: int = 0) -> ReplayMetrics:
    if not cycles:
        return ReplayMetrics(0, 0.0, 0.0, 0.0, {str(i): 0 for i in range(5)}, 0.0, 0.0)
    expected = realized = proposals = accepted_total = coverage = advancement = 0.0
    histogram: Counter[int] = Counter()
    for rows in cycles:
        proposal_rows = rows[:-1]
        first = proposal_rows[0]
        q_first = _q_distribution(first, temperature=temperature, top_k=top_k)
        p_first = dict(zip(first.draft_support_token_ids, first.teacher_probs_on_draft_support))
        expected += sum(min(q_first.get(token, 0.0), float(prob)) for token, prob in p_first.items())
        coverage += sum(
            q_first.get(token, 0.0) for token in first.verifier_topk_ids
        )
        accepted_count = 0
        for row in proposal_rows:
            if not simulated and row.accepted is None:
                break
            q_dist = _q_distribution(row, temperature=temperature, top_k=top_k)
            if simulated:
                sampled = _runtime_proposal(row, q_dist)
                p = dict(
                    zip(
                        row.draft_support_token_ids,
                        row.teacher_probs_on_draft_support,
                    )
                ).get(sampled, 0.0)
                q = q_dist.get(sampled, 0.0)
                if row.request_seed is None or row.rng_position is None:
                    raise DVIArtifactError(
                        "v2 resimulation requires request_seed and rng_position"
                    )
                accept_u = _triton_uniform(
                    row.request_seed, row.rng_position
                )
                accepted = q > 0.0 and accept_u < min(1.0, p / q)
            else:
                accepted = bool(row.accepted)
            proposals += 1.0
            accepted_total += int(accepted)
            accepted_count += int(accepted)
            if not accepted:
                break
        realized += float(bool(accepted_count > 0))
        histogram[min(accepted_count, 4)] += 1
        advancement += accepted_count + 1
    n = float(len(cycles))
    return ReplayMetrics(len(cycles), expected / n, realized / n,
                         accepted_total / proposals if proposals else 0.0,
                         {str(i): histogram.get(i, 0) for i in range(5)},
                         advancement / n, coverage / n)

def evaluate_v2_artifact(artifact_dir: str | Path, *, seed: int = 0,
                         include_first_block: bool = True) -> V2ReplayReport:
    """Evaluate a schema-v2 artifact and return exact and resimulated metrics."""
    reader = DVIArtifactReader(artifact_dir)
    if reader.manifest.schema_version != "v2":
        raise DVIArtifactError("v2 replay requires schema_version='v2'")
    cycles = _group_cycles(reader.iter_evaluation_records())
    mismatches: list[str] = []
    exact_cycles = 0
    for rows in cycles:
        first = rows[0]
        expected = first.accepted_count + 1 if first.accepted_count < first.num_proposals else first.num_proposals + 1
        cycle_label = f"{first.request_id}/{first.cycle_id}"
        if first.advancement != expected or len(first.committed_token_ids) != expected:
            mismatches.append(f"{cycle_label}: advancement")
        accepted_prefix = [row.draft_token_id for row in rows[:-1][:first.accepted_count]]
        if first.committed_token_ids[:first.accepted_count] != accepted_prefix:
            mismatches.append(f"{cycle_label}: committed prefix")
        if first.accepted_count < first.num_proposals and first.correction_token_id != first.committed_token_ids[-1]:
            mismatches.append(f"{cycle_label}: correction token")
        if first.terminal_token_id != first.committed_token_ids[-1]:
            mismatches.append(f"{cycle_label}: terminal token")
        if not any(item for item in mismatches if item.startswith(f"{cycle_label}:")):
            exact_cycles += 1
    sampling = reader.manifest.sampling_config
    # Evaluation rows store raw draft-head logits. The runtime rejection
    # sampler samples those logits at temperature 1.0; rollout temperature
    # applies only to the processed target distribution.
    temperature = float(
        sampling.get("draft_temperature", reader.manifest.draft_temperature)
    )
    top_k = int(
        sampling.get("draft_top_k", sampling.get("top_k", 0)) or 0
    )
    reproduced_proposals, total_proposals, proposal_mismatches = (
        _proposal_reproduction(
            cycles, temperature=temperature, top_k=top_k
        )
    )
    first_cycle_by_request: dict[str, int] = {}
    for rows in cycles:
        key = rows[0].request_id
        first_cycle_by_request[key] = min(first_cycle_by_request.get(key, rows[0].cycle_id), rows[0].cycle_id)
    remainder = [rows for rows in cycles if rows[0].cycle_id != first_cycle_by_request[rows[0].request_id]]
    scoped_cycles = cycles if include_first_block else remainder
    return V2ReplayReport(
        "dvi-v2-replay",
        exact_cycles == len(cycles), exact_cycles, len(cycles), tuple(mismatches),
        reproduced_proposals == total_proposals,
        reproduced_proposals,
        total_proposals,
        proposal_mismatches,
        _metrics(scoped_cycles, temperature=temperature, top_k=top_k),
        _metrics(remainder, temperature=temperature, top_k=top_k),
        _metrics(cycles, temperature=temperature, top_k=top_k, simulated=True, seed=seed),
        _metrics(remainder, temperature=temperature, top_k=top_k, simulated=True, seed=seed),
    )


__all__ = ["ReplayMetrics", "V2ReplayReport", "evaluate_v2_artifact"]
