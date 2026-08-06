"""Replay a trained DVI draft head against captured target trajectories.

The capture artifact stores target boundary hidden states and verifier top-1
tokens. For greedy/no-bonus DVI this is enough to replay a block exactly as
long as the required positions are contiguous. A sampled artifact is not
silently treated as greedy: strict stochastic replay needs the target
probability for every proposed token, which the current top-k-plus-residual
schema does not retain.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors

from vllm.v1.worker.gpu.split_dvi.artifact import (
    DVIArtifactReader,
    DVICaptureRecord,
)

from .draft_head import SplitDVIDraftHead
from .offline import (
    _sha256_file,
    _sha256_tensor,
    load_base_projection,
    load_train_result,
)


REPLAY_SCHEMA_VERSION = "dvi-replay-v1"


def _resolve_base_projection(
    result_dir: Path,
    base_projection: torch.Tensor | None,
) -> torch.Tensor:
    checkpoint_path = result_dir / "draft_head.safetensors"
    checkpoint = load_safetensors(str(checkpoint_path))
    embedded = checkpoint.get("base_projection.weight")
    if base_projection is None:
        if embedded is None:
            raise ValueError(
                "A compact draft checkpoint requires base_projection; pass "
                "--base-checkpoint or base_projection explicitly"
            )
        base_projection = embedded
    base_projection = base_projection.float().cpu().contiguous()
    if embedded is not None and _sha256_tensor(embedded) != _sha256_tensor(
        base_projection
    ):
        raise ValueError("Embedded and supplied base projections do not match")
    return base_projection


def _load_replay_model(
    result_dir: str | Path,
    base_projection: torch.Tensor | None,
    device: torch.device,
) -> SplitDVIDraftHead:
    result_path = Path(result_dir)
    result = load_train_result(result_path)
    base_projection = _resolve_base_projection(result_path, base_projection)
    expected_shape = (result.vocab_size, result.hidden_size)
    if tuple(base_projection.shape) != expected_shape:
        raise ValueError(
            f"base projection shape {tuple(base_projection.shape)} does not "
            f"match expected shape {expected_shape}"
        )
    if _sha256_tensor(base_projection) != result.base_projection_sha256:
        raise ValueError("Base projection checksum mismatch")

    checkpoint = load_safetensors(str(result_path / "draft_head.safetensors"))
    model = SplitDVIDraftHead(
        base_projection,
        rank=result.rank,
        alpha=result.alpha,
        norm=result.norm,
        seed=result.seed,
    ).to(device)
    with torch.no_grad():
        model.lora_a.copy_(checkpoint["lora_A.weight"])
        model.lora_b.copy_(checkpoint["lora_B.weight"])
        if model.norm_weight is not None:
            model.norm_weight.copy_(checkpoint["norm.weight"])
    model.eval()
    return model


def _sampling_is_stochastic(sampling_config: dict[str, Any]) -> bool:
    """Return whether rollout metadata describes non-greedy sampling."""
    temperature = sampling_config.get("rollout_temperature")
    if temperature is None:
        temperature = sampling_config.get("temperature")
    if temperature is not None and float(temperature) != 0.0:
        method = sampling_config.get("method")
        if method not in {"greedy", "argmax"}:
            return True
    method = sampling_config.get("rollout_method")
    return method in {"sample", "stochastic", "rejection_sampling"}


def _group_capture_records(
    reader: DVIArtifactReader,
) -> tuple[dict[str, dict[int, DVICaptureRecord]], int, int]:
    grouped: dict[str, dict[int, DVICaptureRecord]] = {
        request.request_id: {} for request in reader.requests
    }
    terminal_records = 0
    capture_records = 0
    for record in reader.iter_capture_records():
        response_length = len(reader.request_index[record.request_id].response_token_ids)
        if record.position >= response_length:
            terminal_records += 1
            continue
        positions = grouped[record.request_id]
        if record.position in positions:
            raise ValueError(
                f"Duplicate capture position {record.request_id!r}: "
                f"{record.position}"
            )
        positions[record.position] = record
        capture_records += 1
    return grouped, capture_records, terminal_records


def _record_window(
    positions: dict[int, DVICaptureRecord],
    start: int,
    response_length: int,
    draft_length: int,
) -> list[DVICaptureRecord]:
    window: list[DVICaptureRecord] = []
    for position in range(start, min(response_length, start + draft_length)):
        record = positions.get(position)
        if record is None:
            break
        window.append(record)
    return window


def replay_greedy_draft_head(
    result_dir: str | Path,
    artifact_dir: str | Path,
    base_projection: torch.Tensor | None = None,
    *,
    draft_length: int | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Replay a draft head on contiguous captured greedy target windows.

    Each replay window uses captured rows ``h_t ... h_{t+k-1}``. The draft
    head predicts all rows, but the result is consumed sequentially: the
    first mismatch stops the accepted prefix and advancement follows the
    verifier's greedy/no-bonus formula. Windows are partitioned into
    non-overlapping cycles, so a sparse artifact reports its coverage instead
    of presenting independent per-token matches as a full rollout result.
    """
    result_path = Path(result_dir)
    artifact_path = Path(artifact_dir)
    result = load_train_result(result_path)
    reader = DVIArtifactReader(artifact_path)
    if reader.manifest.base_checkpoint_hash != result.source_base_checkpoint_hash:
        raise ValueError(
            "Replay artifact base checkpoint does not match draft training"
        )
    if _sampling_is_stochastic(reader.manifest.sampling_config):
        raise ValueError(
            "Stochastic replay is unsupported by artifact schema v1: capture "
            "must include target probability on every draft support token"
        )

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

    grouped, capture_records, terminal_records = _group_capture_records(reader)
    sampled_hist: Counter[int] = Counter()
    accepted_prefix_hist: Counter[int] = Counter()
    accepted_tokens = 0
    proposed_tokens = 0
    sampled_tokens = 0
    cycles = 0
    replayable_requests = 0
    complete_requests = 0
    incomplete_windows = 0
    gap_count = 0
    response_tokens = 0
    covered_response_tokens = 0
    target_top1_alignment_matches = 0
    target_top1_alignment_samples = 0
    draft_top1_matches_target = 0

    with torch.no_grad():
        for request in reader.requests:
            response_length = len(request.response_token_ids)
            response_tokens += response_length
            positions = grouped[request.request_id]
            covered_response_tokens += len(positions)
            if not positions:
                continue

            ordered_positions = sorted(positions)
            hidden = torch.stack(
                [
                    positions[position].stage_0_hidden.float()
                    for position in ordered_positions
                ]
            ).to(target_device)
            predictions = model(hidden).argmax(dim=-1).cpu().tolist()
            draft_by_position = dict(zip(ordered_positions, predictions))
            for position, prediction in draft_by_position.items():
                target = positions[position].verifier_top1_id
                draft_top1_matches_target += int(prediction == target)
                target_top1_alignment_samples += 1
                target_top1_alignment_matches += int(
                    target == request.response_token_ids[position]
                )

            if len(positions) == response_length and set(positions) == set(
                range(response_length)
            ):
                complete_requests += 1

            ordered_runs: list[list[int]] = []
            run: list[int] = []
            for position in ordered_positions:
                if run and position != run[-1] + 1:
                    ordered_runs.append(run)
                    gap_count += 1
                    run = []
                run.append(position)
            if run:
                ordered_runs.append(run)

            request_had_cycle = False
            for run in ordered_runs:
                cursor = run[0]
                run_end = run[-1] + 1
                while cursor < run_end:
                    window = _record_window(
                        positions, cursor, response_length, k
                    )
                    if len(window) < k and cursor + len(window) < response_length:
                        incomplete_windows += 1
                        break
                    if not window:
                        break

                    draft_ids = [draft_by_position[rec.position] for rec in window]
                    target_ids = [rec.verifier_top1_id for rec in window]
                    accepted = 0
                    for draft_id, target_id in zip(draft_ids, target_ids):
                        if draft_id != target_id:
                            break
                        accepted += 1

                    proposal_count = len(window)
                    advancement = (
                        proposal_count
                        if accepted == proposal_count
                        else accepted + 1
                    )
                    sampled_hist[advancement] += 1
                    accepted_prefix_hist[accepted] += 1
                    accepted_tokens += accepted
                    proposed_tokens += proposal_count
                    sampled_tokens += advancement
                    cycles += 1
                    request_had_cycle = True
                    cursor += advancement
                    if cursor >= run_end:
                        break
                    if cursor not in positions:
                        break
            replayable_requests += int(request_had_cycle)

    capture_coverage = (
        covered_response_tokens / response_tokens if response_tokens else 0.0
    )
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "mode": "greedy",
        "result_dir": str(result_path.resolve()),
        "artifact_dir": str(artifact_path.resolve()),
        "artifact_manifest_sha256": _sha256_file(artifact_path / "manifest.json"),
        "num_requests": len(reader.requests),
        "capture_records": capture_records,
        "terminal_records_excluded": terminal_records,
        "response_tokens": response_tokens,
        "covered_response_tokens": covered_response_tokens,
        "capture_coverage": capture_coverage,
        "replay_scope": "contiguous_capture_windows",
        "complete_trajectory_requests": complete_requests,
        "replayable_requests": replayable_requests,
        "cycles": cycles,
        "incomplete_windows": incomplete_windows,
        "capture_gaps": gap_count,
        "draft_length": k,
        "drafted_tokens": proposed_tokens,
        "proposed_tokens": proposed_tokens,
        "accepted_tokens": accepted_tokens,
        "sampled_tokens": sampled_tokens,
        "sampled_hist": dict(sorted(sampled_hist.items())),
        "accepted_prefix_hist": dict(sorted(accepted_prefix_hist.items())),
        "mean_advancement": sampled_tokens / max(cycles, 1),
        "proposal_acceptance": accepted_tokens / max(proposed_tokens, 1),
        "first_token_acceptance": (
            1.0 - accepted_prefix_hist.get(0, 0) / max(cycles, 1)
        ),
        "draft_top1_target_match_rate": draft_top1_matches_target
        / max(capture_records, 1),
        "target_top1_response_alignment": target_top1_alignment_matches
        / max(target_top1_alignment_samples, 1),
        "target_top1_response_alignment_samples": target_top1_alignment_samples,
        "stochastic_replay_supported": False,
        "stochastic_replay_reason": (
            "schema v1 stores verifier top-k plus residual mass, not the target "
            "probability for arbitrary draft support tokens"
        ),
        "device": str(target_device),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-projection-key")
    parser.add_argument("--draft-length", type=int)
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
    report = replay_greedy_draft_head(
        args.result_dir,
        args.artifact_dir,
        base_projection,
        draft_length=args.draft_length,
        device=args.device,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
