"""CPU tests for exact greedy DVI replay accounting."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as save_safetensors

WORKSPACE = Path(__file__).resolve().parents[3].parent
sys.path.insert(0, str(WORKSPACE / "vllm"))
sys.path.insert(0, str(WORKSPACE / "verl"))

from vllm.v1.worker.gpu.split_dvi.artifact import (  # noqa: E402
    DVIArtifactManifest,
    DVIArtifactWriter,
    DVICaptureRecord,
    DVIRequestRecord,
)
from verl.workers.split_training.dvi import (  # noqa: E402
    DVITrainResult,
    replay_greedy_draft_head,
)
from verl.workers.split_training.dvi.offline import (  # noqa: E402
    _sha256_file,
    _sha256_tensor,
)


def _make_replay_case(
    tmp_path: Path,
    *,
    captured_positions: list[int] | None = None,
    biased_position: int | None = None,
    stochastic: bool = False,
) -> tuple[Path, Path]:
    hidden_size = 8
    vocab_size = 8
    draft_length = 3
    response = list(range(7))
    base_projection = torch.eye(vocab_size, hidden_size)
    base_hash = _sha256_tensor(base_projection)
    manifest = DVIArtifactManifest(
        base_checkpoint="synthetic",
        base_checkpoint_hash=f"sha256:{base_hash}",
        head_adapter_hash="sha256:head",
        tail_adapter_hash="sha256:tail",
        tokenizer_revision="synthetic-v1",
        split_stage_0_size=1,
        split_stage_2_size=1,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        draft_length=draft_length,
        dtype="float32",
        sampling_config=(
            {"rollout_temperature": 0.7}
            if stochastic
            else {"method": "greedy"}
        ),
        max_response_length=len(response),
    )
    request = DVIRequestRecord(
        request_id="request-0",
        prompt_token_ids=[1],
        response_token_ids=response,
    )
    positions = (
        list(range(len(response)))
        if captured_positions is None
        else captured_positions
    )
    capture_records = []
    for position in positions:
        hidden = base_projection[position].clone()
        logits = hidden @ base_projection.T
        logprobs = torch.log_softmax(logits, dim=-1)
        topk_logprobs, topk_ids = torch.topk(logprobs, 4)
        capture_records.append(
            DVICaptureRecord(
                request_id=request.request_id,
                position=position,
                stage_0_hidden=hidden,
                verifier_topk_ids=topk_ids.tolist(),
                verifier_topk_logprobs=topk_logprobs.tolist(),
                verifier_residual_mass=float(1.0 - topk_logprobs.exp().sum()),
                verifier_top1_id=response[position],
            )
        )

    artifact_dir = tmp_path / "artifact"
    with DVIArtifactWriter(artifact_dir, manifest, shard_size=2) as writer:
        writer.write_requests([request])
        writer.write_capture_records(capture_records)

    checkpoint_dir = tmp_path / "result"
    checkpoint_dir.mkdir()
    lora_a = torch.zeros(1, hidden_size)
    lora_b = torch.zeros(vocab_size, 1)
    if biased_position is not None:
        lora_a[0, biased_position] = 1.0
        lora_b[7, 0] = 2.0
    checkpoint_path = checkpoint_dir / "draft_head.safetensors"
    save_safetensors(
        {
            "base_projection.weight": base_projection,
            "lora_A.weight": lora_a,
            "lora_B.weight": lora_b,
        },
        str(checkpoint_path),
    )
    manifest_hash = _sha256_file(artifact_dir / "manifest.json")
    result = DVITrainResult(
        schema_version="v1",
        source_artifact_manifest_sha256=manifest_hash,
        source_base_checkpoint_hash=f"sha256:{base_hash}",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        rank=1,
        alpha=1.0,
        norm="none",
        learning_rate=1.0,
        epochs=1,
        batch_size=1,
        seed=0,
        num_records=len(capture_records),
        initial_kl=1.0,
        final_kl=1.0,
        initial_top1_accuracy=1.0,
        final_top1_accuracy=1.0,
        mean_topk_mass=0.5,
        base_projection_sha256=base_hash,
        draft_checkpoint_sha256=_sha256_file(checkpoint_path),
    )
    result.validate()
    with open(checkpoint_dir / "train_result.json", "w", encoding="utf-8") as file:
        json.dump(asdict(result), file)
    return artifact_dir, checkpoint_dir


def test_replay_reports_exact_greedy_prefix_and_advancement(tmp_path: Path) -> None:
    artifact_dir, result_dir = _make_replay_case(tmp_path)

    report = replay_greedy_draft_head(result_dir, artifact_dir)

    assert report["complete_trajectory_requests"] == 1
    assert report["capture_coverage"] == 1.0
    assert report["cycles"] == 3
    assert report["sampled_hist"] == {1: 1, 3: 2}
    assert report["accepted_prefix_hist"] == {1: 1, 3: 2}
    assert report["accepted_tokens"] == 7
    assert report["sampled_tokens"] == 7
    assert report["mean_advancement"] == pytest.approx(7 / 3)
    assert report["proposal_acceptance"] == 1.0
    assert report["draft_top1_target_match_rate"] == 1.0
    assert report["target_top1_response_alignment"] == 1.0


def test_replay_stops_at_first_reject_and_continues_after_correction(
    tmp_path: Path,
) -> None:
    artifact_dir, result_dir = _make_replay_case(
        tmp_path, biased_position=1
    )

    report = replay_greedy_draft_head(result_dir, artifact_dir)

    assert report["cycles"] == 3
    assert report["accepted_prefix_hist"] == {1: 1, 2: 1, 3: 1}
    assert report["sampled_hist"] == {2: 2, 3: 1}
    assert report["accepted_tokens"] == 6
    assert report["sampled_tokens"] == 7
    assert report["draft_top1_target_match_rate"] < 1.0


def test_sparse_capture_is_reported_as_windows_with_gaps(tmp_path: Path) -> None:
    artifact_dir, result_dir = _make_replay_case(
        tmp_path, captured_positions=[0, 1, 2, 4, 5, 6]
    )

    report = replay_greedy_draft_head(result_dir, artifact_dir)

    assert report["complete_trajectory_requests"] == 0
    assert report["capture_coverage"] == pytest.approx(6 / 7)
    assert report["capture_gaps"] == 1
    assert report["cycles"] == 2
    assert report["incomplete_windows"] == 0


def test_stochastic_artifact_fails_closed_until_teacher_support_is_captured(
    tmp_path: Path,
) -> None:
    artifact_dir, result_dir = _make_replay_case(tmp_path, stochastic=True)

    with pytest.raises(ValueError, match="Stochastic replay is unsupported"):
        replay_greedy_draft_head(result_dir, artifact_dir)
