"""Contract tests for dense schema-v2 evaluation spool merging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vllm.v1.worker.gpu.split_dvi.artifact import (
    DVIArtifactError,
    DVIArtifactManifest,
    DVIArtifactReader,
)
from verl.workers.split_training.dvi.telemetry_merge import (
    DVIPartialSpoolMerger,
)


SAMPLING_CONFIG = {
    "sample_rate": 1.0,
    "max_per_request": 16,
    "seed": 42,
    "top_k": 16,
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _finalize_spool(path: Path) -> None:
    checksums = {}
    for child in sorted(path.iterdir()):
        if child.is_file():
            checksums[child.name] = hashlib.sha256(child.read_bytes()).hexdigest()
    metadata = {
        "run_id": "run0",
        "rollout_id": "rollout0",
        "policy_version": "policy5",
        "base_checkpoint_hash": "base-hash",
        "head_adapter_hash": "head-hash",
        "tail_adapter_hash": "tail-hash",
        "tokenizer_revision": "tokenizer-v1",
        "sampling_config": SAMPLING_CONFIG,
        "capture_mode": "evaluation",
        "draft_length": 4,
        "num_proposals": 3,
        "bonus_token": True,
        "schema_version": "spool-v2",
        "finalized": True,
        "file_checksums": checksums,
        "metrics": {
            "evaluation_stage0_enqueued": 4,
            "evaluation_stage2_enqueued": 4,
            "dropped_records": 0,
        },
    }
    _write_jsonl(path / "spool_metadata_sentinel.jsonl", [])
    # The sentinel is intentionally not part of a real spool; remove it before
    # writing checksums so the helper only publishes data files.
    (path / "spool_metadata_sentinel.jsonl").unlink()
    checksums = {
        child.name: hashlib.sha256(child.read_bytes()).hexdigest()
        for child in sorted(path.iterdir())
        if child.is_file()
    }
    metadata["file_checksums"] = checksums
    (path / "spool_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _make_manifest() -> DVIArtifactManifest:
    return DVIArtifactManifest(
        schema_version="v2",
        base_checkpoint="synthetic",
        base_checkpoint_hash="base-hash",
        head_adapter_hash="head-hash",
        tail_adapter_hash="tail-hash",
        tokenizer_revision="tokenizer-v1",
        split_stage_0_size=1,
        split_stage_2_size=1,
        hidden_size=8,
        vocab_size=1000,
        draft_length=4,
        dtype="float32",
        sampling_config=SAMPLING_CONFIG,
        capture_mode="evaluation",
        policy_version="policy5",
        num_proposals=3,
        bonus_token=True,
        max_response_length=4,
    )


def _make_spools(tmp_path: Path, *, drop_stage2_row: bool = False) -> tuple[Path, Path]:
    stage0_dir = tmp_path / "stage0"
    stage2_dir = tmp_path / "stage2"
    stage0_dir.mkdir()
    stage2_dir.mkdir()
    stage0_rows = []
    stage2_rows = []
    tensors = {}
    for row_index in range(4):
        row_kind = "proposal" if row_index < 3 else "bonus"
        key = {
            "run_id": "run0",
            "rollout_id": "rollout0",
            "policy_version": "policy5",
            "request_id": "r0",
            "cycle_id": 0,
            "row_index": row_index,
        }
        hidden_key = f"h{row_index}"
        tensors[hidden_key] = torch.zeros(8, dtype=torch.float32) + row_index
        stage0_rows.append({
            "key": key,
            "request_id": "r0",
            "cycle_id": 0,
            "generation_id": 1,
            "row_index": row_index,
            "absolute_position": row_index,
            "row_kind": row_kind,
            "num_proposals": 3,
            "draft_token_id": 100 + row_index if row_index < 3 else None,
            "draft_support_token_ids": [100 + row_index, 200 + row_index]
            if row_index < 3 else [],
            "draft_support_logits": [1.0, 0.0] if row_index < 3 else [],
            "hidden_key": hidden_key,
        })
        if drop_stage2_row and row_index == 2:
            continue
        stage2_rows.append({
            "key": key,
            "request_id": "r0",
            "cycle_id": 0,
            "generation_id": 1,
            "row_index": row_index,
            "absolute_position": row_index,
            "row_kind": row_kind,
            "num_proposals": 3,
            "verifier_topk_ids": [20 + row_index],
            "verifier_topk_logprobs": [0.0],
            "verifier_residual_mass": 0.0,
            "teacher_probs_on_draft_support": [0.9, 0.0]
            if row_index < 3 else [],
            "accepted": True if row_index < 3 else None,
            "accepted_count": 3,
            "committed_token_ids": [10, 11, 12, 13] if row_index == 0 else [],
            "correction_token_id": None,
            "terminal_token_id": 13 if row_index == 0 else None,
            "advancement": 4 if row_index == 0 else 0,
        })
    _write_jsonl(stage0_dir / "evaluation-stage0-00000.jsonl", stage0_rows)
    save_file(tensors, str(stage0_dir / "evaluation-stage0-00000.safetensors"))
    _write_jsonl(stage2_dir / "evaluation-stage2-00000.jsonl", stage2_rows)
    _write_jsonl(stage2_dir / "evaluation-finalization-00000.jsonl", [{
        "request_id": "r0",
        "cycle_id": 0,
        "num_rows_expected": 4,
        "num_rows_seen": 4,
        "cycle_status": "complete",
        "finalized": True,
    }])
    _write_jsonl(stage2_dir / "requests.jsonl", [{
        "request_id": "r0",
        "prompt_token_ids": [1],
        "response_token_ids": [10, 11, 12, 13],
    }])
    _finalize_spool(stage0_dir)
    _finalize_spool(stage2_dir)
    return stage0_dir, stage2_dir


def test_dense_evaluation_merge_round_trip(tmp_path: Path) -> None:
    stage0, stage2 = _make_spools(tmp_path)
    out_dir = tmp_path / "artifact"

    report = DVIPartialSpoolMerger(_make_manifest()).merge(
        stage0, stage2, out_dir
    )

    assert report.schema_version == "v2"
    assert report.paired_records == 4
    assert report.incomplete_records == 0
    rows = list(DVIArtifactReader(out_dir).iter_evaluation_records())
    assert len(rows) == 4
    assert rows[-1].row_kind == "bonus"
    assert rows[0].advancement == 4


def test_dense_evaluation_merge_rejects_missing_row(tmp_path: Path) -> None:
    stage0, stage2 = _make_spools(tmp_path, drop_stage2_row=True)

    with pytest.raises(DVIArtifactError, match="missing stage0/stage2 rows"):
        DVIPartialSpoolMerger(_make_manifest()).merge(
            stage0, stage2, tmp_path / "artifact"
        )


def test_dense_evaluation_merge_excludes_whole_overshoot_cycle(tmp_path: Path) -> None:
    stage0, stage2 = _make_spools(tmp_path)
    # Make only the final bonus row overshoot the finalized response. The whole
    # dense cycle must be excluded rather than emitting a partial population.
    for directory in (stage0, stage2):
        path = directory / (
            "evaluation-stage0-00000.jsonl" if directory == stage0
            else "evaluation-stage2-00000.jsonl"
        )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["absolute_position"] = 5
        _write_jsonl(path, rows)
        (directory / "spool_manifest.json").unlink()
        _finalize_spool(directory)

    marker = stage2 / "evaluation-finalization-00000.jsonl"
    marker.write_text(json.dumps({
        "request_id": "r0", "cycle_id": 0,
        "num_rows_expected": 4, "num_rows_seen": 4,
        "cycle_status": "terminal_truncated", "finalized": True,
    }) + "\n")
    (stage2 / "spool_manifest.json").unlink()
    _finalize_spool(stage2)
    report = DVIPartialSpoolMerger(_make_manifest()).merge(
        stage0, stage2, tmp_path / "artifact"
    )
    assert report.terminal_truncated_cycles == 1
    assert report.paired_records == 0
    assert list(DVIArtifactReader(tmp_path / "artifact").iter_evaluation_records()) == []


def test_dense_evaluation_merge_rejects_missing_finalization_marker(tmp_path: Path) -> None:
    stage0, stage2 = _make_spools(tmp_path)
    marker = stage2 / "evaluation-finalization-00000.jsonl"
    marker.unlink()
    (stage2 / "spool_manifest.json").unlink()
    _finalize_spool(stage2)
    with pytest.raises(DVIArtifactError, match="missing cycle finalization"):
        DVIPartialSpoolMerger(_make_manifest()).merge(stage0, stage2, tmp_path / "artifact")
