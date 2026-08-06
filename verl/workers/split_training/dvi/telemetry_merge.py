"""Offline merge of DVI partial spools into schema v1 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors

from vllm.v1.worker.gpu.split_dvi.artifact import (
    DVIArtifactError,
    DVIArtifactManifest,
    DVIEvaluationRecord,
    DVIArtifactWriter,
    DVICaptureRecord,
    DVIRequestRecord,
)


TRAIN_RESULT_SCHEMA_VERSION = "v1"


@dataclass
class MergeReport:
    """Summary of a partial-spool merge operation."""

    schema_version: str = TRAIN_RESULT_SCHEMA_VERSION
    result_dir: str = ""
    stage0_records: int = 0
    stage2_records: int = 0
    paired_records: int = 0
    overshoot_records_excluded: int = 0
    terminal_truncated_cycles: int = 0
    complete_cycle_ids: list[str] = field(default_factory=list)
    terminal_truncated_cycle_ids: list[str] = field(default_factory=list)
    incomplete_records: int = 0
    dropped_records: int = 0
    estimated_bytes: int = 0
    written_bytes: int = 0
    sampled_records: int = 0
    bins: dict[str, dict[str, float]] = field(default_factory=dict)
    topk_mass: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _key_to_tuple(d: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        d["run_id"],
        d["rollout_id"],
        d["policy_version"],
        d["request_id"],
        int(d["position"]),
    )
def _evaluation_key_to_tuple(
    data: dict[str, Any],
) -> tuple[str, str, str, str, int, int]:
    key = data["key"]
    return (
        str(key["run_id"]),
        str(key["rollout_id"]),
        str(key["policy_version"]),
        str(key["request_id"]),
        int(key["cycle_id"]),
        int(key["row_index"]),
    )




def _namespaced_request_id(key: tuple[str, str, str, str, int]) -> str:
    run_id, rollout_id, _, request_id, _ = key
    return f"{run_id}/{rollout_id}/{request_id}"


def _load_spool_manifest(spool_dir: Path) -> dict[str, Any]:
    path = spool_dir / "spool_manifest.json"
    if not path.is_file():
        raise DVIArtifactError(
            f"Missing spool_manifest.json in {spool_dir}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_spool_manifest(spool_dir: Path, manifest: dict[str, Any]) -> None:
    required = [
        "run_id",
        "rollout_id",
        "policy_version",
        "base_checkpoint_hash",
        "head_adapter_hash",
        "tail_adapter_hash",
        "tokenizer_revision",
        "sampling_config",
        "schema_version",
        "finalized",
        "file_checksums",
    ]
    missing = [k for k in required if k not in manifest]
    if missing:
        raise DVIArtifactError(
            f"Spool manifest missing fields: {missing}"
        )
    if manifest.get("schema_version") not in {"spool-v1", "spool-v2"}:
        raise DVIArtifactError(
            f"Unsupported spool schema version: {manifest.get('schema_version')}"
        )
    if manifest.get("schema_version") == "spool-v2":
        if manifest.get("capture_mode") != "evaluation":
            raise DVIArtifactError("spool-v2 requires capture_mode='evaluation'")
        if (
            "draft_length" not in manifest
            or "num_proposals" not in manifest
            or "bonus_token" not in manifest
        ):
            raise DVIArtifactError(
                "spool-v2 requires draft_length, num_proposals and bonus_token metadata"
            )
        if int(manifest["draft_length"]) < 2:
            raise DVIArtifactError("spool-v2 draft_length must be at least 2")
        if int(manifest["num_proposals"]) != int(manifest["draft_length"]) - 1:
            raise DVIArtifactError(
                "spool-v2 num_proposals must equal draft_length - 1"
            )
        if not manifest.get("bonus_token", False):
            raise DVIArtifactError("spool-v2 requires bonus_token=True")
        if int(manifest["num_proposals"]) < 0:
            raise DVIArtifactError("spool-v2 num_proposals must be non-negative")
    if not manifest.get("finalized"):
        raise DVIArtifactError("Spool is not finalized")

    for child in spool_dir.iterdir():
        if child.name == "spool_manifest.json":
            continue
        if child.is_symlink() or child.is_dir():
            raise DVIArtifactError(
                f"Spool must contain only regular files; found {child.name}"
            )
        if child.name.startswith("."):
            raise DVIArtifactError(
                f"Spool contains hidden file: {child.name}"
            )

    checksum_files = set(manifest["file_checksums"].keys())
    actual_files = {
        p.name for p in spool_dir.iterdir() if p.is_file()
    } - {"spool_manifest.json"}
    missing_files = checksum_files - actual_files
    extra_files = actual_files - checksum_files
    if missing_files:
        raise DVIArtifactError(
            f"Spool missing files: {sorted(missing_files)}"
        )
    if extra_files:
        raise DVIArtifactError(
            f"Spool has unexpected files: {sorted(extra_files)}"
        )

    for child in spool_dir.iterdir():
        if child.name == "spool_manifest.json":
            continue
        expected = manifest["file_checksums"].get(child.name)
        if expected is None:
            continue
        actual = _sha256_file(child)
        if actual != expected:
            raise DVIArtifactError(
                f"Checksum mismatch for {child.name}: expected {expected}, "
                f"got {actual}"
            )


def _assert_spools_compatible(
    m0: dict[str, Any],
    m2: dict[str, Any],
    manifest: DVIArtifactManifest,
) -> None:
    fields = [
        "run_id",
        "rollout_id",
        "policy_version",
        "base_checkpoint_hash",
        "head_adapter_hash",
        "tail_adapter_hash",
        "tokenizer_revision",
        "sampling_config",
    ]
    for field_name in fields:
        v0 = m0.get(field_name)
        v2 = m2.get(field_name)
        if v0 != v2:
            raise DVIArtifactError(
                f"Spool mismatch for {field_name}: stage0={v0}, stage2={v2}"
            )

    if manifest.sampling_config != m0.get("sampling_config"):
        raise DVIArtifactError(
            "sampling_config mismatch with artifact manifest"
        )

    if manifest.base_checkpoint_hash != m0.get("base_checkpoint_hash"):
        raise DVIArtifactError("base_checkpoint_hash mismatch with manifest")
    if manifest.head_adapter_hash != m0.get("head_adapter_hash"):
        raise DVIArtifactError("head_adapter_hash mismatch with manifest")
    if manifest.tail_adapter_hash != m0.get("tail_adapter_hash"):
        raise DVIArtifactError("tail_adapter_hash mismatch with manifest")
    if manifest.tokenizer_revision != m0.get("tokenizer_revision"):
        raise DVIArtifactError("tokenizer_revision mismatch with manifest")
    if m0.get("schema_version") == "spool-v2":
        for field_name in (
            "capture_mode",
            "draft_length",
            "num_proposals",
            "bonus_token",
        ):
            if m0.get(field_name) != m2.get(field_name):
                raise DVIArtifactError(
                    f"Spool mismatch for {field_name}: "
                    f"stage0={m0.get(field_name)}, stage2={m2.get(field_name)}"
                )
        if manifest.capture_mode != m0.get("capture_mode"):
            raise DVIArtifactError("capture_mode mismatch with artifact manifest")
        if manifest.draft_length != int(m0["draft_length"]):
            raise DVIArtifactError("draft_length mismatch with artifact manifest")
        if manifest.num_proposals != int(m0["num_proposals"]):
            raise DVIArtifactError("num_proposals mismatch with artifact manifest")
        if manifest.bonus_token != bool(m0["bonus_token"]):
            raise DVIArtifactError("bonus_token mismatch with artifact manifest")


class DVIPartialSpoolMerger:
    """Merge stage_0 and stage_2 partial spools into a v1 artifact."""

    def __init__(self, manifest: DVIArtifactManifest) -> None:
        self.manifest = manifest
        self.manifest.validate()

    def merge(
        self,
        stage0_spool: str | Path,
        stage2_spool: str | Path,
        out_dir: str | Path,
    ) -> MergeReport:
        stage0_dir = Path(stage0_spool)
        stage2_dir = Path(stage2_spool)
        if not stage0_dir.is_dir() or not stage2_dir.is_dir():
            raise DVIArtifactError(
                "stage0_spool and stage2_spool must be directories"
            )

        if Path(out_dir).exists():
            raise DVIArtifactError(
                f"Refusing to overwrite existing output directory: {out_dir}"
            )

        m0 = _load_spool_manifest(stage0_dir)
        m2 = _load_spool_manifest(stage2_dir)
        _validate_spool_manifest(stage0_dir, m0)
        _validate_spool_manifest(stage2_dir, m2)
        if m0.get("schema_version") == "spool-v2":
            if m2.get("schema_version") != "spool-v2":
                raise DVIArtifactError("stage0/stage2 spool schema mismatch")
            if self.manifest.schema_version != "v2":
                raise DVIArtifactError(
                    "spool-v2 evaluation data requires a schema-v2 artifact manifest"
                )
            _assert_spools_compatible(m0, m2, self.manifest)
            return self._merge_evaluation(stage0_dir, stage2_dir, m0, m2, out_dir)
        _assert_spools_compatible(m0, m2, self.manifest)

        session_key = (
            str(m0["run_id"]),
            str(m0["rollout_id"]),
            str(m0["policy_version"]),
        )

        report = MergeReport(
            result_dir=str(out_dir),
            dropped_records=int(
                m0.get("metrics", {}).get("dropped_records", 0)
            )
            + int(m2.get("metrics", {}).get("dropped_records", 0)),
            sampled_records=int(
                m0.get("metrics", {}).get("stage0_enqueued", 0)
            )
            + int(m2.get("metrics", {}).get("stage2_enqueued", 0))
            + int(m2.get("metrics", {}).get("request_enqueued", 0)),
        )

        # Load small stage_2 records into memory; stream stage_0 hidden tensors.
        stage2_records: dict[
            tuple[str, str, str, str, int],
            dict[str, Any],
        ] = {}
        for jsonl_path in sorted(stage2_dir.glob("stage2-*.jsonl")):
            for raw in _load_jsonl(jsonl_path):
                key = _key_to_tuple(raw["key"])
                if key[:3] != session_key:
                    raise DVIArtifactError(
                        f"stage2 record key {key} does not match spool session"
                    )
                if key in stage2_records:
                    raise DVIArtifactError(
                        f"Duplicate stage2 record for key {key}"
                    )
                stage2_records[key] = raw
        report.stage2_records = len(stage2_records)

        # Load request metadata from stage_2 spool.
        requests: dict[str, DVIRequestRecord] = {}
        for raw in _load_jsonl(stage2_dir / "requests.jsonl"):
            req = DVIRequestRecord(
                request_id=raw["request_id"],
                prompt_token_ids=list(raw["prompt_token_ids"]),
                response_token_ids=list(raw["response_token_ids"]),
            )
            if req.request_id in requests:
                raise DVIArtifactError(
                    f"Duplicate request_id {req.request_id!r} in spool"
                )
            requests[req.request_id] = req

        # Build streaming index for stage_0 hidden tensors and detect duplicates.
        stage0_index: dict[
            tuple[str, str, str, str, int],
            tuple[Path, Path, str],
        ] = {}
        for jsonl_path in sorted(stage0_dir.glob("stage0-*.jsonl")):
            st_path = jsonl_path.with_suffix(".safetensors")
            if not st_path.is_file():
                raise DVIArtifactError(
                    f"Missing safetensors pair for {jsonl_path.name}"
                )
            for raw in _load_jsonl(jsonl_path):
                key = _key_to_tuple(raw["key"])
                if key[:3] != session_key:
                    raise DVIArtifactError(
                        f"stage0 record key {key} does not match spool session"
                    )
                if key in stage0_index:
                    raise DVIArtifactError(
                        f"Duplicate stage0 record for key {key}"
                    )
                stage0_index[key] = (jsonl_path, st_path, raw["hidden_key"])
        report.stage0_records = len(stage0_index)

        # Pipeline-parallel scheduling may already have emitted capture rows
        # beyond a request's committed response when the terminal result is
        # observed. Keep terminal rows (position == response length) for
        # diagnostics, but exclude paired beyond-terminal rows explicitly.
        overshoot_keys = {
            key
            for key in stage0_index.keys() & stage2_records.keys()
            if (req := requests.get(key[3])) is not None
            and key[4] > len(req.response_token_ids)
        }
        report.overshoot_records_excluded = len(overshoot_keys)

        # Pre-compute which requests will be captured so requests are written
        # before capture records.
        artifact_requests_by_id: dict[str, DVIRequestRecord] = {}
        for key in stage2_records:
            if key not in stage0_index:
                continue
            if key in overshoot_keys:
                continue
            original_request_id = key[3]
            req = requests.get(original_request_id)
            if req is None:
                continue
            ns_id = _namespaced_request_id(key)
            if ns_id not in artifact_requests_by_id:
                artifact_requests_by_id[ns_id] = DVIRequestRecord(
                    request_id=ns_id,
                    prompt_token_ids=req.prompt_token_ids,
                    response_token_ids=req.response_token_ids,
                )

        if not artifact_requests_by_id:
            raise DVIArtifactError("No paired capture records to merge")

        # Stream stage_0 shards, pair with stage_2 records, and write captures.
        paired_keys: set[tuple[str, str, str, str, int]] = set()
        nonterminal_topk_mass: list[float] = []
        request_index = {
            r.request_id: r for r in artifact_requests_by_id.values()
        }

        with DVIArtifactWriter(out_dir, self.manifest) as writer:
            writer.write_requests(list(artifact_requests_by_id.values()))

            for jsonl_path in sorted(stage0_dir.glob("stage0-*.jsonl")):
                st_path = jsonl_path.with_suffix(".safetensors")
                tensors = load_safetensors(str(st_path))
                for raw in _load_jsonl(jsonl_path):
                    key = _key_to_tuple(raw["key"])
                    if key not in stage2_records:
                        continue
                    if key in overshoot_keys:
                        continue

                    original_request_id = key[3]
                    req = requests.get(original_request_id)
                    if req is None:
                        continue

                    hidden_key = raw["hidden_key"]
                    if hidden_key not in tensors:
                        raise DVIArtifactError(
                            f"Tensor key {hidden_key!r} not found in "
                            f"{st_path.name}"
                        )

                    s2 = stage2_records[key]
                    rec = DVICaptureRecord(
                        request_id=_namespaced_request_id(key),
                        position=key[4],
                        stage_0_hidden=tensors[hidden_key],
                        verifier_topk_ids=list(s2["verifier_topk_ids"]),
                        verifier_topk_logprobs=list(s2["verifier_topk_logprobs"]),
                        verifier_residual_mass=s2["verifier_residual_mass"],
                        verifier_top1_id=s2["verifier_top1_id"],
                    )
                    rec.validate(request_index, self.manifest)
                    writer.write_capture_records([rec])
                    paired_keys.add(key)
                    report.paired_records += 1
                    if rec.position < len(req.response_token_ids):
                        nonterminal_topk_mass.append(
                            1.0 - rec.verifier_residual_mass
                        )

                # Allow the shard tensor dict to be garbage collected.
                del tensors

            if not nonterminal_topk_mass:
                raise DVIArtifactError(
                    "No paired non-terminal capture records to merge"
                )
            report.topk_mass = self._distribution(nonterminal_topk_mass)
            self.manifest.topk_mass_coverage = report.topk_mass["mean"]

        report.written_bytes = sum(
            f.stat().st_size for f in Path(out_dir).rglob("*") if f.is_file()
        )
        report.estimated_bytes = report.written_bytes
        report.incomplete_records = report.stage2_records - report.paired_records - report.overshoot_records_excluded
        Path(str(out_dir) + ".merge_report.json").write_text(
            report.to_json(), encoding="utf-8"
        )
        return report

    def _merge_evaluation(
        self,
        stage0_dir: Path,
        stage2_dir: Path,
        m0: dict[str, Any],
        m2: dict[str, Any],
        out_dir: str | Path,
    ) -> MergeReport:
        """Merge dense stage0/stage2 cycle rows into a schema-v2 artifact."""
        session_key = (
            str(m0["run_id"]),
            str(m0["rollout_id"]),
            str(m0["policy_version"]),
        )
        stage0_rows: dict[
            tuple[str, str, str, str, int, int],
            tuple[dict[str, Any], torch.Tensor],
        ] = {}
        for jsonl_path in sorted(stage0_dir.glob("evaluation-stage0-*.jsonl")):
            st_path = jsonl_path.with_suffix(".safetensors")
            if not st_path.is_file():
                raise DVIArtifactError(
                    f"Missing evaluation stage0 tensor shard for {jsonl_path.name}"
                )
            tensors = load_safetensors(str(st_path))
            for raw in _load_jsonl(jsonl_path):
                key = _evaluation_key_to_tuple(raw)
                if key[:3] != session_key:
                    raise DVIArtifactError(
                        f"evaluation stage0 key {key} does not match spool session"
                    )
                hidden_key = raw.pop("hidden_key")
                if hidden_key not in tensors:
                    raise DVIArtifactError(
                        f"Tensor key {hidden_key!r} not found in {st_path.name}"
                    )
                if key in stage0_rows:
                    raise DVIArtifactError(
                        f"Duplicate evaluation stage0 row for key {key}"
                    )
                stage0_rows[key] = (raw, tensors[hidden_key])
            del tensors

        stage2_rows: dict[
            tuple[str, str, str, str, int, int], dict[str, Any]
        ] = {}
        for jsonl_path in sorted(stage2_dir.glob("evaluation-stage2-*.jsonl")):
            for raw in _load_jsonl(jsonl_path):
                key = _evaluation_key_to_tuple(raw)
                if key[:3] != session_key:
                    raise DVIArtifactError(
                        f"evaluation stage2 key {key} does not match spool session"
                    )
                if key in stage2_rows:
                    raise DVIArtifactError(
                        f"Duplicate evaluation stage2 row for key {key}"
                    )
                stage2_rows[key] = raw

        # Runtime is the sole authority for cycle lifecycle. Merge must not
        # infer terminal status from row positions or response lengths.
        finalization_rows: dict[tuple[str, int], dict[str, Any]] = {}
        finalization_paths = sorted(stage2_dir.glob("evaluation-finalization-*.jsonl"))
        if not finalization_paths:
            raise DVIArtifactError("Evaluation spool is missing cycle finalization markers")
        for jsonl_path in finalization_paths:
            for raw in _load_jsonl(jsonl_path):
                cycle_key = (str(raw.get("request_id")), int(raw.get("cycle_id")))
                if cycle_key in finalization_rows:
                    raise DVIArtifactError(f"Duplicate evaluation finalization marker for cycle {cycle_key}")
                if raw.get("finalized") is not True:
                    raise DVIArtifactError(f"Evaluation finalization marker is not finalized: {cycle_key}")
                if raw.get("cycle_status") not in {"complete", "terminal_truncated", "aborted"}:
                    raise DVIArtifactError(f"Unknown evaluation cycle status for {cycle_key}: {raw.get('cycle_status')!r}")
                expected = int(raw.get("num_rows_expected", -1))
                seen = int(raw.get("num_rows_seen", -1))
                if expected < 0 or seen < 0 or seen > expected:
                    raise DVIArtifactError(
                        f"Invalid evaluation finalization row counts for {cycle_key}: "
                        f"expected={expected}, seen={seen}"
                    )
                finalization_rows[cycle_key] = raw

        requests: dict[str, DVIRequestRecord] = {}
        for raw in _load_jsonl(stage2_dir / "requests.jsonl"):
            request = DVIRequestRecord(
                request_id=str(raw["request_id"]),
                prompt_token_ids=list(raw["prompt_token_ids"]),
                response_token_ids=list(raw["response_token_ids"]),
            )
            if request.request_id in requests:
                raise DVIArtifactError(
                    f"Duplicate request_id {request.request_id!r} in evaluation spool"
                )
            requests[request.request_id] = request
        if not requests:
            raise DVIArtifactError("Evaluation stage2 spool has no requests")

        paired_keys = sorted(stage0_rows.keys() & stage2_rows.keys())
        if not paired_keys:
            raise DVIArtifactError("No paired dense evaluation rows to merge")
        all_keys = set(stage0_rows) | set(stage2_rows)
        incomplete_keys = all_keys - set(paired_keys)
        if incomplete_keys:
            raise DVIArtifactError(
                "Dense evaluation cycle has missing stage0/stage2 rows: "
                f"{len(incomplete_keys)} incomplete rows"
            )

        observed_cycle_keys = {(key[3], key[4]) for key in all_keys}
        marker_cycle_keys = set(finalization_rows)
        missing_markers = observed_cycle_keys - marker_cycle_keys
        extra_markers = marker_cycle_keys - observed_cycle_keys
        if missing_markers or extra_markers:
            raise DVIArtifactError(
                "Evaluation cycle finalization marker population mismatch: "
                f"missing={sorted(missing_markers)}, extra={sorted(extra_markers)}"
            )
        aborted = sorted(
            key for key, marker in finalization_rows.items()
            if marker["cycle_status"] == "aborted"
        )
        if aborted:
            raise DVIArtifactError(f"Evaluation cycle finalization aborted: {aborted}")
        terminal_cycle_keys = {
            key for key, marker in finalization_rows.items()
            if marker["cycle_status"] == "terminal_truncated"
        }
        complete_cycle_keys = {
            key for key, marker in finalization_rows.items()
            if marker["cycle_status"] == "complete"
        }

        # Validate the complete row population before creating the artifact.
        # Every runtime cycle has exactly num_proposals proposal rows followed
        # by one bonus row, and both stages must describe the same row.
        cycles: dict[
            tuple[str, int],
            list[tuple[Any, dict[str, Any], dict[str, Any]]],
        ] = {}
        for key in paired_keys:
            stage0, _hidden = stage0_rows[key]
            stage2 = stage2_rows[key]
            for field_name in (
                "request_id",
                "cycle_id",
                "generation_id",
                "row_index",
                "absolute_position",
                "row_kind",
                "num_proposals",
            ):
                if field_name not in stage0 or field_name not in stage2:
                    raise DVIArtifactError(
                        f"Evaluation row missing {field_name} at key {key}"
                    )
                if stage0[field_name] != stage2[field_name]:
                    raise DVIArtifactError(
                        f"Evaluation stage mismatch for {field_name} at key {key}"
                    )
            cycle_key = (key[3], key[4])
            cycles.setdefault(cycle_key, []).append((key, stage0, stage2))

        for cycle_key, rows in cycles.items():
            if cycle_key in terminal_cycle_keys:
                continue
            proposal_count = int(rows[0][2]["num_proposals"])
            expected_indices = set(range(proposal_count + 1))
            actual_indices = {key[5] for key, _stage0, _stage2 in rows}
            if actual_indices != expected_indices:
                raise DVIArtifactError(
                    "Evaluation cycle row indices are not dense: "
                    f"cycle={cycle_key}, expected={sorted(expected_indices)}, "
                    f"actual={sorted(actual_indices)}"
                )
            if any(
                int(stage2["num_proposals"]) != proposal_count
                or (
                    str(stage2["row_kind"])
                    != ("proposal" if key[5] < proposal_count else "bonus")
                )
                for key, _stage0, stage2 in rows
            ):
                raise DVIArtifactError(
                    f"Evaluation cycle row kind/count mismatch: cycle={cycle_key}"
                )

        unknown_complete = complete_cycle_keys - set(cycles)
        if unknown_complete:
            raise DVIArtifactError(
                "Complete finalization markers have no rows: "
                f"{sorted(unknown_complete)}"
            )
        paired_keys = [
            key for key in paired_keys
            if (key[3], key[4]) in complete_cycle_keys
        ]
        artifact_requests = [
            DVIRequestRecord(
                request_id=f"{session_key[0]}/{session_key[1]}/{request_id}",
                prompt_token_ids=request.prompt_token_ids,
                response_token_ids=request.response_token_ids,
            )
            for request_id, request in sorted(requests.items())
        ]
        report = MergeReport(
            schema_version="v2",
            result_dir=str(out_dir),
            stage0_records=len(stage0_rows),
            stage2_records=len(stage2_rows),
            paired_records=len(paired_keys),
            incomplete_records=len(incomplete_keys),
            overshoot_records_excluded=0,
            terminal_truncated_cycles=len(terminal_cycle_keys),
            complete_cycle_ids=[f"{rid}:{cid}" for rid, cid in sorted(complete_cycle_keys)],
            terminal_truncated_cycle_ids=[f"{rid}:{cid}" for rid, cid in sorted(terminal_cycle_keys)],
            dropped_records=int(
                m0.get("metrics", {}).get("dropped_records", 0)
            )
            + int(m2.get("metrics", {}).get("dropped_records", 0)),
            sampled_records=int(
                m0.get("metrics", {}).get("evaluation_stage0_enqueued", 0)
            )
            + int(m2.get("metrics", {}).get("evaluation_stage2_enqueued", 0)),
        )
        with DVIArtifactWriter(out_dir, self.manifest) as writer:
            writer.write_requests(artifact_requests)
            for key in paired_keys:
                stage0, hidden = stage0_rows[key]
                stage2 = stage2_rows[key]
                request_id = (
                    f"{key[0]}/{key[1]}/{key[3]}"
                )
                record = DVIEvaluationRecord(
                    request_id=request_id,
                    cycle_id=int(stage2["cycle_id"]),
                    generation_id=int(stage2.get("generation_id", 0)),
                    row_index=int(stage2["row_index"]),
                    absolute_position=int(stage2["absolute_position"]),
                    rng_position=(
                        int(stage2["rng_position"])
                        if stage2.get("rng_position") is not None
                        else None
                    ),
                    request_seed=(
                        int(stage2["request_seed"])
                        if stage2.get("request_seed") is not None
                        else None
                    ),
                    row_kind=str(stage2["row_kind"]),
                    num_proposals=int(stage2["num_proposals"]),
                    draft_token_id=(
                        int(stage0["draft_token_id"])
                        if stage0.get("draft_token_id") is not None
                        else None
                    ),
                    draft_support_token_ids=[
                        int(value)
                        for value in stage0.get("draft_support_token_ids", [])
                    ],
                    draft_support_logits=[
                        float(value)
                        for value in stage0.get("draft_support_logits", [])
                    ],
                    stage_0_hidden=hidden,
                    verifier_topk_ids=[
                        int(value) for value in stage2["verifier_topk_ids"]
                    ],
                    verifier_topk_logprobs=[
                        float(value)
                        for value in stage2["verifier_topk_logprobs"]
                    ],
                    verifier_residual_mass=float(
                        stage2["verifier_residual_mass"]
                    ),
                    teacher_probs_on_draft_support=[
                        float(value)
                        for value in stage2.get(
                            "teacher_probs_on_draft_support", []
                        )
                    ],
                    accepted=stage2.get("accepted"),
                    accepted_count=int(stage2["accepted_count"]),
                    committed_token_ids=[
                        int(value)
                        for value in stage2.get("committed_token_ids", [])
                    ],
                    correction_token_id=(
                        int(stage2["correction_token_id"])
                        if stage2.get("correction_token_id") is not None
                        else None
                    ),
                    terminal_token_id=(
                        int(stage2["terminal_token_id"])
                        if stage2.get("terminal_token_id") is not None
                        else None
                    ),
                    advancement=int(stage2.get("advancement", 0)),
                )
                writer.write_evaluation_records([record])
        report.written_bytes = sum(
            path.stat().st_size for path in Path(out_dir).rglob("*")
            if path.is_file()
        )
        report.estimated_bytes = report.written_bytes
        report.bins = self._build_length_bins(artifact_requests)
        Path(str(out_dir) + ".merge_report.json").write_text(
            report.to_json(), encoding="utf-8"
        )
        return report

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)

        def percentile(q: float) -> float:
            position = (len(ordered) - 1) * q
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return ordered[lower]
            fraction = position - lower
            return (
                ordered[lower] * (1.0 - fraction)
                + ordered[upper] * fraction
            )

        return {
            "count": float(len(ordered)),
            "mean": sum(ordered) / len(ordered),
            "min": ordered[0],
            "p05": percentile(0.05),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
        }

    def _build_length_bins(
        self,
        requests: list[DVIRequestRecord],
    ) -> dict[str, dict[str, float]]:
        lengths = [len(r.response_token_ids) for r in requests]
        if not lengths:
            return {}
        lengths.sort()
        n = len(lengths)
        return {
            "count": float(n),
            "min": float(lengths[0]),
            "p25": float(lengths[n // 4]),
            "p50": float(lengths[n // 2]),
            "p75": float(lengths[(3 * n) // 4]),
            "max": float(lengths[-1]),
        }


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "DVIPartialSpoolMerger",
    "MergeReport",
]
