"""Offline merge of DVI partial spools into schema v1 artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors

from vllm.v1.worker.gpu.split_dvi.artifact import (
    DVIArtifactError,
    DVIArtifactManifest,
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
    incomplete_records: int = 0
    dropped_records: int = 0
    estimated_bytes: int = 0
    written_bytes: int = 0
    sampled_records: int = 0
    bins: dict[str, dict[str, float]] = field(default_factory=dict)

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
    if manifest.get("schema_version") != "spool-v1":
        raise DVIArtifactError(
            f"Unsupported spool schema version: {manifest.get('schema_version')}"
        )
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

        # Pre-compute which requests will be captured so requests are written
        # before capture records.
        artifact_requests_by_id: dict[str, DVIRequestRecord] = {}
        for key in stage2_records:
            if key not in stage0_index:
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

                # Allow the shard tensor dict to be garbage collected.
                del tensors

        report.written_bytes = sum(
            f.stat().st_size for f in Path(out_dir).rglob("*") if f.is_file()
        )
        report.estimated_bytes = report.written_bytes
        report.bins = self._build_length_bins(
            list(artifact_requests_by_id.values())
        )

        all_keys = set(stage0_index.keys()) | set(stage2_records.keys())
        report.incomplete_records = len(all_keys - paired_keys)

        report_path = Path(str(out_dir) + ".merge_report.json")
        report_path.write_text(report.to_json(), encoding="utf-8")

        return report

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
