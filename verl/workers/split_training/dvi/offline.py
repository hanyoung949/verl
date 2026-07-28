"""CPU-only training and reporting for the DVI L0 offline phase."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from vllm.v1.worker.gpu.split_dvi.artifact import (
    DVIArtifactReader,
    DVICaptureRecord,
)

from .draft_head import SplitDVIDraftHead


TRAIN_RESULT_SCHEMA_VERSION = "v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def split_request_ids(
    request_ids: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[frozenset[str], frozenset[str], str]:
    """Deterministically split whole requests into train and validation sets."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    unique_ids = sorted(set(request_ids))
    if len(unique_ids) != len(request_ids):
        raise ValueError("request_ids must be unique")
    if not unique_ids:
        raise ValueError("request_ids must not be empty")

    validation_ids: frozenset[str]
    if validation_fraction == 0.0:
        validation_ids = frozenset()
    else:
        if len(unique_ids) < 2:
            raise ValueError(
                "request-level validation requires at least two requests"
            )
        ranked = sorted(
            unique_ids,
            key=lambda request_id: hashlib.sha256(
                f"{seed}\0{request_id}".encode("utf-8")
            ).digest(),
        )
        validation_count = max(
            1, min(len(ranked) - 1, math.ceil(len(ranked) * validation_fraction))
        )
        validation_ids = frozenset(ranked[:validation_count])
    train_ids = frozenset(set(unique_ids) - validation_ids)
    split_payload = json.dumps(
        {"train": sorted(train_ids), "validation": sorted(validation_ids)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return train_ids, validation_ids, hashlib.sha256(split_payload).hexdigest()


def make_synthetic_base_projection(
    hidden_size: int,
    vocab_size: int,
    seed: int,
) -> torch.Tensor:
    """Create the deterministic base projection used by the synthetic PoC."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(vocab_size, hidden_size, generator=generator) / math.sqrt(
        hidden_size
    )


def load_base_projection(
    checkpoint: str | Path,
    tensor_key: str | None = None,
) -> torch.Tensor:
    """Load only the LM projection tensor from a safetensors checkpoint."""
    checkpoint = Path(checkpoint)
    candidates = (
        [tensor_key]
        if tensor_key is not None
        else ["lm_head.weight", "model.embed_tokens.weight"]
    )
    if checkpoint.is_file():
        files = [checkpoint]
        weight_map: dict[str, str] = {}
    elif checkpoint.is_dir():
        index_path = checkpoint / "model.safetensors.index.json"
        if index_path.is_file():
            with open(index_path, encoding="utf-8") as file:
                index = json.load(file)
            weight_map = dict(index.get("weight_map", {}))
            files = []
        else:
            weight_map = {}
            single = checkpoint / "model.safetensors"
            files = [single] if single.is_file() else sorted(
                checkpoint.glob("*.safetensors")
            )
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    for key in candidates:
        shard_name = weight_map.get(key)
        if shard_name is not None:
            shard = checkpoint / shard_name
            if not shard.is_file():
                raise FileNotFoundError(
                    f"Projection shard referenced by index is missing: {shard}"
                )
            with safe_open(shard, framework="pt", device="cpu") as file:
                return file.get_tensor(key)

    for file_path in files:
        if not file_path.is_file():
            continue
        with safe_open(file_path, framework="pt", device="cpu") as file:
            keys = set(file.keys())
            for key in candidates:
                if key in keys:
                    return file.get_tensor(key)
    raise ValueError(
        f"None of the projection tensors {candidates!r} were found in "
        f"{checkpoint}"
    )


@dataclass(frozen=True)
class DVITrainConfig:
    rank: int = 4
    alpha: float = 8.0
    norm: str = "none"
    learning_rate: float = 1e-2
    epochs: int = 20
    batch_size: int = 32
    embed_base_projection: bool = True
    seed: int = 42
    validation_fraction: float = 0.0

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.norm not in {"none", "rmsnorm"}:
            raise ValueError("norm must be 'none' or 'rmsnorm'")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")


@dataclass
class DVITrainResult:
    schema_version: str
    source_artifact_manifest_sha256: str
    source_base_checkpoint_hash: str
    hidden_size: int
    vocab_size: int
    rank: int
    alpha: float
    norm: str
    learning_rate: float
    epochs: int
    batch_size: int
    seed: int
    num_records: int
    initial_kl: float
    final_kl: float
    initial_top1_accuracy: float
    final_top1_accuracy: float
    mean_topk_mass: float
    base_projection_sha256: str
    draft_checkpoint_sha256: str
    terminal_records_excluded: int = 0
    epoch_kl: list[float] = field(default_factory=list)
    num_train_records: int = 0
    num_validation_records: int = 0
    train_request_count: int = 0
    validation_request_count: int = 0
    validation_fraction: float = 0.0
    evaluation_split: str = "train"
    request_split_sha256: str = ""
    initial_train_kl: float | None = None
    final_train_kl: float | None = None
    draft_metadata_sha256: str = ""
    base_projection_embedded: bool = True

    # KL / teacher semantics metadata. Must match the source artifact manifest
    # so downstream reports do not mix forward/reverse/CE/PG objectives.
    kl_direction: str = "forward"
    teacher_distribution_kind: str = "processed_topk_residual"
    teacher_temperature: float | None = None
    topk_mass_coverage: float | None = None

    def validate(self) -> None:
        if self.schema_version != TRAIN_RESULT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported train result schema {self.schema_version!r}")
        if self.hidden_size <= 0 or self.vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        if self.num_records <= 0:
            raise ValueError("num_records must be positive")
        if self.num_train_records or self.num_validation_records:
            if self.num_train_records <= 0:
                raise ValueError("num_train_records must be positive")
            if (
                self.num_train_records + self.num_validation_records
                != self.num_records
            ):
                raise ValueError("train/validation record counts must sum to total")
            if self.train_request_count <= 0:
                raise ValueError("train_request_count must be positive")
            if self.validation_request_count == 0 and self.num_validation_records:
                raise ValueError("validation records require validation requests")
            if not self.request_split_sha256:
                raise ValueError("request_split_sha256 must be non-empty")
        if self.evaluation_split not in {"train", "validation"}:
            raise ValueError("evaluation_split must be train or validation")
        if self.evaluation_split == "validation" and not self.num_validation_records:
            raise ValueError("validation evaluation requires validation records")
        for name in (
            "initial_kl",
            "final_kl",
            "initial_top1_accuracy",
            "final_top1_accuracy",
            "mean_topk_mass",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("initial_train_kl", "final_train_kl"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.terminal_records_excluded < 0:
            raise ValueError("terminal_records_excluded must be non-negative")
        if not self.base_projection_sha256 or not self.draft_checkpoint_sha256:
            raise ValueError("checkpoint hashes must be non-empty")


def _iter_batches(
    reader: DVIArtifactReader,
    batch_size: int,
    request_ids: frozenset[str] | None = None,
) -> Iterator[list[DVICaptureRecord]]:
    """Yield nonterminal records used by the deployed draft head."""
    batch: list[DVICaptureRecord] = []
    for record in reader.iter_capture_records():
        request = reader.request_index[record.request_id]
        if request_ids is not None and record.request_id not in request_ids:
            continue
        if record.position == len(request.response_token_ids):
            # Retained in the artifact for diagnostics, but rollout never
            # drafts after a request has already emitted its terminal token.
            continue
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _lumped_kl(
    logits: torch.Tensor,
    records: list[DVICaptureRecord],
) -> torch.Tensor:
    log_q = torch.log_softmax(logits.float(), dim=-1)
    losses: list[torch.Tensor] = []
    for row, record in enumerate(records):
        ids = torch.tensor(record.verifier_topk_ids, dtype=torch.long)
        teacher_logp = torch.tensor(
            record.verifier_topk_logprobs,
            dtype=torch.float32,
        )
        teacher_prob = teacher_logp.exp()
        draft_logp = log_q[row, ids]
        loss = torch.sum(teacher_prob * (teacher_logp - draft_logp))

        teacher_other = float(record.verifier_residual_mass)
        if teacher_other > 0:
            draft_topk_mass = draft_logp.exp().sum()
            draft_other = (1.0 - draft_topk_mass).clamp_min(1e-12)
            loss = loss + teacher_other * (
                math.log(teacher_other) - torch.log(draft_other)
            )
        losses.append(loss)
    return torch.stack(losses).mean()


def _evaluate(
    model: SplitDVIDraftHead,
    reader: DVIArtifactReader,
    batch_size: int,
    request_ids: frozenset[str] | None = None,
) -> tuple[float, float, float, int]:
    total_loss = 0.0
    total_correct = 0
    total_mass = 0.0
    total_records = 0
    model.eval()
    with torch.no_grad():
        for records in _iter_batches(reader, batch_size, request_ids):
            hidden = torch.stack(
                [record.stage_0_hidden.float() for record in records]
            )
            logits = model(hidden)
            loss = _lumped_kl(logits, records)
            total_loss += float(loss) * len(records)
            predictions = logits.argmax(dim=-1).tolist()
            total_correct += sum(
                prediction == record.verifier_top1_id
                for prediction, record in zip(predictions, records)
            )
            total_mass += sum(
                1.0 - record.verifier_residual_mass for record in records
            )
            total_records += len(records)
    if total_records == 0:
        raise ValueError("Capture artifact contains no training records")
    return (
        total_loss / total_records,
        total_correct / total_records,
        total_mass / total_records,
        total_records,
    )


def _resolve_base_projection(
    reader: DVIArtifactReader,
    config: DVITrainConfig,
    base_projection: torch.Tensor | None,
) -> torch.Tensor:
    manifest = reader.manifest
    if base_projection is None:
        if manifest.base_checkpoint != "synthetic":
            raise ValueError(
                "A base projection is required for non-synthetic artifacts"
            )
        base_projection = make_synthetic_base_projection(
            manifest.hidden_size,
            manifest.vocab_size,
            config.seed,
        )
    expected_shape = (manifest.vocab_size, manifest.hidden_size)
    if tuple(base_projection.shape) != expected_shape:
        raise ValueError(
            f"base projection shape {tuple(base_projection.shape)} does not match "
            f"artifact shape {expected_shape}"
        )
    return base_projection.float().cpu().contiguous()


def train_offline_draft_head(
    artifact_dir: str | Path,
    result_dir: str | Path,
    config: DVITrainConfig,
    base_projection: torch.Tensor | None = None,
) -> DVITrainResult:
    """Train the synthetic/offline draft head and publish a result directory."""
    config.validate()
    torch.manual_seed(config.seed)
    reader = DVIArtifactReader(artifact_dir)
    base_projection = _resolve_base_projection(reader, config, base_projection)
    model = SplitDVIDraftHead(
        base_projection,
        rank=config.rank,
        alpha=config.alpha,
        norm=config.norm,
        seed=config.seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    terminal_records_excluded = sum(
        record.position
        == len(reader.request_index[record.request_id].response_token_ids)
        for record in reader.iter_capture_records()
    )

    train_ids, validation_ids, split_hash = split_request_ids(
        [request.request_id for request in reader.requests],
        config.validation_fraction,
        config.seed,
    )
    initial_train_kl, train_initial_accuracy, train_topk_mass, num_train = (
        _evaluate(model, reader, config.batch_size, train_ids)
    )
    if validation_ids:
        initial_kl, initial_accuracy, topk_mass, num_validation = _evaluate(
            model, reader, config.batch_size, validation_ids
        )
        evaluation_split = "validation"
    else:
        initial_kl = initial_train_kl
        initial_accuracy = train_initial_accuracy
        topk_mass = train_topk_mass
        num_validation = 0
        evaluation_split = "train"
    num_records = num_train + num_validation
    epoch_kl: list[float] = []
    model.train()
    for _ in range(config.epochs):
        epoch_loss = 0.0
        epoch_records = 0
        for records in _iter_batches(reader, config.batch_size, train_ids):
            hidden = torch.stack(
                [record.stage_0_hidden.float() for record in records]
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _lumped_kl(model(hidden), records)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(records)
            epoch_records += len(records)
        epoch_kl.append(epoch_loss / epoch_records)

    final_train_kl, train_final_accuracy, _, _ = _evaluate(
        model, reader, config.batch_size, train_ids
    )
    if validation_ids:
        final_kl, final_accuracy, _, _ = _evaluate(
            model, reader, config.batch_size, validation_ids
        )
    else:
        final_kl = final_train_kl
        final_accuracy = train_final_accuracy

    result_dir = Path(result_dir)
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    if result_dir.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {result_dir}")
    temp_dir = Path(
        tempfile.mkdtemp(prefix=result_dir.name + ".", dir=result_dir.parent)
    )
    try:
        checkpoint_path = temp_dir / "draft_head.safetensors"
        trainable = model.trainable_state_dict()
        checkpoint = {
            "lora_A.weight": trainable["lora_a"],
            "lora_B.weight": trainable["lora_b"],
        }
        if config.embed_base_projection:
            checkpoint["base_projection.weight"] = (
                model.base_weight.detach().cpu().contiguous()
            )
        if "norm_weight" in trainable:
            checkpoint["norm.weight"] = trainable["norm_weight"]
        save_safetensors(checkpoint, str(checkpoint_path))
        checkpoint_hash = _sha256_file(checkpoint_path)
        metadata_path = temp_dir / "draft_head.json"
        base_projection_hash = _sha256_tensor(base_projection)
        metadata = {
            "norm": config.norm,
            "rank": config.rank,
            "alpha": config.alpha,
            "vocab_size": reader.manifest.vocab_size,
            "hidden_size": reader.manifest.hidden_size,
        }
        if not config.embed_base_projection:
            metadata.update(
                base_projection_source="target_model",
                base_projection_sha256=base_projection_hash,
            )
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                indent=2,
                sort_keys=True,
            )
        metadata_hash = _sha256_file(metadata_path)

        result = DVITrainResult(
            schema_version=TRAIN_RESULT_SCHEMA_VERSION,
            source_artifact_manifest_sha256=_sha256_file(
                Path(artifact_dir) / "manifest.json"
            ),
            source_base_checkpoint_hash=reader.manifest.base_checkpoint_hash,
            hidden_size=reader.manifest.hidden_size,
            vocab_size=reader.manifest.vocab_size,
            rank=config.rank,
            alpha=config.alpha,
            norm=config.norm,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.batch_size,
            seed=config.seed,
            num_records=num_records,
            initial_kl=initial_kl,
            final_kl=final_kl,
            initial_top1_accuracy=initial_accuracy,
            final_top1_accuracy=final_accuracy,
            mean_topk_mass=topk_mass,
            base_projection_sha256=base_projection_hash,
            terminal_records_excluded=terminal_records_excluded,
            draft_checkpoint_sha256=checkpoint_hash,
            draft_metadata_sha256=metadata_hash,
            base_projection_embedded=config.embed_base_projection,
            epoch_kl=epoch_kl,
            num_train_records=num_train,
            num_validation_records=num_validation,
            train_request_count=len(train_ids),
            validation_request_count=len(validation_ids),
            validation_fraction=config.validation_fraction,
            evaluation_split=evaluation_split,
            request_split_sha256=split_hash,
            initial_train_kl=initial_train_kl,
            final_train_kl=final_train_kl,
            kl_direction=reader.manifest.kl_direction,
            teacher_distribution_kind=reader.manifest.teacher_distribution_kind,
            teacher_temperature=reader.manifest.teacher_temperature,
            topk_mass_coverage=reader.manifest.topk_mass_coverage,
        )
        result.validate()
        with open(temp_dir / "train_result.json", "w", encoding="utf-8") as file:
            json.dump(asdict(result), file, indent=2, sort_keys=True)
        os.rename(temp_dir, result_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return result


def load_train_result(result_dir: str | Path) -> DVITrainResult:
    result_dir = Path(result_dir)
    with open(result_dir / "train_result.json", "r", encoding="utf-8") as file:
        result = DVITrainResult(**json.load(file))
    result.validate()
    checkpoint_path = result_dir / "draft_head.safetensors"
    metadata_path = result_dir / "draft_head.json"
    if _sha256_file(checkpoint_path) != result.draft_checkpoint_sha256:
        raise ValueError("Draft checkpoint checksum mismatch")
    if result.draft_metadata_sha256:
        if not metadata_path.is_file():
            raise ValueError("Draft checkpoint metadata is missing")
        if _sha256_file(metadata_path) != result.draft_metadata_sha256:
            raise ValueError("Draft checkpoint metadata checksum mismatch")
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    elif not result.base_projection_embedded:
        raise ValueError("Compact draft checkpoint metadata is missing")
    checkpoint = load_safetensors(str(checkpoint_path))
    runtime_embedded = {
        "base_projection.weight",
        "lora_A.weight",
        "lora_B.weight",
    }
    runtime_trainable = {"lora_A.weight", "lora_B.weight"}
    legacy_expected = {"base_weight", "lora_a", "lora_b"}
    if result.norm == "rmsnorm":
        runtime_embedded.add("norm.weight")
        runtime_trainable.add("norm.weight")
        legacy_expected.add("norm_weight")
    if result.base_projection_embedded:
        if set(checkpoint) == runtime_embedded:
            base_key = "base_projection.weight"
        elif set(checkpoint) == legacy_expected:
            base_key = "base_weight"
        else:
            raise ValueError(
                f"Unexpected embedded draft checkpoint tensors: "
                f"{sorted(checkpoint)}"
            )
        if _sha256_tensor(checkpoint[base_key]) != result.base_projection_sha256:
            raise ValueError("Base projection checksum mismatch")
    else:
        if set(checkpoint) != runtime_trainable:
            raise ValueError(
                f"Unexpected compact draft checkpoint tensors: "
                f"{sorted(checkpoint)}"
            )
        if metadata.get("base_projection_source") != "target_model":
            raise ValueError(
                "Compact draft checkpoint must declare target_model as its "
                "base projection source"
            )
        if metadata.get("base_projection_sha256") != result.base_projection_sha256:
            raise ValueError("Compact base projection checksum mismatch")
    return result


def build_offline_report(
    result_dir: str | Path,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    result = load_train_result(result_dir)
    if artifact_dir is not None:
        source_hash = _sha256_file(Path(artifact_dir) / "manifest.json")
        if source_hash != result.source_artifact_manifest_sha256:
            raise ValueError("Training result does not match the source artifact")
    reduction = (result.initial_kl - result.final_kl) / max(
        abs(result.initial_kl),
        1e-12,
    )
    return {
        "phase": "L0-offline-train",
        "evaluation_split": result.evaluation_split,
        "num_records": result.num_records,
        "num_train_records": result.num_train_records,
        "num_validation_records": result.num_validation_records,
        "train_request_count": result.train_request_count,
        "validation_request_count": result.validation_request_count,
        "validation_fraction": result.validation_fraction,
        "request_split_sha256": result.request_split_sha256,
        "initial_train_kl": result.initial_train_kl,
        "final_train_kl": result.final_train_kl,
        "terminal_records_excluded": result.terminal_records_excluded,
        "initial_kl": result.initial_kl,
        "final_kl": result.final_kl,
        "kl_reduction_fraction": reduction,
        "initial_top1_accuracy": result.initial_top1_accuracy,
        "final_top1_accuracy": result.final_top1_accuracy,
        "mean_topk_mass": result.mean_topk_mass,
        "base_projection_embedded": result.base_projection_embedded,
        "train_config": {
            "rank": result.rank,
            "alpha": result.alpha,
            "norm": result.norm,
            "learning_rate": result.learning_rate,
            "epochs": result.epochs,
            "batch_size": result.batch_size,
            "seed": result.seed,
            "validation_fraction": result.validation_fraction,
        },
        "go_no_go": "not_evaluated",
        "go_no_go_reason": (
            "Advancement length and end-to-end speedup require phase-4 block eval."
        ),
    }
