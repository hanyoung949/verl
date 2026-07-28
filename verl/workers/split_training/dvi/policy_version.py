"""Policy version helpers for DVI artifact session identity."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


def compute_policy_version(
    base_checkpoint: str | Path,
    head_adapter: str | Path | None = None,
    tail_adapter: str | Path | None = None,
    tokenizer_revision: str = "main",
    base_checkpoint_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Compute a stable policy version string for a DVI artifact session.

    The policy version identifies the exact model weights + tokenizer used for
    a rollout. It must change whenever the base model, LoRA adapters, or
    tokenizer revision changes, so that artifact merger can reject mixed-policy
    spools.

    The implementation is content-based: it hashes the bytes of weight files,
    adapter files, and tokenizer files rather than paths or mtimes. This keeps
    the version deterministic across nodes and immutable for a given artifact.

    Args:
        base_checkpoint: Path to the base model checkpoint directory or single
            weight file.
        head_adapter: Path to the head LoRA adapter directory/file, or a
            pre-computed SHA256 hex digest.
        tail_adapter: Path to the tail LoRA adapter directory/file, or a
            pre-computed SHA256 hex digest.
        tokenizer_revision: Tokenizer revision string.
        base_checkpoint_hash: Optional pre-computed hash for the base
            checkpoint. If provided, the function will not re-read the
            (potentially large) base checkpoint files. The driver is expected
            to compute/cache this once per weight sync.
        extra: Extra key/value pairs to include in the hash.

    Raises:
        FileNotFoundError: If ``base_checkpoint`` does not exist and no
            ``base_checkpoint_hash`` is provided.
        ValueError: If a checkpoint directory contains no recognizable weight
            files, or if an adapter path does not exist.
    """
    hasher = hashlib.sha256()
    if base_checkpoint_hash is not None:
        hasher.update(base_checkpoint_hash.encode("utf-8"))
    else:
        hasher.update(_hash_checkpoint(base_checkpoint).encode("utf-8"))
    hasher.update(_hash_adapter(head_adapter).encode("utf-8"))
    hasher.update(_hash_adapter(tail_adapter).encode("utf-8"))
    hasher.update(_hash_tokenizer(base_checkpoint, tokenizer_revision).encode("utf-8"))
    if extra:
        for key in sorted(extra):
            hasher.update(f"{key}={extra[key]}".encode("utf-8"))
    return hasher.hexdigest()[:32]


def _hash_file_stream(
    path: Path, hasher: hashlib._Hash, chunk_size: int = 4 * 1024 * 1024
) -> None:
    """Update *hasher* with the contents of *path* using chunked reads."""
    st = path.stat()
    hasher.update(str(st.st_size).encode("utf-8"))
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)


def _hash_directory(path: Path, pattern: tuple[str, ...] | None = None) -> str:
    """Hash all files under *path* matching *pattern* (sorted, relative path + content)."""
    hasher = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if pattern is not None:
        files = [p for p in files if any(p.name.endswith(ext) for ext in pattern)]
    for child in files:
        rel = str(child.relative_to(path))
        hasher.update(rel.encode("utf-8"))
        _hash_file_stream(child, hasher)
    return hasher.hexdigest()[:32]


def _hash_checkpoint(value: str | Path) -> str:
    """Hash a base checkpoint directory or single weight file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If a directory contains no recognizable weight or index
            files.
    """
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {path}")
    if path.is_file():
        hasher = hashlib.sha256()
        _hash_file_stream(path, hasher)
        return hasher.hexdigest()[:32]

    weight_exts = (
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
    )
    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    hasher = hashlib.sha256()
    matched_any = False
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = str(child.relative_to(path))
        is_weight = child.name.endswith(weight_exts) or child.name in index_names
        if not is_weight:
            continue
        matched_any = True
        hasher.update(rel.encode("utf-8"))
        _hash_file_stream(child, hasher)

    if not matched_any:
        raise ValueError(
            f"Base checkpoint directory contains no weight files: {path}"
        )
    return hasher.hexdigest()[:32]


def _hash_adapter(value: str | Path | None) -> str:
    """Return a stable hash for an adapter path or a pre-computed hex digest."""
    if value is None:
        return ""
    if isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    ):
        return value
    path = Path(value)
    if not path.exists():
        raise ValueError(f"Adapter path does not exist: {path}")
    if path.is_file():
        hasher = hashlib.sha256()
        _hash_file_stream(path, hasher)
        return hasher.hexdigest()[:32]
    return _hash_directory(path)


def _hash_tokenizer(checkpoint: str | Path, tokenizer_revision: str) -> str:
    """Hash tokenizer files that affect tokenization behavior.

    We first look inside the checkpoint directory; if no tokenizer files are
    present, the revision string still contributes to the hash, but note that
    this is less strict than hashing the actual tokenizer artifacts.
    """
    path = Path(checkpoint)
    hasher = hashlib.sha256()
    hasher.update(tokenizer_revision.encode("utf-8"))
    tokenizer_suffixes = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
    )
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.name.endswith(tokenizer_suffixes)
        )
        for child in files:
            rel = str(child.relative_to(path))
            hasher.update(rel.encode("utf-8"))
            _hash_file_stream(child, hasher)
    return hasher.hexdigest()[:32]
