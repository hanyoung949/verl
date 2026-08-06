"""DVI telemetry orchestration for the standard verl agent loop.

This module is opt-in through ``rollout.agent.agent_loop_manager_class``.  It
keeps the default AgentLoopManager unchanged while wrapping each rollout in a
transactional dual-sided telemetry session.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils.ray_utils import auto_await


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def hash_tensor_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash deployed tensor names, shapes, dtypes, and exact contiguous bytes."""
    if not state_dict:
        raise ValueError("DVI policy state_dict must not be empty")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"DVI policy tensor {name!r} is not a torch.Tensor")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_live_policy_metadata(
    *,
    base_checkpoint_hash: str,
    tokenizer_revision: str,
    lora_state_dict: dict[str, torch.Tensor],
    peft_config: dict[str, Any],
) -> dict[str, str]:
    """Build the exact identity of one successfully deployed merged adapter."""
    adapter_hash = hash_tensor_state_dict(lora_state_dict)
    payload = json.dumps(
        {
            "base_checkpoint_hash": base_checkpoint_hash,
            "merged_adapter_sha256": adapter_hash,
            "peft_config": peft_config,
            "tokenizer_revision": tokenizer_revision,
        },
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    policy_version = hashlib.sha256(payload).hexdigest()[:32]
    deployed_hash = f"sha256:{adapter_hash}"
    return {
        "policy_version": policy_version,
        # The rollout engine receives one merged adapter. Preserve its exact
        # deployed identity in both legacy side fields rather than inventing
        # unavailable pre-merge head/tail hashes.
        "head_adapter_hash": deployed_hash,
        "tail_adapter_hash": deployed_hash,
    }


@dataclass(frozen=True)
class DVIRolloutTelemetryConfig:
    spool_root: Path
    run_id: str
    base_checkpoint_hash: str
    tokenizer_revision: str
    quota_bytes: int
    sampling_config: dict[str, Any]
    initial_policy: dict[str, str] | None = None
    handoff_pool_size: int = 64
    capture_mode: str = "train"
    capture_dtype: str | None = None
    draft_length: int = 4
    bonus_token: bool = False

    @classmethod
    def from_rollout_config(cls, rollout_config: Any) -> DVIRolloutTelemetryConfig:
        if getattr(rollout_config, "name", None) != "vllm" or not bool(
            getattr(rollout_config, "enable_layerwise_split", False)
        ):
            raise ValueError(
                "DVIAgentLoopManager requires vLLM layer-wise split rollout"
            )
        custom = getattr(rollout_config, "custom", None) or {}
        raw = custom.get("dvi_telemetry") if hasattr(custom, "get") else None
        if not raw or not bool(raw.get("enabled", False)):
            raise ValueError(
                "DVIAgentLoopManager requires "
                "rollout.custom.dvi_telemetry.enabled=true"
            )
        required = (
            "spool_root",
            "run_id",
            "base_checkpoint_hash",
            "tokenizer_revision",
        )
        missing = [name for name in required if not raw.get(name)]
        if missing:
            raise ValueError(
                f"DVI rollout telemetry config is missing {sorted(missing)}"
            )
        run_id = str(raw["run_id"])
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError(f"Unsafe DVI run_id {run_id!r}")
        quota_bytes = int(raw.get("quota_bytes", 10 * 1024**3))
        handoff_pool_size = int(raw.get("handoff_pool_size", 64))
        if quota_bytes <= 0 or handoff_pool_size <= 0:
            raise ValueError("DVI quota_bytes and handoff_pool_size must be positive")
        capture_mode = str(raw.get("capture_mode", "train"))
        if capture_mode not in {"train", "evaluation"}:
            raise ValueError(
                "DVI capture_mode must be 'train' or 'evaluation'"
            )
        sampling = dict(raw.get("sampling_config") or {})
        sampling.setdefault("sample_rate", 0.05)
        sampling.setdefault("max_per_request", 16)
        sampling.setdefault("seed", 42)
        sampling.setdefault("top_k", 8)
        if not 0.0 < float(sampling["sample_rate"]) <= 1.0:
            raise ValueError("DVI sample_rate must be in (0, 1]")
        if int(sampling["max_per_request"]) <= 0 or int(sampling["top_k"]) <= 0:
            raise ValueError("DVI max_per_request and top_k must be positive")
        draft_length = int(raw.get("draft_length", 4))
        bonus_token = bool(raw.get("bonus_token", capture_mode == "evaluation"))
        if draft_length < 2:
            raise ValueError("DVI draft_length must be at least 2")
        if capture_mode == "evaluation" and not bonus_token:
            raise ValueError(
                "evaluation capture requires bonus_token=True for stochastic DVI"
            )

        initial_policy = raw.get("initial_policy")
        if initial_policy is not None:
            initial_policy = dict(initial_policy)
            policy_required = {
                "policy_version",
                "head_adapter_hash",
                "tail_adapter_hash",
            }
            if policy_required - set(initial_policy):
                raise ValueError("DVI initial_policy is incomplete")
        return cls(
            spool_root=Path(str(raw["spool_root"])),
            run_id=run_id,
            base_checkpoint_hash=str(raw["base_checkpoint_hash"]),
            tokenizer_revision=str(raw["tokenizer_revision"]),
            quota_bytes=quota_bytes,
            sampling_config=sampling,
            initial_policy=initial_policy,
            handoff_pool_size=handoff_pool_size,
            capture_mode=capture_mode,
            capture_dtype=raw.get("capture_dtype"),
            draft_length=draft_length,
            bonus_token=bonus_token,
        )


class DVIRolloutTelemetryCoordinator:
    """Transactionally wrap complete agent-loop rollouts in DVI sessions."""

    def __init__(self, config: DVIRolloutTelemetryConfig, replicas: list[Any]):
        if not replicas:
            raise ValueError("DVI rollout telemetry requires at least one replica")
        self.config = config
        self.replicas = replicas
        self._lock = asyncio.Lock()
        self._counter = 0

    def _rollout_id(self, prompts: Any) -> str:
        meta = getattr(prompts, "meta_info", {}) or {}
        explicit = meta.get("dvi_rollout_id")
        if explicit is not None:
            value = str(explicit)
            if not _SAFE_ID.fullmatch(value):
                raise ValueError(f"Unsafe DVI rollout_id {value!r}")
            return value
        step = meta.get("global_steps", "unknown")
        value = f"step_{step}_{self._counter:06d}"
        self._counter += 1
        if not _SAFE_ID.fullmatch(value):
            raise ValueError(f"Unsafe generated DVI rollout_id {value!r}")
        return value

    async def run(
        self,
        prompts: Any,
        policy: dict[str, str] | None | Callable[[], dict[str, str] | None],
        generate: Callable[[], Awaitable[Any]],
    ) -> Any:
        async with self._lock:
            resolved_policy = policy() if callable(policy) else policy
            required_policy = {
                "policy_version",
                "head_adapter_hash",
                "tail_adapter_hash",
            }
            if resolved_policy is None:
                raise RuntimeError(
                    "DVI capture has no deployed policy identity; run a "
                    "successful LoRA update or configure initial_policy "
                    "before rollout"
                )
            missing = required_policy - set(resolved_policy)
            if missing:
                raise RuntimeError(
                    f"DVI policy identity is missing {sorted(missing)}"
                )
            rollout_id = self._rollout_id(prompts)
            rollout_root = (
                self.config.spool_root / self.config.run_id / rollout_id
            )
            if rollout_root.exists():
                raise FileExistsError(
                    f"Refusing to reuse DVI rollout spool: {rollout_root}"
                )
            rollout_root.mkdir(parents=True)
            metadata: dict[str, Any] = {
                "run_id": self.config.run_id,
                "rollout_id": rollout_id,
                "base_checkpoint_hash": self.config.base_checkpoint_hash,
                "tokenizer_revision": self.config.tokenizer_revision,
                "quota_bytes": self.config.quota_bytes,
                "handoff_pool_size": self.config.handoff_pool_size,
                "capture_mode": self.config.capture_mode,
                "draft_length": self.config.draft_length,
                "num_proposals": self.config.draft_length - 1,
                "bonus_token": self.config.bonus_token,
                **resolved_policy,
            }
            if self.config.capture_dtype is not None:
                metadata["capture_dtype"] = self.config.capture_dtype

            started: list[tuple[int, Any]] = []
            try:
                for index, replica in enumerate(self.replicas):
                    await replica.start_dvi_session(
                        str(rollout_root / f"replica_{index}"),
                        metadata,
                        self.config.sampling_config,
                    )
                    started.append((index, replica))
            except BaseException:
                await asyncio.gather(
                    *(replica.close_dvi_session() for _, replica in started),
                    return_exceptions=True,
                )
                raise

            try:
                output = await generate()
            except BaseException as generation_error:
                close_results = await self._close_started(started)
                violations = self._close_violations(close_results)
                if violations and hasattr(generation_error, "add_note"):
                    generation_error.add_note(
                        f"DVI close also failed: {violations}"
                    )
                raise

            close_results = await self._close_started(started)
            violations = self._close_violations(close_results)
            if violations:
                raise RuntimeError(
                    f"DVI rollout telemetry close failed: {violations}"
                )
            output.meta_info = dict(getattr(output, "meta_info", {}) or {})
            output.meta_info["dvi_telemetry"] = {
                "run_id": self.config.run_id,
                "rollout_id": rollout_id,
                "policy_version": resolved_policy["policy_version"],
                "spool_root": str(rollout_root),
                "replica_results": close_results,
            }
            return output

    async def run_exclusive(
        self,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Serialize a policy update against complete capture sessions."""
        async with self._lock:
            return await operation()

    @staticmethod
    async def _close_started(
        started: list[tuple[int, Any]],
    ) -> dict[str, Any]:
        results = await asyncio.gather(
            *(replica.close_dvi_session() for _, replica in started),
            return_exceptions=True,
        )
        return {
            f"replica_{index}": result
            for (index, _), result in zip(started, results, strict=True)
        }

    @staticmethod
    def _close_violations(results: dict[str, Any]) -> list[str]:
        violations: list[str] = []
        for replica, result in results.items():
            if isinstance(result, BaseException):
                violations.append(f"{replica}: {result}")
            elif not isinstance(result, dict) or result.get("ok") is not True:
                violations.append(f"{replica}: invalid close result {result!r}")
        return violations


class DVIAgentLoopManager(AgentLoopManager):
    """AgentLoopManager with opt-in fail-closed DVI capture per rollout."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        config = DVIRolloutTelemetryConfig.from_rollout_config(
            self.rollout_config
        )
        self._dvi_config = config
        self._dvi_policy = config.initial_policy
        self._dvi_coordinator: DVIRolloutTelemetryCoordinator | None = None

    async def _initialize_llm_servers(self):
        await super()._initialize_llm_servers()
        self._dvi_coordinator = DVIRolloutTelemetryCoordinator(
            self._dvi_config,
            self.rollout_replicas,
        )

    @auto_await
    async def update_lora_adapter(
        self,
        lora_state_dict: dict[str, torch.Tensor],
        peft_config: dict,
    ) -> bool:
        if self._dvi_coordinator is None:
            raise RuntimeError("DVI telemetry coordinator is not initialized")

        async def deploy_and_record() -> bool:
            candidate_policy = build_live_policy_metadata(
                base_checkpoint_hash=self._dvi_config.base_checkpoint_hash,
                tokenizer_revision=self._dvi_config.tokenizer_revision,
                lora_state_dict=lora_state_dict,
                peft_config=peft_config,
            )
            try:
                results = await asyncio.gather(
                    *(
                        replica.update_lora_adapter(
                            lora_state_dict, peft_config
                        )
                        for replica in self.rollout_replicas
                    ),
                    return_exceptions=True,
                )
            except BaseException:
                self._dvi_policy = None
                raise

            errors = [
                (index, result)
                for index, result in enumerate(results)
                if isinstance(result, BaseException)
            ]
            if errors:
                self._dvi_policy = None
                details = ", ".join(
                    f"replica_{index}: {error!r}"
                    for index, error in errors
                )
                raise RuntimeError(
                    f"DVI LoRA deployment failed: {details}"
                ) from errors[0][1]

            if not all(result is True for result in results):
                self._dvi_policy = None
                return False

            self._dvi_policy = candidate_policy
            return True

        return await self._dvi_coordinator.run_exclusive(deploy_and_record)

    @auto_await
    async def generate_sequences(self, prompts: Any) -> Any:
        if self._dvi_coordinator is None:
            raise RuntimeError("DVI telemetry coordinator is not initialized")
        return await self._dvi_coordinator.run(
            prompts,
            lambda: self._dvi_policy,
            lambda: super(DVIAgentLoopManager, self).generate_sequences(prompts),
        )


__all__ = [
    "DVIAgentLoopManager",
    "DVIRolloutTelemetryConfig",
    "DVIRolloutTelemetryCoordinator",
    "build_live_policy_metadata",
    "hash_tensor_state_dict",
]
