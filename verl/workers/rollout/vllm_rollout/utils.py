# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes
import json
import logging
import os
import platform
import re
import signal
import threading
from types import MethodType
from typing import Any, Literal, Optional, get_args

import torch
from vllm.outputs import RequestOutput

from verl.utils.device import is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

try:
    from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

    from verl.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack

    _VLLM_OMNI_AVAILABLE = True
except (ImportError, RuntimeError):  # optional stack; ImportError if missing, RuntimeError e.g. diffusers/transformers
    CustomPipelineWorkerExtension = None  # type: ignore[assignment]
    OmniTensorLoRARequest = None  # type: ignore[assignment]
    VLLMOmniHijack = None  # type: ignore[assignment]
    _VLLM_OMNI_AVAILABLE = False

# Use object as fallback base so the class definition is always valid even when
# vllm_omni is not installed (None is not a valid base class).
_OmniWorkerBase = CustomPipelineWorkerExtension if _VLLM_OMNI_AVAILABLE else object

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
            npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
            return "NPU-" + npu_visible_devices[device_id]
        else:
            return f"NPU-{device_id}"
    else:
        return current_platform.get_device_uuid(device_id)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMHijack.hijack()
        # 2. patch online fp8 quant
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        vllm_config = kwargs.get("vllm_config")
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        _is_modelopt_qat = type(quant_config).__name__ == "ModelOptNvFp4Config"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
        elif _is_modelopt_qat:
            from verl.utils.modelopt import apply_modelopt_nvfp4_patches

            apply_modelopt_nvfp4_patches()
            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        instance._is_modelopt_qat = _is_modelopt_qat
        return instance

    def monkey_patch_model(self, vocab_size: int):
        # patch compute_logits to avoid sampling OOV token
        monkey_patch_compute_logits(self.model_runner.model, vocab_size)
        # patch weight loader to support MoE model
        patch_vllm_moe_model_weight_loader(self.model_runner.model)

    def update_weights_from_ipc(
        self,
        peft_config: dict = None,
        base_sync_done=False,
        use_shm: bool = False,
        weight_sync_addrs: list[str] = None,
    ):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightReceiver,
            TcpBucketedWeightReceiver,
            _format_tcp_address,
            _get_free_tcp_port,
            _get_worker_node_ip,
        )

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        use_standard_weight_load = not (peft_config and base_sync_done) and not is_fp8_model(
            self.model_runner.vllm_config
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            prepare_qat_for_load_weights(self.model_runner.model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif use_standard_weight_load:
            # Re-apply here because async IPC weight sync can happen long after init and lose MoE weight_loader attrs.
            patch_vllm_moe_model_weight_loader(self.model_runner.model)

        assert self.device is not None

        if weight_sync_addrs is not None:
            # Cross-node TCP path (C2-2): the coordinator bound one endpoint per
            # worker and passed the connect addresses to us.  We pick our own
            # address by global rank and connect back to the server.
            rank = getattr(self, "rank", self.local_rank)
            assert 0 <= rank < len(weight_sync_addrs), (
                f"Worker rank {rank} out of range for {len(weight_sync_addrs)} addresses"
            )
            connect_addr = weight_sync_addrs[rank]
            logger.info(
                "Worker rank=%d connecting to TCP weight endpoint %s",
                rank,
                connect_addr,
            )
            receiver = TcpBucketedWeightReceiver(
                zmq_handle=connect_addr,
                device=self.device,
            )
        else:
            # Single-node IPC path (default, backwards compatible).
            receiver = BucketedWeightReceiver(
                zmq_handle=self._get_zmq_handle(),
                device=self.device,
                use_shm=use_shm,
            )

        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            manual_process_weights_after_loading(self.model_runner.model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            model = self.model_runner.model
            model_config = self.model_runner.vllm_config.model_config
            process_weights_after_loading(model, model_config, self.device)

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, self.model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                self.model_runner.model.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.
        Uses replica_rank + local_rank to form handle so it matches the sender side
        regardless of CUDA_VISIBLE_DEVICES differences, and avoids collisions
        when multiple replicas share the same node.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        return f"ipc:///tmp/rl-colocate-zmq-replica-{replica_rank}-rank-{self.local_rank}.sock"

    def update_lora_adapter(
        self,
        lora_state_dict: dict[str, torch.Tensor],
        peft_config: dict,
    ) -> bool:
        """Load a LoRA adapter from an in-memory state dict.

        This is invoked by split training's WeightSyncManager via
        ``vLLMHttpServer.update_lora_adapter`` / ``collective_rpc``.  Constructing
        the ``TensorLoRARequest`` inside the worker (instead of in the server
        actor) avoids msgspec deserialization dropping the subclass fields.
        """
        try:
            self.remove_lora(VLLM_LORA_INT_ID)
        except Exception:
            pass

        lora_request = TensorLoRARequest(
            lora_name=VLLM_LORA_NAME,
            lora_int_id=VLLM_LORA_INT_ID,
            lora_path=VLLM_LORA_PATH,
            peft_config=peft_config,
            lora_tensors=lora_state_dict,
        )
        return self.add_lora(lora_request)

    def read_lora_weight_norms(self) -> dict[str, float]:
        """Read L2 norms of currently loaded LoRA weights for verification.

        Returns a dict mapping parameter name to its L2 norm (float).
        Returns an empty dict if no LoRA adapter is loaded.
        """
        result: dict[str, float] = {}
        try:
            model = self.model_runner.model
            for name, param in model.named_parameters():
                if "lora" in name and param.requires_grad:
                    result[name] = float(param.data.norm().item())
        except Exception as e:
            logger.warning("read_lora_weight_norms failed: %s", e)
        return result

    # ------------------------------------------------------------------
    # DVI L0 telemetry lifecycle
    # ------------------------------------------------------------------
    _DVI_DEFAULT_QUOTA_BYTES = 10 * 1024**3  # 10 GiB
    _DVI_STAGE0_QUOTA_PERCENT = 90

    @classmethod
    def _dvi_side_quota(cls, total_bytes: int, *, stage0: bool) -> int:
        """Split one run-level quota between the two independent spools."""
        if total_bytes < 2:
            raise ValueError("DVI quota_bytes must be at least 2")
        stage0_bytes = total_bytes * cls._DVI_STAGE0_QUOTA_PERCENT // 100
        stage0_bytes = max(1, min(stage0_bytes, total_bytes - 1))
        if stage0:
            return stage0_bytes
        return total_bytes - stage0_bytes

    @staticmethod
    def _dvi_capture_dtype(
        spool_metadata: dict[str, Any],
        runner_dtype: Any,
    ) -> torch.dtype | None:
        """Resolve and persist the stage-0 capture dtype for both spools."""
        configured = spool_metadata.get("capture_dtype")
        if configured is None:
            if not isinstance(runner_dtype, torch.dtype):
                return None
            dtype = runner_dtype
        elif isinstance(configured, torch.dtype):
            dtype = configured
        else:
            dtype_name = str(configured).removeprefix("torch.")
            dtype = getattr(torch, dtype_name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"Unsupported DVI capture_dtype: {configured!r}"
                )
        spool_metadata["capture_dtype"] = str(dtype).removeprefix("torch.")
        return dtype

    def _is_dvi_stage0_capture_owner(self, model_runner: Any) -> bool:
        """Return True if this worker owns the stage-0 hidden spool writer.

        Owner is the first PP-rank worker with tensor-parallel rank 0
        (fail-closed, same rule as the stage-2 owner check).
        """
        if model_runner is None:
            return False
        if not getattr(model_runner, "is_first_pp_rank", False):
            return False
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
            )

            return get_tensor_model_parallel_rank() == 0
        except Exception:
            return False

    def _is_dvi_capture_owner(self, model_runner: Any) -> bool:
        """Return True if this worker should own the stage-2 spool writer.

        Owner is the last PP-rank worker with tensor-parallel rank 0. This
        avoids multiple workers writing to the same spool directory. The check
        is fail-closed: if we cannot positively confirm rank 0, we return
        ``False`` so the session is safe even when distributed state is in an
        unexpected state.
        """
        if model_runner is None:
            return False
        if not getattr(model_runner, "is_last_pp_rank", False):
            return False
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
            )

            return get_tensor_model_parallel_rank() == 0
        except Exception:
            # Cannot confirm rank; fail-closed rather than risk multiple owners.
            return False

    def _start_dvi_stage0_side(
        self,
        spool_dir: str,
        spool_metadata: dict[str, Any],
        sampling_config: dict[str, Any],
    ) -> None:
        """Create the stage_0 hidden writer/producer/probe on this worker.

        Uses a sibling spool dir (``<spool_dir>-stage0``): the offline merger
        expects separate stage_0/stage_2 spool directories, and two writers
        must not share a spool manifest.  The sampling config is identical on
        both sides so both stages reserve the same deterministic position set.
        Transactional like the stage_2 setup: on failure the writer is closed
        and references cleared.
        """
        from pathlib import Path

        from vllm.v1.worker.gpu.split_dvi.hooks import DVIStage0TelemetryProbe
        from vllm.v1.worker.gpu.split_dvi.telemetry import (
            DVIAsyncTensorHandoff,
            DVIPartialSpoolWriter,
            DVISamplingConfig,
            DVIStage0TelemetryProducer,
            DVITensorSpec,
        )

        model_runner = getattr(self, "model_runner", None)
        merged_metadata = dict(spool_metadata)
        merged_metadata["sampling_config"] = dict(sampling_config)
        runner_dtype = getattr(model_runner, "dtype", None)
        capture_dtype = self._dvi_capture_dtype(merged_metadata, runner_dtype)
        total_quota_bytes = int(
            merged_metadata.get("quota_bytes", self._DVI_DEFAULT_QUOTA_BYTES)
        )
        quota_bytes = self._dvi_side_quota(total_quota_bytes, stage0=True)
        stage0_dir = Path(spool_dir).with_name(Path(spool_dir).name + "-stage0")
        writer = DVIPartialSpoolWriter(
            spool_dir=stage0_dir,
            spool_metadata=merged_metadata,
            quota_bytes=quota_bytes,
            queue_maxsize=int(merged_metadata.get("queue_maxsize", 1024)),
        )
        try:
            cfg = DVISamplingConfig(
                sample_rate=float(sampling_config["sample_rate"]),
                max_per_request=int(sampling_config["max_per_request"]),
                seed=int(sampling_config["seed"]),
                top_k=int(sampling_config.get("top_k", 8)),
            )
            handoff = None
            device = getattr(model_runner, "device", None)
            hidden_size = None
            model_config = getattr(model_runner, "model_config", None)
            if model_config is not None and hasattr(
                model_config, "get_hidden_size"
            ):
                value = model_config.get_hidden_size()
                if type(value) is int:
                    hidden_size = value
            if (
                isinstance(device, torch.device)
                and device.type == "cuda"
                and hidden_size is not None
            ):
                if capture_dtype is None:
                    raise ValueError("CUDA DVI capture requires a concrete dtype")
                handoff = DVIAsyncTensorHandoff(
                    {"hidden": DVITensorSpec((hidden_size,), capture_dtype)},
                    pool_size=int(merged_metadata.get("handoff_pool_size", 64)),
                )
            producer = DVIStage0TelemetryProducer(writer, cfg, handoff=handoff)
            probe = DVIStage0TelemetryProbe(producer, enabled=True)

            self._dvi_stage0_writer: Any = writer
            self._dvi_stage0_producer: Any = producer
            self._dvi_stage0_probe: Any = probe
            if model_runner is not None and hasattr(
                model_runner, "set_dvi_stage0_probe"
            ):
                model_runner.set_dvi_stage0_probe(probe)
        except Exception:
            producer = locals().get("producer")
            handoff = locals().get("handoff")
            try:
                if producer is not None:
                    producer.close()
                elif handoff is not None:
                    handoff.close()
            except Exception:
                pass
            try:
                writer.close()
            except Exception:
                pass
            self._dvi_stage0_writer = None
            self._dvi_stage0_producer = None
            self._dvi_stage0_probe = None
            if model_runner is not None and hasattr(
                model_runner, "set_dvi_stage0_probe"
            ):
                model_runner.set_dvi_stage0_probe(None)
            raise

    def start_dvi_session(
        self,
        spool_dir: str,
        spool_metadata: dict[str, Any],
        sampling_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Start a new stage-2 DVI telemetry session on this worker.

        Each rollout should use its own spool directory. Calling this while a
        session already exists will close the previous session first.

        Only the designated capture-owner worker creates a writer; other workers
        disable telemetry locally. The driver should pass a per-server unique
        ``spool_dir`` (or root) to avoid conflicts across server actors.

        The start is transactional: if hook/producer initialization fails after
        the writer is created, the writer is closed and all references are
        cleared before re-raising.

        Returns:
            ``{"owner": True, "spool_dir": str}`` if this worker owns the
            writer, or ``{"owner": False}`` otherwise.
        """
        self.close_dvi_session()

        model_runner = getattr(self, "model_runner", None)

        # stage_0 hidden side: independent owner on the first PP rank; shares
        # the same spool dir and sampling config so both stages finalize to
        # identical deterministic sample sets.
        stage0_owner = self._is_dvi_stage0_capture_owner(model_runner)
        if stage0_owner:
            self._start_dvi_stage0_side(spool_dir, spool_metadata, sampling_config)
        elif model_runner is not None and hasattr(
            model_runner, "set_dvi_stage0_probe"
        ):
            model_runner.set_dvi_stage0_probe(None)

        if not self._is_dvi_capture_owner(model_runner):
            if model_runner is not None and hasattr(
                model_runner, "set_dvi_telemetry_hook"
            ):
                model_runner.set_dvi_telemetry_hook(None)
            if stage0_owner:
                return {"owner": False, "stage0_owner": True}
            return {"owner": False}

        from pathlib import Path

        from vllm.v1.worker.gpu.split_dvi.hooks import DVIStage2TelemetryHook
        from vllm.v1.worker.gpu.split_dvi.telemetry import (
            DVIAsyncTensorHandoff,
            DVIPartialSpoolWriter,
            DVISamplingConfig,
            DVIStage2TelemetryProducer,
            DVITensorSpec,
        )

        merged_metadata = dict(spool_metadata)
        merged_metadata["sampling_config"] = dict(sampling_config)
        runner_dtype = getattr(model_runner, "dtype", None)
        self._dvi_capture_dtype(merged_metadata, runner_dtype)

        total_quota_bytes = int(
            merged_metadata.get("quota_bytes", self._DVI_DEFAULT_QUOTA_BYTES)
        )
        quota_bytes = self._dvi_side_quota(total_quota_bytes, stage0=False)
        writer = DVIPartialSpoolWriter(
            spool_dir=Path(spool_dir),
            spool_metadata=merged_metadata,
            quota_bytes=quota_bytes,
            queue_maxsize=int(merged_metadata.get("queue_maxsize", 1024)),
        )

        try:
            cfg = DVISamplingConfig(
                sample_rate=float(sampling_config["sample_rate"]),
                max_per_request=int(sampling_config["max_per_request"]),
                seed=int(sampling_config["seed"]),
                top_k=int(sampling_config.get("top_k", 8)),
            )
            handoff = None
            device = getattr(model_runner, "device", None)
            if isinstance(device, torch.device) and device.type == "cuda":
                top_k = cfg.top_k
                specs = {
                    "topk_ids": DVITensorSpec((top_k,), torch.int32),
                    "topk_logprobs": DVITensorSpec((top_k,), torch.float32),
                    "residual_mass": DVITensorSpec((), torch.float32),
                    "top1_id": DVITensorSpec((), torch.int32),
                    "valid_count": DVITensorSpec((), torch.int32),
                }
                handoff = DVIAsyncTensorHandoff(
                    specs,
                    pool_size=int(merged_metadata.get("handoff_pool_size", 64)),
                )
            producer = DVIStage2TelemetryProducer(writer, cfg, handoff=handoff)
            hook = DVIStage2TelemetryHook(producer, enabled=True)

            self._dvi_writer: Any = writer
            self._dvi_stage2_producer: Any = producer
            self._dvi_hook: Any = hook
            self._dvi_finalize_no_match_count = 0
            self._dvi_finalize_ambiguous_count = 0

            if model_runner is not None and hasattr(
                model_runner, "set_dvi_telemetry_hook"
            ):
                model_runner.set_dvi_telemetry_hook(hook)
        except Exception:
            # Roll back the writer thread and references on any failure.
            producer = locals().get("producer")
            handoff = locals().get("handoff")
            try:
                if producer is not None:
                    producer.close()
                elif handoff is not None:
                    handoff.close()
            except Exception:
                pass
            try:
                writer.close()
            except Exception:
                pass
            self._dvi_writer = None
            self._dvi_stage2_producer = None
            self._dvi_hook = None
            self._dvi_finalize_no_match_count = 0
            self._dvi_finalize_ambiguous_count = 0
            if model_runner is not None and hasattr(
                model_runner, "set_dvi_telemetry_hook"
            ):
                model_runner.set_dvi_telemetry_hook(None)
            raise

        return {
            "owner": True,
            "spool_dir": str(spool_dir),
            "stage0_owner": stage0_owner,
        }

    def finalize_dvi_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        response_token_ids: list[int],
    ) -> None:
        """Finalize a request in the active stage-2 DVI session.

        vLLM randomizes internal request ids to avoid collisions, so the worker
        buffers are keyed by an internal id such as ``{external_id}-{8hex}``.
        The driver should ensure ``request_id`` is unique within the session;
        we resolve it to the internal id using a strict regex and fail-fast if
        the mapping is ambiguous.
        """
        producer = getattr(self, "_dvi_stage2_producer", None)
        if producer is None:
            return
        session_key = producer.session_key
        expected_len = len(session_key) + 1
        exact_key = session_key + (request_id,)

        # Exact match: either randomization is disabled or the caller supplied
        # the internal id directly.
        if exact_key in producer._buffers:
            matched = [request_id]
        else:
            pattern = re.compile(
                rf"^{re.escape(request_id)}-[0-9a-f]{{8}}$"
            )
            candidates = (
                producer._evaluation_request_ids
                if producer.evaluation_mode
                else (
                    buf_key[-1]
                    for buf_key in producer._buffers.keys()
                    if len(buf_key) == expected_len
                    and buf_key[: len(session_key)] == session_key
                )
            )
            matched = [candidate for candidate in candidates if pattern.match(candidate)]
            if len(matched) > 1:
                self._dvi_finalize_ambiguous_count += 1
                raise RuntimeError(
                    f"Ambiguous request id mapping for {request_id!r}: "
                    f"multiple internal ids matched {matched}"
                )

        if not matched:
            # No records captured for this request; nothing to finalize.
            self._dvi_finalize_no_match_count += 1
            return

        producer.finalize_request(
            session_key,
            matched[0],
            prompt_token_ids,
            response_token_ids,
        )

    def close_dvi_session(self) -> dict[str, Any]:
        """Close the active DVI session(s) and return writer metrics.

        Raises any writer error so the RPC caller knows the spool may not be
        finalized. Hooks are removed before handoffs drain and writers close.

        Returns:
            A dict containing ``owner`` plus writer metrics and finalize
            counters. Non-owner workers return ``{"owner": False}``.
        """
        model_runner = getattr(self, "model_runner", None)

        # stage_0 hidden side (independent owner on the first PP rank).
        stage0_writer = getattr(self, "_dvi_stage0_writer", None)
        stage0_producer = getattr(self, "_dvi_stage0_producer", None)
        stage0_metrics: dict[str, Any] = {}
        if stage0_writer is not None:
            if model_runner is not None and hasattr(
                model_runner, "set_dvi_stage0_probe"
            ):
                model_runner.set_dvi_stage0_probe(None)
            try:
                try:
                    if stage0_producer is not None:
                        stage0_producer.close()
                finally:
                    stage0_metrics = stage0_writer.close()
            finally:
                self._dvi_stage0_writer = None
                self._dvi_stage0_producer = None
                self._dvi_stage0_probe = None

        writer = getattr(self, "_dvi_writer", None)
        if writer is None:
            if stage0_writer is not None:
                return {
                    "owner": False,
                    "stage0_owner": True,
                    "stage0_metrics": stage0_metrics,
                }
            return {"owner": False}

        metrics: dict[str, Any] = {}
        producer = getattr(self, "_dvi_stage2_producer", None)
        if model_runner is not None and hasattr(
            model_runner, "set_dvi_telemetry_hook"
        ):
            model_runner.set_dvi_telemetry_hook(None)
        try:
            try:
                if producer is not None:
                    producer.close()
            finally:
                metrics = writer.close()
        finally:
            self._dvi_writer = None
            self._dvi_stage2_producer = None
            self._dvi_hook = None

        return {
            "owner": True,
            **metrics,
            "stage0_owner": stage0_writer is not None,
            "stage0_metrics": stage0_metrics or None,
            "finalize_no_match_count": getattr(
                self, "_dvi_finalize_no_match_count", 0
            ),
            "finalize_ambiguous_count": getattr(
                self, "_dvi_finalize_ambiguous_count", 0
            ),
        }


class vLLMOmniColocateWorkerExtension(_OmniWorkerBase):
    """
    The class for vLLM-Omni's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    """

    def __new__(cls, **kwargs):
        assert _VLLM_OMNI_AVAILABLE, "vLLM-Omni is required to use vLLMOmniColocateWorkerExtension"
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = OmniTensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM-Omni load weights, loaded_params: {len(weights)}")
        else:
            logger.info("Loading standard weights (async)")
            self.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.
        Uses replica_rank + local_rank to form handle so it matches the sender side
        regardless of CUDA_VISIBLE_DEVICES differences, and avoids collisions
        when multiple replicas share the same node.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        return f"ipc:///tmp/rl-colocate-zmq-replica-{replica_rank}-rank-{self.local_rank}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


def extract_prompt_logprobs(output: RequestOutput, num_prompt_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None:
        return

    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls
