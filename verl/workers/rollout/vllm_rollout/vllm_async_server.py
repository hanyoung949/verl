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
import argparse
import asyncio
import inspect
import json
import logging
import os
import uuid
from pprint import pprint
from typing import Any, Callable, Optional

import ray
import torch
import vllm.entrypoints.cli.serve
from packaging import version
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import run_headless
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_resource_name, get_visible_devices_keyword, is_torch_npu_available
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler import DistProfiler, build_vllm_profiler_args
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings, qwen2_5_vl_dedup_image_tokens, run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    SuppressSignalInThread,
    build_cli_args_from_config,
    extract_prompt_logprobs,
    get_vllm_max_lora_rank,
)

_VLLM_VERSION = version.parse(vllm.__version__)

# Treat upstream dev builds (e.g. 0.1.devXXXX+gHASH) as a recent vLLM version.
if _VLLM_VERSION.is_devrelease:
    _VLLM_VERSION = version.parse("0.13.0")

if _VLLM_VERSION > version.parse("0.11.0"):
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if _VLLM_VERSION == version.parse("0.12.0"):
        from vllm.entrypoints.harmony_utils import get_encoding

    elif _VLLM_VERSION >= version.parse("0.13.0"):
        from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

    else:
        get_encoding = None

    if get_encoding is not None and os.getenv("VERL_USE_GPT_OSS", "0") == "1":
        get_encoding()
else:
    from vllm.utils import FlexibleArgumentParser


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMHttpServer:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
        placement_group: Any = None,
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
            cuda_visible_devices (str): cuda visible devices.
        """
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices
        os.environ["VERL_REPLICA_RANK"] = str(replica_rank)

        self.config = self._init_config(config)
        self.model_config = self._init_model_config(model_config)
        self._validate_configs()

        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self.placement_group = placement_group
        # model weights version, set by ServerAdapter when update weights.
        self.global_steps = None

        if self.placement_group is not None:
            world_size = (
                self.config.tensor_model_parallel_size
                * self.config.pipeline_model_parallel_size
                * self.config.data_parallel_size
            )
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(str(i) for i in range(world_size))

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling vllm server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            # used for torch.distributed.init_process_group
            self._master_port, self._master_sock = get_free_port(self._server_address, with_alive_sock=True)
            # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
            self._dp_rpc_port, self._dp_rpc_sock = get_free_port(self._server_address, with_alive_sock=True)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address, with_alive_sock=True)
        else:
            self._master_address = None
            self._master_port = None
            self._dp_rpc_port = None
            self._dp_master_port = None

        self._post_init(cuda_visible_devices)

    def get_master_address(self):
        """Get master address and port for data parallel.
        Returns:
            tuple: (master_address, master_port, dp_rpc_port)
        """
        return self._master_address, self._master_port, self._dp_rpc_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    @property
    def lora_as_adapter(self) -> bool:
        return (
            self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
        ) and not self.model_config.lora.get("merge", False)

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port and dp_rpc_port, (
                "non-master node should provide master_address, master_port and dp_rpc_port"
            )
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get(self._get_engine_kwargs_key(), {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

        # Promote first-class layer-wise split config into vllm engine_kwargs.
        # engine_kwargs is still accepted for backward compatibility, but
        # RolloutConfig-level fields take precedence and conflicts are caught
        # in RolloutConfig.__post_init__.
        if self.config.name == "vllm" and self.config.get("enable_layerwise_split", False):
            engine_kwargs["enable_layerwise_split"] = True
            if self.config.get("split_stage_0_size", 0) > 0:
                engine_kwargs["split_stage_0_size"] = self.config.split_stage_0_size
            if self.config.get("split_stage_2_size", 0) > 0:
                engine_kwargs["split_stage_2_size"] = self.config.split_stage_2_size
            if self.config.get("split_stage_1_tensor_parallel_size", 1) > 1:
                engine_kwargs["split_stage_1_tensor_parallel_size"] = self.config.split_stage_1_tensor_parallel_size

        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        self._preprocess_engine_kwargs(engine_kwargs)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = self._get_override_generation_config()
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        quantization, hf_overrides = self._apply_quantization()

        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")

        # FULL cuda graph is not yet supported with DCP, downgrade to PIECEWISE
        dcp_size = engine_kwargs.get("decode_context_parallel_size", 1) or 1
        if dcp_size > 1 and compilation_config["cudagraph_mode"] == "FULL_AND_PIECEWISE":
            logger.warning(
                "FULL cuda graph is not supported with DCP (decode_context_parallel_size=%d), "
                "downgrading cudagraph_mode to PIECEWISE.",
                dcp_size,
            )
            compilation_config["cudagraph_mode"] = "PIECEWISE"

        compilation_config = json.dumps(compilation_config)
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "ray" if self.nnodes > 1 else "mp",
            "worker_extension_cls": self._get_worker_extension_cls(),
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_model_parallel_size,
            "seed": self.replica_rank + self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": compilation_config,
            **engine_kwargs,
        }

        # update profiler args
        profiler_args = build_vllm_profiler_args(
            self.profiler_controller.config, self.profiler_controller.tool_config, self.replica_rank
        )
        if _VLLM_VERSION >= version.parse("0.13.0"):
            # vLLM >= 0.13.0 supports profiler config via CLI args; env vars still work but will be deprecated
            args.update(profiler_args)

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        # mtp (None for diffusion models; only LLM models use speculative decoding)
        if self.config.mtp is not None and self.config.mtp.enable and self.config.mtp.enable_rollout:
            speculative_config = {
                "method": self.config.mtp.method,
                "num_speculative_tokens": self.config.mtp.num_speculative_tokens,
            }
            args["speculative_config"] = speculative_config

        if self.config.data_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to "
                f"dp_size_local ({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"
            )
            dp_args = {
                "data_parallel_size": self.config.data_parallel_size,
                "data_parallel_size_local": data_parallel_size_local,
                "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                "data_parallel_address": self._master_address,
                "data_parallel_rpc_port": self._dp_rpc_port,
            }
            args.update(dp_args)

        args.update({"enable_expert_parallel": self.config.expert_parallel_size > 1})

        # used for torch.distributed.init_process_group
        # Layer-wise split with a non-uniform stage-node map is orchestrated by
        # vLLM's RayExecutorV2 using the externally supplied placement group, so
        # do not pass the uniform multi-node args that assume nnodes divides
        # world_size.
        is_split_with_map = (
            getattr(self.config, "enable_layerwise_split", False)
            and getattr(self.config, "split_stage_node_map", None) is not None
        )
        if self.nnodes > 1 and not is_split_with_map:
            args.update(
                {
                    "master_addr": self._master_address,
                    "master_port": self._master_port,
                    "node_rank": self.node_rank,
                    "nnodes": self.nnodes,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # update lora-related args
        lora_rank = self.model_config.lora.get("rank", 0)
        if lora_rank <= 0:
            lora_rank = (
                self.model_config.lora_rank
            )  # FIXME: fallback to lora_rank for now, we should unify lora settings.

        if self.model_config.lora.get("merge", False):
            lora_rank = 0

        if lora_rank > 0:
            lora_args = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(lora_rank),
            }
            if self.model_config.lora.get("fully_sharded_loras", False):
                lora_args["fully_sharded_loras"] = True
            args.update(lora_args)

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        server_args = ["serve", self.model_config.local_path] + build_cli_args_from_config(args)

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = self._get_cli_modules()
        parser = FlexibleArgumentParser(description=self._get_cli_description())
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args)
        else:
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        if self.placement_group is not None:
            vllm_config.parallel_config.placement_group = self.placement_group
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()
        await engine_client.collective_rpc(
            method="monkey_patch_model", kwargs={"vocab_size": len(self.model_config.tokenizer)}
        )

        build_app_sig = inspect.signature(build_app)
        supported_tasks: tuple[Any, ...] = ()
        if "supported_tasks" in build_app_sig.parameters:
            supported_tasks = await engine_client.get_supported_tasks()
            app = build_app(args, supported_tasks)
        else:
            app = build_app(args)

        init_app_sig = inspect.signature(init_app_state)
        if "vllm_config" in init_app_sig.parameters:
            await init_app_state(engine_client, vllm_config, app.state, args)
        elif "supported_tasks" in init_app_sig.parameters:
            await init_app_state(engine_client, app.state, args, supported_tasks)
        else:
            await init_app_state(engine_client, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        args.api_server_count = 0

        def run_headless_wrapper():
            with SuppressSignalInThread():
                run_headless(args)

        def on_run_headless_done(future: asyncio.Future):
            try:
                exc = future.exception()
                if exc:
                    logger.exception(f"run_headless failed with exception: {exc}")
                else:
                    logger.warning("run_headless completed successfully, but it's not expected.")
            except Exception as e:
                logger.exception(f"get result from run_headless failed: {e}")
            finally:
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run_headless_wrapper))
        self.task.add_done_callback(on_run_headless_done)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
        lora_request: Optional[LoRARequest] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        prompt_ids = normalize_token_ids(prompt_ids)

        # Calculate the maximum possible new tokens based on available context space
        # This serves as a safety upper bound
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        # Determine max_tokens from sampling_params or use configured response_length as default
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            # support sglang-style 'max_new_tokens' param
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # Default to a calculation that considers configured lengths
            # Cap max_tokens by response_length to ensure tensor alignment,
            # and by remaining budget to prevent OOM in multi-turn rollouts.
            max_tokens = min(
                self.config.response_length, self.config.prompt_length + self.config.response_length - len(prompt_ids)
            )

        # Clamp max_tokens to the valid range [0, max_possible_tokens]
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        assert max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        # Add lora request
        if lora_request is None and self.lora_as_adapter:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        extra_fields = {"global_steps": self.global_steps}
        extract_prompt_logprobs(
            output=final_res,
            num_prompt_logprobs=sampling_params.prompt_logprobs,
            result_dict=extra_fields,
        )
        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts

        # Determine stop reason from finish_reason
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        num_preempted = None

        if hasattr(final_res.outputs[0], "num_preempted"):
            num_preempted = final_res.outputs[0].num_preempted

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def wake_up(self):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            await self.engine.wake_up(tags=self._get_wake_up_tags())
            await self.engine.reset_prefix_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a LoRA adapter into the running vLLM engine."""
        if self.node_rank != 0:
            return True
        return await self.engine.add_lora(lora_request)

    async def update_lora_adapter(
        self,
        lora_state_dict: dict[str, torch.Tensor],
        peft_config: dict,
    ) -> bool:
        """Load a LoRA adapter from an in-memory state dict into the running engine.

        This is used by split training to push head/tail LoRA weights directly to
        the rollout engine without writing an adapter directory to disk.  We reuse
        the existing bucketed weight transfer path (:meth:`update_weights_from_ipc`)
        so tensors stay as ``torch.Tensor`` on the worker side instead of being
        serialized to Python lists by the engine RPC layer.

        For single-node deployments the legacy IPC/SHM backend is used.  When the
        rollout spans multiple nodes, a TCP raw-bytes backend plus a per-sync Ray
        endpoint registry discovers every worker's address before sending.
        """
        if self.node_rank != 0:
            return True

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightSender,
            TcpBucketedWeightSender,
            WeightEndpointRegistry,
        )

        use_shm = self.config.get("use_shm_weight_update", False)
        bucket_size_mb = self.config.get("update_weights_bucket_megabytes", 512)
        use_tcp = (
            self.nnodes > 1
            or os.environ.get("VERL_SPLIT_WEIGHT_SYNC_TCP", "0") == "1"
        )

        if use_tcp:
            # Cross-node path: bind one TCP endpoint per vLLM worker, pass the
            # connect addresses to the workers, and send the same LoRA weights to
            # all of them concurrently.  This avoids needing a shared registry
            # actor inside the vLLM subprocesses.
            from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
                _format_tcp_address,
                _get_free_tcp_port,
                _get_worker_node_ip,
            )

            world_size = len(self.workers)
            server_ip = _get_worker_node_ip()
            senders: list[TcpBucketedWeightSender] = []
            connect_addrs: list[str] = []
            for _ in range(world_size):
                port = _get_free_tcp_port()
                bind_addr = f"tcp://*:{port}"
                connect_addr = _format_tcp_address(server_ip, port)
                senders.append(
                    TcpBucketedWeightSender(
                        zmq_handle=bind_addr,
                        bucket_size_mb=bucket_size_mb,
                    )
                )
                connect_addrs.append(connect_addr)

            logger.info(
                "update_lora_adapter (TCP): bound %d endpoints on %s",
                world_size,
                server_ip,
            )

            receive_task = asyncio.create_task(
                self.engine.collective_rpc(
                    "update_weights_from_ipc",
                    kwargs={
                        "peft_config": peft_config,
                        "base_sync_done": True,
                        "use_shm": False,
                        "weight_sync_addrs": connect_addrs,
                    },
                )
            )

            try:
                async def _send_with_sender(sender: TcpBucketedWeightSender):
                    async def _weights():
                        for name, tensor in lora_state_dict.items():
                            yield name, tensor

                    await sender.async_send_weights(_weights())

                await asyncio.gather(*(_send_with_sender(s) for s in senders))
                await receive_task
                return True
            except Exception:
                logger.exception("update_lora_adapter (TCP) failed")
                receive_task.cancel()
                return False

        # Single-node IPC/SHM path (default, backwards compatible).
        # Number of local vLLM workers that will receive weights.  This equals
        # the number of GPUs managed by this server actor (TP * local PP).  The
        # receiver socket is indexed by the worker's local rank, so we send to
        # every local rank in [0, num_local_workers).
        num_local_workers = self.gpus_per_node

        receive_task = asyncio.create_task(
            self.engine.collective_rpc(
                "update_weights_from_ipc",
                kwargs={
                    "peft_config": peft_config,
                    "base_sync_done": True,
                    "use_shm": use_shm,
                },
            )
        )

        async def _send_to_rank(local_rank: int):
            zmq_handle = (
                f"ipc:///tmp/rl-colocate-zmq-replica-{self.replica_rank}"
                f"-rank-{local_rank}.sock"
            )
            sender = BucketedWeightSender(
                zmq_handle=zmq_handle,
                bucket_size_mb=bucket_size_mb,
                use_shm=use_shm,
            )

            async def _weights():
                for name, tensor in lora_state_dict.items():
                    yield name, tensor

            await sender.async_send_weights(_weights())

        try:
            await asyncio.gather(
                *(_send_to_rank(r) for r in range(num_local_workers))
            )
            await receive_task
            return True
        except Exception:
            logger.exception("update_lora_adapter failed")
            receive_task.cancel()
            return False

    async def remove_lora(self, lora_int_id: int) -> bool:
        """Remove a loaded LoRA adapter from the running vLLM engine."""
        if self.node_rank != 0:
            return True
        return await self.engine.remove_lora(lora_int_id)

    async def list_loras(self) -> set[int]:
        """List LoRA adapter ids currently loaded in the engine."""
        if self.node_rank != 0:
            return set()
        return await self.engine.list_loras()

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            await self._sleep_hybrid()
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def start_profile(self, **kwargs):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.engine.start_profile(**kwargs)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.engine.stop_profile()

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self.engine.reset_prefix_cache()

    async def set_global_steps(self, global_steps: int):
        """Set the global steps of the model weights."""
        self.global_steps = global_steps

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all ongoing generation requests.

        On vLLM >= 0.12.0, uses AsyncLLM.pause_generation() to abort in-flight
        requests, drain, and clear caches. The engine remains paused after this
        call — use resume_generation() to accept new requests (e.g. before
        validation).

        On vLLM < 0.12.0, manually aborts each request and resets prefix cache.

        Returns:
            dict[str, Any]: Dictionary containing:
                - aborted_count: Number of requests aborted
                - request_ids: List of aborted request IDs
        """
        try:
            if _VLLM_VERSION >= version.parse("0.12.0"):
                # Snapshot request IDs before pausing for reporting
                request_ids = list(self.engine.output_processor.request_states.keys())

                # pause_generation with wait_for_inflight_requests=False will:
                # 1. Set engine to paused state (blocks new generate calls)
                # 2. Abort all in-flight requests
                # 3. Wait for requests to drain
                # 4. Clear prefix and mm caches if clear_cache=True
                await self.engine.pause_generation(
                    wait_for_inflight_requests=False,
                    clear_cache=reset_prefix_cache,
                )
            else:
                # Take an atomic snapshot to avoid race conditions with the vLLM engine thread
                request_states_snapshot = list(self.engine.output_processor.request_states.items())
                request_ids = [req_id for req_id, _ in request_states_snapshot]

                if not request_ids:
                    return {"aborted_count": 0, "request_ids": []}

                # For each request, create an abort output and put it to its queue
                # This allows the generator to receive the aborted result
                from vllm.v1.engine import FinishReason

                for _, req_state in request_states_snapshot:
                    request_output = req_state.make_request_output(
                        [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
                    )
                    req_state.queue.put(request_output)

                # Abort requests in the output processor and engine core
                self.engine.output_processor.abort_requests(request_ids)
                await self.engine.engine_core.abort_requests_async(request_ids)

                # Try to reset prefix cache to ensure clean state
                if reset_prefix_cache:
                    await self.clear_kv_cache()
                    logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def resume_generation(self):
        """Resume generation after abort_all_requests (pause_generation).

        Only effective on vLLM >= 0.12.0 where pause_generation is used.
        No-op on older versions.
        """
        if self.node_rank != 0:
            return
        if _VLLM_VERSION >= version.parse("0.12.0"):
            await self.engine.resume_generation()

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}

    # -----------------------------------------------------------------------
    # Hook methods for subclass overrides
    # -----------------------------------------------------------------------

    def _init_config(self, config):
        """Initialise config. Override when a specific dataclass_type is needed."""
        return omega_conf_to_dataclass(config)

    def _init_model_config(self, model_config):
        """Initialise model_config. Override when a specific dataclass_type is needed."""
        return omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)

    def _validate_configs(self) -> None:
        """Validate config/model_config after initialisation."""
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        else:
            if self.config.max_model_len > max_position_embeddings:
                raise ValueError(
                    f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                    f"max_position_embeddings ({max_position_embeddings})"
                )

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Called at the end of __init__. Default logs server metadata."""
        logger.info(
            f"{self.__class__.__name__}, replica_rank: {self.replica_rank}, node_rank: {self.node_rank}, "
            f"{get_visible_devices_keyword()}: {cuda_visible_devices}, "
            f"master_address: {self._master_address}, master_port: {self._master_port}, "
            f"data_parallel_rpc_port: {self._dp_rpc_port}, data_parallel_master_port: {self._dp_master_port}"
        )

    def _get_engine_kwargs_key(self) -> str:
        """Return the key under config.engine_kwargs for this engine (e.g. 'vllm')."""
        return "vllm"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        """Mutate engine_kwargs in-place before the CLI args dict is built. No-op by default."""
        pass

    def _get_override_generation_config(self) -> dict:
        """Return the override_generation_config dict."""
        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        return dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )

    def _apply_quantization(self) -> tuple[Optional[str], dict]:
        """Process quantization config. Returns (quantization_str, hf_overrides)."""
        quantization = self.config.quantization
        hf_overrides = {}

        if is_torch_npu_available(check_device=False):
            from verl.utils.vllm.npu_vllm_patch import check_vllm_ascend_before_server_launch

            check_vllm_ascend_before_server_launch()

        # Handle QAT (Quantization-Aware Training) configuration
        qat_config_dict = getattr(self.config, "qat", {}) or {}
        if qat_config_dict.get("enable", False):
            from verl.utils.qat import QATConfig, load_quantization_config

            qat_config = QATConfig(**qat_config_dict)
            quantization_config_dict = load_quantization_config(qat_config)
            quant_method = quantization_config_dict.get("quant_method", None)

            if quant_method == "modelopt":
                from verl.utils.modelopt import apply_modelopt_nvfp4_patches

                apply_modelopt_nvfp4_patches()
                quantization = "modelopt"
            elif quant_method == "compressed-tensors":
                from verl.utils.qat import apply_qat_patches

                apply_qat_patches()
                quantization = "compressed-tensors"
            else:
                raise ValueError(f"Unsupported quant_method: {quant_method}")

            logger.info(f"QAT quantization config injected (quant_method={quant_method})")
            hf_overrides["quantization_config"] = quantization_config_dict
        elif quantization is not None:
            # Handle other quantization methods (fp8, torchao)
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao", "ascend"]
            if quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {quantization}")

            if quantization == "fp8":
                # Ignore MoE router layers for FP8 quantization
                all_mlp_gate_layers = []
                for layer in range(self.model_config.hf_config.num_hidden_layers):
                    all_mlp_gate_layers.append(f"model.layers.{layer}.mlp.gate")

                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                    "ignored_layers": all_mlp_gate_layers,
                }
                hf_overrides["quantization_config"] = dict(FP8_BLOCK_QUANT_KWARGS)
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()
                # for subprocesses patching
                os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"

        if quantization is not None and self.config.quantization_config_file is not None:
            hf_overrides["quantization_config_file"] = self.config.quantization_config_file

        return quantization, hf_overrides

    def _get_worker_extension_cls(self) -> str:
        """Return the fully-qualified colocate worker extension class name."""
        return "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        """Return the list of CLI command modules used for argument parsing."""
        return [vllm.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        """Return the description string for the CLI argument parser."""
        return "vLLM CLI"

    def _get_wake_up_tags(self) -> list[str]:
        """Return the tags passed to engine.wake_up(). Default includes kv_cache."""
        return ["kv_cache", "weights"]

    async def _sleep_hybrid(self):
        """HYBRID sleep: lora adapters only need level=1; full weights need level=2."""
        # Don't use engine.sleep(level=2) here
        # lora only update adapter weights, so set sleep level to 1
        if self.lora_as_adapter:
            sleep_level = 1
        else:
            sleep_level = 2
        await self.engine.collective_rpc("sleep", kwargs={"level": sleep_level})
        if _VLLM_VERSION >= version.parse("0.17.0"):
            await self.engine.reset_encoder_cache()


class vLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = ray.remote(vLLMHttpServer)

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        self._validate_launch_requirements()

        if self._is_split_with_map:
            await self._launch_split_servers()
            return

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            prefix = self._get_server_name_prefix()
            if self.is_reward_model:
                name = f"{prefix}server_reward_{self.replica_rank}_{node_rank}{self.name_suffix}"
            elif self.is_teacher_model:
                name = f"{prefix}server_teacher_{self.replica_rank}_{node_rank}{self.name_suffix}"
            else:
                name = f"{prefix}server_{self.replica_rank}_{node_rank}{self.name_suffix}"
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                        # To prevent hanging or crash during synchronization of weights between actor and rollout
                        # in disaggregated mode. See:
                        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
                        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
                        "NCCL_CUMEM_ENABLE": "0",
                        # Propagate split-related env vars from driver to vLLM server actors.
                        # VLLM_USE_V2_MODEL_RUNNER=0 is required for the current layer-wise split impl.
                        **{k: v for k, v in os.environ.items() if k.startswith("VLLM_SPLIT_") or k == "VLLM_USE_V2_MODEL_RUNNER"},
                        # NCCL network env vars may also be required for cross-node split.
                        **{k: os.environ[k] for k in [
                            "VLLM_HOST_IP",
                            "NCCL_SOCKET_IFNAME",
                            "NCCL_IB_DISABLE",
                            "NCCL_NET",
                            "NCCL_P2P_DISABLE",
                        ] if k in os.environ},
                    }
                },
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def _launch_split_servers(self):
        """Launch a single vLLM frontend for cross-node layer-wise split.

        The engine workers are managed by vLLM's RayExecutorV2 inside the shared
        cross-node placement group created by ``RolloutReplica.init_standalone``.
        Only the master node (where rank 0 / stage_0 is placed) runs the HTTP
        frontend; remote stages are pure Ray worker actors.
        """
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        worker_infos = self._split_worker_infos
        pg = self._split_placement_group

        master_node_id = worker_infos[0][0]
        master_gpu_ids = [info[1] for info in worker_infos]
        # Master node may hold multiple ranks (stage_0 and stage_2 for TP=1).
        # The vLLMHttpServer actor itself does not use local GPUs, but we keep
        # the visible-devices string consistent with the legacy path.
        master_cuda_visible_devices = ",".join(
            info[1] for info in worker_infos if info[0] == master_node_id
        )

        logger.info(
            "Launching cross-node split vLLM server on master node %s with CUDA_VISIBLE_DEVICES=%s",
            master_node_id,
            master_cuda_visible_devices,
        )

        prefix = self._get_server_name_prefix()
        if self.is_reward_model:
            name = f"{prefix}server_reward_{self.replica_rank}_0{self.name_suffix}"
        elif self.is_teacher_model:
            name = f"{prefix}server_teacher_{self.replica_rank}_0{self.name_suffix}"
        else:
            name = f"{prefix}server_{self.replica_rank}_0{self.name_suffix}"

        bundle_indices = ",".join(str(i) for i in range(self.world_size))
        server = self.server_class.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=master_node_id,
                soft=False,
            ),
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "NCCL_CUMEM_ENABLE": "0",
                    "VLLM_RAY_BUNDLE_INDICES": bundle_indices,
                    **{k: v for k, v in os.environ.items() if k.startswith("VLLM_SPLIT_") or k == "VLLM_USE_V2_MODEL_RUNNER"},
                    # Do NOT propagate VLLM_HOST_IP: split workers must bind
                    # to their own node's IP.  NCCL_SOCKET_IFNAME is enough to
                    # keep the control plane on the right interface.
                    **{k: os.environ[k] for k in [
                        "NCCL_SOCKET_IFNAME",
                        "NCCL_IB_DISABLE",
                        "NCCL_NET",
                        "NCCL_P2P_DISABLE",
                    ] if k in os.environ},
                }
            },
            name=name,
            max_concurrency=self.max_concurrency,
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=[None] * self.world_size,
            replica_rank=self.replica_rank,
            node_rank=0,
            gpus_per_node=len(master_gpu_ids),
            nnodes=self.nnodes,
            cuda_visible_devices=master_cuda_visible_devices,
            placement_group=pg,
        )
        self.servers.append(server)

        await server.launch_server.remote()
        server_address, server_port = await server.get_server_address.remote()
        self._server_handle = server
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        """Abort all ongoing generation requests across all servers.

        Returns:
            dict[str, Any]: Combined abort results from all servers.
        """
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def resume_generation(self):
        """Resume generation on all servers after abort_all_requests."""
        await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])

    async def update_lora_adapter(
        self,
        lora_state_dict: dict[str, Any],
        peft_config: dict,
    ) -> bool:
        """Push an in-memory LoRA adapter to every server in this replica."""
        results = await asyncio.gather(
            *[server.update_lora_adapter.remote(lora_state_dict, peft_config) for server in self.servers]
        )
        return all(results)

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Tries all servers since we don't know which one has it.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Abort result.
        """
        # TODO(petersh6): we should only abort on the server that has the request.
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}

    # -----------------------------------------------------------------------
    # Hook methods for subclass overrides
    # -----------------------------------------------------------------------

    def _validate_launch_requirements(self) -> None:
        """Validate requirements before launching. Override in subclasses."""
        # NOTE: We always use MP Executor backend whether it's single-node or multi-node.
        # For multi-node without DP (e.g TP=16), need vllm>=0.11.1, https://github.com/vllm-project/vllm/pull/23691
        if self.config.data_parallel_size == 1 and self.nnodes > 1:
            assert _VLLM_VERSION >= version.parse("0.11.1"), (
                "For multi-node MP Executor, either (1) set data_parallel_size > 1 or (2) upgrade vLLM to >= 0.11.1"
            )

    def _get_server_name_prefix(self) -> str:
        """Return the Ray actor name prefix (e.g. 'vllm_' or 'vllm_omni_')."""
        return "vllm_"
