# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Bucketed weight transfer via ZMQ + IPC (or shared memory fallback).

Not recommended depending on vllm for this file.
"""

import gc
import logging
import os
import pickle
import socket
import threading
from multiprocessing import shared_memory
from typing import Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


class BucketedWeightSender:
    """
    Send model weights via bucketed IPC transfer over ZMQ.

    Packs weight tensors into a fixed-size communication buffer and sends them
    in buckets to the receiver. Supports CUDA IPC and shared memory fallback.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/rl-colocate-zmq-<uuid>.sock")
        bucket_size_mb: Communication buffer size in MB
        use_shm: Use shared memory instead of CUDA IPC (for NPU compatibility)
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    async def async_send_weights(self, weights):
        """
        Send weights to the receiver. Accepts a sync generator or async iterator.

        Args:
            weights: Generator or async iterator yielding (name, tensor) pairs
        """
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            # dtype = PrecisionType.to_dtype(self.config.dtype)
            async for name, weight in ensure_async_iterator(weights):
                # model parameters are in fp32 full precision
                # (vermouth1992) we should not force cast weight here because some parameters
                # (such as moe gate) have to keep fp32 precision. If a weight is bf16 in the rollout side,
                # the rollout should automatically cast on demand. However, this would incur a higher weight
                # transfer volume.
                # weight = weight.to(dtype, non_blocking=True)

                # fill the tensor bucket
                if offset + weight.nbytes > self.bucket_size:
                    get_torch_device().synchronize()
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self.socket.recv()
                    bucket_meta = {}
                    offset = 0

                # TODO: slice embedding layer weight into chunks
                assert offset + weight.nbytes <= self.bucket_size, (
                    f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                    f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                )
                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                }
                self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
                offset += weight.nbytes

            # send the last bucket
            get_torch_device().synchronize()
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self.socket.recv()
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REQ socket and bind."""
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        """build communication buffer"""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            # Create unique name for shared memory
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()


# -----------------------------------------------------------------------------
# Cross-node TCP backend (C2-2)
# -----------------------------------------------------------------------------
# This backend is used when vLLM split rollout workers live on different nodes
# than the coordinating server actor.  It sends raw tensor bytes over ZMQ TCP
# instead of CUDA-IPC / POSIX shared memory, which are node-local only.


def _get_free_tcp_port() -> int:
    """Pick a free TCP port on all interfaces."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _get_interface_ip(iface: str) -> str | None:
    """Return the IPv4 address assigned to ``iface``, or ``None``."""
    try:
        import fcntl
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr_bytes = fcntl.ioctl(
            sock.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack("256s", iface.encode("utf-8")[:15]),
        )[20:24]
        sock.close()
        return socket.inet_ntoa(addr_bytes)
    except Exception:
        return None


def _get_worker_node_ip() -> str:
    """Return an externally reachable IP for this worker.

    Mirrors the heuristic used by vLLM split endpoint discovery:
      1. ``VLLM_HOST_IP`` if explicitly configured per node.
      2. The IP of the interface named in ``NCCL_SOCKET_IFNAME`` (first
         interface if comma-separated).  This aligns ZMQ binds with the
         interface NCCL will use and avoids Docker/bridge IPs.
      3. Ray node IP address.
      4. Hostname resolution as a last resort.
    """
    import os
    import socket

    host_ip = os.environ.get("VLLM_HOST_IP")
    if host_ip:
        return host_ip

    nccl_iface = os.environ.get("NCCL_SOCKET_IFNAME", "")
    if nccl_iface:
        # NCCL supports comma-separated interface names; use the first one.
        first_iface = nccl_iface.split(",")[0].strip()
        if first_iface:
            iface_ip = _get_interface_ip(first_iface)
            if iface_ip:
                return iface_ip

    try:
        import ray

        return ray.util.get_node_ip_address()
    except Exception:
        pass

    return socket.gethostbyname(socket.gethostname())


def _format_tcp_address(ip: str, port: int) -> str:
    """Format an IPv4 or IPv6 address for ZMQ TCP connect/bind."""
    # For IPv6 ZMQ requires brackets, e.g. tcp://[::1]:port
    if "." not in ip and ":" in ip:
        return f"tcp://[{ip}]:{port}"
    return f"tcp://{ip}:{port}"


class TcpBucketedWeightSender:
    """Send bucketed weights over ZMQ TCP using raw CPU bytes.

    Unlike ``BucketedWeightSender`` this transport does not use CUDA-IPC or
    shared memory; the payload is copied to CPU and sent as multipart ZMQ
    frames, so it works across machines.  The receiver must bind first and
    advertise its ``tcp://<ip>:<port>`` endpoint.
    """

    def __init__(self, zmq_handle: str, bucket_size_mb: int = 512):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None

    async def async_send_weights(self, weights):
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()

            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}

            async for name, weight in ensure_async_iterator(weights):
                if offset + weight.nbytes > self.bucket_size:
                    await self._send_bucket(bucket_meta, is_last=False)
                    bucket_meta = {}
                    offset = 0

                assert offset + weight.nbytes <= self.bucket_size, (
                    f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                    f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                )
                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                }
                # Copy to CPU byte buffer.  For LoRA weights this is cheap enough;
                # for full-model weights a pinned/zero-copy path would be better.
                self.buffer[offset : offset + weight.nbytes].copy_(
                    weight.view(-1).view(torch.uint8).cpu(), non_blocking=False
                )
                offset += weight.nbytes

            await self._send_bucket(bucket_meta, is_last=True)
        finally:
            self._cleanup()

    def _init_socket(self):
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        """Allocate CPU buffer and perform the TCP handshake."""
        self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device="cpu")
        self.socket.send_pyobj({"transport": "tcp", "bucket_size": self.bucket_size})
        handshake = self.socket.recv_pyobj()
        assert handshake == "ok", f"Unexpected TCP receiver handshake: {handshake}"

    async def _send_bucket(self, bucket_meta: dict[str, TensorMetadata], is_last: bool):
        get_torch_device().synchronize()
        # Compute the actual byte payload for this bucket.
        total_bytes = sum(
            meta["shape"].numel() * meta["dtype"].itemsize for meta in bucket_meta.values()
        )
        payload = self.buffer[:total_bytes].numpy().tobytes()
        metadata = {"bucket_meta": bucket_meta, "is_last": is_last}
        self.socket.send_multipart([pickle.dumps(metadata), payload])
        self.socket.recv()

    def _cleanup(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        self.buffer = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()


class TcpBucketedWeightReceiver:
    """Receive bucketed weights over ZMQ TCP and reconstruct tensors on device."""

    def __init__(self, zmq_handle: str, device: torch.device):
        self.zmq_handle = zmq_handle
        self.device = device

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None

    def receive_weights(self, on_bucket_received: callable):
        try:
            self._init_socket()
            self._init_buffer()

            while True:
                frames = self.socket.recv_multipart()
                assert len(frames) == 2, f"Expected 2 frames, got {len(frames)}"
                metadata = pickle.loads(frames[0])
                payload = frames[1]

                # Copy payload into the CPU buffer so views line up with offsets.
                # ``payload`` is a read-only bytes object; wrap it in a writable
                # ``bytearray`` first so ``torch.frombuffer`` does not warn about
                # non-writable buffers.
                payload_size = len(payload)
                if payload_size > 0:
                    self.buffer[:payload_size].copy_(
                        torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                    )

                weights = []
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                    size = dtype.itemsize * shape.numel()
                    tensor = (
                        self.buffer[offset : offset + size]
                        .view(dtype=dtype)
                        .view(shape)
                        .to(self.device, non_blocking=False)
                    )
                    weights.append((name, tensor))

                on_bucket_received(weights)
                get_torch_device().synchronize()
                self.socket.send(b"")
                del weights

                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REP socket and connect to the sender's bound address."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        """Receive sender handshake and allocate a matching CPU buffer."""
        handshake = self.socket.recv_pyobj()
        assert isinstance(handshake, dict) and handshake.get("transport") == "tcp", (
            f"Unexpected TCP sender handshake: {handshake}"
        )
        bucket_size = handshake["bucket_size"]
        self.buffer = torch.empty(bucket_size, dtype=torch.uint8, device="cpu")
        self.socket.send_pyobj("ok")

    def _cleanup(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        get_torch_device().synchronize()
        self.buffer = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()


# -----------------------------------------------------------------------------
# Endpoint registry for cross-node weight sync
# -----------------------------------------------------------------------------
try:
    import ray
except ImportError:
    ray = None  # type: ignore

if ray is not None:

    @ray.remote
    class WeightEndpointRegistry:
        """Small rendezvous actor for TCP weight-sync endpoints.

        One registry is created per ``update_lora_adapter`` call.  Every vLLM
        worker registers its own ``tcp://<ip>:<port>``; the coordinator waits
        until ``world_size`` endpoints are present and then returns the list.
        """

        def __init__(self, world_size: int):
            self._world_size = world_size
            self._endpoints: set[str] = set()
            self._event = threading.Event()

        def ping(self) -> bool:
            return True

        def register(self, addr: str) -> bool:
            self._endpoints.add(addr)
            if len(self._endpoints) >= self._world_size:
                self._event.set()
            return True

        def get_endpoints(self, timeout: float = 600.0) -> list[str]:
            if not self._event.wait(timeout):
                raise TimeoutError(
                    f"WeightEndpointRegistry timed out: got {len(self._endpoints)} / "
                    f"{self._world_size} endpoints after {timeout}s"
                )
            return list(self._endpoints)


class BucketedWeightReceiver:
    """
    Receive model weights via bucketed IPC transfer over ZMQ.

    Receives weight tensors from BucketedWeightSender and passes each
    bucket to a callback for processing (e.g., loading into the model).

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: Target device for received tensors
        use_shm: Use shared memory instead of CUDA IPC
    """

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def receive_weights(self, on_bucket_received: callable):
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)]) called per bucket.
        """
        try:
            self._init_socket()
            self._init_buffer()

            # receive bucket and update weights
            while True:
                metadata = self.socket.recv_pyobj()
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                on_bucket_received(weights)
                get_torch_device().synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REP socket and connect to the sender's bound address."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        # Synchronize before releasing the buffer to ensure all async ops
        # referencing it (e.g. clone, .to()) have completed.
        get_torch_device().synchronize()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()
