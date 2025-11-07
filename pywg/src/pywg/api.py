"""Public API for pywg."""

from __future__ import annotations

from typing import Callable, Optional, TypedDict

import numpy as np

from . import codegen_wgsl, dsl
from .runtime import cpu_map, cpu_matmul, cpu_reduce, cpu_scan
from .runtime.pyodide_bridge import is_webgpu_available as _bridge_is_webgpu_available
from .utils import BroadcastResult, broadcast_arrays, ensure_supported_dtype


class DeviceInfo(TypedDict):
    backend: str
    adapter_name: str
    device_name: str
    limits: dict[str, object]


KernelFunction = Callable[..., object]


def kernel(fn: KernelFunction) -> KernelFunction:
    return dsl.kernel(fn)


def _compute_wgsl_map(fn: KernelFunction, broadcast: BroadcastResult, out: np.ndarray) -> str:
    ir = dsl.build_map_ir(fn, broadcast.operands, out)
    return codegen_wgsl.generate_wgsl(ir)


def _compute_wgsl_reduce(fn: KernelFunction, input_array: np.ndarray, out: np.ndarray) -> str:
    ir = dsl.build_reduce_ir(fn, input_array, out)
    return codegen_wgsl.generate_wgsl(ir)


def _compute_wgsl_scan(fn: KernelFunction, input_array: np.ndarray, out: np.ndarray) -> str:
    ir = dsl.build_scan_ir(fn, input_array, out)
    return codegen_wgsl.generate_wgsl(ir)


def _compute_wgsl_matmul(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> str:
    ir = dsl.build_matmul_ir(a, b, out)
    return codegen_wgsl.generate_wgsl(ir)


def map(fn: KernelFunction, *inputs: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    if not inputs:
        raise ValueError("map requires at least one input array")
    arrays = [ensure_supported_dtype(np.asarray(arr)) for arr in inputs]
    broadcast = broadcast_arrays(*arrays)
    dtype = broadcast.operands[0].dtype
    if out is None:
        out_arr = np.empty(broadcast.shape, dtype=dtype)
    else:
        out_arr = ensure_supported_dtype(out)
        if out_arr.shape != broadcast.shape:
            raise ValueError("Output shape mismatch for map")
    _compute_wgsl_map(fn, broadcast, out_arr)
    return cpu_map(fn, *broadcast.operands, out=out_arr)


def reduce(
    fn: KernelFunction,
    input: np.ndarray,
    init: float | int | bool = 0,
) -> np.generic | float | int | bool:
    arr = ensure_supported_dtype(np.asarray(input))
    out = np.empty(1, dtype=arr.dtype)
    _compute_wgsl_reduce(fn, arr, out)
    return cpu_reduce(fn, arr, init)


def scan(
    fn: KernelFunction,
    input: np.ndarray,
    init: float | int | bool = 0,
    inclusive: bool = False,
) -> np.ndarray:
    arr = ensure_supported_dtype(np.asarray(input))
    out = np.empty_like(arr)
    _compute_wgsl_scan(fn, arr, out)
    return cpu_scan(fn, arr, init, inclusive)


def matmul(
    A: np.ndarray,
    B: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    a_arr = ensure_supported_dtype(np.asarray(A))
    b_arr = ensure_supported_dtype(np.asarray(B))
    result_shape = (a_arr.shape[0], b_arr.shape[1])
    dtype = np.result_type(a_arr.dtype, b_arr.dtype)
    if out is None:
        out_arr = np.empty(result_shape, dtype=dtype)
    else:
        out_arr = ensure_supported_dtype(out)
        if out_arr.shape != result_shape:
            raise ValueError("Output shape mismatch for matmul")
    _compute_wgsl_matmul(a_arr, b_arr, out_arr)
    return cpu_matmul(a_arr, b_arr, out=out_arr)


def get_device_info() -> DeviceInfo:
    if _bridge_is_webgpu_available():  # pragma: no cover - requires browser
        return DeviceInfo(
            backend="webgpu",
            adapter_name="webgpu-adapter",
            device_name="webgpu-device",
            limits={"maxWorkgroupSize": 256},
        )
    return DeviceInfo(
        backend="cpu",
        adapter_name="numpy",
        device_name="numpy",
        limits={"maxWorkgroupSize": 256},
    )


def is_webgpu_available() -> bool:
    return _bridge_is_webgpu_available()
