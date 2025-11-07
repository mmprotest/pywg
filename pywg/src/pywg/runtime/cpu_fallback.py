"""NumPy based fallback implementations of the pywg API."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..utils import broadcast_arrays, ensure_supported_dtype


def _prepare_out(shape: tuple[int, ...], dtype: np.dtype, out: Optional[np.ndarray]) -> np.ndarray:
    if out is None:
        return np.empty(shape, dtype=dtype)
    if out.shape != shape:
        raise ValueError(f"Output array has incorrect shape {out.shape!r}; expected {shape!r}")
    if out.dtype != dtype:
        raise ValueError(f"Output array has dtype {out.dtype!r} but expected {dtype!r}")
    return ensure_supported_dtype(out)


def cpu_map(
    fn: Callable[..., object],
    *inputs: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not inputs:
        raise ValueError("map requires at least one input array")
    broadcast = broadcast_arrays(*inputs)
    dtype = broadcast.operands[0].dtype
    result = _prepare_out(broadcast.shape, dtype, out)
    flat_result = result.reshape(-1)
    for linear_index, multi_index in enumerate(np.ndindex(broadcast.shape)):
        args = [operand[multi_index] for operand in broadcast.operands]
        try:
            value = fn(linear_index, *args)
        except TypeError:
            value = fn(*args)
        flat_result[linear_index] = value
    return result


def cpu_reduce(
    fn: Callable[[object, object], object],
    input_array: np.ndarray,
    init: object,
) -> object:
    arr = ensure_supported_dtype(np.asarray(input_array))
    accumulator = init
    for value in arr.flat:
        accumulator = fn(accumulator, value)
    return accumulator


def cpu_scan(
    fn: Callable[[object, object], object],
    input_array: np.ndarray,
    init: object,
    inclusive: bool,
) -> np.ndarray:
    arr = ensure_supported_dtype(np.asarray(input_array))
    result = np.empty_like(arr)
    accumulator = init
    if inclusive:
        for index, value in enumerate(arr.flat):
            accumulator = fn(accumulator, value)
            result.flat[index] = accumulator
    else:
        for index, value in enumerate(arr.flat):
            result.flat[index] = accumulator
            accumulator = fn(accumulator, value)
    return result.reshape(arr.shape)


def cpu_matmul(
    a: np.ndarray,
    b: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    a_arr = ensure_supported_dtype(np.asarray(a))
    b_arr = ensure_supported_dtype(np.asarray(b))
    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError("matmul expects 2-D arrays")
    if a_arr.shape[1] != b_arr.shape[0]:
        raise ValueError("Inner dimensions do not align for matmul")
    dtype = np.result_type(a_arr.dtype, b_arr.dtype)
    result = _prepare_out((a_arr.shape[0], b_arr.shape[1]), dtype, out)
    result[...] = a_arr @ b_arr
    return result
