"""Utility helpers shared across runtime backends and the compiler."""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

SUPPORTED_DTYPES = {
    np.dtype("float32"),
    np.dtype("int32"),
}


def ensure_supported_dtype(array: np.ndarray) -> np.ndarray:
    """Ensure *array* uses a supported dtype.

    A defensive copy is returned when the array is non-contiguous. This makes the
    runtime logic easier because WGSL storage buffers expect tightly packed data.
    """

    if array.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"Unsupported dtype {array.dtype!r}; supported dtypes: {sorted(SUPPORTED_DTYPES)}")
    if not array.flags.c_contiguous:
        return np.ascontiguousarray(array)
    return array


@dataclass
class BroadcastResult:
    """Description of broadcasted operands."""

    shape: Tuple[int, ...]
    operands: List[np.ndarray]


def broadcast_arrays(*arrays: np.ndarray) -> BroadcastResult:
    """Broadcast arrays using NumPy semantics and return a `BroadcastResult`."""

    if not arrays:
        raise ValueError("Expected at least one array to broadcast")
    np_arrays = [ensure_supported_dtype(np.asarray(arr)) for arr in arrays]
    broadcasted = np.broadcast_arrays(*np_arrays)
    shape = broadcasted[0].shape
    return BroadcastResult(shape=shape, operands=[np.asarray(b) for b in broadcasted])


def total_size(shape: Sequence[int]) -> int:
    return functools.reduce(operator.mul, shape, 1)


def flatten_index(index: Tuple[int, ...], shape: Sequence[int]) -> int:
    return int(np.ravel_multi_index(index, shape, mode="raise"))


def as_shape(value: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in value)
