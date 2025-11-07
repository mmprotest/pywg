"""Lightweight type system used by the pywg IR and code generation steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .utils import as_shape


@dataclass(frozen=True)
class ScalarType:
    name: str

    def to_wgsl(self) -> str:
        return self.name


@dataclass(frozen=True)
class TensorType:
    dtype: ScalarType
    shape: Tuple[int, ...]

    def rank(self) -> int:
        return len(self.shape)

    def to_wgsl(self) -> str:
        return self.dtype.to_wgsl()


F32 = ScalarType("f32")
I32 = ScalarType("i32")
U32 = ScalarType("u32")
BOOL = ScalarType("bool")

DTYPE_TO_SCALAR = {
    np.dtype("float32"): F32,
    np.dtype("int32"): I32,
}


def scalar_from_numpy(dtype: np.dtype) -> ScalarType:
    try:
        return DTYPE_TO_SCALAR[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype for scalar conversion: {dtype!r}") from exc


def tensor_from_numpy(array: np.ndarray) -> TensorType:
    return TensorType(dtype=scalar_from_numpy(array.dtype), shape=as_shape(array.shape))
