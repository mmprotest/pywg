from __future__ import annotations

import numpy as np
import pytest

from pywg.runtime.cpu_fallback import cpu_map, cpu_matmul, cpu_reduce, cpu_scan


def test_cpu_map_non_contiguous() -> None:
    arr = np.arange(9, dtype=np.float32).reshape(3, 3)[:, ::2]
    result = cpu_map(lambda i, x: x + 1, arr)
    expected = arr + 1
    assert np.allclose(result, expected)


def test_cpu_reduce_empty() -> None:
    @np.vectorize
    def add(x: float, y: float) -> float:
        return x + y

    arr = np.zeros((0,), dtype=np.float32)
    total = cpu_reduce(lambda x, y: x + y, arr, 0.0)
    assert total == 0.0


def test_cpu_scan_inclusive() -> None:
    arr = np.arange(4, dtype=np.float32)
    result = cpu_scan(lambda x, y: x + y, arr, 0.0, inclusive=True)
    expected = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
    assert np.allclose(result, expected)


def test_cpu_matmul_shape_mismatch() -> None:
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        cpu_matmul(a, b)
