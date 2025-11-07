from __future__ import annotations

import numpy as np

from pywg import kernel, map, matmul, reduce, scan


@kernel
def _add_index(i: int, a: float, b: float) -> float:
    return a + b + float(i)


def test_map_broadcasting() -> None:
    a = np.ones((4,), dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    result = map(_add_index, a, b)
    expected = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    assert np.allclose(result, expected)


@kernel
def _add(x: float, y: float) -> float:
    return x + y


def test_reduce_sum() -> None:
    arr = np.ones((128,), dtype=np.float32)
    total = reduce(_add, arr, init=0.0)
    assert float(total) == 128.0


def test_scan_exclusive() -> None:
    arr = np.arange(5, dtype=np.float32)
    result = scan(_add, arr, init=0.0, inclusive=False)
    expected = np.array([0.0, 0.0 + 0.0, 1.0, 3.0, 6.0], dtype=np.float32)
    assert np.allclose(result, expected)


def test_matmul_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((16, 8), dtype=np.float32)
    b = rng.random((8, 4), dtype=np.float32)
    expected = a @ b
    result = matmul(a, b)
    assert np.allclose(result, expected, atol=1e-5)
