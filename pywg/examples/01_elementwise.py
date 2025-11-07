from __future__ import annotations

import numpy as np

from pywg import kernel, map


@kernel
def fuse(i: int, a: float, b: float) -> float:
    """Simple fused arithmetic example."""

    return a + 2.0 * b


if __name__ == "__main__":
    N = 1_000_000
    a = np.ones(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    out = map(fuse, a, b)
    print(out[:5])
else:
    N = 16
    a = np.ones(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    map(fuse, a, b)
