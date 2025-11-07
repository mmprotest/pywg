from __future__ import annotations

import numpy as np

from pywg import matmul


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.random((128, 128), dtype=np.float32)
    b = rng.random((128, 128), dtype=np.float32)
    out = matmul(a, b)
    reference = a @ b
    print(np.max(np.abs(out - reference)))
else:
    rng = np.random.default_rng(0)
    a = rng.random((32, 32), dtype=np.float32)
    b = rng.random((32, 32), dtype=np.float32)
    matmul(a, b)
