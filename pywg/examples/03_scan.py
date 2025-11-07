from __future__ import annotations

import numpy as np

from pywg import kernel, scan


@kernel
def add(x: float, y: float) -> float:
    return x + y


if __name__ == "__main__":
    arr = np.arange(16, dtype=np.float32)
    prefix = scan(add, arr, init=0.0, inclusive=False)
    print(prefix[:8])
else:
    arr = np.arange(16, dtype=np.float32)
    scan(add, arr, init=0.0, inclusive=False)
