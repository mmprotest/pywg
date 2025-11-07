from __future__ import annotations

import numpy as np

from pywg import kernel, reduce


@kernel
def add(x: float, y: float) -> float:
    return x + y


if __name__ == "__main__":
    arr = np.ones(1_048_576, dtype=np.float32)
    s = reduce(add, arr, init=0.0)
    print(float(s))
else:
    arr = np.ones(32, dtype=np.float32)
    reduce(add, arr, init=0.0)
