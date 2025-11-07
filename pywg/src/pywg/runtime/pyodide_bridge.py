"""Pyodide and WebGPU integration layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import RuntimeErrorWebGPU

try:  # pragma: no cover - exercised in browser only
    from js import navigator  # type: ignore
    from pyodide.ffi import to_js  # type: ignore

    _HAS_PYODIDE = True
except Exception:  # pragma: no cover - executed on CPython
    navigator = None  # type: ignore
    to_js = None  # type: ignore
    _HAS_PYODIDE = False


@dataclass
class WebGPUBridge:
    """Placeholder WebGPU bridge used for type checking."""

    def run(self) -> None:
        raise RuntimeErrorWebGPU("WebGPU execution is only available in Pyodide")


def is_webgpu_available() -> bool:
    if not _HAS_PYODIDE:
        return False
    try:  # pragma: no cover
        return getattr(navigator, "gpu", None) is not None
    except Exception:
        return False


def _unavailable() -> RuntimeErrorWebGPU:
    return RuntimeErrorWebGPU("WebGPU runtime is not available in this environment")


def run_map(*args: object, **kwargs: object) -> np.ndarray:  # pragma: no cover - requires browser
    raise _unavailable()


def run_reduce(*args: object, **kwargs: object) -> np.ndarray:  # pragma: no cover - requires browser
    raise _unavailable()


def run_scan(*args: object, **kwargs: object) -> np.ndarray:  # pragma: no cover - requires browser
    raise _unavailable()


def run_matmul(*args: object, **kwargs: object) -> np.ndarray:  # pragma: no cover - requires browser
    raise _unavailable()
