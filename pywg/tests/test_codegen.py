from __future__ import annotations

import numpy as np

from pywg import kernel
from pywg.codegen_wgsl import generate_wgsl
from pywg.dsl import build_map_ir


@kernel
def _double(i: int, value: float) -> float:
    return value * 2.0


def test_generate_wgsl_contains_bindings() -> None:
    arr = np.ones((4,), dtype=np.float32)
    ir = build_map_ir(_double, [arr], arr)
    shader = generate_wgsl(ir)
    assert "@group(0) @binding(0)" in shader
    assert "@compute @workgroup_size(" in shader
    assert "Python kernel source" in shader
