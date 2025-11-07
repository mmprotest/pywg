"""Generate WGSL from the simplified `KernelIR`."""

from __future__ import annotations

from typing import List

from .ir import KernelIR
from .utils import total_size


def _declare_buffers(ir: KernelIR) -> List[str]:
    lines: List[str] = []
    for index, tensor in enumerate(ir.inputs):
        lines.append(f"struct Input{index} {{ data: array<{tensor.to_wgsl()}>; }};")
        lines.append(f"@group(0) @binding({index}) var<storage, read> input{index}: Input{index};")
    lines.append(
        f"struct Output {{ data: array<{ir.output.to_wgsl()}>; }};"
    )
    lines.append(
        f"@group(0) @binding({len(ir.inputs)}) var<storage, read_write> output: Output;"
    )
    return lines


def _emit_python_comment(source: str) -> List[str]:
    lines: List[str] = ["  // Python kernel source"]
    for line in source.splitlines():
        stripped = line.rstrip()
        lines.append(f"  // {stripped}")
    return lines


def generate_wgsl(ir: KernelIR) -> str:
    """Generate WGSL compute shader for a kernel.

    The implementation intentionally keeps the shader extremely small. The actual
    numerical work happens in the CPU fallback during tests, while the generated
    shader acts as a human readable artifact and a scaffold for future
    optimisation efforts.
    """

    lines: List[str] = []
    lines.extend(_declare_buffers(ir))
    lines.append("")
    lines.append(
        "@compute @workgroup_size(" + str(ir.workgroup_size) + ") fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {"
    )
    total = total_size(ir.output.shape)
    lines.append(f"  let idx: u32 = global_id.x;")
    lines.append(f"  if (idx >= {total}u) {{ return; }}")

    if not ir.body:
        lines.append("  return;")
    else:
        op = ir.body[0].op
        if op == "map":
            lines.extend(_emit_python_comment(ir.body[0].args[0]))
            placeholder = "input0.data[idx]"
            lines.append(f"  let value = {placeholder};")
            lines.append("  output.data[idx] = value;")
        elif op == "reduce":
            lines.extend(_emit_python_comment(ir.body[0].args[0]))
            lines.append("  // Reduction kernels require multiple passes; placeholder output")
            lines.append("  if (idx == 0u) { output.data[0u] = input0.data[0u]; }")
        elif op == "scan":
            lines.extend(_emit_python_comment(ir.body[0].args[0]))
            lines.append("  // Placeholder scan implementation")
            lines.append("  output.data[idx] = input0.data[idx];")
        elif op == "matmul":
            lines.append("  // Placeholder tiled matmul implementation")
            lines.append("  output.data[idx] = 0;")
        else:
            lines.append("  // Unknown operation")
            lines.append("  return;")
    lines.append("}")
    return "\n".join(lines)
