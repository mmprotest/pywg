"""High level DSL helper utilities."""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .errors import CompilationError
from .ir import IRInstruction, KernelIR
from .type_system import tensor_from_numpy


@dataclass
class KernelDefinition:
    fn: Callable[..., object]
    source: str


def _strip_decorators(source: str) -> str:
    lines = source.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    return "\n".join(lines)


def kernel(fn: Callable[..., object]) -> Callable[..., object]:
    """Decorator that marks a Python function as kernel compatible."""

    try:
        source = inspect.getsource(fn)
    except OSError as exc:  # pragma: no cover - extremely rare
        raise CompilationError("Unable to retrieve source for kernel", source=None) from exc
    source = textwrap.dedent(source)
    stripped = _strip_decorators(source)
    definition = KernelDefinition(fn=fn, source=stripped)
    setattr(fn, "__pywg_kernel__", definition)
    return fn


def get_kernel_definition(fn: Callable[..., object]) -> KernelDefinition:
    definition = getattr(fn, "__pywg_kernel__", None)
    if definition is None:
        raise CompilationError("Function is not marked with @kernel")
    return definition


def build_map_ir(fn: Callable[..., object], inputs: Iterable[np.ndarray], output: np.ndarray) -> KernelIR:
    definition = get_kernel_definition(fn)
    input_types = [tensor_from_numpy(arr) for arr in inputs]
    output_type = tensor_from_numpy(output)
    ir = KernelIR(name=fn.__name__, inputs=input_types, output=output_type, body=[])
    ir.add_instruction(IRInstruction(op="map", args=[definition.source]))
    return ir


def build_reduce_ir(fn: Callable[..., object], input_array: np.ndarray, output: np.ndarray) -> KernelIR:
    definition = get_kernel_definition(fn)
    input_types = [tensor_from_numpy(input_array)]
    output_type = tensor_from_numpy(output)
    ir = KernelIR(name=fn.__name__, inputs=input_types, output=output_type, body=[])
    ir.add_instruction(IRInstruction(op="reduce", args=[definition.source]))
    return ir


def build_scan_ir(fn: Callable[..., object], input_array: np.ndarray, output: np.ndarray) -> KernelIR:
    definition = get_kernel_definition(fn)
    input_types = [tensor_from_numpy(input_array)]
    output_type = tensor_from_numpy(output)
    ir = KernelIR(name=fn.__name__, inputs=input_types, output=output_type, body=[])
    ir.add_instruction(IRInstruction(op="scan", args=[definition.source]))
    return ir


def build_matmul_ir(a: np.ndarray, b: np.ndarray, output: np.ndarray) -> KernelIR:
    input_types = [tensor_from_numpy(a), tensor_from_numpy(b)]
    output_type = tensor_from_numpy(output)
    ir = KernelIR(name="matmul", inputs=input_types, output=output_type, body=[])
    ir.add_instruction(IRInstruction(op="matmul", args=[]))
    return ir
