"""Simplified intermediate representation used for WGSL code generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .type_system import ScalarType, TensorType


@dataclass
class IRValue:
    name: str
    type: ScalarType


@dataclass
class IRInstruction:
    op: str
    args: Sequence[str] = field(default_factory=list)
    result: str | None = None


@dataclass
class KernelIR:
    name: str
    inputs: List[TensorType]
    output: TensorType
    body: List[IRInstruction]
    workgroup_size: int = 256

    def add_instruction(self, instr: IRInstruction) -> None:
        self.body.append(instr)
