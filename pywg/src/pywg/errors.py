"""Error types used throughout the pywg package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class PyWGError(Exception):
    """Base class for pywg specific errors."""


class CompilationError(PyWGError):
    """Raised when Python source cannot be translated to the internal IR."""

    def __init__(self, message: str, source: Optional[str] = None) -> None:
        super().__init__(message)
        self.source = source

    def __str__(self) -> str:
        if self.source is None:
            return super().__str__()
        return f"{super().__str__()}\nSource:\n{self.source}"


class RuntimeErrorWebGPU(PyWGError):
    """Raised when the WebGPU runtime reports a failure."""


@dataclass(eq=False)
class TypeErrorWGSL(PyWGError):
    """Raised when type inference fails for WGSL code generation."""

    message: str

    def __str__(self) -> str:
        return self.message
