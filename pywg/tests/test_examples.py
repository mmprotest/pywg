from __future__ import annotations

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_examples_execute() -> None:
    examples_dir = PROJECT_ROOT / "examples"
    for script in sorted(examples_dir.glob("0*_*.py")):
        runpy.run_path(str(script))
