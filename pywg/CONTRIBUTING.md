# Contributing to pywg

Thank you for your interest in contributing to `pywg`! We welcome improvements to the DSL, compiler, runtime, documentation, and demo experience.

## Development setup

1. Fork and clone the repository.
2. Create a virtual environment with Python 3.9+.
3. Install dependencies with `pip install -e .[dev]`.
4. Enable the pre-commit hooks using `pre-commit install`.

## Pull requests

* Keep PRs focused and include tests.
* Ensure `ruff`, `black`, `mypy`, and `pytest` pass locally before submitting.
* Update documentation for user-facing changes.
* Add yourself to the changelog when appropriate.

## Code style

* Python code must be formatted with `black` and linted with `ruff`.
* Type hints are required for new code. The project runs `mypy --strict`.
* Prefer pure functions and functional style when working on the DSL and compiler.

## Reporting issues

Please use GitHub Issues for bug reports and feature requests. Include reproduction steps and environment details where possible.
