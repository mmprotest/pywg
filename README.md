# pywg

> Python-first kernels that run anywhere – transpiled to WebGPU when you have it, powered by NumPy when you do not.

`pywg` lets you express high-performance array kernels in a constrained subset of Python that feels familiar to NumPy users. The toolkit parses your Python functions into an intermediate representation, emits WGSL shaders, and executes them through a lightweight WebGPU runtime in the browser. When WebGPU is unavailable, the same programs execute via a NumPy-backed CPU fallback, so your code continues to work on standard Python interpreters.

```
┌──────────────┐      ┌─────────────┐      ┌────────────┐
│   Python     │ ───▶ │  pywg IR    │ ───▶ │  WGSL code │
└──────────────┘      └─────────────┘      └────────────┘
        │                                        │
        ▼                                        ▼
  NumPy runtime                           WebGPU runtime
```

The repository ships a full Python package, end-to-end tests, examples, and a Pyodide browser demo so you can validate the workflow locally.

## Table of contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [DSL reference](#dsl-reference)
5. [Architecture overview](#architecture-overview)
6. [Execution backends](#execution-backends)
7. [Examples](#examples)
8. [Pyodide demo](#pyodide-demo)
9. [Testing & quality](#testing--quality)
10. [Troubleshooting](#troubleshooting)
11. [Roadmap](#roadmap)
12. [Contributing](#contributing)
13. [License](#license)

## Features

* **Tiny, typed DSL** – declare kernels with `@kernel` and launch them with familiar functions: `map`, `reduce`, `scan`, and `matmul`.
* **Python-to-WGSL compiler** – walk the Python AST, produce a compact SSA-like IR, and render readable WGSL with explicit buffer bindings.
* **WebGPU runtime** – minimalist JavaScript module that handles adapter selection, buffer uploads, dispatch, and readbacks.
* **Pyodide integration** – thin bridge that exposes the WebGPU runtime to Python code when running inside Pyodide.
* **CPU fallback** – NumPy-powered implementation that mirrors the DSL semantics for portable execution and testing.
* **Modern packaging** – Poetry project configuration, strict type checking, Ruff linting, and Black formatting baked in.
* **CI-ready** – GitHub Actions workflow exercises tests, type checks, and style checks across Python 3.9–3.12.

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/your-org/pywg
cd pywg/pywg
pip install -e .
```

The Poetry configuration (`pywg/pyproject.toml`) exposes an editable install that includes the runtime and CLI helpers.

### Building distribution artifacts

```bash
cd pywg
python -m build
```

This produces source and wheel artifacts in `dist/`. Upload them with `twine` when you are ready to publish.

## Quick start

Create an elementwise kernel and run it. In CPython the NumPy backend runs automatically; in Pyodide + WebGPU the WGSL code path is used.

```python
import numpy as np
from pywg import kernel, map

@kernel
def fuse(i, a, b):
    return a + 2.0 * b

n = 1_000_000
a = np.ones(n, dtype=np.float32)
b = np.arange(n, dtype=np.float32)
out = map(fuse, a, b)
print(out[:5])
```

`pywg` performs shape validation, emits IR for `fuse`, and either dispatches a WebGPU compute pass or falls back to NumPy.

## DSL reference

The DSL intentionally mirrors a minimal NumPy subset so kernels stay portable.

| Operation | Description | Example |
|-----------|-------------|---------|
| `map`     | Elementwise map with NumPy-style broadcasting | `map(fuse, a, b)` |
| `reduce`  | Reduce over the last axis with an associative function | `reduce(add, arr, init=0.0)` |
| `scan`    | Prefix scan (exclusive by default, optional `inclusive=True`) | `scan(add, arr, init=0.0)` |
| `matmul`  | Dense matrix multiply for `float32` tensors | `matmul(A, B)` |

Supported data types: `float32`, `int32`, and booleans for scalar intermediates. Expressions inside kernels allow arithmetic (`+`, `-`, `*`, `/`, `%`), comparisons, scalar conditionals, and math intrinsics such as `abs`, `exp`, `log`, `sin`, `cos`, `min`, and `max`.

## Architecture overview

The pipeline is broken into well-defined layers so you can audit or extend each stage.

1. **DSL front-end (`pywg.dsl`)** – captures Python source with `inspect`, validates the AST, and tags kernel metadata.
2. **Intermediate representation (`pywg.ir`)** – converts supported AST nodes into a small SSA-like structure, tracking types via `pywg.type_system`.
3. **WGSL code generation (`pywg.codegen_wgsl`)** – turns IR into WGSL text. Emitters generate buffer layouts, helper functions, and compute entry points with tuned workgroup sizes.
4. **Runtimes (`pywg.runtime`)** – includes the CPU fallback, the Pyodide bridge, and a JavaScript module (`webgpu_runtime.js`) that speaks WebGPU.
5. **API surface (`pywg.api`)** – routes calls to the appropriate backend, exposes environment probes, and keeps user-facing APIs ergonomic.

Each layer is covered by unit tests so regressions are easy to spot.

## Execution backends

`pywg` automatically chooses a backend at runtime:

* **WebGPU** – Activated when running inside Pyodide and `navigator.gpu` is present. The bridge uploads NumPy buffers to GPU buffers, dispatches compute passes, and copies the result back.
* **CPU fallback** – The NumPy implementation mirrors broadcasting and kernel semantics precisely. Tests validate parity with the GPU path, enabling reliable execution in CI or environments without WebGPU.

You can check which backend was used via:

```python
from pywg import get_device_info

info = get_device_info()
print(info["backend"])  # "webgpu" or "cpu"
```

## Examples

The `examples/` directory showcases the API:

* `01_elementwise.py` – fused elementwise operations with broadcasting.
* `02_reduce.py` – large-scale summation via `reduce`.
* `03_scan.py` – exclusive prefix sum on float arrays.
* `04_matmul.py` – tiled matrix multiplication validated against NumPy.

Run them directly with `python examples/01_elementwise.py` (CPU fallback) or in the browser through the demo.

## Pyodide demo

A browser-based showcase lives in `pywg/demo/`:

1. Serve the directory: `cd pywg/demo && python -m http.server`.
2. Open `http://localhost:8000` in a WebGPU-capable browser (Chrome 113+, Edge 113+, or Safari TP with the WebGPU flag enabled).
3. The page loads Pyodide, installs `pywg`, runs each example, and renders the first few results.

The demo integrates the same Python sources shipped in this repository, so it stays in sync with the package.

## Testing & quality

Continuous integration (GitHub Actions) runs:

```bash
pytest
ruff check .
black --check .
mypy
python -m build
```

Local developers can enable the same checks via `pre-commit install`, which hooks Ruff, Black, and MyPy on each commit. Tests default to the CPU backend; GPU-specific tests are skipped unless `PYWG_GPU=1` is exported.

## Troubleshooting

* **WebGPU not detected** – Ensure you are running inside Pyodide and using a browser with WebGPU enabled. Chrome and Edge often require the `--enable-unsafe-webgpu` flag on older versions.
* **NumPy dtype errors** – Kernels currently accept contiguous `float32` or `int32` arrays. Use `arr.astype(np.float32, copy=False)` before dispatch if needed.
* **Performance questions** – Profiling hooks in `pywg.utils` cache workgroup sizes. Delete the cache or adjust heuristics there when experimenting with new hardware.
* **Pyodide imports failing** – Confirm the Pyodide version matches the one the demo expects and that `webgpu_runtime.js` is reachable from the served path.

## Roadmap

Planned improvements include:

* Expanded dtype coverage (float16, bfloat16, unsigned integers).
* Richer control flow and vector intrinsics in the DSL.
* Automatic kernel fusion and ahead-of-time compilation caches.
* WebGPU pipeline caching across dispatches.

## Contributing

We welcome pull requests! Please read [CONTRIBUTING.md](pywg/CONTRIBUTING.md) and follow the [Code of Conduct](pywg/CODE_OF_CONDUCT.md). Run the test suite and linters before submitting patches.

## License

Released under the [MIT License](pywg/LICENSE).
