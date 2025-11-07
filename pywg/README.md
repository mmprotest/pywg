# pywg

`pywg` is a proof-of-concept toolkit that lets you write numeric kernels in a restricted subset of Python and run them in the browser using WebGPU – no hand-written WGSL required. Kernels can also run on CPU-only machines through a NumPy based fallback path, so your code keeps working even when WebGPU is unavailable.

```
┌──────────────┐      ┌─────────────┐      ┌────────────┐
│   Python     │ ───▶ │  pywg IR    │ ───▶ │  WGSL code │
└──────────────┘      └─────────────┘      └────────────┘
        │                                        │
        ▼                                        ▼
  NumPy runtime                           WebGPU runtime
```

## Features

* Tiny, well-typed DSL with a familiar NumPy feel: `map`, `reduce`, `scan`, `matmul`.
* Python-to-WGSL compiler with a compact IR and readable generated shader code.
* Minimal WebGPU runtime written in modern JavaScript.
* Pyodide bridge that makes the WebGPU runtime available in the browser.
* CPU fallback that mirrors the GPU semantics for testing and non-browser environments.
* Thorough documentation, tests, and CI configuration.

## Installation

```bash
pip install pywg
```

For development:

```bash
git clone https://github.com/your-org/pywg
cd pywg
pip install -e .
```

## Quick start

```python
import numpy as np
from pywg import kernel, map

@kernel
def fuse(i, a, b):
    return a + 2.0 * b

arr_a = np.ones(16, dtype=np.float32)
arr_b = np.arange(16, dtype=np.float32)
out = map(fuse, arr_a, arr_b)
print(out[:4])
```

`pywg` will automatically select a WebGPU backend when available (Pyodide + WebGPU-enabled browser). When WebGPU is not available a NumPy based implementation is used.

## Documentation

### Python DSL

The DSL mirrors a subset of NumPy. Kernels are ordinary Python functions decorated with `@kernel` and can use straight-line arithmetic, conditionals, and calls to a curated list of math intrinsics.

| Operation | Description | Example |
|-----------|-------------|---------|
| `map`     | Elementwise map with broadcasting | `map(fuse, a, b)` |
| `reduce`  | Reduction over the last axis | `reduce(add, arr, init=0.0)` |
| `scan`    | Prefix scan (inclusive/exclusive) | `scan(add, arr, init=0.0, inclusive=False)` |
| `matmul`  | Dense matrix multiply (float32 / int32) | `matmul(A, B)` |

Supported dtypes: `float32`, `int32`.

### Intermediate Representation

When a kernel is compiled, the function source is captured and converted into a compact IR object (`KernelIR`). The IR tracks inputs, outputs, bindings, and a high-level operation kind. This makes the code generator deterministic and easy to inspect during debugging.

### WGSL generation

`pywg.codegen_wgsl.generate_wgsl` converts the IR into a readable WGSL shader. The emitter produces the buffer declarations, assigns bindings, and sketches out the structure of the compute entry-point. The generated shader is intentionally small: it contains rich comments with the original Python source, making it ideal for further manual optimisation.

### Runtime backends

* **NumPy fallback (`pywg.runtime.cpu_fallback`)** – portable implementation that mirrors the DSL semantics. Used in tests and when WebGPU is unavailable.
* **Pyodide bridge (`pywg.runtime.pyodide_bridge`)** – thin adaptor that exposes WebGPU functionality to the Python API when running inside the browser. The bridge gracefully reports errors on CPython.
* **JavaScript runtime (`pywg/runtime/webgpu_runtime.js`)** – ES module that owns buffer management, pipeline compilation, and command submission.

### Demo

The `demo/` directory contains a Pyodide-driven browser demo. It mirrors the `pywg` sources into the Pyodide filesystem at runtime and executes the examples inside the browser. Start a static server (`python -m http.server`) and open the page in a WebGPU-capable browser.

### Limitations

* The compiler currently records kernel source and emits placeholder WGSL expressions; extending it with full expression support is tracked in future milestones.
* WebGPU execution requires Pyodide and a browser with WebGPU enabled. The CPU fallback ensures deterministic behaviour elsewhere.
* Only contiguous arrays with supported dtypes are accepted.

## Testing

Run the unit tests and static analysis tools with:

```bash
pytest
ruff check .
black --check .
mypy
```

To build distribution artifacts use:

```bash
python -m build
```

## License

MIT License. See [LICENSE](LICENSE).
