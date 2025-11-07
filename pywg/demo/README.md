# pywg browser demo

This directory contains a lightweight demo that runs the `pywg` examples inside Pyodide.

## Running locally

```bash
cd demo
python -m http.server 8000
```

Open <http://localhost:8000> in a WebGPU-enabled browser (Chrome, Edge, or Safari Technology Preview). The page downloads Pyodide, mirrors the `pywg` sources into the virtual filesystem, and runs the example kernels. If WebGPU is unavailable the computations will still execute on the CPU via NumPy.

## Troubleshooting

* Ensure WebGPU is enabled via `chrome://flags/#enable-unsafe-webgpu` when running on Chrome desktop.
* The examples may take a couple of seconds to run on first load while Pyodide initialises.
* Check the browser console for detailed error logs if a script fails.
