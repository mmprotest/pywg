import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.mjs";

const status = document.getElementById("status");
const outputs = document.getElementById("outputs");

function renderResult(title, data) {
  const container = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = title;
  container.appendChild(heading);
  const pre = document.createElement("pre");
  pre.textContent = data;
  container.appendChild(pre);
  outputs.appendChild(container);
}

async function copyFile(pyodide, source, target) {
  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${source}: ${response.status}`);
  }
  const text = await response.text();
  const directory = target.substring(0, target.lastIndexOf("/"));
  pyodide.FS.mkdirTree(directory);
  pyodide.FS.writeFile(target, text);
}

async function bootstrapSources(pyodide) {
  const base = "..";
  const files = [
    "/src/pywg/__init__.py",
    "/src/pywg/api.py",
    "/src/pywg/dsl.py",
    "/src/pywg/ir.py",
    "/src/pywg/type_system.py",
    "/src/pywg/codegen_wgsl.py",
    "/src/pywg/utils.py",
    "/src/pywg/errors.py",
    "/src/pywg/version.py",
    "/src/pywg/runtime/__init__.py",
    "/src/pywg/runtime/cpu_fallback.py",
    "/src/pywg/runtime/pyodide_bridge.py",
    "/examples/__init__.py",
    "/examples/01_elementwise.py",
    "/examples/02_reduce.py",
    "/examples/03_scan.py",
    "/examples/04_matmul.py",
  ];
  for (const file of files) {
    await copyFile(pyodide, `${base}${file}`, `/pywg${file}`);
  }
}

async function main() {
  try {
    status.textContent = "Downloading Pyodide…";
    const pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/" });
    status.textContent = "Loading packages…";
    await pyodide.loadPackage(["numpy"]);
    await bootstrapSources(pyodide);
    await pyodide.runPythonAsync(
      "import sys\nsys.path.append('/pywg/src')\n"
    );
    status.textContent = "Running pywg examples…";
    const scripts = [
      { title: "Elementwise", path: "/pywg/examples/01_elementwise.py" },
      { title: "Reduce", path: "/pywg/examples/02_reduce.py" },
      { title: "Scan", path: "/pywg/examples/03_scan.py" },
      { title: "Matmul", path: "/pywg/examples/04_matmul.py" },
    ];
    for (const { title, path } of scripts) {
      const result = await pyodide.runPythonAsync(
        `import runpy\nrunpy.run_path('${path}', run_name='__main__')`
      );
      renderResult(title, JSON.stringify(result, null, 2));
    }
    status.textContent = "Done";
  } catch (error) {
    status.textContent = `Error: ${error}`;
    console.error(error);
  }
}

main();
