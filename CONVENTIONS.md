# Project Conventions & Architecture

> **Context for AI Agents:** This is a hybrid-stack AI engineering project involving **Rust** (WebGPU/WASM), **Python** (PyTorch), and **TypeScript** (React). Strict architectural separation is required.

---

## 1. The Architecture Map

Understanding the directory boundaries is critical. **Do not cross-contaminate logic.**

| Directory | Stack | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **`inference/`** | **Rust** | `no_std` compatible, `wgpu`, `wasm-bindgen` | The runtime engine. **CRITICAL:** Must compile to WASM. Do not use `std::fs`, `std::thread`, or blocking I/O here. |
| **`rust_tokenizer/`**| **Rust/Py** | `pyo3`, `tokenizers` | BPE Tokenizer logic. Compiles to a Python extension module. |
| **`web/`** | **TS/React**| `vite` | The UI. Will interact with the Rust engine via Web Workers. |
| **`training/`** | **Python** | `torch`, `cuda` | Research & Training. Can be recreated when needed. Output must be serializable to flat binary `.bin`. |
| **`scripts/`** | **Python** | Various utilities for model export and training | Helper scripts (to be created when needed). |

---

## 2. Agent Directives (Read Carefully)

If you are an LLM or Agent working on this codebase, adhere to these strict constraints:

### A. The "WASM Wall"
* **Scope:** `inference/`
* **Rule:** This code runs in a browser environment.
    * ❌ **NO** direct file access (`File::open`).
    * ❌ **NO** os-level threading (`std::thread::spawn`). Use `async`/`await`.
    * ❌ **NO** heavy crates that lack WASM support (e.g., `tokio` full features).
    * ✅ **YES** use `wgpu` for compute.
    * ✅ **YES** use `bytemuck` for casting slices.

### B. The "Golden Rule" of Data
* **Rule:** Never generate git commands that add `.bin`, `.pt`, `.safetensors`, or `.gguf` files.
* **Action:** If a user asks to save a model, write a Python script to save it to `models/` but ensure `.gitignore` is respected.

### C. Math & Kernels
* **Scope:** `inference/src/kernels/*.wgsl`
* **Rule:** WGSL is sensitive to memory layout.
    * Ensure struct members are aligned to 16 bytes (vec4) where required by WebGPU `std140`/`std430` layout.
    * Do not hallucinate WGSL built-ins; stick to the official WebGPU spec.

---

## 3. Human Workflow

### Setup
1.  **Engine:** `cargo build --workspace`
2.  **Web:** `cd web && npm install` (when package.json exists)
3.  **Python:** `uv sync` or `python -m pip install -e .`

### Commit Standards
We use **Conventional Commits**. Please define the scope clearly.

* `feat(inf):` New feature in Rust engine.
* `fix(web):` Bug fix in React UI.
* `feat(tok):` Tokenizer changes.
* `exp(train):` Experimental change in training logic.
* `chore:` Config or documentation.

**Example:** `feat(inf): implement RMSNorm shader in WGSL`

### Testing
* **Rust:** `cargo test --workspace`
* **Python:** `pytest` (when tests exist)
* **Tokenizer:** `cd rust_tokenizer && ./run_tests_and_bench.sh`

---

## 4. Current Project State

**Implemented:**
- ✅ Basic workspace structure with `rust_tokenizer` and `inference` crates
- ✅ Tokenizer with Python bindings (`pyo3`)
- ✅ Web UI scaffold with React/TypeScript
- ✅ Python training pipeline structure

**In Progress:**
- 🚧 Inference engine WebGPU implementation
- 🚧 Model export and conversion scripts
- 🚧 Training data pipeline

**Not Yet Implemented:**
- ❌ WGSL kernels for matrix operations
- ❌ WASM compilation and browser integration
- ❌ Model weight loading and inference
- ❌ Web Worker communication

---

## 5. License
MIT.
