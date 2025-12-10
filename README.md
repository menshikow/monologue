# Monologue

> **Private "System 2" AI Reasoning at the Edge.**
>
> *A 560M parameter reasoning Large Language Model build from scratch and ran locally in your browser via Rust and WebGPU.*

## The Concept

**Monologue** is an learning/engineering exploration into the modern LLM pipeline, built to solve two specific problems: **Privacy** and **Reasoning Capability**.

Monologue will be trained to generate an internal "Chain of Thought" (`<think>...</think>`) before answering. This mimics the "System 2" reasoning found in frontier models (like OpenAI o1), but it runs entirely on the user's device—ensuring that these private thoughts never leave the browser.

## Technical Architecture

This project is a **hybrid-stack application** demonstrating the full lifecycle of an AI product:

```mermaid
graph LR
    subgraph "Phase 1: The Brain (Python)"
        A["Dataset Mix: Chat + Reasoning"] --> B["PyTorch Training Loop"]
        B --> C["Export Weights"]
    end

    subgraph "Phase 2: The Engine (Rust)"
        C --> D["WASM Loader"]
        D --> E["wgpu / WebGPU Backend"]
        E --> F["Transformer Kernels (WGSL)"]
    end

    subgraph "Phase 3: The Body (React)"
        F --> G["Web Worker"]
        G --> H["User Interface"]
    end
````

### Key Engineering Challenges needed to be solve

1. **System 2 Alignment:** I want to create a mixed dataset (General Chat + Reasoning Traces) to teach the model to "pause and plan" before outputting a final answer.
2. **Cross-Platform Inference:** Replace Python dependencies with a custom Rust inference engine compiled to WebAssembly.
3. **Hardware Acceleration:** Implement a custom Matrix Multiplication (MatMul) kernels in WGSL (WebGPU Shading Language) to utilize the user's GPU directly from the browser.

## Project Structure

This repo is organized into modular components:

| Directory | Description | Stack |
| :--- | :--- | :--- |
| **`training/`** | The "Brain". PyTorch code to train the 560M param GPT model. | Python, PyTorch |
| **`rust_tokenizer/`** | The "Builder". A custom BPE tokenizer trainer. | Rust |
| **`inference/`** | The "Engine". A raw inference runtime (KV-Cache, MatMul). | Rust, wgpu, WGSL |
| **`web/`** | The "Body". React frontend that runs the Rust engine via WASM. | TypeScript, React |
| **`speedrun.sh`** | Automation script to build the tokenizer, prep data, and train. | Bash |

## 🚀 Quick Start (The Speedrun)

You can go from zero to a training run with a single script.

### Prerequisites

  * Python 3.10+
  * Rust (Cargo)
  * NVIDIA GPU (Recommended for training)

### Run the Pipeline

The `speedrun.sh` script handles everything:

1. Compiles the Rust Tokenizer.
2. Downloads FineWeb-Edu & OpenR1 datasets.
3. Tokenizes the data into binary files.
4. Launches the PyTorch training loop.

<!-- end list -->

```bash
# 1. Clone the repo
git clone [https://github.com/your-username/monologue.git](https://github.com/your-username/monologue.git)
cd monologue

# 2. Make executable
chmod +x speedrun.sh

# 3. Launch
./speedrun.sh
```

## Tech Stack Details

| Domain | Technology | Usage |
| :--- | :--- | :--- |
| **AI / ML** | PyTorch | Training the 560M parameter Transformer model. |
| **Systems** | Rust | Core inference logic, KV-cache management, Safe memory handling. |
| **Compute** | WebGPU / wgpu | Hardware accelerated linear algebra in the browser. |
| **Web** | TypeScript / React | UI, State management, and WASM bindings. |
| **Ops** | WASM | Compiling the Rust engine for universal deployment. |

## TODO

### Phase 1: Model Training

  - [ ] **Architecture:** Define 560M param GPT (Nanochat config) in PyTorch.
  - [ ] **Data Pipeline:** Script to blend "Instruction Following" with "Reasoning Trace" datasets.
  - [ ] **Tokenizer:** Train Byte-Pair Encoding (BPE) with special control tokens (`<think>`).
  - [ ] **Training:** Converge model on 1B+ tokens.

### Phase 2: Serialization

  - [ ] **Export:** Python script to flatten weights to binary `.bin`.
  - [ ] **Quantization:** Implement Int8 quantization for browser efficiency.

### Phase 3: Rust & WebGPU

  - [ ] **Kernels:** Write WGSL shaders for MatMul, RMSNorm, and RoPE.
  - [ ] **Inference Loop:** Implement autoregressive generation in Rust.
  - [ ] **WASM:** Bind Rust functions to JavaScript context.

### Phase 4: Frontend

  - [ ] **UI:** React chat interface with collapsible "Thought Process" accordion.
  - [ ] **Workers:** Offload inference to Web Workers for non-blocking UI.

## License

MIT License.
