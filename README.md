# Monologue

> *An engineering exploration: Building a 560M parameter reasoning Small Language Model from scratch to run locally in the browser via Rust and WebGPU.*

## The Concept

**Monologue** is a proposed solution to two specific problems in the current AI landscape: **Privacy** and **Reasoning Capability**.

The goal is to train a model to generate an internal "Chain of Thought" (`<think>...</think>`) before answering. This mimics the "System 2" reasoning found in frontier models, but the architecture is designed to run entirely on the user's device.

**The Promise:** Deep reasoning capabilities where your private thoughts never leave the browser.

## The Planned Architecture

This project is a **hybrid-stack application** that will demonstrate the full lifecycle of an AI product.

```md
[ PyTorch Training ]  ->  [ Model Weights (.bin) ]  ->  [ Rust Engine ]
                                                               |
                                                          (Compiles to)
                                                               v
                                                      [ WebAssembly / UI ]
```                                                      

### Core Engineering Goals

1. **System 2 Alignment:** Create a mixed dataset (General Chat + Reasoning Traces).
2. **Cross-Platform Inference:** Eliminate Python dependencies for the end-user by building a custom inference engine in Rust, compiled to WebAssembly.
3. **Hardware Acceleration:** Implement custom Matrix Multiplication (MatMul) kernels in **WGSL** (WebGPU Shading Language) to utilize the user's GPU directly from the browser context.

---

## Proposed Structure

The repository is being scaffolded into the following modular components:

| Directory | Description | Stack |
| --- | --- | --- |
| **`training/`** | **The "Brain"**. PyTorch code to define and train the 560M param GPT model. | Python, PyTorch |
| **`rust_tokenizer/`** | **The "Builder"**. A custom BPE tokenizer trainer to be built in Rust for speed. | Rust |
| **`inference/`** | **The "Engine"**. The raw inference runtime (KV-Cache, MatMul). | Rust, wgpu, WGSL |
| **`web/`** | **The "Body"**. React frontend to host the Rust engine via WASM. | TypeScript, React |
| **`scripts/`** | Automation scripts for data prep and environment setup. | Bash |

---

## The Vision: "Zero to Training"

The end goal for the developer experience is a single script, `run.sh`, that automates the entire pipeline.

### Target Workflow

*Once implemented, the setup process will look like this:*

```bash
# 1. Clone the repo
git clone https://github.com/your-username/monologue.git
cd monologue

# 2. Make executable
chmod +x run.sh

# 3. Launch
./run.sh

```

**What `run.sh` will do:**

1. Compile the Rust Tokenizer.
2. Download FineWeb-Edu & OpenR1 datasets.
3. Tokenize the data into binary files.
4. Launch the PyTorch training loop.

---

## Tech Stack Strategy

| Domain | Technology | Reason for Choice |
| --- | --- | --- |
| **AI / ML** | **PyTorch** | Industry standard for defining the Transformer architecture. |
| **Systems** | **Rust** | Memory safety and performance are critical for the inference engine. |
| **Compute** | **WebGPU** | The only viable path for high-performance hardware acceleration in the web. |
| **Web** | **React / TS** | Efficient UI state management to handle the streaming token output. |
| **Ops** | **WASM** | To port the heavy lifting of Rust into the universal browser environment. |

---

## Roadmap & TODOs

### Phase 1: Model Training (Current Focus)

* [ ] **Architecture:** Define 560M param GPT (Nanochat config) in PyTorch.
* [ ] **Data Pipeline:** Create script to blend "Instruction Following" with "Reasoning Trace" datasets.
* [ ] **Tokenizer:** Implement Byte-Pair Encoding (BPE) trainer with special control tokens (`<think>`).
* [ ] **Training:** Achieve loss convergence on initial subset (1B tokens).

### Phase 2: Serialization

* [ ] **Export:** Write Python script to flatten model weights to binary `.bin`.
* [ ] **Quantization:** Investigate Int8 quantization strategies for browser memory efficiency.

### Phase 3: Rust & WebGPU

* [ ] **Kernels:** Write WGSL shaders for MatMul, RMSNorm, and RoPE.
* [ ] **Inference Loop:** Implement the autoregressive generation loop in Rust.
* [ ] **WASM:** Bind Rust functions to the JavaScript context.

### Phase 4: Frontend

* [ ] **UI:** Build React chat interface with collapsible "Thought Process" accordion.
* [ ] **Workers:** Offload inference to Web Workers to ensure the UI remains non-blocking.

---

## Acknowledgements

❤️ This project is heavily influenced by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). The goal is to take those principles and push them into the browser environment.

## License

MIT License.
