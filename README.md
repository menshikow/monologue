# Monologue

![Image](https://github.com/user-attachments/assets/dee80dce-bc8d-4b9c-b9fd-7b2ffc20141a)

> *An engineering exploration: Building a 560M parameter reasoning Small Language Model from scratch to run locally in the browser via Rust and WebGPU.*

## The Architecture

This project is a **hybrid-stack application** that demonstrates the full lifecycle of an AI product.

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

## Project Structure

```md
monologue/
├── Cargo.toml                    # Workspace root configuration
├── Cargo.lock                    # Dependency lock file
│
├── rust_tokenizer/               # RUST CRATE: Custom BPE Builder
│   ├── Cargo.toml                
│   └── src/
│       ├── lib.rs                # Core tokenizer library API
│       └── tokenizer.rs          # BPE implementation logic
│
├── models/                       # Local model storage
│   ├── qwen-0.5b/                # Placeholder model artifacts
│   │   ├── model.bin             # Flattened binary weights
│   │   └── tokenizer.json        # Qwen BPE config
│   └── monologue-560m/           # (Future) Trained model 
│
├── inference/                    # The Engine (Rust + WebGPU)
│   ├── Cargo.toml                # wgpu, WGSL dependencies
│   └── src/
│       ├── lib.rs                # Core inference library
│       ├── model.rs              # Binary loader & Structs
│       ├── kernels/              # WGSL shader modules (MatMul, RoPE)
│       └── cache.rs              # KV-cache implementation
│
├── training/                     # PyTorch 
│   ├── src/                      # Training scripts
│   └── data/                     # FineWeb-Edu & OpenR1 datasets
│
├── web/                          # The Body (React + WASM)
│   ├── src/                      # React/TypeScript source
│   ├── public/                   # Static assets
│   └── pkg/                      # Compiled WASM output
│
└── scripts/
    ├── run.sh                    # Main automation script
    ├── train.sh                    # Main automation script
    └── export_qwen.py            # Converts Qwen safetensors to binary

```

---

### 1. Setup the Engine (Free)

```bash
# 1. Clone the repo
git clone https://github.com/menshikow/monologue.git

# 2. Download & Convert Qwen (The Placeholder)
# This downloads Qwen2.5-0.5B and flattens it to 'models/qwen-0.5b/model.bin'
python3 scripts/export_qwen.py

# 3. Run the Rust Inference (CPU/Native check)
cargo run --bin inference -- --model models/qwen-0.5b/model.bin --prompt "Hello!"

```

### 2. Train the Brain (Paid)

```bash
# Launch the training pipeline (Requires GPU)
./scripts/run_training.sh
```

---

## Roadmap

### Phase 1: The Engine (Qwen Integration)

* [ ] **Artifacts:** Download `Qwen2.5-0.5B-Instruct` and export to raw binary (`.bin`).
* [ ] **Loader:** Implement Rust `mmap` loader to read the binary model file.
* [ ] **Math:** Implement WGSL shaders for:
* [ ] RMSNorm (Root Mean Square Normalization)
* [ ] RoPE (Rotary Positional Embeddings)
* [ ] SwiGLU Activation
* [ ] MatMul (Matrix Multiplication)

* [ ] **Verification:** Pass "Sanity Check" (Rust logits == PyTorch logits).

### Phase 2: The Brain (Custom Training)

*Focus: Replacing the generic brain with a reasoning one.*

* [ ] **Data Pipeline:** Blend "Instruction Following" with "Reasoning Trace" datasets.
* [ ] **Training:** PyTorch loop to train a 560M param model from scratch.
* [ ] **Drop-in:** Replace the Qwen `.bin` file with the Monologue `.bin` file.

### Phase 3: The Application (Web/WASM)

* [ ] **WASM:** Bind Rust inference `step()` function to JavaScript.
* [ ] **UI:** React chat interface with collapsible `<think>` accordion.
* [ ] **Optimization:** Implement KV-Cache paging for long-context performance.

---

## Tech Stack

| Domain | Technology | Reason for Choice |
| --- | --- | --- |
| **AI / ML** | **PyTorch** | Industry standard for defining the Transformer architecture. |
| **Systems** | **Rust** | Memory safety and performance are critical for the inference engine. |
| **Compute** | **WebGPU** | The only viable path for high-performance hardware acceleration in the web. |
| **Web** | **React / TS** | Efficient UI state management to handle the streaming token output. |
| **Ops** | **WASM** | To port the heavy lifting of Rust into the universal browser environment. |

## Acknowledgements

❤️ This project is heavily influenced by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and the [Burn](https://github.com/tracel-ai/burn) project.

## License

MIT License.
