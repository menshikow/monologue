# Monologue

> *Building a browser-native LLM inference engine with Rust and WebGPU, starting with CPU-based inference using Qwen as a placeholder model.*

## The Architecture

This project is a **hybrid-stack application** that demonstrates the full lifecycle of an AI product.

```md
[ PyTorch Training ]  ->  [ Model Weights (.bin) ]  ->  [ Rust Engine (CPU) ]
                                                               |
                                                           (Future: WebGPU)
                                                                v
                                                       [ WebAssembly / UI ]
```

### Core Engineering Goals

1. **CPU-First Inference:** Build a production-ready inference engine using existing Rust tensor libraries.
2. **Qwen Integration:** Use Qwen2.5-0.5B as a placeholder model while developing custom training capabilities.
3. **WebGPU Future:** Plan for GPU acceleration via WGSL shaders after CPU implementation is complete.
4. **Browser Deployment:** Eliminate Python dependencies by compiling to WebAssembly.

---

## Project Structure

```md
monologue/
├── Cargo.toml                      # Workspace root configuration
├── TODO.md                         # Detailed implementation roadmap
│
├── rust_tokenizer/                 # ✅ PRODUCTION-READY: Custom BPE + Python bindings
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                  # Core tokenizer library + pyo3 module
│   │   └── tests.rs                # Rust unit tests
│   ├── benches/
│   │   └── benchmark.rs            # Performance benchmarks
│   └── tests/
│       ├── integration_tests.rs    # Rust integration tests
│       └── test_tokenizer.py       # Python tests
│
├── inference/                      # 🚧 IN DEVELOPMENT: CPU inference engine
│   ├── Cargo.toml                  # Tensor library dependencies
│   └── src/
│       ├── lib.rs                  # Core inference library
│       ├── main.rs                 # CLI interface
│       └── [planned modules]       # model.rs, tensor.rs, loader.rs, etc.
│
├── training/                       # 📋 PLANNED: PyTorch training pipeline
│   └── src/                        # Training scripts (placeholder)
│
├── web/                            # 📋 PLANNED: React + WASM interface
│   └── src/
│       └── App.tsx                 # React scaffold
```

---

## Current Status

### ✅ **Completed**
- **Production-ready BPE tokenizer** with Python bindings and comprehensive testing
- **Well-structured project architecture** with clear separation of concerns
- **Detailed implementation roadmap** (see TODO.md)

### 🚧 **In Development** 
- **CPU inference engine** using existing Rust tensor libraries
- **Qwen2.5-0.5B integration** as placeholder model

### 📋 **Planned**
- **WebGPU acceleration** via WGSL shaders (future phase)
- **Custom model training** pipeline
- **Browser WASM deployment** with React interface

---

## Quick Start

### Prerequisites
- Rust 1.70+ 
- Python 3.12+ with `uv` or `pip`
- Node.js 18+ (for web development)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/menshikow/monologue.git
cd monologue

# 2. Build the workspace
cargo build --workspace

# 3. Test the tokenizer (working component)
cd rust_tokenizer
cargo test
./run_tests_and_bench.sh

# 4. Python setup for future model conversion
cd ..
uv sync  # or: pip install -e .
```

### Development Status

**Working Components:**
```bash
# Tokenizer with Python bindings
cargo run --bin rust_tokenizer  # (when implemented)
python -c "import rust_tokenizer; print('Tokenizer works!')"
```

**Planned Components (not yet implemented):**
```bash
# CPU inference (future)
cargo run --bin inference -- --model models/qwen-0.5b/model.bin --prompt "Hello!"

# Model conversion (future)  
python scripts/export_qwen.py  # scripts to be created

# Web interface (future)
cd web && npm run dev
```

---

## Implementation Roadmap

See **[TODO.md](./TODO.md)** for the detailed 6-phase implementation plan:

### **Phase 1-3: CPU Inference Engine** (Current Focus)
- Tensor library integration and model loading
- Qwen architecture implementation  
- Generation pipeline and CLI interface

### **Phase 4-6: Production & Web Prep**
- Model conversion and comprehensive testing
- WASM compatibility and browser API design
- Documentation and polish

**Estimated Timeline:** 10-15 weeks total for complete CPU inference implementation.

---

## Tech Stack

| Domain | Technology | Status |
| --- | --- | --- |
| **Tokenizer** | **Rust + PyO3** | ✅ Production-ready |
| **Inference** | **Rust + Tensor Library** | 🚧 In development |
| **Training** | **PyTorch** | 📋 Planned |
| **Web** | **React + TypeScript** | 📋 Scaffold only |
| **Compute** | **CPU → WebGPU** | 🚧 CPU first, WebGPU future |
| **Deployment** | **WASM** | 📋 Planned |

---

## Architecture Decisions

### **Why CPU First?**
- Reduces complexity and development time
- Allows focus on model architecture and correctness
- WebGPU integration planned after CPU implementation is stable
- Easier debugging and validation

### **Why Existing Tensor Libraries?**
- Leverages battle-tested linear algebra implementations
- Faster development than building custom tensor operations
- Better performance optimization and SIMD support
- Options: candle-core, tch, burn, or ndarray

### **Why Qwen as Placeholder?**
- No training costs while developing inference engine
- Well-documented architecture specifications
- Similar size to target 560M parameter model
- Easy to replace with custom model later

---

## Contributing

This project follows the conventions outlined in **[CONVENTIONS.md](./CONVENTIONS.md)**. Key guidelines:

- **Strict architectural separation** between Rust, Python, and TypeScript components
- **WASM compatibility** requirements for the inference engine
- **Conventional commits** with clear scope definitions
- **Comprehensive testing** for all components

---

## Acknowledgements

❤️ This project is heavily influenced by:
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) for transformer architecture guidance
- The [Burn](https://github.com/tracel-ai/burn) project for Rust ML framework inspiration
- Hugging Face's [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B) for the placeholder model

---

## License

MIT License.