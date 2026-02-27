# boostr

**ML framework built on numr — attention, quantization, model architectures.**

[![Crates.io](https://img.shields.io/crates/v/boostr)](https://crates.io/crates/boostr) [![Docs](https://docs.rs/boostr/badge.svg)](https://docs.rs/boostr) [![License](https://img.shields.io/crates/l/boostr)](LICENSE)

boostr extends [numr](https://github.com/ml-rust/numr) with production-grade ML primitives. It provides attention mechanisms, quantization support, model architectures, and inference infrastructure — all built on numr's foundational tensors, runtimes, and ops. No reimplementation. No wrappers. Pure extension traits.

## Key Capabilities

### Quantization

- **26 formats** (GGUF-compatible): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K–Q8K, IQ1S–IQ4XS, TQ1_0, TQ2_0
- **QuantTensor type** for block-quantized data
- **Per-backend kernels**: Native SIMD (CPU), PTX (CUDA), WGSL (WebGPU)
- Zero-copy GGUF loading with memory mapping

### Attention

- **Flash Attention v2/v3** with fused QKV projection
- **Multi-Head Latent Attention (MLA)** — compressed KV cache
- **Grouped Query Attention (GQA)** and multi-head variants
- **Paged attention** for memory-efficient inference
- **Variable-length attention** with ragged tensors
- **Prefix caching** for context reuse

### Position Encodings

- **RoPE**: Split-half, interleaved, ALiBi variants
- **YaRN** for length extrapolation
- Efficient fused implementations on all backends

### Model Architectures

- **LLaMA** — standard and tensor-parallelized
- **Mamba2** — state space models with SSD kernels
- **Hybrid** — mixed transformer/SSM models
- Extensible architecture system for custom models

### Neural Network Modules

- **Linear** — standard and quantized variants
- **Embedding** and **QuantEmbedding** for low-rank token embedding
- **LayerNorm**, **RMSNorm** with fused implementations
- **MoE layers** with expert routing and load balancing

### Inference Infrastructure

- **Paged KV cache** with block allocator for memory efficiency
- **Request scheduler** with continuous batching
- **Prefix caching** for prompt reuse
- **Speculative decoding** support
- **Flash decoding** for long sequences

### Training

- **Optimizers**: AdamW, Lamb, SGD with gradient clipping
- **Mixed precision** (AMP) with automatic loss scaling
- **Gradient accumulation** and checkpointing
- **Learning rate scheduling** (warmup, cosine, linear decay)
- **Distributed training**:
  - ZeRO stage 1/2/3 (parameter/gradient/optimizer sharding)
  - Tensor parallelism with communicators
  - Pipeline parallelism (1F1B, Gpipe, ZeroBubble schedules)

### Model Loading

- **SafeTensors**: Zero-copy memory-mapped loading
- **GGUF**: Full format support with block-quantized tensors
- Format auto-detection

### Multi-Backend

- **CPU**: SIMD kernels (AVX2, NEON), native ops
- **CUDA**: PTX kernels, Flash Attention v2/v3, fused ops (CUDA 12.x)
- **WebGPU**: WGSL shaders, cross-platform GPU support

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    boostr                             │
│   (attention, RoPE, MoE, quantization, model loaders) │
└──────────────────────────┬──────────────────────────┘
                           │
                        (uses)
                           │
┌──────────────────────────▼──────────────────────────┐
│                      numr                            │
│   (tensors, ops, runtime, autograd, linalg, FFT)     │
└──────────────────────────────────────────────────────┘
```

**Design principles:**

- **Extension traits**: ML ops (AttentionOps, RoPEOps) implemented on numr's clients — not new types
- **QuantTensor**: Separate type for quantized data with custom kernels
- **impl_generic**: Composite ops composed from numr primitives, same logic on all backends
- **Custom kernels**: Dequant, quantized matmul, fused attention use per-backend optimizations (SIMD/PTX/WGSL)
- **Vendor-agnostic**: No cuBLAS, cuDNN, or MKL; all native kernels

## Quick Start

### Installation

Add to `Cargo.toml`:

```toml
[dependencies]
boostr = "0.1"

# With CUDA support (requires CUDA 12.x)
# boostr = { version = "0.1", features = ["cuda"] }

# With WebGPU support
# boostr = { version = "0.1", features = ["wgpu"] }
```

### Build

```bash
# CPU build
cargo build --release

# CUDA support (requires CUDA 12.x)
cargo build --release --features cuda

# WebGPU support
cargo build --release --features wgpu

# Run tests
cargo test
cargo test --features cuda
```

### Basic Usage

```rust
use boostr::*;
use numr::runtime::cpu::{CpuClient, CpuDevice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create CPU runtime
    let device = CpuDevice::new();
    let client = CpuClient::new(device);

    // Create tensors via numr (boostr re-exports)
    let queries = Tensor::randn::<_, f32>(
        &client,
        &[batch_size, seq_len, num_heads, head_dim],
    )?;

    // Use boostr's extension traits
    use boostr::AttentionOps;
    let output = client.multi_head_attention(
        &queries, &keys, &values,
        None, // no causal mask
        1.0,  // scale
    )?;

    Ok(())
}
```

### Loading a Model

```rust
use boostr::format::GgufLoader;

// Load a GGUF model with quantized weights
let loader = GgufLoader::from_path("model.gguf")?;
let model_config = loader.metadata()?;
let weights = loader.load_tensors(&client)?;
```

### Inference with KV Cache

```rust
use boostr::inference::PagedKvCache;

// Create a paged KV cache for efficient inference
let mut kv_cache = PagedKvCache::new(
    &client,
    num_layers,
    batch_size,
    max_seq_len,
    head_dim,
)?;

// Process tokens with cache
for token_idx in 0..seq_len {
    // ... forward pass using kv_cache ...
    kv_cache.update(layer_idx, &k, &v)?;
}
```

## Feature Flags

| Feature | Purpose                      | Dependencies      |
| ------- | ---------------------------- | ----------------- |
| `cpu`   | CPU backend (default)        | numr              |
| `cuda`  | CUDA GPU acceleration        | numr/cuda, cudarc |
| `nccl`  | Multi-GPU via NCCL           | numr/nccl         |
| `wgpu`  | WebGPU cross-platform GPU    | numr/wgpu         |
| `f16`   | Half-precision float support | numr/f16          |
| `fp8`   | FP8 precision support        | numr/fp8          |

## Module Overview

- **`ops/`** — ML-specific operations (attention, RoPE, MoE, etc.)
- **`quant/`** — Quantized tensors and kernels (26 formats)
- **`nn/`** — Neural network modules (Linear, Embedding, LayerNorm, RMSNorm, MoE)
- **`model/`** — Model architectures (LLaMA, Mamba2, Hybrid)
- **`format/`** — Model loaders (SafeTensors, GGUF)
- **`inference/`** — Inference infrastructure (KV cache, scheduling, batching)
- **`optimizer/`** — Training optimizers (AdamW, Lamb, SGD)
- **`trainer/`** — Training utilities and distributed training (ZeRO, tensor/pipeline parallelism)
- **`distributed/`** — Multi-GPU coordination

## Performance

boostr provides production-grade performance through:

- **Fused kernels** — Attention, layer norm, optimizer steps compiled to single kernels
- **Custom quantization** — Per-format SIMD/PTX/WGSL kernels for dequant and quantized matmul
- **Memory efficiency** — Paged KV cache, prefix caching, gradient checkpointing
- **Distributed training** — ZeRO stages, tensor/pipeline parallelism with minimal communication overhead
- **Zero-copy loading** — Memory-mapped GGUF with quantized weights

## Ecosystem

boostr is part of the [ml-rust](https://github.com/ml-rust) organization:

- **[numr](https://github.com/ml-rust/numr)** — Foundational numerical computing (tensors, ops, linalg, FFT)
- **[boostr](https://github.com/ml-rust/boostr)** — ML framework (this project)
- **[oxidizr](https://github.com/ml-rust/oxidizr)** — Training framework for Mamba2, MLA, MoE (uses boostr)
- **[blazr](https://github.com/ml-rust/blazr)** — Inference server with OpenAI-compatible API (uses boostr)
- **[compressr](https://github.com/ml-rust/compressr)** — Model quantization and compression (uses boostr)
- **[splintr](https://github.com/ml-rust/splintr)** — High-performance BPE tokenizer (10-12x faster than tiktoken)

## Building from Source

### Requirements

- Rust 1.85+
- For CUDA: CUDA 12.x and cudarc dependencies
- For WebGPU: wgpu and platform GPU drivers

### Clone and Build

```bash
git clone https://github.com/ml-rust/boostr.git
cd boostr

# CPU
cargo build --release

# CUDA
cargo build --release --features cuda

# Run tests
cargo test --all-features

# Format and lint
cargo fmt --all
cargo clippy --all-targets
```

## Documentation

- [API Documentation](https://docs.rs/boostr) — Full reference for public API
- [numr Documentation](https://docs.rs/numr) — Tensor and runtime types
- [CLAUDE.md](./CLAUDE.md) — Architecture and development guidelines

## Testing

```bash
# Run all tests
cargo test --all-features

# Specific test suite
cargo test ops::attention --all-features

# Verbose output
cargo test --all-features -- --nocapture
```

## Contributing

Contributions are welcome! Please see the main repository's contribution guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

boostr builds on the numerical foundation provided by [numr](https://github.com/ml-rust/numr) and is designed to power production ML infrastructure across training (oxidizr) and inference (blazr).
