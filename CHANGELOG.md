# Changelog

All notable changes to boostr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
boostr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-06-07

### ⚠️ Breaking changes

- **Distributed** — fallible paths that previously `unwrap`/`expect`-ed now return `Result`. Callers must propagate or handle the new errors.
- **Checkpoint** — `ShardingMeta` flattened to a single global strategy record. Checkpoints written by 0.1.0 must be re-saved; code matching the old per-tensor shape must update.
- **Trainer** — CUDA graph capture API moved to a destination-passing convention; existing capture call sites must pass an output buffer.
- **nn** — `var_contiguous` and `repeat_kv` consolidated into `nn::var_ops`; update import paths.
- Several internal ops now propagate errors from `contiguous()` instead of panicking, changing affected signatures to `Result`.

### Added

- **Encoder / embedding models** — `EmbeddingPipeline` with a variable-length (packed) forward path; XLM-RoBERTa (position ids, embedding LayerNorm), NomicBert, and Gemma-embedding architectures; `forward_hidden` and `ImageEmbedder` for embedding extraction; F16 compute path with auto-managed CUDA graphs.
- **Audio** — Whisper decoder and Kokoro TTS pipeline, including model/config support (`Kokoro` config, Whisper decoder fields) and usage examples.
- **Ops** — `RoPEPackedOps` for position-id-aware packed sequences; variable-length attention extended to support GQA.
- **CUDA** — varlen attention kernels extended to `head_dim=256` with corrected FP16 atomics.
- **nn** — AdaIN, LSTM, and weight-norm modules.
- **Format** — PyTorch `.pt` loader; EOS token exposed from the GGUF tokenizer.
- **GGUF** — streaming dequant path for large quantized tensors.

### Changed

- **Attention** — standard attention fwd/bwd lifted into a shared `impl_generic` implementation.
- **Encoder** — `model.rs` split into a directory module with CUDA graph support; padded sequence length aligned to multiples of 16; token ids truncated to `max_position_embeddings` before padding.
- **Quantization (CUDA)** — Q4_K GEMM kernel rewritten with an output-tiled 16×16 layout.
- Adopted let-chain syntax throughout for Rust 1.88.

### Fixed

- `PyTorchEmbedding` initializer now uses a real N(0,1) distribution.
- Kokoro `source_filter` wired to the real noise/phase path; dead scaffolding removed.
- GPU block allocator capacity uses `checked_div` to avoid divide-by-zero.
- Encoder frees layer intermediates during inference to prevent VRAM accumulation.
- CLS pooling made contiguous before reshape.

---

## [0.1.0] - 2026-03-15

> First tagged release of boostr — production-grade LLM primitives built on
> [numr](https://github.com/ml-rust/numr) via extension traits, with CPU, CUDA,
> and WebGPU backends.

### Added

#### Quantization

- 26 GGUF-compatible formats — Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1, Q2K–Q8K, IQ1S–IQ4XS, TQ1_0, TQ2_0
- `QuantTensor` type for block-quantized data, with per-backend dequant and quantized-matmul kernels (native SIMD / PTX / WGSL)
- Zero-copy GGUF loading with memory mapping

#### Attention

- Flash Attention v2/v3 with fused QKV projection
- Multi-Head Latent Attention (MLA) with compressed KV cache
- Grouped Query Attention (GQA) and multi-head variants
- Paged attention and variable-length (ragged) attention
- Prefix caching for context reuse

#### Position encodings

- RoPE (split-half, interleaved) and ALiBi variants
- YaRN for length extrapolation, with fused implementations on all backends

#### Model architectures

- LLaMA — standard and tensor-parallelized
- Mamba2 — state-space models with SSD kernels
- Hybrid transformer/SSM models, plus an extensible architecture system

#### Neural network modules

- `Linear` (standard and quantized), `Embedding`, `LayerNorm`, `RMSNorm`
- MoE layers with expert routing and load balancing

#### Inference

- Paged KV cache with a block allocator
- Request scheduler with continuous batching
- Speculative decoding with adaptive draft depth and verification kernels
- Flash decoding for single-token decode (CUDA, auto-selected when S_q=1)

#### Training

- Optimizers — AdamW, Lamb, SGD with gradient clipping
- Mixed precision (AMP) with automatic loss scaling
- Gradient accumulation and checkpointing; LR scheduling (warmup, cosine, linear decay)
- Distributed training — ZeRO stages 1/2/3, tensor parallelism with communicators, pipeline parallelism (1F1B, GPipe, ZeroBubble)

#### Model loading

- SafeTensors — zero-copy memory-mapped loading
- GGUF — full format support with block-quantized tensors; format auto-detection

#### Backends

- CPU — SIMD kernels (AVX2, NEON)
- CUDA — PTX kernels, Flash Attention v2/v3, fused ops (CUDA 12.x)
- WebGPU — WGSL shaders, cross-platform GPU support

---

[0.2.0]: https://github.com/ml-rust/boostr/releases/tag/v0.2.0
[0.1.0]: https://github.com/ml-rust/boostr/releases/tag/v0.1.0
