# Changelog

All notable changes to boostr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
boostr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-10-09

### ⚠️ Breaking changes

- **Minimum Rust** — 1.95.
- **Dependencies** — boostr requires numr 0.9. With the `wgpu` feature, the WebGPU
  types that `boostr::runtime::wgpu` re-exports from numr come from wgpu 30. A caller
  that passes or receives these types must depend on wgpu 30.

### Changed

- **Dependencies** — wgpu 30, zip 9 and serde-saphyr 1.3.

### Fixed

- **WebGPU** — a failed buffer mapping during the fused FP8 unscale-and-clip readback
  returns an error instead of panicking.

---

## [0.3.0] - 2026-10-09

### ⚠️ Breaking changes

- **Dependencies** — boostr requires numr 0.8 and splintr 0.21. The minimum Rust
  version is 1.89.
- **Tokenizers move to splintr.** `GgufTokenizer` and `WhisperTokenizer` are removed.
  `format::extract_gguf_vocab` lifts `tokenizer.ggml.*` metadata into a
  `splintr::GgufVocab`, and splintr builds the tokenizer from it.
  `EmbeddingPipeline<R, T: Tokenize>` becomes `EmbeddingPipeline<R>` and holds a
  `splintr::AnyTokenizer`.
- **Encoder** — `HiddenAct` is removed. Encoder configs use `nn::Activation`.
- **MoE** — `MoEActivation::GeLU` becomes `MoEActivation::Gelu` and now means the
  exact erf form. `MoEActivation::GeluTanh` is the tanh approximation.
- **Audio is gated per architecture.** `model::audio` models sit behind the `whisper`,
  `kokoro`, `neucodec`, `voxcpm`, `silero-vad`, `vits` and `clap` features. `audio`
  gates only the tensor-level signal ops (`stft`, `istft`, `mel`).
- **Sample-level audio moves to voxr.** boostr no longer holds `g2p`, `wav_encode`,
  `tts_bundle`, `fft` or `reflection_pad`. WAV and codec I/O, resampling, G2P, voices
  and the TTS, ASR and VAD pipelines live in the `voxr` crate. `WhisperBundle` is
  replaced by `WhisperCheckpoint`; voxr builds the bundle with its tokenizer.
- **Fallible allocation** — model, ops, KV cache, SSM and data APIs that allocate
  tensors return `Result` instead of panicking.
- **LoRA** — `LoraLinear` implements `Module`, gains merge and a trainable flag, and
  MoE `Expert` projections accept LoRA adapters. Code that builds these types by
  field changes.
- **Quantization** — the `prism` low-bit quant family is renamed `lowbit`.

### Added

- **Model gates** — `whisper`, `kokoro`, `neucodec`, `voxcpm`, `silero-vad`, `vits`
  and `clap` model features, plus `spacy-tagger` and the `vocab-*` / `vocabs`
  features that forward splintr's bundled vocabularies.
- **Audio models**
  - VITS / MMS-TTS: text encoder, duration predictors, flows and HiFi-GAN, with
    end-to-end synthesis.
  - VoxCPM2: AudioVAE, MiniCPM4 `base_lm`, local DiT, FSQ layer and the end-to-end
    orchestrator. It loads from SafeTensors, GGUF and TCF. It supports batched and
    continuous-batched generation, streaming windowed decode, zero-shot rendering,
    LoRA / QLoRA fine-tuning and CUDA graph replay.
  - NeuCodec: semantic and acoustic encoders, FSQ quantizer and decoder.
  - CLAP: HTSAT audio tower and RoBERTa text tower.
  - Silero VAD with a streaming LSTM step.
  - Whisper: language detection, cross-attention capture and the word-timestamp
    alignment matrix.
- **spaCy tagger** — `model::text::spacy_tagger`, the `tok2vec` + `tagger` network of
  `en_core_web_sm`.
- **Qwen3.5** — `Qwen35Model`: Gated DeltaNet and full-attention layers with
  interleaved MRoPE, GGUF loading, and the Qwen3-VL vision tower for image input.
- **Encoder** — Qwen3-Embedding and Jina-BERT v2/v3 support (ALiBi, QK-norm, RoPE
  base), and `EmbeddingPipeline` over any splintr tokenizer.
- **Mamba** — Mamba1 and Mamba3 variants next to Mamba2, with `init()` for training
  from scratch.
- **Inference**
  - Multi-lane batched decode: `LaneDecodeSession`, a lane driver, and per-lane KV
    ranges and GDN state.
  - Per-row KV ends and per-row RoPE positions in paged and flash decode.
  - CUDA graph decode for Qwen3.5 (`Qwen35DecodeGraph`) with one replay serving lanes
    of different lengths.
  - `SamplingScratch`: allocation-free sampling into caller-owned buffers.
  - `generate_greedy` for autoregressive generation.
- **Attention and KV cache** — FP8 and INT4 KV-cache flash attention, per-tensor and
  per-head FP8 KV quantization, `copy_blocks` / `swap_blocks`, multi-layer batched KV
  update, fused ALiBi flash forward, sliding-window flash backward, per-row `kv_start`
  padding, and a token-major output layout.
- **Quantization**
  - Low-bit formats Q1_0, Q2_0, PQ2_0 and PTQ1_0 with CPU, CUDA GEMV and MMQ kernels.
  - CPU quantizers for the K-quants, IQ4_NL and IQ4_XS, checked against llama.cpp.
  - Importance-matrix (imatrix) weighted quantization and AWQ-style smoothing.
  - Feature-major tensor-core MMQ kernels for the GGUF formats, with a per-device
    split-K or tile-parallel schedule.
  - Hadamard-rotated linear and embedding layers and the GGUF rotation contract.
- **TCF** — `boostr::tcf`, the tensor contract file container, and `TcfLoader`, with
  CPU, CUDA and WebGPU kernels for its native quantized encodings.
- **nn** — LoRA and QLoRA adapters with named loading, conditional flow-matching loss,
  finite scalar quantizer, tanh-approximate and QuickGELU activations, LongRoPE and
  YaRN scaling, and seeded weight initialization.
- **CUDA build** — kernels compile to multi-arch fatbins embedded in the binary.
  `BOOSTR_CUDA_ARCH` (falling back to `NUMR_CUDA_ARCH`) selects the architectures.

### Fixed

- **Paged attention** — prefill applies the causal mask. Block tables narrower than
  `seq_len_k` are rejected.
- **Encoder** — batches pad with the model's pad token id. Special-token wrapping
  truncates content before it adds `[CLS]` / `[SEP]`. GGUF
  `tokenizer.ggml.precompiled_charsmap` is now read.
- **Activations** — Whisper and the Qwen3-VL merger use exact GELU.
- **Kokoro** — vocoder, source and STFT padding match the reference.
- **Mamba2** — a shared causal conv1d removes conv window drift, and `dt_bias` applies
  inside softplus.
- **CUDA attention** — bottom-right causal alignment in varlen, paged, MQA/GQA and
  KV-cache flash kernels. GQA/MQA out-of-bounds writes and dQ accumulation in flash
  backward. Rounding of per-element atomic adds in the backward kernels.
- **Quantization** — IQ1_M type id, IQ4_XS scale layout, Q4/Q5/Q5_K nibble indexing,
  TQ1_0/TQ2_0 scale offset and trit order, and CUDA GGUF dequant for four formats.
- **Optimizers** — LAMB, AdaGrad and SGD keep F32 master weights. Gradient clipping
  covers only the optimized parameters and accepts strided gradients.
- **Autograd** — `Embedding::forward`, MoE routing, `repeat_kv` and `Conv2d::forward`
  keep their gradient edges.
- **nn** — seeded Kaiming / Xavier init reads `fan_in` from the correct dimension.
- **Format** — SafeTensors writes each dtype at its native byte width.

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
