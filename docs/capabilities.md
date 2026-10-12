# Capabilities

This page lists what boostr implements. Each entry names the module that holds it.
For the feature flags, see the [README](../README.md#feature-flags).

## Who It Is For

- **LLM trainers**: ZeRO stages 1, 2, and 3, tensor and pipeline parallelism, mixed precision, and fused optimizers.
- **Inference engineers**: Flash attention, paged KV cache, continuous batching, speculative decoding, and prefix caching.
- **Quantization researchers**: 27 block-quantized formats, a `QuantTensor` type, and per-backend dequantization and quantized-matmul kernels.
- **Architecture researchers**: LLaMA, Mamba2, hybrid transformer and SSM models, and encoder models. Custom architectures use the same extension traits.
- **WebAssembly and browser developers**: The WebGPU backend needs no CUDA. `Gguf::from_bytes` loads a model from an in-memory buffer.

## Quantization

`src/quant/` holds the `QuantFormat` enum and the `QuantTensor` type for block-quantized data.

- **Simple blocks**: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`
- **K-quants**: `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `Q8K`
- **IQ formats**: `IQ1S`, `IQ1M`, `IQ2XXS`, `IQ2XS`, `IQ2S`, `IQ3XXS`, `IQ3S`, `IQ4NL`, `IQ4XS`
- **Ternary**: `TQ1_0`, `TQ2_0`
- **Low-bit and other**: `Q1_0`, `Q2_0`, `PQ2_0`, `PTQ1_0`
- **Kernels**: SIMD on CPU, PTX on CUDA, and WGSL on WebGPU.

## Attention

- **Flash attention v2 and v3**: `ops/traits/attention/flash.rs`, with fused QKV projection (`fused_qkv.rs`).
- **Multi-head latent attention (MLA)**: compressed KV cache (`mla.rs`).
- **Grouped query attention (GQA)** and multi-head variants (`multi_head.rs`).
- **Paged attention**: `paged_attention.rs` and `inference/kv_cache/paged.rs`.
- **Variable-length attention**: ragged batches (`varlen_attention.rs`).
- **Sliding window** through the `window_size` argument of `flash_attention_fwd`.
- **ALiBi bias**: `flash_alibi.rs`.
- **Prefix caching** for context reuse (`inference/prefix_cache/`).

## Position Encodings

- **RoPE**: split-half, interleaved, and partial variants (`ops/impl_generic/attention/rope/`).
- **ALiBi**: slopes follow the ALiBi paper, including head counts that are not a power of two.
- **YaRN** and **LongRoPE** for length extrapolation.

## Model Architectures

`src/model/` holds these architectures.

- **LLaMA**: standard, and tensor-parallel through `LlamaTp`.
- **Llama-family variant blocks**: `UniversalConfig.block` (`BlockConfig`) adds LayerNorm, GELU MLPs, parallel residuals, Gemma 2 sandwich norms, partial and interleaved RoPE, logit softcaps, and per-layer sliding windows on the LLaMA path. `HuggingFaceConfig::to_universal` maps Gemma, Gemma 2, StarCoder2, GPT-NeoX, Falcon, DBRX, and Cohere configs onto it.
- **Mixture of experts** on the LLaMA path: `LlamaFfn` routes `[B, S, hidden]` input.
- **Mamba2**: state space models with SSD kernels.
- **Hybrid**: mixed transformer and SSM models.
- **Qwen3.5**: Gated DeltaNet and attention hybrid.
- **Qwen3-VL**: vision encoder for image input.
- **Encoders**: BERT, XLM-RoBERTa, Nomic BERT, Jina BERT v2 and v3, Qwen3 embedding, and Gemma embedding (`model/encoder/`), with varlen batching and an embedding pipeline.
- **Extensible**: custom models implement the same traits.

`format::restructure_hf_checkpoint` maps a Hugging Face checkpoint into the LLaMA tensor layout. It splits fused QKV and gate-up projections, stacks MoE experts, and adds Gemma's norm offset.

## Neural Network Modules

`src/nn/` holds these modules.

- **Embedding**, **RmsNorm**, **LayerNorm**, and **GroupNorm**
- **Conv1d** and **Conv2d**
- **LoRA linear** layers (`maybe_lora.rs`)
- **Quantized embedding** (`quant_embedding.rs`)

## Inference

`src/inference/` holds the inference infrastructure.

- **Paged KV cache** with a block allocator (`memory/`).
- **Request scheduler** with continuous batching (`scheduler/`).
- **Prefix caching** for prompt reuse (`prefix_cache/`).
- **Speculative decoding** with adaptive draft depth and verification kernels (`speculative/`).
- **Flash decoding** for single-token decode on CUDA, selected when `S_q == 1`.

## Training

- **Optimizers**: AdamW, AdaGrad, Lamb, and SGD (`src/optimizer/`).
- **Gradient clipping** by norm and by value.
- **Mixed precision** (AMP) with automatic loss scaling (`trainer/amp.rs`, `optimizer/grad_scaler.rs`).
- **Gradient accumulation** and checkpointing, with an asynchronous checkpoint writer.
- **Learning rate schedules**: constant, linear warmup, cosine annealing, Warmup-Stable-Decay, and step decay.
- **Distributed training** (`src/distributed/`):
  - ZeRO stages 1, 2, and 3 for parameter, gradient, and optimizer sharding
  - Tensor parallelism with communicators
  - Pipeline parallelism with 1F1B, GPipe, and ZeroBubble schedules

## Model Loading

- **SafeTensors**: single-file and sharded checkpoints (`format/safetensors_loader.rs`).
- **GGUF**: block-quantized and f32 tensors, metadata, and an in-memory loader (`format/gguf/`).
- **GGUF memory mapping**: `Gguf::open_with_mmap(path, true)` maps the file. `Gguf::release_tensor` drops a tensor's mapped pages after a copy.
- **PyTorch `.pt`** state dicts (`format/torch_pt/`).
- **TCF container**: stores BLAKE3 digests (`src/tcf/`).

## Backends

- **CPU**: SIMD kernels for AVX2 and NEON, and native ops.
- **CUDA**: PTX kernels, FlashAttention, and fused ops. Requires CUDA 12.8+ or 13.x.
- **WebGPU**: WGSL shaders, including sampling and grammar DFA kernels.

boostr uses no cuBLAS, cuDNN, or MKL. Every kernel is native.

## Bundled Tokenizer Vocabularies

The `vocab-*` features each enable one bundled [splintr](https://github.com/ml-rust/splintr) vocabulary:

`vocab-cl100k`, `vocab-o200k`, `vocab-gpt-oss`, `vocab-phi4`, `vocab-olmo2`, `vocab-llama3`, `vocab-llama2`, `vocab-deepseek`, `vocab-qwen`, `vocab-glm`, `vocab-gemma2`, `vocab-gemma3`, `vocab-gemma4`, `vocab-kimi`, `vocab-mistral`, and `vocab-modernbert`.

None is on by default. `vocabs` enables all of them.

## Audio

Audio support is gated per model. `audio` provides the tensor-level signal ops: STFT, iSTFT, and mel. Each model gate implies `audio`.

- **Speech recognition**: Whisper.
- **Speech synthesis**: Kokoro, VITS (MMS-TTS), and VoxCPM2.
- **Codec**: NeuCodec.
- **Voice activity detection**: Silero VAD.
- **Audio and text towers**: CLAP.
- **Part-of-speech tagging**: the spaCy `en_core_web_sm` CNN tagger (`spacy-tagger`, text only).

boostr holds tensor code only. Audio decoding, resampling, grapheme-to-phoneme, voices, and the TTS, ASR, and VAD pipelines live in the [voxr](https://github.com/ml-rust/voxr) crate.

## Design Notes

- **Fused ops**: fused QKV projection, a fused distributed optimizer (`distributed/training/fused_optimizer.rs`), and fused CUDA attention kernels.
- **Per-backend kernels**: dequantization and quantized matmul use SIMD, PTX, or WGSL kernels.
- **Memory**: paged KV cache, prefix caching, and gradient checkpointing.
- **Distributed training**: ZeRO stages and tensor and pipeline parallelism.
- **Memory-mapped GGUF**: see [Model Loading](#model-loading).
