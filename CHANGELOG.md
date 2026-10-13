# Changelog

All notable changes to boostr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
boostr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **`format::peft`** — reads a PEFT LoRA adapter. `PeftAdapter::load` takes `adapter_config.json` and `adapter_model.safetensors` and returns per-module `lora_A`/`lora_B` pairs with their scaling. Scaling follows `use_rslora`, `rank_pattern` and `alpha_pattern`. Keys under `model.layers.N` and `model.language_model.layers.N` normalize to `model.layers.N`. The load refuses DoRA, `fan_in_fan_out`, `lora_bias`, `modules_to_save`, `.bin` weights, unpaired factors and rank mismatches.
- **`Qwen35Model::forward_qwen35_gather`** — runs the token forward over the caches you pass and returns the normed hidden rows at chosen flat positions `b * seq + t`, with no `lm_head`. A right-padded batch gives each sequence the rows it gets alone, up to its last real token. A bad gather index is an error that names the index, and it returns before any cache is written.
- **`Qwen35Model::candidate_logits`** — F32 logits of `K` candidate tokens per hidden row. It reads only those `lm_head` rows and works for dense, tied, Hadamard-rotated and block-quantized heads. A LoRA-adapted head is an error.
- **`MaybeRotatedLinear::row_logits_f32`** — the projection's outputs at chosen weight rows per input row, in F32. A block-quantized weight dequantizes only those rows. An AWQ/GPTQ weight has no row gather, so it computes the full output for the given input rows and picks the chosen columns.
- **`inference::decision::run`** — answers decision questions over one shared prefix and many rows. `DecisionPlan` holds the prefix tokens, each row's suffix tokens and the positions to read. `run_decision` runs the prefix once, packs the rows longest first into passes of right-padded lanes, and returns the gathered rows in plan order. A `DecisionHead` scores them. One device softmax applies each question's temperature, and the answers come from one read-back. An isolated Score question takes one `[no, yes]` row per level. `prefill_prefix` builds a `PrefixState` that a caller can keep and reuse for later requests with the same prefix. A gather at a padded or missing position, a reused prefix of the wrong length, and a row past the model's positions are errors that name the value.
- **`DecisionBackbone` for `Qwen35Model`** — the lanes of a pass attend to the one cached prefix without copying its keys and values into each lane. Each lane starts from a copy of the prefix's GDN state.
- **`model::decision::LabelTokenHead`** — the System One readout of decision models whose options carry one-token labels, such as Jebadiah and the Mapika decider. Each scoring row is a `LabelTokenItem`: the gathered answer slot and the label token of each option. All rows go through one `candidate_logits` call, and the valid logits are selected on the device. An isolated Score question takes one item per level with labels `[no, yes]`. A slot past the gathered rows, an empty or repeated label set, and a label outside the vocabulary are errors that name the item and the value.
- **`DecisionBackbone::vocab_size`** — the number of token ids the backbone scores, so a head can check its candidate ids before any device work.
- **`Qwen35AttentionBlock::forward_shared_prefix`** — attention for lanes that share one batch-1 prefix cache. Each lane attends to every prefix key and to its own suffix under one softmax. An empty prefix runs the ordinary forward.
- **`inference::decision::plan_from_rows`** — builds a `DecisionPlan` from full prompt rows and their answer slots. The prefix is the rows' longest common token prefix, cut back so it ends at or before the earliest slot and leaves every row at least one token. A single row, or a prefix shorter than `min_prefix`, gives no prefix. Repeated slots share one gather, and `RowsPlan::slot_gathers` gives each slot's gather index. A missing row, an empty row and a slot past its row's end are errors that name the slot and the value.
- **`merge_requests` and `split_result`** — run several requests that have no prefix as one. `merge_requests` joins their rows, gathers, head items and questions in request order and records where each request sits. Requests with one temperature policy keep it, and mixed policies keep each question's own temperature. `split_result` cuts the result back into one result per request. A head item that names gathers by index implements `GatherIndexed`, as `LabelTokenItem` does. A request with its own prefix is refused.
- **`DecisionBackbone::lane_bytes`** — the cache bytes one lane of a pass holds, so a server can bound lanes by memory. `RunnerConfig::max_lane_bytes` caps the lanes of each pass by that budget. A pass whose single lane needs more than the budget is an error that names both sizes.
- **`CancelToken`** — a stop flag a caller sets from any thread. `RunnerConfig::cancel` makes a decision run read it before the prefix, before each pass and before scoring. A cancelled run returns the new `Error::Cancelled`, which names the point the run stopped at.
- **`fit_temperature_isolated`** — fits one temperature to isolated Score questions. Each `IsolatedSample` holds the `[no, yes]` logits of every level and the gold level or a target distribution. The fit minimises the likelihood loss of the combined level distribution the answer stage forms. That loss can have more than one local minimum, so the fit scans the temperature range, refines the best bracket by golden-section search, and polishes with Newton steps.
- **`fit_temperature_mixed`** — fits one temperature to Score answers read both ways: listed `CalibrationSample`s and isolated `IsolatedSample`s, by their pooled mean likelihood loss. With one side empty it is `fit_temperature` or `fit_temperature_isolated`. `evaluate_mixed` gives the pooled loss and its derivatives.
- **`TemperaturePolicy::PerQuestion`** — one temperature per question, in question order. `TemperaturePolicy::for_question` resolves the temperature of any policy for one question, including the level rows of an isolated Score question.
- **`QuestionMeta::scoring_rows`** — the scoring rows a question takes.
- **`Qwen35Model::derive_with_adapter`** — a second model over a loaded base's weights, with a PEFT LoRA adapter attached unmerged. No weight is read again or copied: dense, block-quantized and Hadamard sign tensors are shared with the base. The base is not modified and stays usable, so one base serves several adapters at once. A base that already carries an adapter is refused, since adapters do not stack. The attach errors are those of `from_varbuilder_with_adapter`. `LoadedModel::derive_with_adapter` does the same for a loaded model and refuses every model type but `qwen35`.
- **`LoadedModel::adapter_refusal`** and **`LoadedModel::adapted_modules`** — why a loaded model takes no PEFT adapter (the error `derive_with_adapter` returns), checked without an adapter in hand, and the modules an attached adapter targets.
- **`alias` on the `qwen35` modules** — `Embedding`, `QuantEmbedding`, `MaybeQuantEmbedding`, `RotatedEmbedding`, `MaybeRotatedEmbedding`, `RotatedLinear`, `MaybeRotatedLinear`, `RotatedMlp`, `QkNorm`, `GdnBlock` and `Qwen35AttentionBlock` gain `alias`: a duplicate over the same storage that keeps each weight's `TensorId`.

### Changed

- **`restructure_hf_checkpoint`** — the client must implement `TypeConversionOps`. Generic callers add the bound.
- **`RunnerConfig`** — gains `max_lane_bytes` and `cancel`, and is no longer `Copy`. A struct literal must name the new fields or use `..RunnerConfig::default()`.
- **`DecisionBackbone`** — every implementation must provide `lane_bytes`.
- **`run_decision`** — checks the pass layout before it runs the prefix, so a bad config fails before any device work.
- **`Qwen35AttentionBlock::forward`** — refuses a cache whose lanes hold different numbers of keys. The shared append writes every lane at one position, so such a cache produced wrong keys before. The error names each lane's length.

### Fixed

- **Chunked GDN prefill precision** — chunked prefill of Qwen3.5-style gated delta net layers lost precision on prompts longer than a few dozen tokens. The error grew through the layers, so the logits drifted from the reference model. The UT transform of each chunk now runs as forward substitution through numr's batched triangular solve, in one call for all chunks and heads. F16 and BF16 inputs solve in F32.
- **Qwen3.5 checkpoint edits in half precision** — `restructure_hf_checkpoint` computes the norm offset `1 + w` and `a_neg_exp = -exp(A_log)` of a Qwen3.5 checkpoint in F32, as the llama.cpp converter does. An F16 or BF16 tensor is cast to F32 on its device, edited, and cast back to its own dtype, so it rounds once. An F32 checkpoint takes the edits in F32 with no cast.

---

## [0.8.0] - 2026-10-12

### Added

- **`BlockConfig`** — `UniversalConfig.block` describes a non-Llama block on the Llama path: LayerNorm, GELU and plain MLPs, parallel residuals, Gemma 2 sandwich norms, partial and interleaved RoPE, attention and final logit softcaps, `logit_scale`, `clip_qkv`, a scaled ALiBi bias and per-layer sliding windows. `HuggingFaceConfig::to_universal` fills it for Gemma, Gemma 2, StarCoder2, GPT-NeoX, Falcon, DBRX and Cohere.
- **`format::restructure_hf_checkpoint`** — puts a Hugging Face checkpoint into the Llama tensor layout. It splits fused QKV and gate-up projections, stacks MoE experts, adds Gemma's norm offset, folds the query scale and reorders partial-RoPE rows.
- **`RoPE::precompute_freqs_partial`** — RoPE over the leading `rotary_dim` dims of each head.
- **`LoadedModel::paged_attention_refusal`** — names a model feature the paged kernel does not apply.
- **SafeTensors `U8`** — load and save. GPT-NeoX stores its mask buffer as `U8`.

### Changed

- **`HuggingFaceConfig::to_universal`** returns `Result`. An unsupported activation, an Lp expert norm with p other than 1, or Cohere `use_qk_norm: true` is an error.
- **`ModelClient`** requires `SpecialFunctions`. Exact-GELU gated MLPs need it.
- **New public fields** — `UniversalConfig.block` and `MoeConfig.norm_topk_prob`. A struct literal must set them.
- **`HuggingFaceConfig`** — `rope_theta` is `Option<f32>` and `tie_word_embeddings` is `Option<bool>`. It gains fields for the new architectures.
- **`build_block_from_varbuilder`** takes the layer index.
- **splintr** — requires `0.22`.

### Fixed

- **ALiBi slopes** — every backend used `2^(-8h/H)`, so head 0 got slope 1. The slopes now follow the ALiBi paper and `transformers`, including head counts that are not a power of two.
- **MoE on the Llama path** — `LlamaFfn` flattens `[B, S, hidden]` before routing. Routing refused 3-D input, so no Mixtral, Qwen2-MoE or DBRX model ran.
- **Qwen2-MoE routing** — `norm_topk_prob: false` keeps the full-softmax weights. The shared expert takes its sigmoid gate, and experts take the configured activation.
- **DBRX expert names** — the fused `w1`, `v1`, `w2` tensors map to gate, up and down. The old map swapped gate and up and matched no real tensor name.
- **Qwen2 sliding window** — `use_sliding_window: false` turns the window off. The window applied to every layer before.
- **LongRoPE** — the cache uses `short_factor` below `original_max_position_embeddings` and `long_factor` from there on. A cache longer than the original used `long_factor` at every position. The attention factor follows the config lengths, as `transformers` computes it. A top-level `original_max_position_embeddings` (Phi-3) is read.
- **Falcon names** — `post_attention_layernorm` maps. Falcon-RW failed to load before.
- **Paged attention** — the paged path refuses ALiBi, softcaps and a sliding window shorter than the sequence. It ignored them before.
- **transformers v5 configs** — `rope_parameters.rope_theta` is read.

---

## [0.7.0] - 2026-10-11

### Added

- **WebGPU sampling and grammar kernels** — `SamplingOps` and `GrammarDfaOps` run on `WgpuClient`:
  sampling penalties, logits-to-token and grammar DFA masking. Parity tests hold them to the CPU.
- **GGUF name table** — `format::gguf::hf_to_gguf_name` inverts `gguf_to_hf_name`. Both read one
  shared table.
- **GGUF RoPE row order** — `format::gguf::rope_rows` holds `RopeRowOrder`, `ROPE_PERMUTED_ARCHS`,
  `rope_row_heads`, `rope_row_heads_in`, `rope_row_sources` and `reorder_rope_rows`. They convert
  llama.cpp's pair-ordered query and key rows to and from the HuggingFace order.
- **GGUF loads in HF order** — `Gguf::load_tensor_f32_hf_order` and
  `Gguf::load_tensor_quantized_hf_order` load a tensor with HuggingFace row order.
  `VarMap::from_gguf_reader` builds a `VarMap` from a GGUF the caller already opened, such as one
  from `Gguf::from_bytes`.
- **GGUF memory release** — `Gguf::is_memory_mapped`, `Gguf::release_tensor` and `Gguf::tensor_slice`.
  A loader releases a memory-mapped tensor's pages once it copies the tensor out. A mapped file then
  never adds a second copy of the model to resident memory.
- **`rope_freqs` RoPE scaling** — the per-dimension divisors of a GGUF `rope_freqs.weight` tensor,
  carried in `long_factor`. They reproduce llama3 scaling.
- **`SafeTensorsLoader::from_bytes`** — loads a SafeTensors image from memory, with no model directory.
- **`LoadedModel::tp_communicator`** — the tensor-parallel communicator of a `LlamaTp` shard. Every
  other variant returns `None`.

### Changed

- **Dependencies** — boostr requires numr 0.12. Its WebGPU backend copies sub-word and offset views.
- **`VarMap::from_gguf`** — a llama-architecture file gets its query and key rows back in
  HuggingFace order. The file opens memory-mapped, and each tensor's pages leave resident memory
  after the load.
- **SafeTensors on F32-only backends** — F16 and BF16 payloads decode to F32 on the host when the
  backend does not compute those dtypes.
- **`LlamaTp::forward_with_kv_cache`** — gathers the vocabulary shards, so every rank returns full
  logits.
- **CUDA grammar kernel** — reads the F32 table layout every backend shares, and bounds-checks the
  DFA state.
- **WebGPU K-quants** — the Q4_K and Q6_K dequant and quant matmul shaders share one block decoder.
  CPU dequant writes straight into the output tensor, so it holds one F32 copy instead of two.

### Fixed

- **`gather_from_ranks`** — concatenates the shards along the gather dim. Before, any dim other than 0
  interleaved the shards. An out-of-range dim returns an error.
- **CUDA top-k** — the threshold keeps at most `top_k` tokens. Before, it kept more.
- **WebGPU RoPE** — a narrowed cos/sin cache copies to a dense buffer first. Before, a view at a
  position offset read the wrong rows.

---

## [0.6.0] - 2026-10-11

### Added

- **`SequenceScheduler::schedule_with_limit`** — `schedule` with a cap on batch growth. Waiting and
  preempted sequences join the running set only while it holds fewer than the cap. Running sequences
  keep decoding above it, and `max_batch_size` still applies.
- **Kokoro on every backend** — the full synthesis path, with the harmonic source and the
  per-stage noise conditioning, runs on any runtime: CPU, CUDA and WebGPU.
- **`KokoroClient<R>`** — the client bound of the Kokoro synthesis path, exported from
  `model::audio::kokoro`. Every numr backend client implements it.
- **`KokoroSynthOptions`** — `speed` and `seed: Option<u64>` for
  `KokoroModelV2::synthesize` and `forward_to_spectrogram`. One seed on one backend
  always gives the same samples. `None` keeps the unseeded draws.
- **Tests** — `kokoro_cuda` holds CUDA to the CPU port of the same checkpoint and checks
  that a seed reproduces the waveform. `vits_synthesis_cuda` runs the VITS reference
  parity on CUDA.

### Changed

- **Dependencies** — boostr requires numr 0.11. CUDA `scatter_reduce` there is deterministic, so Kokoro's
  inverse STFT gives the same samples on every run.
- **`KokoroModelV2::forward_to_spectrogram`** — runs the decoder's noise-conditioned
  path and takes `&KokoroSynthOptions` in place of `speed: f32`.
- **`Decoder::forward`** (Kokoro) — runs the generator's noise-conditioned path and
  takes a trailing `seed: Option<u64>`.
- **`IStftNetGenerator::forward`** — runs the full vocoder with the harmonic source and
  takes a trailing `seed: Option<u64>`. A generator without `noise_convs` / `noise_res`
  runs the upsampling stages alone, as before.
- **`SineGen::forward` and `SourceModuleHnNSF::forward`** — take a trailing
  `seed: Option<u64>`. The initial phase draws from `seed` and the noise from
  `seed + 1`.
- **`SineGen` on WebGPU** — the phase sums in the input dtype, since WGSL has no 64-bit
  float. CPU and CUDA still sum in f64.
- **`IStftNetGenerator::has_noise_path`** — available on every runtime.

### Removed

- **CPU-only Kokoro methods** — use the runtime-neutral method in each case:

  | Removed | Use |
  | --- | --- |
  | `KokoroModelV2::synthesize_cpu` | `KokoroModelV2::synthesize` |
  | `KokoroModelV2::forward_to_spectrogram_cpu` | `KokoroModelV2::forward_to_spectrogram` |
  | `Decoder::forward_cpu_full` | `Decoder::forward` |
  | `IStftNetGenerator::forward_cpu_full` | `IStftNetGenerator::forward` |
  | `IStftNetGenerator::forward_cpu_with_spectrum` | `IStftNetGenerator::forward_with_spectrum` |
  | `IStftNetGenerator::excitation_cpu` | `IStftNetGenerator::excitation` |
  | `IStftNetGenerator::harmonic_excitation_spec_cpu` | `IStftNetGenerator::harmonic_excitation_spec` |
  | `IStftNetGenerator::source_spectrum_cpu` | `IStftNetGenerator::source_spectrum` |

- **Kokoro's generic forward without the source** — the old generic
  `forward_to_spectrogram`, `Decoder::forward` and `IStftNetGenerator::forward` skipped
  the harmonic source and gave degraded audio. The methods of those names now run the
  full path.

---

## [0.5.0] - 2026-10-10

### ⚠️ Breaking changes

- **Dependencies** — boostr requires numr 0.10.
- **MMA kernel modules** — the feature-major and token-major MMA kernels now load
  from one CUDA module per quant format, named `quant_mmq_mma_<format>` (for example
  `quant_mmq_mma_q4_k`). `quant::cuda::kernels::QUANT_MMQ_MMA_MODULE` is removed.
  `QUANT_MMQ_MMA_MODULES` lists every module, and `quant_mmq_mma_module(infix)` returns
  the module for a format's kernel-name infix. Kernel names do not change.

### Changed

- **CUDA build** — the MMA kernels compile as 22 separate units, one per quant format,
  instead of one. The build script compiles them in parallel, and each nvcc run needs
  a fraction of the memory the single unit needed. A multi-arch build
  (`NUMR_CUDA_ARCH=all`) no longer spends hours on one file.
- **cudarc** — default features are off. boostr enables the driver API and module
  loading only, so cuBLAS, cuBLASLt, cuRAND and the CUDA runtime library are not built.

### Fixed

- **Stale CUDA kernels** — a kernel in a subdirectory of a kernel root was not rebuilt
  after a change to a header in a parent directory. The up-to-date check now scans the
  whole kernel root.

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
