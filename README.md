<div align="center">

# boostr

<h3>ML framework in Rust. Write once. Run on any backend.</h3>

<p>
Attention, quantization, model architectures, training, and inference built on <a href="https://github.com/ml-rust/numr">numr</a>, for CPU, CUDA, and WebGPU.
</p>

<p>
  <a href="https://docs.rs/boostr"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/boostr"><strong>Crate</strong></a>
  ·
  <a href="#what-works"><strong>What works</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="#feature-flags"><strong>Feature Flags</strong></a>
  ·
  <a href="#testing"><strong>Testing</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://github.com/ml-rust/boostr/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ml-rust/boostr/ci.yml?branch=main&label=ci" alt="CI status">
  </a>
  <a href="https://crates.io/crates/boostr">
    <img src="https://img.shields.io/crates/v/boostr" alt="crates.io">
  </a>
  <a href="https://docs.rs/boostr">
    <img src="https://img.shields.io/docsrs/boostr" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/boostr" alt="License">
  </a>
  <a href="https://github.com/ml-rust/boostr/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/boostr?style=social" alt="GitHub stars">
  </a>
</p>

</div>

boostr extends [numr](https://github.com/ml-rust/numr) with ML operations and adds no tensor type or runtime wrapper.
Its `Tensor`, `Runtime`, and client types come from numr.
ML operations attach to numr's clients as extension traits.

## What works

| Backend | Feature | Needs | Limits |
| --- | --- | --- | --- |
| CPU | `cpu`, on by default | Nothing | None |
| NVIDIA through CUDA | `cuda` | CUDA toolkit 12.8 or newer, `nvcc` on `PATH`, compute capability 7.5 (Turing) or newer | Flash attention v3 needs compute capability 9.0 (Hopper) |
| WebGPU | `wgpu` | Vulkan, Metal, or DX12 driver | Computes in F32 only |

| Area | Contents | Feature |
| --- | --- | --- |
| Text models | LLaMA, Mamba2, hybrid transformer and SSM, Qwen3.5, Qwen3-VL vision encoder, text encoders | None |
| Audio models | Whisper, Kokoro, VITS, VoxCPM2, NeuCodec, Silero VAD, CLAP | One per model, listed in [Feature Flags](#feature-flags) |
| Text tagging | spaCy `en_core_web_sm` part-of-speech tagger | `spacy-tagger` |
| Image models | `SrVggCompact`, `RrdbNet` | `super-resolution` |
| Quantization | Simple blocks, K-quants, IQ, ternary, and low-bit formats | None |

CI has no GPU. Some ops run on some backends only. [docs/capabilities.md](docs/capabilities.md) lists every capability, and [docs/architecture.md](docs/architecture.md) holds the layout and design rules.

## Quick Start

### Install

Needs Rust 1.95 or newer.

```toml
[dependencies]
boostr = "0.11"                                    # CPU
# boostr = { version = "0.11", features = ["cuda"] }
# boostr = { version = "0.11", features = ["wgpu"] }
```

### Basic Usage

Run flash attention on the CPU backend:

```rust
use boostr::ops::traits::attention::{AttnOutLayout, FlashAttentionOps};
use boostr::{CpuClient, CpuDevice, DType};
use boostr::RandomOps;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = CpuClient::new(CpuDevice::new());

    // Random tensors via numr's RandomOps: [batch, heads, seq, dim]
    let q = client.randn(&[1, 8, 32, 64], DType::F32)?;
    let k = client.randn(&[1, 8, 32, 64], DType::F32)?;
    let v = client.randn(&[1, 8, 32, 64], DType::F32)?;

    let (output, _lse) = client.flash_attention_fwd(
        &q,
        &k,
        &v,
        8,                         // num_heads
        8,                         // num_kv_heads
        64,                        // head_dim
        true,                      // causal
        0,                         // window_size (0 = no sliding window)
        None,                      // kv_seq_len
        None,                      // kv_start
        AttnOutLayout::HeadMajor,  // output layout: [B, H, S_q, D]
    )?;

    Ok(())
}
```

### Load a GGUF Model

```rust
use boostr::format::Gguf;
use boostr::{CpuDevice, CpuRuntime};

// Open a GGUF file with buffered reads
let mut gguf = Gguf::open("model.gguf")?;
let device = CpuDevice::new();

// Load quantized tensors as QuantTensor and the others as f32
for name in gguf.tensor_names().map(|s| s.to_string()).collect::<Vec<_>>() {
    let info = gguf.tensor_info(&name)?;
    if info.ggml_type.is_quantized() {
        let qt = gguf.load_tensor_quantized::<CpuRuntime>(&name, &device)?;
    } else {
        let t = gguf.load_tensor_f32::<CpuRuntime>(&name, &device)?;
    }
}
```

Use `Gguf::open_with_mmap(path, true)` to memory-map the file. Each loader call releases mapped pages after it copies a tensor.

### Paged KV Cache

```rust
use boostr::inference::PagedKvCache;
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType, Tensor};
use boostr::RandomOps;

let device = CpuDevice::new();
let client = CpuClient::new(device.clone());
let (num_blocks, block_size, num_heads, head_dim) = (16, 16, 8, 64);

let cache = PagedKvCache::<CpuRuntime>::new(
    num_blocks, block_size, num_heads, head_dim, DType::F32, &device,
)?;

// key and value: [num_tokens, num_heads, head_dim]
let k = client.randn(&[4, num_heads, head_dim], DType::F32)?;
let v = client.randn(&[4, num_heads, head_dim], DType::F32)?;

// slot_mapping: [num_tokens], I64. Slot s maps to block s / block_size, offset s % block_size.
let slots = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[4], &device)?;
cache.update(&k, &v, &slots, &client)?;
```

## Build from Source

Copy `.cargo/config.toml.example` to `.cargo/config.toml` first. See [CONTRIBUTING.md](CONTRIBUTING.md#set-up).

```bash
cargo build --release                  # CPU
cargo build --release --features cuda  # CUDA
cargo build --release --features wgpu  # WebGPU
```

- CUDA binaries: `build.rs` compiles kernels for the GPUs on the build host. Set `NUMR_CUDA_ARCH=all` for a binary that runs on another machine.
- boostr kernels only: `BOOSTR_CUDA_ARCH` overrides `NUMR_CUDA_ARCH`.

```bash
NUMR_CUDA_ARCH=all cargo build --release --features cuda
```

## Feature Flags

Backends:

| Feature | Adds |
| --- | --- |
| `cpu` | CPU backend (default) |
| `cuda` | CUDA GPU backend |
| `wgpu` | WebGPU cross-platform GPU backend |
| `nccl` | Multi-GPU through NCCL |
| `distributed` | Distributed inference over nexar. Implies `f16` |

Precision:

| Feature | Adds |
| --- | --- |
| `f16` | Half-precision floats |
| `fp8` | FP8 precision |

Models and signal ops. No model feature implies another:

| Feature | Adds |
| --- | --- |
| `audio` | Tensor-level signal ops (STFT, iSTFT, mel) |
| `whisper` | Whisper speech recognition. Implies `audio` |
| `kokoro` | Kokoro speech synthesis. Implies `audio` |
| `vits` | VITS / MMS-TTS speech synthesis. Implies `audio` |
| `neucodec` | NeuCodec audio codec. Implies `audio` |
| `voxcpm` | VoxCPM2 speech synthesis. Implies `audio` and `f16` |
| `silero-vad` | Silero voice activity detection. Implies `audio` |
| `clap` | CLAP audio and text towers. Implies `audio` |
| `spacy-tagger` | spaCy CNN part-of-speech tagger |
| `super-resolution` | `SrVggCompact` for `realesr-general-x4v3`, and `RrdbNet` ×2 for `RealESRGAN_x2plus` and `2x_PureVision`. Half precision also needs `f16` |

Tokenizer vocabularies:

| Feature | Adds |
| --- | --- |
| `vocab-*` | One bundled splintr vocabulary each. See [docs/capabilities.md](docs/capabilities.md#bundled-tokenizer-vocabularies) |
| `vocabs` | Every bundled splintr vocabulary |

## Testing

```bash
# CPU tests, no model gates
cargo test

# Every model gate on the CPU backend
cargo test --features whisper,kokoro,neucodec,voxcpm,silero-vad,vits,clap,spacy-tagger,super-resolution,f16

# CUDA tests
cargo test --features cuda,f16

# Show test output
cargo test -- --nocapture
```

Run the format and lint checks in [CONTRIBUTING.md](CONTRIBUTING.md#build-and-test) before you open a pull request.

## Ecosystem

boostr is part of the [ml-rust](https://github.com/ml-rust) organization:

- **[numr](https://github.com/ml-rust/numr)**: Foundational numerical computing (tensors, ops, linalg, FFT)
- **[boostr](https://github.com/ml-rust/boostr)**: ML framework (this project)
- **[oxidizr](https://github.com/ml-rust/oxidizr)**: Training framework for Mamba2, MLA, and MoE (uses boostr)
- **[blazr](https://github.com/ml-rust/blazr)**: Inference server with an OpenAI-compatible API (uses boostr)
- **[compressr](https://github.com/ml-rust/compressr)**: Model quantization and compression (uses boostr)
- **[voxr](https://github.com/ml-rust/voxr)**: Audio and speech on top of boostr's models
- **[splintr](https://github.com/ml-rust/splintr)**: BPE tokenizer

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

Parts of boostr port code from other open-source projects. Their copyright
lines and licenses are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
