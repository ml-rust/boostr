<div align="center">

# boostr

<h3>ML framework in Rust. Write once. Run on any backend.</h3>

<p>
Attention, quantization, model architectures, training, and inference built on <a href="https://github.com/ml-rust/numr">numr</a>. It ships 27 quantization formats, and the same code runs on CPU, CUDA, and WebGPU.
</p>

<p>
  <a href="https://docs.rs/boostr"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/boostr"><strong>Crate</strong></a>
  ·
  <a href="#feature-flags"><strong>Feature Flags</strong></a>
  ·
  <a href="#testing"><strong>Testing</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
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

Capabilities are in [docs/capabilities.md](docs/capabilities.md). The layout and design rules are in [docs/architecture.md](docs/architecture.md).

## Quick Start

### Install

Add boostr to `Cargo.toml`:

```toml
[dependencies]
boostr = "0.9"
```

Enable a backend with a feature:

```toml
# CUDA: requires CUDA 12.8+ or 13.x
boostr = { version = "0.9", features = ["cuda"] }

# WebGPU: requires Vulkan, Metal, or DX12 GPU drivers
boostr = { version = "0.9", features = ["wgpu"] }
```

### Build from Source

Requirements:

- Rust 1.95 or newer.
- For `cuda`: CUDA 12.8+ or 13.x with `nvcc` on `PATH`. The GPU needs compute capability 7.5 (Turing) or newer.
- For `wgpu`: platform GPU drivers.

Run these commands in a checkout of the repository:

```bash
cargo build --release                  # CPU
cargo build --release --features cuda  # CUDA
cargo build --release --features wgpu  # WebGPU
```

By default, `build.rs` compiles CUDA kernels for the GPUs on the build host. Set `NUMR_CUDA_ARCH=all` for any binary that runs on another machine.
`BOOSTR_CUDA_ARCH` overrides `NUMR_CUDA_ARCH` for boostr's kernels.

```bash
NUMR_CUDA_ARCH=all cargo build --release --features cuda
```

### Basic Usage

Run flash attention on the CPU backend:

```rust
use boostr::ops::traits::attention::{AttnOutLayout, FlashAttentionOps};
use boostr::{CpuClient, CpuDevice, DType};
use numr::ops::RandomOps;

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
use numr::ops::RandomOps;

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

## Feature Flags

| Feature        | Purpose                                                   | Dependencies            |
| -------------- | --------------------------------------------------------- | ----------------------- |
| `cpu`          | CPU backend (default)                                     | numr                    |
| `cuda`         | CUDA GPU acceleration (CUDA 12.8+ or 13.x)                | numr/cuda, cudarc       |
| `nccl`         | Multi-GPU via NCCL                                        | numr/nccl               |
| `wgpu`         | WebGPU cross-platform GPU                                 | numr/wgpu, wgpu         |
| `distributed`  | Distributed inference over nexar (implies `f16`)          | nexar, anyhow, bytemuck |
| `f16`          | Half-precision float support                              | numr/f16                |
| `fp8`          | FP8 precision support                                     | numr/fp8                |
| `audio`        | Tensor-level signal ops (STFT, iSTFT, mel)                | —                       |
| `whisper`      | Whisper speech recognition model (implies `audio`)        | —                       |
| `kokoro`       | Kokoro speech synthesis model (implies `audio`)           | —                       |
| `vits`         | VITS / MMS-TTS speech synthesis model (implies `audio`)   | —                       |
| `neucodec`     | NeuCodec audio codec (implies `audio`)                    | —                       |
| `voxcpm`       | VoxCPM2 speech synthesis model (implies `audio`, `f16`)   | —                       |
| `silero-vad`   | Silero voice activity detection (implies `audio`)         | —                       |
| `clap`         | CLAP audio and text towers (implies `audio`)              | —                       |
| `spacy-tagger` | spaCy CNN part-of-speech tagger                           | —                       |
| `vocab-*`      | One bundled splintr vocabulary each (see [docs/capabilities.md](docs/capabilities.md#bundled-tokenizer-vocabularies)) | splintr |
| `vocabs`       | Every bundled splintr vocabulary                          | splintr                 |

## Testing

```bash
# CPU tests, no model gates
cargo test

# Every model gate on the CPU backend
cargo test --features whisper,kokoro,neucodec,voxcpm,silero-vad,vits,clap,spacy-tagger,f16

# CUDA tests
cargo test --features cuda,f16

# Show test output
cargo test -- --nocapture
```

Run the format and lint checks in [CONTRIBUTING.md](CONTRIBUTING.md#local-quality-checks) before you open a pull request.

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
