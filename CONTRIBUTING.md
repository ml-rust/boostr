# Contributing

Thanks for contributing to [boostr](https://crates.io/crates/boostr). This guide covers the architecture conventions and quality gates.

## Before you start

- Rust 1.95 or newer (edition 2024).
- For CUDA work: CUDA 12.8+ or 13.x with `nvcc` on `PATH`.
- For WebGPU work: platform GPU drivers (Vulkan, Metal, or DX12).
- A clean working tree before you open a pull request.

Useful contributions:

- **New ML primitives**: attention variants, position encodings, quantization formats, model architectures, or inference and training machinery.
- Bug fixes and numerical-accuracy improvements.
- Backend coverage (CPU SIMD, CUDA PTX, WebGPU WGSL) for ops that run on some backends only.

Open an issue before you write a non-trivial op or architecture. Describe the addition, the reference paper, and the crate it belongs in. Small, self-contained fixes can go straight to a pull request.

### Which crate

Place work by what it is:

| Crate | Scope |
| --- | --- |
| [numr](https://github.com/ml-rust/numr) | Foundational primitives: tensor ops, dtypes, the `Runtime`/backend abstraction and new backends, autograd, FFT, core linear algebra (matmul, LU/QR/SVD/eigen, `solve`). |
| [solvr](https://github.com/ml-rust/solvr) | Scientific and solving algorithms built from numr primitives: optimization, ODE/DAE/PDE, interpolation, statistics, signal, spatial. |
| **boostr** (this crate) | AI/ML building blocks: attention, positional encodings, mixture-of-experts, state-space kernels, quantization, neural-network layers, model architectures, training and inference machinery. |

Quick test:

- A low-level primitive (tensor op, FFT, linear-algebra factorization) or a new backend: **numr**.
- A domain solver a scientist or engineer reaches for: **solvr**.
- Makes sense only for neural networks or LLMs: **boostr**.

When in doubt, propose the work in an issue.

If numr lacks a primitive you need, add it to numr. Do not reimplement it here or work around it with a host-side loop. boostr is built on numr, never alongside it.

## Architecture

boostr extends numr. It uses numr's `Tensor<R>`, `Runtime`/client, autograd, and ops directly. It adds ML capabilities through two mechanisms:

- **Extension traits**: a local trait implemented on numr's client type.
- **`QuantTensor`**: a separate type for block-quantized data.

```rust
// boostr defines the trait (local)
pub trait AttentionOps<R: Runtime> {
    fn multi_head_attention(&self, ...) -> Result<Tensor<R>>;
}

// boostr implements it for numr's client (orphan rule: local trait on foreign type = OK)
impl AttentionOps<CpuRuntime> for numr::runtime::cpu::CpuClient {
    fn multi_head_attention(&self, ...) -> Result<Tensor<CpuRuntime>> {
        // compose from numr ops
    }
}
```

boostr never reimplements tensors, storage, or the runtime. It never wraps numr runtimes, such as a `BoostrCpuRuntime`. It never defines its own `DType` for standard types. Run `rg "fn softmax" ../numr/src/` before you write an op. If numr has it, use it.

### Two kinds of ops

- **Composite ops** (attention, RoPE, MoE) are algorithms built from numr primitives. Each lives once in `impl_generic/`, and every backend delegates to it. A fused kernel, such as CUDA FlashAttention, can replace the delegation. It must produce results identical to `impl_generic`.
- **Primitive ops** (dequant, quantized matmul) are the kernel. The code differs fundamentally per backend, so these have no `impl_generic`. Each backend has its own SIMD, PTX, or WGSL kernel.

### Module layout

Each op lives in exactly one place. The same file name mirrors across the trait, the generic algorithm, and each backend:

```
src/ops/
├── traits/<domain>/<op>.rs        # trait definition + option/result types
├── impl_generic/<domain>/<op>.rs  # the algorithm: fn <op>_impl<R, C>(...)   (composite ops only)
├── cpu/<domain>/<op>.rs           # impl Trait for CpuClient  — delegates to *_impl
├── cuda/<domain>/<op>.rs          # impl Trait for CudaClient — delegates OR fused kernel
└── wgpu/<domain>/<op>.rs          # impl Trait for WgpuClient — delegates to *_impl
```

- Group ops into domain subdirectories: `attention/`, `position/`, `cache/`, `training/`, `quantization/`, `architecture/`, `inference/`. Never add flat files directly to `ops/traits/`, `ops/cpu/`, or similar directories.
- Create a new subdirectory only for a genuinely new domain.
- `mod.rs` contains only `pub mod` and `pub use`. It holds no logic, traits, or types.
- One op is one file. Add files for a new op. Do not expand existing files.
- Backend directories are required: `cpu/`, `cuda/`, `wgpu/`, not `cpu.rs`. Kernels go in `cuda/kernels/<domain>/` and `wgpu/shaders/`.

### Runtime-generic algorithms

- Be generic over `R: Runtime`. Operate on `Tensor<R>`, never on `&[f32]`, `Vec<f32>`, or struct fields of those types.
- Build computation from numr ops, not scalar `for` loops. numr uses SIMD on CPU and kernels on GPU. Scalar loops are slower and not portable.
- Respect backend dtype limits, such as WebGPU being F32-only. Return a clear error instead of degrading silently.

### No GPU↔CPU transfers in hot paths

Host/device transfers cost far more than the computation. Inside ops, never call `tensor.to_vec()` or `Tensor::from_slice(...)`. Two transfers are allowed:

- At the public API boundary, for user input and returned output.
- A single scalar pulled to the host for a convergence or control-flow check.

Keep state in `Tensor<R>`. Keep loops on the device.

## File size limits

| File type | Soft limit | Hard limit |
| --- | --- | --- |
| `traits/*.rs` | 100 | 200 |
| `impl_generic/*.rs` | 300 | 500 |
| `cpu/*.rs`, `cuda/*.rs` | 200 | 400 |
| Kernel files | 300 | 500 |
| `mod.rs` | 10 | 30 |
| `nn/*.rs` | 200 | 400 |
| Model architectures | 300 | 500 |

Prefer many small, focused files over few large ones.

## Build

```bash
cargo build --release                  # CPU (default)
cargo build --release --features cuda  # CUDA (requires CUDA 12.8+ or 13.x)
cargo build --release --features wgpu  # WebGPU
```

## Testing

- Put unit tests in the same file as the code, in `#[cfg(test)] mod tests`. Put public-API behavior tests under `tests/`.
- For composite ops, assert numerical correctness against a reference result. Check that a fused kernel matches its `impl_generic` output, not only that it returns `Ok`.
- A backend-specific test skips gracefully when no device is available.

```bash
cargo test                             # CPU
cargo test --features cuda             # CUDA
cargo test --features wgpu             # WebGPU
cargo test --all-features
```

## Quality checks

Run these before you submit. CI runs clippy with `-D warnings`, so treat a warning as an error.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

If you touch GPU backends, also run clippy with `--features cuda` and `--features wgpu`.

## Pull requests

- Keep each pull request focused and scoped.
- Keep the module structure, domain subdirectories, and `impl_generic` pattern.
- Include tests for behavioral changes. Check parity across backends, and check fused kernels against `impl_generic`.
- Update docs when public APIs or features change.
- **No `.unwrap()` or `.expect()` in library code.** Return a typed error with context.
- Do not add reimplemented numr ops, runtime wrappers, GPU↔CPU transfers in hot paths, or vendor libraries (cuBLAS, cuDNN, MKL).

## Commit messages

Use Conventional Commits with a clear, imperative summary:

```
feat(attention): add sliding-window support to flash_attention_fwd
fix(quant): correct Q4_K block scale unpacking on the CPU kernel
```
