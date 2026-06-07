# Contributing

Thanks for contributing to [boostr](https://crates.io/crates/boostr). This guide
covers the architecture conventions and quality gates the project expects.

## Prerequisites

- Rust 1.88+ (edition 2024).
- For CUDA work: a CUDA 12.x toolchain (cudarc does not support CUDA 13.x yet).
- For WebGPU work: platform GPU drivers (Vulkan/Metal/DX12).
- A clean working tree before opening a pull request.

## What to contribute

The most valuable contributions are usually **new ML primitives** — attention
variants, position encodings, quantization formats, model architectures, or
inference/training machinery — and bug fixes, numerical-accuracy improvements,
and additional backend coverage (CPU SIMD, CUDA PTX, WebGPU WGSL) for ops that
only run on some backends today.

Before writing a non-trivial op or architecture, **open an issue first**
describing what you want to add, the reference/paper, and which crate it belongs
in (see below). This avoids duplicated effort and lets us agree on placement and
API up front. Small, self-contained fixes can go straight to a pull request.

## Which crate: numr, solvr, or boostr

boostr is one layer of a stack, and a contribution only belongs here if it fits
this layer. Place new work by what it _is_, not where it's convenient:

- **[numr](https://github.com/ml-rust/numr)** — foundational primitives that
  everything else builds on: tensor ops, dtypes, the `Runtime`/backend
  abstraction (and **new backends** themselves), autograd, FFT, and core linear
  algebra (matmul, LU/QR/SVD/eigen, `solve`). If it's a building block reused
  across domains, or it adds/touches a hardware backend, it goes in numr.
- **[solvr](https://github.com/ml-rust/solvr)** — scientific/solving algorithms
  composed from numr primitives: optimization, ODE/DAE/PDE, interpolation,
  statistics, signal, and spatial.
- **boostr** (this crate) — AI/ML-specific building blocks: attention,
  positional encodings, mixture-of-experts, state-space kernels, quantization,
  neural-network layers, model architectures, and training/inference machinery.

Quick test:

- Is it a low-level primitive (a tensor op, an FFT, a linear-algebra
  factorization) or a new backend? → **numr**.
- Is it a domain solver a scientist/engineer would reach for? → **solvr**.
- Does it only make sense for neural networks / LLMs? → **boostr**.

When in doubt, propose it in an issue and we'll help place it. **If a primitive
you need is missing in numr, add it to numr** — don't reimplement it here or work
around it with a host-side loop. boostr is built _on_ numr, never alongside it.

## Architecture

boostr extends numr rather than reimplementing it. It uses numr's `Tensor<R>`,
`Runtime`/client, autograd, and ops directly, and adds ML capabilities through
**extension traits** (a local trait implemented on numr's foreign client type)
plus a separate `QuantTensor` type for block-quantized data.

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

boostr **never** reimplements tensors, storage, or runtime; never wraps numr
runtimes (no `BoostrCpuRuntime`); and never defines its own `DType` for standard
types. Run `rg "fn softmax" ../numr/src/` before writing your own — if numr has
it, use it.

### Two kinds of ops

- **Composite ops** (attention, RoPE, MoE) are an algorithm composed of numr
  primitives. They live once in `impl_generic/`, and every backend delegates to
  it. A fused kernel (e.g. CUDA FlashAttention) may replace the delegation, but
  it **must produce identical results** to `impl_generic`.
- **Primitive ops** (dequant, quantized matmul) _are_ the kernel — the code
  differs fundamentally per backend, so there is **no** `impl_generic`. Each
  backend has its own SIMD/PTX/WGSL kernel.

### The `impl_generic` pattern and module layout

Each op exists in exactly one place, with the **same file name** mirrored across
trait, generic algorithm, and each backend:

```
src/ops/
├── traits/<domain>/<op>.rs        # trait definition + option/result types
├── impl_generic/<domain>/<op>.rs  # the algorithm: fn <op>_impl<R, C>(...)   (composite ops only)
├── cpu/<domain>/<op>.rs           # impl Trait for CpuClient  — delegates to *_impl
├── cuda/<domain>/<op>.rs          # impl Trait for CudaClient — delegates OR fused kernel
└── wgpu/<domain>/<op>.rs          # impl Trait for WgpuClient — delegates to *_impl
```

Ops are grouped into **domain subdirectories** — `attention/`, `position/`,
`cache/`, `training/`, `quantization/`, `architecture/`, `inference/`. Never add
flat files directly to `ops/traits/`, `ops/cpu/`, etc. New ops go into one of
these groups; if a genuinely new domain emerges, add a new subdirectory.

- `mod.rs` contains **only** `pub mod` / `pub use` — no logic, traits, or types.
- One op = one file. Adding an op means adding files, not expanding existing ones.
- Backend dirs are required (`cpu/`, `cuda/`, `wgpu/` — not `cpu.rs`). Kernels go
  in dedicated subdirs (`cuda/kernels/<domain>/`, `wgpu/shaders/`).

### Runtime-generic algorithms

- Be generic over `R: Runtime`; operate on `Tensor<R>`, never on `&[f32]` /
  `Vec<f32>` parameters or struct fields.
- Build computation out of numr ops, not scalar `for` loops — numr uses SIMD on
  CPU and kernels on GPU, so scalar loops are both slower and not portable.
- Respect backend dtype limits (e.g. the WebGPU backend is F32-only) and surface
  a clear error rather than silently degrading.

### No GPU↔CPU transfers in hot paths

Host/device transfers cost far more than the computation itself. Inside ops, do
**not** call `tensor.to_vec()` or `Tensor::from_slice(...)`. The only acceptable
transfers are at the public API boundary (user input / returned output) and a
single scalar pulled to the host for a convergence/control-flow check. Keep state
in `Tensor<R>` and keep loops on-device.

## File size limits

| File type               | Soft | Hard |
| ----------------------- | ---- | ---- |
| `traits/*.rs`           | 100  | 200  |
| `impl_generic/*.rs`     | 300  | 500  |
| `cpu/*.rs`, `cuda/*.rs` | 200  | 400  |
| kernel files            | 300  | 500  |
| `mod.rs`                | 10   | 30   |
| `nn/*.rs`               | 200  | 400  |
| model architectures     | 300  | 500  |

Prefer many small, focused files over few large ones.

## Building with backends

```bash
cargo build --release                  # CPU (default)
cargo build --release --features cuda  # CUDA (requires a CUDA 12.x toolchain)
cargo build --release --features wgpu  # WebGPU
```

## Testing

- Put unit tests in the same file as the code under test
  (`#[cfg(test)] mod tests`). Use integration tests under `tests/` for public-API
  behavior.
- For composite ops, assert numerical correctness against a reference result —
  and verify a fused kernel matches its `impl_generic` output, not just that it
  returns `Ok`.
- A backend-specific test should skip gracefully when no device is available
  rather than fail.

```bash
cargo test                             # CPU
cargo test --features cuda             # CUDA
cargo test --features wgpu             # WebGPU
cargo test --all-features
```

## Local quality checks

Run these before submitting. Clippy is run with `-D warnings` to match CI, so a
warning is a failure — treat it as one locally too.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

If you touch GPU backends, also run clippy with `--features cuda` and
`--features wgpu`.

## Pull request guidelines

- Keep PRs focused and scoped.
- Preserve the module structure, domain subdirectories, and `impl_generic`
  pattern described above.
- Include tests for behavioral changes; verify parity across backends and
  against `impl_generic` for fused kernels.
- Update docs when public APIs or features change.
- **No `.unwrap()` / `.expect()` in library code** — return a typed error with
  context.
- No reimplemented numr ops, no runtime wrappers, no GPU↔CPU transfers in hot
  paths, no vendor libraries (cuBLAS/cuDNN/MKL).

## Commit messages

Use Conventional Commits with a clear, imperative summary, for example:

```
feat(attention): add sliding-window support to flash_attention_fwd
fix(quant): correct Q4_K block scale unpacking on the CPU kernel
```
