//! WGSL shader generators for TCF native quantized weights.
//!
//! # Why this file holds a second copy of the format
//!
//! `tcf-core` IS the reference codec, and boostr's CPU kernels call it rather
//! than restating it. A WGSL shader cannot: a compute shader has no path back
//! into the Rust crate. So the read-direction rules below are written once,
//! here, in `decoder`, and every WebGPU TCF kernel goes through them, exactly as
//! `quant/cuda/kernels/tcf.cuh` does for CUDA.
//!
//! What is duplicated is deliberately minimal, and is exactly the part that
//! cannot be hoisted to the host:
//!   - the nibble/bit position of one element inside one tile (Section 14.1,
//!     Section 14.2, Section 14.3),
//!   - the 6-bit sub-field position inside one super-block (Section 14.6),
//!   - the reconstruction math (Section 13.0, Section 13.0.1, Section 13.3,
//!     Section 13.4).
//!
//! What is NOT duplicated: plane order, plane sizes, and every byte offset
//! between planes. The host computes those from `tcf-core`'s own `QuantLayout`
//! through [`crate::quant::tcf::TcfPlanes`] — the same type the CUDA launches
//! use — and passes them in as uniform fields.
//!
//! # What WGSL forces
//!
//! WGSL has no 8-bit or 16-bit scalar type and no `f16` without an optional
//! extension, so a payload arrives as `array<u32>` and every byte, nibble,
//! 2-bit field, 6-bit field and binary16 is extracted by shifting and masking.
//! The generated decoder holds all of it, including ONE `tcf_binary16` function that
//! reproduces `tcf_core::binary16::bits_to_f32` by pure integer bit
//! manipulation — no `exp2`, no float multiply — so its result cannot drift
//! from the CPU value by a library's exponentiation accuracy.
//!
//! # Why a two-level scale is divided in integers
//!
//! Section 15.7.4 of the WGSL specification gives f32 `*` and `+` correct
//! rounding but f32 `/` only 2.5 ULP. A two-level effective scale is
//! `(super * sub) / 255` — or `/ 63`, `/ 31` — and an adapter this project was
//! measured on returned the neighbouring float for `Q6S16D_T64`, one ULP off
//! the CPU. `fma` cannot repair it: the same section gives `fma(x, y, z)` the
//! accuracy of `x * y + z`, so a single rounding is permitted and never
//! required, and naga emits `mad` on HLSL and expands to `(x * y + z)` on a
//! GLSL target without `fma`. So `tcf_scaled_quotient` computes the quotient by
//! integer long division and rounds to nearest even itself. The numerator is a
//! binary16's 11-bit significand times an 8-bit factor, at most 19 bits, so it
//! is exact as an integer and the quotient's correctly rounded f32 is a matter
//! of bookkeeping, not of any adapter's divider.
//!
//! # Why the asymmetric product is pinned before the add
//!
//! Section 15.7.4 permits, and never requires, contracting `a * b + c` into a
//! single rounding, and it gives `fma` the accuracy of the unfused expression,
//! so no WGSL spelling of `d * code + m` is pinned by the specification alone.
//! An adapter this project was measured on fused it and returned `0x3f450002`
//! for `Q4AS32D_T64` where the CPU's two roundings give `0x3f450000` — two ULP
//! of the sum, which is half an ULP of the product. `Q4AS32D_T64` is the only
//! encoding that reconstructs from a two-level `d` AND a two-level `m`, so it
//! is the only one where the cancellation makes a fused rounding visible. So
//! the product goes through `tcf_settled`, a `bitcast` round trip ORed with an
//! always-zero uniform, which breaks the float dependency chain the fusion
//! needs and survives a driver that folds a bare bitcast pair.
//!
//! # Alignment
//!
//! Every storage binding is `array<u32>` or `array<f32>`, so its only
//! requirement is a 4-byte buffer size; numr's WebGPU allocator rounds every
//! allocation up to 4 bytes, and byte addressing inside the payload is done by
//! the decoder rather than by the binding. The uniform block is 16 `u32` fields,
//! 64 bytes, which satisfies WGSL's 16-byte uniform struct alignment with no
//! padding member beyond the single trailing one that rounds 60 up to 64. A
//! `u32` never needs interior padding, so the Rust `#[repr(C)]` mirror and the
//! WGSL struct agree field for field.
//!
//! # The access pattern, and what plane-major costs on a WebGPU device
//!
//! Both kernels split their work the way the CUDA kernels do. Phase 1 resolves
//! a group's parameters ONCE per (tile, group) — at most four per tile — so a
//! scale plane is read once per 16, 32 or 64 weights rather than once per
//! weight, and the reads that phase issues are consecutive entries of one
//! plane. Phase 2 streams the code plane, which is one contiguous run per tile
//! and adjacent between tiles, so neighbouring invocations read neighbouring
//! bytes.
//!
//! The `array<u32>` binding makes this sharper than it is on CUDA rather than
//! softer. WGSL forces every byte through a word load, so a decode reads whole
//! aligned 32-bit words no matter what; a scale interleaved into the code
//! stream every 16 or 32 weights — what a GGUF block layout does — puts a
//! code run at an arbitrary byte phase inside those words and costs extra
//! shifting on every element. TCF's dense code plane starts each tile at a
//! 32-byte boundary for 4-bit and 6-bit codes and at 64 bytes for 8-bit, so
//! the word index is a clean function of the element index. What plane-major
//! costs here is concurrent read STREAMS — two for a flat symmetric encoding,
//! up to five for the two-level asymmetric one — not locality.
//!
//! The 6-bit encodings are the same exception CUDA found. Section 14.2 splits
//! their code plane into a low-nibble sub-plane and a high-two-bit sub-plane,
//! so `Q6S32_T64` and `Q6S16D_T64` cost two code streams per tile instead of
//! one, and `tcf_code` issues two `tcf_byte` word loads per element rather
//! than one. Interleaving the two sub-planes per tile would halve that. This
//! is the third backend to reach the same conclusion, which CONFORMANCE.md
//! Section 8.1 leaves open pending benchmark.
//!
//! The gate is `tests/backend_parity/quant_tcf_wgpu.rs`: every encoding,
//! decoded by these shaders and by `tcf_core::unpack` + `tcf_core::dequantize`,
//! must agree.

mod decoder;
mod kernels;

pub use kernels::{
    DEQUANT_ENTRY, DEQUANT_TILES_PER_GROUP, DEQUANT_WORKGROUP, MATMUL_ENTRY, MATMUL_TILE,
    generate_tcf_dequant_shader, generate_tcf_matmul_shader,
};
