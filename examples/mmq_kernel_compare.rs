//! Launches the dp4a and tensor-core MMQ kernels back to back in one process,
//! on identical inputs, so a profiler attributes instruction counts to each
//! kernel without an A/B rebuild. Covers Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K,
//! Q5_K, Q6_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS and IQ2_S via
//! `--format`.
//!
//! Q4_0, Q4_1, Q5_0, Q5_1, Q5_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS
//! and IQ2_S have no token-major kernel of either kind — neither a `quant_mmq_<fmt>_q8_1` nor
//! its `_mma` twin exists — so for those formats the tool runs and checks the
//! feature-major kernels alone and skips the token-major comparison rather
//! than resolving a symbol that is not compiled.
//!
//! ```text
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q8_0 --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q4_0 --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q4_1 --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q5_0 --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q5_1 --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q4_k --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q5_k --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q6_k --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q3_k --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q2_k --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format iq4_nl --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format iq4_xs --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format iq2_xxs --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format iq2_xs --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format iq2_s --n 4096 --k 14336 --m 512
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("mmq_kernel_compare needs --features cuda");
}

#[cfg(feature = "cuda")]
use boostr::quant::cuda::kernels::{self, QUANT_GEMV_MODULE, QUANT_MMQ_MMA_MODULE};
// The ONE set of grids and the ONE sign table, shared with the CPU
// dequantizers the IQ2 references mirror.
#[cfg(feature = "cuda")]
use boostr::quant::cpu::kernels::iq_grid::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, KSIGNS};
// The ONE codebook, shared with the CPU dequantizer the references mirror.
#[cfg(feature = "cuda")]
use boostr::quant::tables::KVALUES_IQ4NL;
#[cfg(feature = "cuda")]
use cudarc::driver::PushKernelArg;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::LaunchConfig;
#[cfg(feature = "cuda")]
use numr::runtime::Device;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::{Runtime, RuntimeClient};
#[cfg(feature = "cuda")]
use numr::tensor::Tensor;

/// Mirror of the production selection rule. The authoritative copy lives in
/// `src/quant/cuda/quant_matmul/mmq_feat_major.rs`; keep the two in step.
///
/// Compiled feature-major token-tile variants, ascending. A variant needs a
/// 128-row weight tile at the FORMAT'S stride plus an `mmq_x`-row activation
/// tile at a 36-int stride. The weight stride differs per format — Q4_K's row
/// is wider than Q8_0's — so every call here threads the format's stride
/// through, or the harness opts a kernel in to the wrong size. The list skips
/// every `mmq_x` the kernel's granularity rule rejects: below 48 the tile steps
/// by 8, at and above it by 16.
#[cfg(feature = "cuda")]
const MMQ_X_VARIANTS: &[u32] = &[8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];

/// Dynamic shared memory one feature-major variant needs. The larger ones are
/// above the 48KB static limit, so it is allocated dynamically and opted into.
///
/// `act_scratch` is the format's per-token scratch region after the activation
/// tile, which only Q2_K asks for. It MUST be threaded through here as well as
/// through the stride: this mirror has desynced from
/// `mmq_feat_major::dispatch::smem_bytes` before, and a kernel opted in to less
/// shared memory than it indexes fails in a way that reads as a kernel bug.
#[cfg(feature = "cuda")]
const fn mmq_x_smem_bytes(x_stride: u32, act_scratch: u32, mmq_x: u32) -> u32 {
    4 * (128 * x_stride + mmq_x * (36 + act_scratch))
}

/// Picks the feature-major variant that launches the fewest token tiles for
/// `m`, breaking ties toward the smaller tile because it costs fewer registers
/// and less shared memory. `smem_limit` is the device's per-block opt-in
/// maximum.
#[cfg(feature = "cuda")]
fn select_mmq_x(m: u32, smem_limit: u32, x_stride: u32, act_scratch: u32) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for &mmq_x in MMQ_X_VARIANTS {
        if mmq_x_smem_bytes(x_stride, act_scratch, mmq_x) > smem_limit {
            continue;
        }
        let tiles = m.div_ceil(mmq_x);
        if best.is_none_or(|(_, b)| tiles < b) {
            best = Some((mmq_x, tiles));
        }
        if tiles == 1 {
            break;
        }
    }
    best.map(|(mmq_x, _)| mmq_x)
}

/// Exact reference for one output element, in f64, plus the accumulated
/// magnitude of the sum: the sum over `k`-blocks of `|contribution|`.
///
/// This is the ground truth every kernel is checked against. It is not another
/// GPU kernel: the per-block int32 dot product is exact in integer arithmetic,
/// so the only inexact step anywhere is the accumulation across blocks, and
/// doing that in f64 bounds every f32 kernel's error regardless of the order it
/// sums in. That is what lets stream-k — which reassociates the k sum across
/// blocks by construction — be checked at all. The magnitude is what f32
/// accumulation error is actually proportional to; see
/// [`check_against_reference`].
#[cfg(feature = "cuda")]
fn q8_0_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    let bpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for b in 0..bpr {
        let wb = (feat * bpr + b) * 34;
        let ab = (token * bpr + b) * 36;
        let wd = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
        let mut dot = 0i64;
        for p in 0..32 {
            dot += i64::from(weight[wb + 2 + p] as i8) * i64::from(act[ab + 4 + p] as i8);
        }
        let contribution = dot as f64 * ad * wd;
        sum += contribution;
        magnitude += contribution.abs();
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for a Q4_0 weight against a
/// Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_q4_0` in
/// `src/quant/cpu/kernels/dequant_simple.rs`: per 32-element block of 18
/// bytes, `d`@0 (f16) then 16 nibble-packed quants@2, where element `j`
/// (0..15) is the LOW nibble of `qs[j]`, element `j + 16` is the HIGH nibble of
/// the same byte, and the value is `d * (q - 8)` with `q` unsigned 4-bit. The
/// magnitude is what f32 accumulation error is actually proportional to; see
/// [`check_against_reference`].
#[cfg(feature = "cuda")]
fn q4_0_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    let bpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for b in 0..bpr {
        let wb = (feat * bpr + b) * 18;
        let ab = (token * bpr + b) * 36;
        let wd = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
        let mut dot = 0i64;
        for j in 0..16 {
            let byte = weight[wb + 2 + j];
            let lo = i64::from(byte & 0x0F) - 8;
            let hi = i64::from(byte >> 4) - 8;
            dot += lo * i64::from(act[ab + 4 + j] as i8);
            dot += hi * i64::from(act[ab + 4 + j + 16] as i8);
        }
        let contribution = dot as f64 * ad * wd;
        sum += contribution;
        magnitude += contribution.abs();
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for a Q4_1 weight against a
/// Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_q4_1` in
/// `src/quant/cpu/kernels/dequant_simple.rs`: per 32-element block of 20
/// bytes, `d`@0 (f16), `m`@2 (f16), then 16 nibble-packed quants@4, where
/// element `j` (0..15) is the LOW nibble of `qs[j]`, element `j + 16` is the
/// HIGH nibble of the same byte, and the value is `d * q + m` with `q`
/// unsigned 4-bit. The minimum is ADDITIVE, unlike Q4_K's, so it enters the
/// sum as `+m * (activation scale * block sum)`.
#[cfg(feature = "cuda")]
fn q4_1_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    let bpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for b in 0..bpr {
        let wb = (feat * bpr + b) * 20;
        let ab = (token * bpr + b) * 36;
        let wd = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let wm = f64::from(half::f16::from_le_bytes([weight[wb + 2], weight[wb + 3]]).to_f32());
        let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
        let mut dot = 0i64;
        let mut asum = 0i64;
        for j in 0..16 {
            let byte = weight[wb + 4 + j];
            let (lo, hi) = (i64::from(byte & 0x0F), i64::from(byte >> 4));
            let (al, ah) = (
                i64::from(act[ab + 4 + j] as i8),
                i64::from(act[ab + 4 + j + 16] as i8),
            );
            dot += lo * al + hi * ah;
            asum += al + ah;
        }
        let contribution = dot as f64 * ad * wd + asum as f64 * ad * wm;
        sum += contribution;
        magnitude += contribution.abs();
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for a Q5_0 weight against a
/// Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_q5_0` in
/// `src/quant/cpu/kernels/dequant_simple.rs`: per 32-element block of 22
/// bytes, `d`@0 (f16), a 32-bit `qh`@2 holding one fifth bit per element, then
/// 16 low-nibble bytes@6. Element `j` (0..15) takes the LOW nibble of `qs[j]`
/// plus bit `j` of `qh`; element `j + 16` takes the HIGH nibble of the same
/// byte plus bit `j + 16`. The value is `d * (q - 16)`.
#[cfg(feature = "cuda")]
fn q5_0_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    let bpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for b in 0..bpr {
        let wb = (feat * bpr + b) * 22;
        let ab = (token * bpr + b) * 36;
        let wd = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
        let qh = u32::from_le_bytes([
            weight[wb + 2],
            weight[wb + 3],
            weight[wb + 4],
            weight[wb + 5],
        ]);
        let mut dot = 0i64;
        for j in 0..16 {
            let byte = weight[wb + 6 + j];
            let lo = i64::from((byte & 0x0F) | ((((qh >> j) & 1) as u8) << 4)) - 16;
            let hi = i64::from((byte >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)) - 16;
            dot += lo * i64::from(act[ab + 4 + j] as i8);
            dot += hi * i64::from(act[ab + 4 + j + 16] as i8);
        }
        let contribution = dot as f64 * ad * wd;
        sum += contribution;
        magnitude += contribution.abs();
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for a Q5_1 weight against a
/// Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_q5_1` in
/// `src/quant/cpu/kernels/dequant_simple.rs`: per 32-element block of 24
/// bytes, `d`@0 (f16), `m`@2 (f16), a 32-bit `qh`@4, then 16 low-nibble
/// bytes@8. The 5-bit assembly is Q5_0's and the value is `d * q + m`, so the
/// minimum enters ADDITIVELY, exactly as in Q4_1.
#[cfg(feature = "cuda")]
fn q5_1_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    let bpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for b in 0..bpr {
        let wb = (feat * bpr + b) * 24;
        let ab = (token * bpr + b) * 36;
        let wd = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let wm = f64::from(half::f16::from_le_bytes([weight[wb + 2], weight[wb + 3]]).to_f32());
        let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
        let qh = u32::from_le_bytes([
            weight[wb + 4],
            weight[wb + 5],
            weight[wb + 6],
            weight[wb + 7],
        ]);
        let mut dot = 0i64;
        let mut asum = 0i64;
        for j in 0..16 {
            let byte = weight[wb + 8 + j];
            let lo = i64::from((byte & 0x0F) | ((((qh >> j) & 1) as u8) << 4));
            let hi = i64::from((byte >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4));
            let (al, ah) = (
                i64::from(act[ab + 4 + j] as i8),
                i64::from(act[ab + 4 + j + 16] as i8),
            );
            dot += lo * al + hi * ah;
            asum += al + ah;
        }
        let contribution = dot as f64 * ad * wd + asum as f64 * ad * wm;
        sum += contribution;
        magnitude += contribution.abs();
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for a Q4_K weight against
/// a Q8_1 activation, plus the accumulated magnitude of the sum: the sum over
/// every element of `|contribution|`.
///
/// Dequant math and byte offsets are ground-truthed against
/// `dequant_q4k` and `unpack_q4k_q5k_scales` in
/// `src/quant/cpu/kernels/dequant_k_quants/q4k_q5k.rs`: per 256-element
/// super-block, `d`@0 and `dmin`@2 (f16), a 12-byte packed 6-bit scale/min
/// array@4, and 128 nibble-packed quants@16, unpacked into eight 32-element
/// sub-blocks as `x = d * sc[j] * q - dmin * m[j]`. The magnitude is what f32
/// accumulation error is actually proportional to; see
/// [`check_against_reference`].
#[cfg(feature = "cuda")]
fn q4_k_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    const SUPER: usize = 256;
    const BYTES: usize = 144;
    let bpr = k / SUPER;
    let abpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for sup in 0..bpr {
        let wb = (feat * bpr + sup) * BYTES;
        let d = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let dmin = f64::from(half::f16::from_le_bytes([weight[wb + 2], weight[wb + 3]]).to_f32());
        let sc = &weight[wb + 4..wb + 16];
        let qs = &weight[wb + 16..wb + 144];

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc[i] & 0x3F;
            mins[i] = sc[i + 4] & 0x3F;
        }
        for i in 4..8 {
            scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
            mins[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
        }

        for j in 0..8 {
            let dl = d * f64::from(scales[j]);
            let ml = dmin * f64::from(mins[j]);
            let qs_base = (j / 2) * 32;
            let is_high = j % 2 == 1;
            let ab = (token * abpr + sup * 8 + j) * 36;
            let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
            for l in 0..32 {
                let q = f64::from(if is_high {
                    (qs[qs_base + l] >> 4) & 0x0F
                } else {
                    qs[qs_base + l] & 0x0F
                });
                let w = dl * q - ml;
                let aq = f64::from(act[ab + 4 + l] as i8);
                let contribution = w * ad * aq;
                sum += contribution;
                magnitude += contribution.abs();
            }
        }
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for a Q5_K weight against
/// a Q8_1 activation, plus the accumulated magnitude of the sum: the sum over
/// every element of `|contribution|`.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_q5k` and
/// `unpack_q4k_q5k_scales` in
/// `src/quant/cpu/kernels/dequant_k_quants/q4k_q5k.rs`: per 256-element
/// super-block, `d`@0 and `dmin`@2 (f16), a 12-byte packed 6-bit scale/min
/// array@4, 32 bytes of fifth bits@16, and 128 nibble-packed low quants@48,
/// unpacked into eight 32-element sub-blocks as `x = d * sc[j] * q - dmin *
/// m[j]` with `q` five bits wide. The scale/min packing is the same as Q4_K's.
/// Sub-block PAIRS share one 32-byte run of `qs` (even sub-block reads the low
/// nibbles, odd the high nibbles of the same bytes), and in `qh` the BYTE index
/// is the element within the sub-block while the BIT index is the sub-block
/// number. The magnitude is what f32 accumulation error is actually
/// proportional to; see [`check_against_reference`].
#[cfg(feature = "cuda")]
fn q5_k_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    const SUPER: usize = 256;
    const BYTES: usize = 176;
    let bpr = k / SUPER;
    let abpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for sup in 0..bpr {
        let wb = (feat * bpr + sup) * BYTES;
        let d = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let dmin = f64::from(half::f16::from_le_bytes([weight[wb + 2], weight[wb + 3]]).to_f32());
        let sc = &weight[wb + 4..wb + 16];
        let qh = &weight[wb + 16..wb + 48];
        let qs = &weight[wb + 48..wb + 176];

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc[i] & 0x3F;
            mins[i] = sc[i + 4] & 0x3F;
        }
        for i in 4..8 {
            scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
            mins[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
        }

        for j in 0..8 {
            let dl = d * f64::from(scales[j]);
            let ml = dmin * f64::from(mins[j]);
            let qs_base = (j / 2) * 32;
            let is_high = j % 2 == 1;
            let ab = (token * abpr + sup * 8 + j) * 36;
            let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
            for l in 0..32 {
                let low4 = if is_high {
                    (qs[qs_base + l] >> 4) & 0x0F
                } else {
                    qs[qs_base + l] & 0x0F
                };
                let high1 = (qh[l] >> j) & 0x01;
                let q = f64::from(low4 | (high1 << 4));
                let w = dl * q - ml;
                let aq = f64::from(act[ab + 4 + l] as i8);
                let contribution = w * ad * aq;
                sum += contribution;
                magnitude += contribution.abs();
            }
        }
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for a Q6_K weight against
/// a Q8_1 activation, plus the accumulated magnitude of the sum: the sum over
/// every element of `|contribution|`.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_q6k` in
/// `src/quant/cpu/kernels/dequant_k_quants/q6k_q8k.rs`: per 256-element
/// super-block, 128-byte `ql` (low nibbles)@0, 64-byte `qh` (high bit
/// pairs)@128, 16 signed 8-bit sub-block scales@192 (one per 16 elements),
/// f16 `d`@208 (last, not first). Each half of the super-block (128 elements)
/// merges `ql`/`qh` into four biased 6-bit levels per lane, `x = d * scale *
/// (q - 32)`. The magnitude is what f32 accumulation error is actually
/// proportional to; see [`check_against_reference`].
#[cfg(feature = "cuda")]
fn q6_k_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    const SUPER: usize = 256;
    const BYTES: usize = 210;
    let bpr = k / SUPER;
    let abpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for sup in 0..bpr {
        let wb = (feat * bpr + sup) * BYTES;
        let ql = &weight[wb..wb + 128];
        let qh = &weight[wb + 128..wb + 192];
        let sc = &weight[wb + 192..wb + 208];
        let d = f64::from(half::f16::from_le_bytes([weight[wb + 208], weight[wb + 209]]).to_f32());

        for n in 0..2 {
            let y_base = n * 128;
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 = i64::from((ql[ql_base + l] & 0x0F) | ((qh[qh_base + l] & 0x03) << 4)) - 32;
                let q2 = i64::from(
                    (ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 0x03) << 4),
                ) - 32;
                let q3 =
                    i64::from((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 0x03) << 4)) - 32;
                let q4 =
                    i64::from((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 0x03) << 4))
                        - 32;

                let positions = [
                    y_base + l,
                    y_base + l + 32,
                    y_base + l + 64,
                    y_base + l + 96,
                ];
                let qvals = [q1, q2, q3, q4];
                let scs = [
                    sc[sc_base + is] as i8,
                    sc[sc_base + is + 2] as i8,
                    sc[sc_base + is + 4] as i8,
                    sc[sc_base + is + 6] as i8,
                ];
                for idx in 0..4 {
                    let w = d * f64::from(scs[idx]) * qvals[idx] as f64;
                    let elem = sup * SUPER + positions[idx];
                    let ab = (token * abpr + elem / 32) * 36;
                    let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
                    let aq = f64::from(act[ab + 4 + elem % 32] as i8);
                    let contribution = w * ad * aq;
                    sum += contribution;
                    magnitude += contribution.abs();
                }
            }
        }
    }
    (sum, magnitude)
}

/// Unpacks Q3_K's 16 signed 6-bit scales from the 12 packed bytes.
///
/// Ground-truthed against `unpack_q3k_scales` in
/// `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`, restated here per
/// scale rather than as four u32 lanes. With `g = j / 4`, scale `j` takes its
/// low nibble from byte `j % 4 + 4 * (g & 1)` (nibble `g / 2`) and its high
/// bit pair from byte `8 + j % 4` (bit pair `g`), and the 6-bit result is
/// biased by 32.
#[cfg(feature = "cuda")]
fn q3_k_scales(sc: &[u8]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (j, slot) in out.iter_mut().enumerate() {
        let g = j / 4;
        let lo = i32::from(sc[j % 4 + 4 * (g & 1)] >> (4 * (g / 2))) & 0x0F;
        let hi = i32::from(sc[8 + j % 4] >> (2 * g)) & 0x03;
        *slot = (lo | (hi << 4)) - 32;
    }
    out
}

/// Exact reference for one output element, in f64, for a Q3_K weight against
/// a Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_q3k` in
/// `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`: per 256-element
/// super-block, 32-byte `hmask`@0, 64-byte `qs`@32, 12 packed 6-bit signed
/// scales@96 (one per 16 elements), f16 `d`@108 (last, not first). Element
/// `pos` takes two low bits from `qs[32 * (pos / 128) + pos % 32]` at shift
/// `2 * ((pos % 128) / 32)` and one high bit from `hmask[pos % 32]` at bit
/// `4 * (pos / 128) + (pos % 128) / 32`. That high bit is INVERTED: a SET bit
/// means do NOT subtract 4, so the quant is `low2 - (bit ? 0 : 4)`, in
/// [-4, 3], and `x = d * scale * q`.
#[cfg(feature = "cuda")]
fn q3_k_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    const SUPER: usize = 256;
    const BYTES: usize = 110;
    let bpr = k / SUPER;
    let abpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for sup in 0..bpr {
        let wb = (feat * bpr + sup) * BYTES;
        let hmask = &weight[wb..wb + 32];
        let qs = &weight[wb + 32..wb + 96];
        let scales = q3_k_scales(&weight[wb + 96..wb + 108]);
        let d = f64::from(half::f16::from_le_bytes([weight[wb + 108], weight[wb + 109]]).to_f32());

        for pos in 0..SUPER {
            let n = pos / 128;
            let t = (pos % 128) / 32;
            let r = pos % 32;
            let low2 = i32::from((qs[32 * n + r] >> (2 * t)) & 3);
            let high_sub = if hmask[r] & (1u8 << (4 * n + t)) != 0 {
                0
            } else {
                4
            };
            let w = d * f64::from(scales[pos / 16]) * f64::from(low2 - high_sub);
            let elem = sup * SUPER + pos;
            let ab = (token * abpr + elem / 32) * 36;
            let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
            let aq = f64::from(act[ab + 4 + elem % 32] as i8);
            let contribution = w * ad * aq;
            sum += contribution;
            magnitude += contribution.abs();
        }
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for a Q2_K weight against
/// a Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_q2k` in
/// `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`: per 256-element
/// super-block, 16 `scales`@0 (low nibble the scale, high nibble the minimum,
/// one pair per 16 elements), 64-byte `qs`@16, f16 `d`@80, f16 `dmin`@82.
///
/// Element `pos` takes two UNSIGNED bits from
/// `qs[32 * (pos / 128) + 16 * ((pos % 32) / 16) + pos % 16]` at shift
/// `2 * ((pos % 128) / 32)`, and its scale pair is `scales[pos / 16]`, so
/// `x = d * (sc & 0x0F) * q - dmin * (sc >> 4)`. The minimum does not depend
/// on `q` — it is the term the kernel folds into a rank-1 correction against
/// the activation's per-16 quant sum.
#[cfg(feature = "cuda")]
fn q2_k_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    const SUPER: usize = 256;
    const BYTES: usize = 84;
    let bpr = k / SUPER;
    let abpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for sup in 0..bpr {
        let wb = (feat * bpr + sup) * BYTES;
        let scales = &weight[wb..wb + 16];
        let qs = &weight[wb + 16..wb + 80];
        let d = f64::from(half::f16::from_le_bytes([weight[wb + 80], weight[wb + 81]]).to_f32());
        let dmin = f64::from(half::f16::from_le_bytes([weight[wb + 82], weight[wb + 83]]).to_f32());

        for pos in 0..SUPER {
            let n = pos / 128;
            let t = (pos % 128) / 32;
            let h = (pos % 32) / 16;
            let l = pos % 16;
            let q = f64::from(i32::from((qs[32 * n + 16 * h + l] >> (2 * t)) & 3));
            let sc = scales[pos / 16];
            let w = d * f64::from(i32::from(sc & 0x0F)) * q - dmin * f64::from(i32::from(sc >> 4));
            let elem = sup * SUPER + pos;
            let ab = (token * abpr + elem / 32) * 36;
            let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
            let aq = f64::from(act[ab + 4 + elem % 32] as i8);
            let contribution = w * ad * aq;
            sum += contribution;
            magnitude += contribution.abs();
        }
    }
    (sum, magnitude)
}

/// Output positions sampled for the reference check. A full f64 pass is
/// O(M*N*K) and far too slow at benchmark shapes, so a fixed spread of
/// positions is checked instead — enough to catch a wrong index map, a dropped
/// k-block, or a double-counted partial, which are the failure modes that
/// matter here.
#[cfg(feature = "cuda")]
const REFERENCE_SAMPLES: usize = 256;

/// Error bound against the f64 reference, shared by every format.
///
/// The check divides the absolute error by the accumulated magnitude of the
/// dot product (see [`check_against_reference`]), not by the result. Q4_K and
/// Q6_K dequantize through cancellation-heavy terms — `d * sc * q - dmin * m`
/// and `d * sc * (q - 32)` — where individual block contributions run far
/// larger than the final result, so f32 rounding error is proportional to
/// that summed magnitude, not to the result. Dividing by magnitude instead of
/// result measures error against the quantity f32 rounding is actually
/// proportional to, so one tight bound serves Q8_0, Q4_K, Q6_K and Q3_K alike
/// regardless of how much cancellation each carries.
///
/// The bound must clear the f32 accumulation floor. Summing `K/32` block
/// products in f32 walks to roughly `sqrt(K/32) * f32::EPSILON`, which at the
/// shapes this tool runs is a few parts in a million, so a bound at 1e-6 sits
/// under the noise and fails on arithmetic rather than on defects.
///
/// A real defect — a wrong index map, a dropped k-block, a double-counted
/// partial — moves the result by a sizeable fraction of the magnitude and
/// still lands orders of magnitude above this bound.
///
/// One bound covers every kernel here. The feature-major family stages its
/// weight scales as f32, and the activation record's quant sum is an exact
/// int16, so the only half-rounded factor left anywhere is the activation
/// record's `d`, and that rounding stays under this bound. A kernel that put the weight scales back through `half` would
/// not, and would need its own.
#[cfg(feature = "cuda")]
const REFERENCE_RTOL: f64 = 1e-5;

/// One reference implementation: quantized weight and activation bytes plus an
/// output position, returning that output's dot product and the accumulated
/// magnitude the error is measured against.
#[cfg(feature = "cuda")]
type ReferenceFn = fn(&[u8], &[u8], usize, usize, usize) -> (f64, f64);

/// Exact reference for one output element, in f64, for an IQ4_NL weight
/// against a Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_iq4_nl`
/// in `src/quant/cpu/kernels/dequant_iq4.rs`: per 32-element block of 18
/// bytes, `d`@0 (f16) then 16 bytes@2 holding 32 4-bit fields. Each field is
/// an INDEX into the 16-entry signed codebook `KVALUES_IQ4NL`, never a
/// magnitude. The nibble order is Q4_0's split-half: element `j` (0..15) is the
/// LOW nibble of `qs[j]`, element `j + 16` the HIGH nibble of the same byte.
#[cfg(feature = "cuda")]
fn iq4_nl_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    let bpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for b in 0..bpr {
        let wb = (feat * bpr + b) * 18;
        let ab = (token * bpr + b) * 36;
        let wd = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
        let mut dot = 0i64;
        for j in 0..16 {
            let byte = weight[wb + 2 + j];
            let lo = i64::from(KVALUES_IQ4NL[usize::from(byte & 0x0F)]);
            let hi = i64::from(KVALUES_IQ4NL[usize::from(byte >> 4)]);
            dot += lo * i64::from(act[ab + 4 + j] as i8);
            dot += hi * i64::from(act[ab + 4 + j + 16] as i8);
        }
        let contribution = dot as f64 * ad * wd;
        sum += contribution;
        magnitude += contribution.abs();
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for an IQ4_XS weight
/// against a Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_iq4_xs`
/// in `src/quant/cpu/kernels/dequant_iq4.rs`: per 256-element super-block of
/// 136 bytes, `d`@0 (f16), `scales_h`@2 (u16), `scales_l[4]`@4, `qs[128]`@8.
/// Eight sub-blocks of 32 elements each take a 6-bit scale `ls`, four low bits
/// from a `scales_l` nibble and two high bits from `scales_h`, applied as
/// `d * (ls - 32)`. `scales_h` is TWO bytes and carries high bits for all
/// EIGHT sub-blocks. The quant fields index `KVALUES_IQ4NL` in the same
/// split-half nibble order as IQ4_NL, within each sub-block.
#[cfg(feature = "cuda")]
fn iq4_xs_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    const SUPER: usize = 256;
    const BYTES: usize = 136;
    let bpr = k / SUPER;
    let abpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for sup in 0..bpr {
        let wb = (feat * bpr + sup) * BYTES;
        let d = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let scales_h = u16::from_le_bytes([weight[wb + 2], weight[wb + 3]]);
        for sb in 0..8 {
            let sl = (weight[wb + 4 + sb / 2] >> (4 * (sb % 2))) & 0x0F;
            let sh = ((scales_h >> (2 * sb)) & 0x03) as u8;
            let ls = i32::from(sl | (sh << 4));
            let wd = d * f64::from(ls - 32);
            for j in 0..16 {
                let byte = weight[wb + 8 + sb * 16 + j];
                let lo = f64::from(KVALUES_IQ4NL[usize::from(byte & 0x0F)]);
                let hi = f64::from(KVALUES_IQ4NL[usize::from(byte >> 4)]);
                for (val, pos) in [(lo, j), (hi, j + 16)] {
                    let elem = sup * SUPER + sb * 32 + pos;
                    let ab = (token * abpr + elem / 32) * 36;
                    let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
                    let aq = f64::from(act[ab + 4 + elem % 32] as i8);
                    let contribution = wd * val * ad * aq;
                    sum += contribution;
                    magnitude += contribution.abs();
                }
            }
        }
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for an IQ2_XXS weight
/// against a Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_iq2_xxs`
/// in `src/quant/cpu/kernels/dequant_iq2.rs`: per 256-element block of 66
/// bytes, `d`@0 (f16) then `qs[64]`@2 read as eight pairs of little-endian
/// `u32`. The first `u32` of a pair holds four 8-bit INDICES into
/// `IQ2XXS_GRID`, whose entry expands to eight magnitude bytes; the second
/// holds the group's 4-bit scale in its top nibble over four 7-bit indices
/// into `KSIGNS`, one sign bit per expanded component. The group scale is
/// `d * (0.5 + s) * 0.25` and covers 32 elements.
#[cfg(feature = "cuda")]
fn iq2_xxs_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    const SUPER: usize = 256;
    const BYTES: usize = 66;
    let bpr = k / SUPER;
    let abpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for blk in 0..bpr {
        let wb = (feat * bpr + blk) * BYTES;
        let d = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        for group in 0..8 {
            let off = wb + 2 + group * 8;
            let indices = u32::from_le_bytes([
                weight[off],
                weight[off + 1],
                weight[off + 2],
                weight[off + 3],
            ]);
            let aux = u32::from_le_bytes([
                weight[off + 4],
                weight[off + 5],
                weight[off + 6],
                weight[off + 7],
            ]);
            let db = d * (0.5 + f64::from(aux >> 28)) * 0.25;
            for sub in 0..4 {
                let point = IQ2XXS_GRID[((indices >> (8 * sub)) & 0xFF) as usize];
                let signs = KSIGNS[((aux >> (7 * sub)) & 0x7F) as usize];
                for j in 0..8 {
                    let mag = f64::from((point >> (8 * j)) as u8);
                    let val = if (signs >> j) & 1 != 0 { -mag } else { mag };
                    let elem = blk * SUPER + group * 32 + sub * 8 + j;
                    let ab = (token * abpr + elem / 32) * 36;
                    let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
                    let aq = f64::from(act[ab + 4 + elem % 32] as i8);
                    let contribution = db * val * ad * aq;
                    sum += contribution;
                    magnitude += contribution.abs();
                }
            }
        }
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for an IQ2_XS weight
/// against a Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_iq2_xs`
/// in `src/quant/cpu/kernels/dequant_iq2.rs`: per 256-element block of 74
/// bytes, `d`@0 (f16), `qs[64]`@2 read as 32 little-endian `u16`, then
/// `scales[8]`@66. Each `u16` holds a 9-bit INDEX into `IQ2XS_GRID`, whose
/// entry expands to eight magnitude bytes, under a 7-bit index into `KSIGNS`,
/// one sign bit per expanded component. The scale for entry `e` is the 4-bit
/// field `k = e / 2` of `scales`, packed two per byte — `scales[k / 2]`,
/// nibble `k % 2` — giving `d * (0.5 + s) * 0.25` over 16 elements.
#[cfg(feature = "cuda")]
fn iq2_xs_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    const SUPER: usize = 256;
    const BYTES: usize = 74;
    let bpr = k / SUPER;
    let abpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for blk in 0..bpr {
        let wb = (feat * bpr + blk) * BYTES;
        let d = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        for entry in 0..32 {
            let q = u16::from_le_bytes([weight[wb + 2 + entry * 2], weight[wb + 3 + entry * 2]]);
            let point = IQ2XS_GRID[usize::from(q & 511)];
            let signs = KSIGNS[usize::from(q >> 9)];
            let sk = entry / 2;
            let s = (weight[wb + 66 + sk / 2] >> (4 * (sk % 2))) & 0x0F;
            let db = d * (0.5 + f64::from(s)) * 0.25;
            for j in 0..8 {
                let mag = f64::from((point >> (8 * j)) as u8);
                let val = if (signs >> j) & 1 != 0 { -mag } else { mag };
                let elem = blk * SUPER + entry * 8 + j;
                let ab = (token * abpr + elem / 32) * 36;
                let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
                let aq = f64::from(act[ab + 4 + elem % 32] as i8);
                let contribution = db * val * ad * aq;
                sum += contribution;
                magnitude += contribution.abs();
            }
        }
    }
    (sum, magnitude)
}

/// Exact reference for one output element, in f64, for an IQ2_S weight against
/// a Q8_1 activation, plus the accumulated magnitude of the sum.
///
/// Dequant math and byte offsets are ground-truthed against `dequant_iq2_s` in
/// `src/quant/cpu/kernels/dequant_iq2.rs`: per 256-element block of 82 bytes,
/// `d`@0 (f16), `qs[32]`@2, `signs[32]`@34, `qh[8]`@66, `scales[8]`@74. Entry
/// `e` takes eight index bits from `qs[e]` and two more from field
/// `2 * (e % 4)` of `qh[e / 4]`, selecting one of the 1024 points of
/// `IQ2S_GRID`. Its signs are the explicit bits of `signs[e]`, with no sign
/// table. The scale is the 4-bit field `k = e / 2` of `scales`, packed two per
/// byte, giving `d * (0.5 + s) * 0.25` over 16 elements.
#[cfg(feature = "cuda")]
fn iq2_s_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    const SUPER: usize = 256;
    const BYTES: usize = 82;
    let bpr = k / SUPER;
    let abpr = k / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for blk in 0..bpr {
        let wb = (feat * bpr + blk) * BYTES;
        let d = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        for entry in 0..32 {
            let high = usize::from((weight[wb + 66 + entry / 4] >> (2 * (entry % 4))) & 0x03);
            let point = IQ2S_GRID[usize::from(weight[wb + 2 + entry]) | (high << 8)];
            let signs = weight[wb + 34 + entry];
            let sk = entry / 2;
            let s = (weight[wb + 74 + sk / 2] >> (4 * (sk % 2))) & 0x0F;
            let db = d * (0.5 + f64::from(s)) * 0.25;
            for j in 0..8 {
                let mag = f64::from((point >> (8 * j)) as u8);
                let val = if (signs >> j) & 1 != 0 { -mag } else { mag };
                let elem = blk * SUPER + entry * 8 + j;
                let ab = (token * abpr + elem / 32) * 36;
                let ad = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
                let aq = f64::from(act[ab + 4 + elem % 32] as i8);
                let contribution = db * val * ad * aq;
                sum += contribution;
                magnitude += contribution.abs();
            }
        }
    }
    (sum, magnitude)
}

/// The quantized operands and the shape one reference sweep runs over. Bundled
/// because every reference reads the same five values.
#[cfg(feature = "cuda")]
struct RefCase<'a> {
    weight: &'a [u8],
    act: &'a [u8],
    m: usize,
    n: usize,
    k: usize,
}

/// Checks sampled outputs against the f64 reference for `format`
/// ([`q8_0_reference`], [`q4_0_reference`], [`q4_1_reference`],
/// [`q5_0_reference`], [`q5_1_reference`], [`q4_k_reference`],
/// [`q5_k_reference`], [`q6_k_reference`], [`q3_k_reference`],
/// [`q2_k_reference`], [`iq4_nl_reference`], [`iq4_xs_reference`],
/// [`iq2_xxs_reference`], [`iq2_xs_reference`] or [`iq2_s_reference`]),
/// panicking with the position and both values on the first breach.
#[cfg(feature = "cuda")]
fn check_against_reference(label: &str, format: MmqFormat, got: &[f32], case: &RefCase) {
    let rtol = REFERENCE_RTOL;
    let reference: ReferenceFn = match format {
        MmqFormat::Q8_0 => q8_0_reference,
        MmqFormat::Q40 => q4_0_reference,
        MmqFormat::Q4K => q4_k_reference,
        MmqFormat::Q5K => q5_k_reference,
        MmqFormat::Q6K => q6_k_reference,
        MmqFormat::Q3K => q3_k_reference,
        MmqFormat::Q2K => q2_k_reference,
        MmqFormat::Q41 => q4_1_reference,
        MmqFormat::Q50 => q5_0_reference,
        MmqFormat::Q51 => q5_1_reference,
        MmqFormat::IQ4NL => iq4_nl_reference,
        MmqFormat::IQ4XS => iq4_xs_reference,
        MmqFormat::IQ2XXS => iq2_xxs_reference,
        MmqFormat::IQ2XS => iq2_xs_reference,
        MmqFormat::IQ2S => iq2_s_reference,
    };
    let RefCase {
        weight,
        act,
        m,
        n,
        k,
    } = *case;
    let total = m * n;
    let stride = (total / REFERENCE_SAMPLES).max(1);
    let mut checked = 0usize;
    // Tracked so the line below reports headroom against the bound, which is
    // what says whether the bound is calibrated or merely passing.
    let mut worst = 0.0f64;
    for idx in (0..total).step_by(stride) {
        let (token, feat) = (idx / n, idx % n);
        let (want, magnitude) = reference(weight, act, token, feat, k);
        let have = f64::from(got[idx]);
        let err = (have - want).abs() / magnitude.max(f64::MIN_POSITIVE);
        assert!(
            err <= rtol,
            "{label} disagrees with the f64 reference at (token={token}, feat={feat}): \
             got {have}, want {want}, magnitude-relative error {err:.3e} exceeds {rtol:.0e}"
        );
        worst = worst.max(err);
        checked += 1;
    }
    println!(
        "{label}: {checked} sampled outputs within {rtol:.0e} of the f64 reference \
         (worst {worst:.2e})"
    );
}

/// Calls timed per kernel, after warmup.
#[cfg(feature = "cuda")]
const ITERS: usize = 100;
/// Calls made before timing starts, to cover module load and any autotune.
#[cfg(feature = "cuda")]
const WARMUP: usize = 20;

/// Deterministic pseudo-random `i8` quant, varied by block index and
/// position so no permutation of the payload coincides with another.
#[cfg(feature = "cuda")]
fn quant_byte(block: usize, pos: usize) -> i8 {
    (((block * 131 + pos * 17) % 251) as i32 - 125) as i8
}

/// Plausible per-block f16 scale, varied by block index.
#[cfg(feature = "cuda")]
fn block_scale(block: usize) -> half::f16 {
    half::f16::from_f32(0.01 + (block as f32 * 0.003) % 0.5)
}

/// Plausible per-block f16 minimum for the additive formats Q4_1 and Q5_1,
/// varied by block index and signed so the minimum term cannot cancel to a
/// constant across blocks.
#[cfg(feature = "cuda")]
fn block_min(block: usize) -> half::f16 {
    half::f16::from_f32(-0.4 + (block as f32 * 0.011) % 0.8)
}

/// The MMQ formats this tool can compare, selected by `--format`.
///
/// Kernel names and module constants come from
/// `src/quant/cuda/quant_matmul/format_dispatch.rs`'s `dispatch_matmul`,
/// which is the authoritative dispatch this tool mirrors.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum MmqFormat {
    Q8_0,
    Q40,
    Q4K,
    Q5K,
    Q6K,
    Q3K,
    Q2K,
    Q41,
    Q50,
    Q51,
    IQ4NL,
    IQ4XS,
    IQ2XXS,
    IQ2XS,
    IQ2S,
}

#[cfg(feature = "cuda")]
impl MmqFormat {
    fn parse(s: &str) -> Self {
        match s {
            "q8_0" => MmqFormat::Q8_0,
            "q4_0" => MmqFormat::Q40,
            "q4_k" => MmqFormat::Q4K,
            "q5_k" => MmqFormat::Q5K,
            "q6_k" => MmqFormat::Q6K,
            "q3_k" => MmqFormat::Q3K,
            "q2_k" => MmqFormat::Q2K,
            "q4_1" => MmqFormat::Q41,
            "q5_0" => MmqFormat::Q50,
            "q5_1" => MmqFormat::Q51,
            "iq4_nl" => MmqFormat::IQ4NL,
            "iq4_xs" => MmqFormat::IQ4XS,
            "iq2_xxs" => MmqFormat::IQ2XXS,
            "iq2_xs" => MmqFormat::IQ2XS,
            "iq2_s" => MmqFormat::IQ2S,
            other => panic!(
                "unknown --format {other}, expected one of: \
                 q8_0, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k, q3_k, q2_k, iq4_nl, iq4_xs, \
                 iq2_xxs, iq2_xs, iq2_s"
            ),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            MmqFormat::Q8_0 => "q8_0",
            MmqFormat::Q40 => "q4_0",
            MmqFormat::Q4K => "q4_k",
            MmqFormat::Q5K => "q5_k",
            MmqFormat::Q6K => "q6_k",
            MmqFormat::Q3K => "q3_k",
            MmqFormat::Q2K => "q2_k",
            MmqFormat::Q41 => "q4_1",
            MmqFormat::Q50 => "q5_0",
            MmqFormat::Q51 => "q5_1",
            MmqFormat::IQ4NL => "iq4_nl",
            MmqFormat::IQ4XS => "iq4_xs",
            MmqFormat::IQ2XXS => "iq2_xxs",
            MmqFormat::IQ2XS => "iq2_xs",
            MmqFormat::IQ2S => "iq2_s",
        }
    }

    /// Elements per weight block: 32 for the legacy formats Q8_0, Q4_0, Q4_1,
    /// Q5_0, Q5_1 and IQ4_NL, 256 for the K-quant super-blocks, IQ4_XS and the
    /// IQ2 formats.
    /// `k` must be a whole number of these, which
    /// mirrors the `k.is_multiple_of(...)` guards in `dispatch_matmul` and the
    /// `k_multiple` field of each `FeatMajorFormat`.
    fn block_elems(&self) -> usize {
        match self {
            MmqFormat::Q8_0
            | MmqFormat::Q40
            | MmqFormat::Q41
            | MmqFormat::Q50
            | MmqFormat::Q51
            | MmqFormat::IQ4NL => 32,
            MmqFormat::Q4K
            | MmqFormat::Q5K
            | MmqFormat::Q6K
            | MmqFormat::Q3K
            | MmqFormat::Q2K
            | MmqFormat::IQ4XS
            | MmqFormat::IQ2XXS
            | MmqFormat::IQ2XS
            | MmqFormat::IQ2S => 256,
        }
    }

    /// Token-major dp4a MMQ kernel, or `None` for a format that has none.
    /// Q4_0, Q4_1, Q5_0, Q5_1, Q5_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ2_XXS,
    /// IQ2_XS and IQ2_S have no `quant_mmq_*_q8_1` twin: their only
    /// pre-feature-major GEMM path is the dequantize-then-f32 kernel, which
    /// this tool does not time.
    fn dp4a_kernel(&self) -> Option<&'static str> {
        match self {
            MmqFormat::Q8_0 => Some("quant_mmq_q8_0_q8_1"),
            MmqFormat::Q40 => None,
            MmqFormat::Q4K => Some("quant_mmq_q4_k_q8_1"),
            MmqFormat::Q5K => None,
            MmqFormat::Q6K => Some("quant_mmq_q6_k_q8_1"),
            MmqFormat::Q3K => None,
            MmqFormat::Q2K => None,
            MmqFormat::Q41
            | MmqFormat::Q50
            | MmqFormat::Q51
            | MmqFormat::IQ4NL
            | MmqFormat::IQ4XS
            | MmqFormat::IQ2XXS
            | MmqFormat::IQ2XS
            | MmqFormat::IQ2S => None,
        }
    }

    /// Token-major tensor-core MMQ kernel, or `None` for a format that has
    /// none. Q4_0, Q4_1, Q5_0, Q5_1, Q5_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS,
    /// IQ2_XXS, IQ2_XS and IQ2_S have no `_mma` twin; all twelve went straight
    /// to the feature-major family.
    fn mma_kernel(&self) -> Option<&'static str> {
        match self {
            MmqFormat::Q8_0 => Some("quant_mmq_q8_0_q8_1_mma"),
            MmqFormat::Q40 => None,
            MmqFormat::Q4K => Some("quant_mmq_q4_k_q8_1_mma"),
            MmqFormat::Q5K => None,
            MmqFormat::Q6K => Some("quant_mmq_q6_k_q8_1_mma"),
            MmqFormat::Q3K => None,
            MmqFormat::Q2K => None,
            MmqFormat::Q41
            | MmqFormat::Q50
            | MmqFormat::Q51
            | MmqFormat::IQ4NL
            | MmqFormat::IQ4XS
            | MmqFormat::IQ2XXS
            | MmqFormat::IQ2XS
            | MmqFormat::IQ2S => None,
        }
    }

    /// Format name inside the feature-major kernel symbols. `None` marks a
    /// format the family does not compile; every format this tool knows is
    /// compiled today. Mirrors the `FeatMajorFormat`
    /// constants in `src/quant/cuda/quant_matmul/mmq_feat_major.rs` and the
    /// `MMQ_FM_KERNEL` instantiations in `quant_mmq_mma.cu`.
    fn feat_major_infix(&self) -> Option<&'static str> {
        match self {
            MmqFormat::Q8_0 => Some("q8_0"),
            MmqFormat::Q40 => Some("q4_0"),
            MmqFormat::Q4K => Some("q4_k"),
            MmqFormat::Q5K => Some("q5_k"),
            MmqFormat::Q6K => Some("q6_k"),
            MmqFormat::Q3K => Some("q3_k"),
            MmqFormat::Q2K => Some("q2_k"),
            MmqFormat::Q41 => Some("q4_1"),
            MmqFormat::Q50 => Some("q5_0"),
            MmqFormat::Q51 => Some("q5_1"),
            MmqFormat::IQ4NL => Some("iq4_nl"),
            MmqFormat::IQ4XS => Some("iq4_xs"),
            MmqFormat::IQ2XXS => Some("iq2_xxs"),
            MmqFormat::IQ2XS => Some("iq2_xs"),
            MmqFormat::IQ2S => Some("iq2_s"),
        }
    }

    /// Weight row stride in the shared tile, in ints, mirroring the
    /// `FeatMajorFormat` constants. Q8_0 stages 64 quant words, 8 f32 scales
    /// and 4 ints of bank padding; Q4_0 stages that row byte for byte, biasing
    /// its unsigned nibbles by 8 so the staged lanes are signed. Q4_K stages 64
    /// quant words, 8 `float2` scale/min pairs (16 ints) and 4 ints of padding
    /// — its pair is two f32
    /// rather than one packed word, so its row is 8 ints wider. Q5_K stages
    /// the Q4_K row verbatim; its extra bit changes staging, not layout. Q6_K
    /// reaches the same 84 by a different split: 64 quant words, 16 f32 group
    /// scales (its scale changes every 16 elements) and 4 ints of padding.
    /// Q3_K stages the Q6_K row verbatim: it shares that 16-element scale
    /// granularity and, like Q6_K, has no minimum term. Q2_K is the widest at
    /// 100: 64 quant words, 16 `float2` scale/min pairs (32 ints, twice Q4_K's
    /// because the granularity is 16 rather than 32) and 4 ints of padding.
    /// Q5_0 stages the Q8_0 row, biasing its 5-bit quants by 16. Q4_1 and Q5_1
    /// stage the Q4_K row instead, because both carry a minimum: their pair is
    /// `(d, +m)`, not Q4_K's `(d * sc, -dmin * m)`, since their value is
    /// `d * q + m` rather than `d * sc * q - dmin * m`. IQ4_NL and IQ4_XS
    /// stage the Q8_0 row too: their 4-bit fields are indices into a 16-entry
    /// SIGNED codebook, so the resolved lanes need no bias, and IQ4_XS's scale
    /// changes every 32 elements, which is the granularity the shared
    /// `vec_dot` already indexes at. IQ2_XXS stages the Q8_0 row as well: its
    /// grid point expands to eight magnitudes and its sign table supplies one
    /// bit each, both folded in during staging, and its scale granularity is
    /// also 32 elements. IQ2_XS and IQ2_S expand the same 8-component grid but
    /// stage the Q6_K row: their 4-bit scale is packed two to a `scales` byte,
    /// one per two grid entries, so it changes every 16 elements and needs the
    /// 16 f32 record.
    fn feat_major_x_stride(&self) -> u32 {
        match self {
            MmqFormat::Q8_0
            | MmqFormat::Q40
            | MmqFormat::Q50
            | MmqFormat::IQ4NL
            | MmqFormat::IQ4XS
            | MmqFormat::IQ2XXS => 76,
            MmqFormat::Q4K
            | MmqFormat::Q5K
            | MmqFormat::Q6K
            | MmqFormat::Q3K
            | MmqFormat::Q41
            | MmqFormat::Q51
            | MmqFormat::IQ2XS
            | MmqFormat::IQ2S => 84,
            MmqFormat::Q2K => 100,
        }
    }

    /// Per-token ints of shared scratch after the activation tile, mirroring
    /// `FeatMajorFormat::act_scratch_ints_per_token`. Only Q2_K asks for it:
    /// its minimum changes every 16 elements while the activation record
    /// stores one sum per 32, so the kernel derives the per-16 split into this
    /// region once per staged activation tile.
    fn feat_major_act_scratch(&self) -> u32 {
        match self {
            MmqFormat::Q2K => 4,
            _ => 0,
        }
    }

    fn build_weight(&self, n: usize, k: usize) -> Vec<u8> {
        match self {
            MmqFormat::Q8_0 => build_q8_0_weight(n, k),
            MmqFormat::Q40 => build_q4_0_weight(n, k),
            MmqFormat::Q4K => build_q4_k_weight(n, k),
            MmqFormat::Q5K => build_q5_k_weight(n, k),
            MmqFormat::Q6K => build_q6_k_weight(n, k),
            MmqFormat::Q3K => build_q3_k_weight(n, k),
            MmqFormat::Q2K => build_q2_k_weight(n, k),
            MmqFormat::Q41 => build_q4_1_weight(n, k),
            MmqFormat::Q50 => build_q5_0_weight(n, k),
            MmqFormat::Q51 => build_q5_1_weight(n, k),
            MmqFormat::IQ4NL => build_iq4_nl_weight(n, k),
            MmqFormat::IQ4XS => build_iq4_xs_weight(n, k),
            MmqFormat::IQ2XXS => build_iq2_xxs_weight(n, k),
            MmqFormat::IQ2XS => build_iq2_xs_weight(n, k),
            MmqFormat::IQ2S => build_iq2_s_weight(n, k),
        }
    }
}

/// Builds a Q8_0 weight buffer: `n * (k / 32)` blocks of 34 bytes, half scale
/// at byte 0, 32 `i8` quants at byte 2.
#[cfg(feature = "cuda")]
fn build_q8_0_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; n * bpr * 34];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 34;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            for pos in 0..32 {
                out[base + 2 + pos] = quant_byte(block, pos) as u8;
            }
        }
    }
    out
}

/// Builds a Q4_0 weight buffer: `n * (k / 32)` blocks of 18 bytes, half scale
/// at byte 0, 16 nibble-packed quant bytes at byte 2.
///
/// The nibble layout is the inverse of `dequant_q4_0` in
/// `src/quant/cpu/kernels/dequant_simple.rs`: element `j` (0..15) goes in the
/// LOW nibble of `qs[j]` and element `j + 16` in the HIGH nibble of the same
/// byte. Quants are unsigned 0..15; the dequantizer subtracts 8, so the value
/// this encodes is `d * (q - 8)`.
#[cfg(feature = "cuda")]
fn build_q4_0_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; n * bpr * 18];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 18;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            for j in 0..16 {
                let lo = quant_byte(block, j) as u8 & 0x0F;
                let hi = quant_byte(block, j + 16) as u8 & 0x0F;
                out[base + 2 + j] = lo | (hi << 4);
            }
        }
    }
    out
}

/// Builds a Q4_1 weight buffer: `n * (k / 32)` blocks of 20 bytes, half scale
/// at byte 0, half minimum at byte 2, 16 nibble-packed quant bytes at byte 4.
///
/// The nibble layout is the inverse of `dequant_q4_1` in
/// `src/quant/cpu/kernels/dequant_simple.rs`: element `j` (0..15) goes in the
/// LOW nibble of `qs[j]` and element `j + 16` in the HIGH nibble of the same
/// byte. Quants are unsigned 0..15 and the value is `d * q + m`.
#[cfg(feature = "cuda")]
fn build_q4_1_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; n * bpr * 20];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 20;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&block_min(block).to_le_bytes());
            for j in 0..16 {
                let lo = quant_byte(block, j) as u8 & 0x0F;
                let hi = quant_byte(block, j + 16) as u8 & 0x0F;
                out[base + 4 + j] = lo | (hi << 4);
            }
        }
    }
    out
}

/// Builds a Q5_0 weight buffer: `n * (k / 32)` blocks of 22 bytes, half scale
/// at byte 0, 32-bit fifth-bit field at byte 2, 16 low-nibble bytes at byte 6.
///
/// The layout is the inverse of `dequant_q5_0` in
/// `src/quant/cpu/kernels/dequant_simple.rs`: element `j` (0..15) takes the
/// LOW nibble of `qs[j]` and bit `j` of `qh`, element `j + 16` the HIGH nibble
/// of the same byte and bit `j + 16`. Quants are unsigned 0..31; the
/// dequantizer subtracts 16, so this encodes `d * (q - 16)`.
#[cfg(feature = "cuda")]
fn build_q5_0_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; n * bpr * 22];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 22;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            let mut qh = 0u32;
            for j in 0..16 {
                let lo = quant_byte(block, j) as u8 & 0x1F;
                let hi = quant_byte(block, j + 16) as u8 & 0x1F;
                out[base + 6 + j] = (lo & 0x0F) | ((hi & 0x0F) << 4);
                qh |= u32::from(lo >> 4) << j;
                qh |= u32::from(hi >> 4) << (j + 16);
            }
            out[base + 2..base + 6].copy_from_slice(&qh.to_le_bytes());
        }
    }
    out
}

/// Builds a Q5_1 weight buffer: `n * (k / 32)` blocks of 24 bytes, half scale
/// at byte 0, half minimum at byte 2, 32-bit fifth-bit field at byte 4, 16
/// low-nibble bytes at byte 8.
///
/// The layout is the inverse of `dequant_q5_1` in
/// `src/quant/cpu/kernels/dequant_simple.rs`. The 5-bit assembly is Q5_0's;
/// quants are unsigned 0..31 and the value is `d * q + m`.
#[cfg(feature = "cuda")]
fn build_q5_1_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; n * bpr * 24];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 24;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&block_min(block).to_le_bytes());
            let mut qh = 0u32;
            for j in 0..16 {
                let lo = quant_byte(block, j) as u8 & 0x1F;
                let hi = quant_byte(block, j + 16) as u8 & 0x1F;
                out[base + 8 + j] = (lo & 0x0F) | ((hi & 0x0F) << 4);
                qh |= u32::from(lo >> 4) << j;
                qh |= u32::from(hi >> 4) << (j + 16);
            }
            out[base + 4..base + 8].copy_from_slice(&qh.to_le_bytes());
        }
    }
    out
}

/// Builds an IQ4_NL weight buffer: `n * (k / 32)` blocks of 18 bytes, half
/// scale at byte 0, 16 nibble-packed codebook indices at byte 2.
///
/// The layout is the inverse of `dequant_iq4_nl` in
/// `src/quant/cpu/kernels/dequant_iq4.rs`: element `j` (0..15) goes in the LOW
/// nibble of `qs[j]` and element `j + 16` in the HIGH nibble of the same byte.
/// Each nibble is an INDEX into `KVALUES_IQ4NL`, so `quant_byte` is reduced to
/// the 0..15 index range rather than masked as a magnitude.
#[cfg(feature = "cuda")]
fn build_iq4_nl_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; n * bpr * 18];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 18;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            for j in 0..16 {
                let lo = quant_byte(block, j) as u8 & 0x0F;
                let hi = quant_byte(block, j + 16) as u8 & 0x0F;
                out[base + 2 + j] = lo | (hi << 4);
            }
        }
    }
    out
}

/// Builds an IQ4_XS weight buffer: `n * (k / 256)` super-blocks of 136 bytes —
/// half `d` at byte 0, `scales_h` (u16) at byte 2, `scales_l[4]` at byte 4,
/// 128 nibble-packed codebook indices at byte 8.
///
/// The layout is the inverse of `dequant_iq4_xs` in
/// `src/quant/cpu/kernels/dequant_iq4.rs`. Each sub-block's 6-bit scale is
/// split into a `scales_l` nibble and two `scales_h` bits. The values chosen
/// sweep the whole 0..63 range and skip 32, so every `scales_h` bit pattern is
/// exercised and no sub-block scale is zero — a dropped or shifted scale
/// cannot cancel out.
#[cfg(feature = "cuda")]
fn build_iq4_xs_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 256;
    let mut out = vec![0u8; n * bpr * 136];
    for row in 0..n {
        for s in 0..bpr {
            let sup = row * bpr + s;
            let base = sup * 136;
            out[base..base + 2].copy_from_slice(&block_scale(sup).to_le_bytes());
            let mut scales_h = 0u16;
            for sb in 0..8 {
                // 6-bit scale, varied per sub-block and super-block. 32 is
                // skipped because `d * (32 - 32)` zeroes the sub-block.
                let raw = ((sup * 7 + sb * 5) % 63) as u8;
                let ls = if raw >= 32 { raw + 1 } else { raw };
                out[base + 4 + sb / 2] |= (ls & 0x0F) << (4 * (sb % 2));
                scales_h |= u16::from((ls >> 4) & 0x03) << (2 * sb);
            }
            out[base + 2..base + 4].copy_from_slice(&scales_h.to_le_bytes());
            for sb in 0..8 {
                for j in 0..16 {
                    let lo = quant_byte(sup, sb * 32 + j) as u8 & 0x0F;
                    let hi = quant_byte(sup, sb * 32 + j + 16) as u8 & 0x0F;
                    out[base + 8 + sb * 16 + j] = lo | (hi << 4);
                }
            }
        }
    }
    out
}

/// Builds an IQ2_XXS weight buffer: `n * (k / 256)` blocks of 66 bytes, half
/// scale at byte 0, then eight pairs of `u32` at byte 2.
///
/// The layout is the inverse of `dequant_iq2_xxs` in
/// `src/quant/cpu/kernels/dequant_iq2.rs`. Every grid index 0..255 is legal, so
/// the payload bytes go in unmasked; each sign field is masked to its 7 bits
/// and each group's 4-bit scale is varied so no two groups of a block share
/// one — a dropped or shifted scale cannot cancel out. No scale is degenerate:
/// `d * (0.5 + s) * 0.25` is non-zero even at `s = 0`.
#[cfg(feature = "cuda")]
fn build_iq2_xxs_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 256;
    let mut out = vec![0u8; n * bpr * 66];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 66;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            for group in 0..8 {
                let mut indices = 0u32;
                let mut aux = 0u32;
                for sub in 0..4 {
                    let idx = quant_byte(block, group * 4 + sub) as u8;
                    indices |= u32::from(idx) << (8 * sub);
                    let signs = quant_byte(block, 32 + group * 4 + sub) as u8 & 0x7F;
                    aux |= u32::from(signs) << (7 * sub);
                }
                aux |= (((block * 3 + group) % 16) as u32) << 28;
                let off = base + 2 + group * 8;
                out[off..off + 4].copy_from_slice(&indices.to_le_bytes());
                out[off + 4..off + 8].copy_from_slice(&aux.to_le_bytes());
            }
        }
    }
    out
}

/// Builds an IQ2_XS weight buffer: `n * (k / 256)` blocks of 74 bytes — half
/// scale at byte 0, 32 `u16` at byte 2, 8 packed scale bytes at byte 66.
///
/// The layout is the inverse of `dequant_iq2_xs` in
/// `src/quant/cpu/kernels/dequant_iq2.rs`. Every 9-bit grid index 0..511 is
/// legal and every 7-bit sign index is, so both fields go in masked only to
/// their widths. Each entry's 4-bit scale is varied so no two 16-element
/// groups of a block share one — a dropped or shifted scale cannot cancel out.
/// No scale is degenerate: `d * (0.5 + s) * 0.25` is non-zero even at `s = 0`.
#[cfg(feature = "cuda")]
fn build_iq2_xs_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 256;
    let mut out = vec![0u8; n * bpr * 74];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 74;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            for entry in 0..32 {
                let idx = u16::from(quant_byte(block, entry) as u8)
                    | (u16::from(quant_byte(block, 64 + entry) as u8 & 0x01) << 8);
                let signs = u16::from(quant_byte(block, 32 + entry) as u8 & 0x7F);
                let q = (idx & 511) | (signs << 9);
                out[base + 2 + entry * 2..base + 4 + entry * 2].copy_from_slice(&q.to_le_bytes());
            }
            // One 4-bit scale per 16-element group, two groups per byte.
            for sk in 0..16 {
                let s = ((block * 5 + sk * 3) % 16) as u8;
                out[base + 66 + sk / 2] |= s << (4 * (sk % 2));
            }
        }
    }
    out
}

/// Builds an IQ2_S weight buffer: `n * (k / 256)` blocks of 82 bytes — half
/// scale at byte 0, `qs[32]` at byte 2, `signs[32]` at byte 34, `qh[8]` at
/// byte 66, 8 packed scale bytes at byte 74.
///
/// The layout is the inverse of `dequant_iq2_s` in
/// `src/quant/cpu/kernels/dequant_iq2.rs`. Every 10-bit grid index 0..1023 is
/// legal and every sign byte is, so both go in unmasked. Each entry's 4-bit
/// scale is varied so no two 16-element groups of a block share one, and the
/// two `qh` bits are varied per entry so the high half of the index is
/// exercised rather than left at zero.
#[cfg(feature = "cuda")]
fn build_iq2_s_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 256;
    let mut out = vec![0u8; n * bpr * 82];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 82;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            for entry in 0..32 {
                out[base + 2 + entry] = quant_byte(block, entry) as u8;
                out[base + 34 + entry] = quant_byte(block, 32 + entry) as u8;
                let high = ((block + entry) % 4) as u8;
                out[base + 66 + entry / 4] |= high << (2 * (entry % 4));
            }
            // One 4-bit scale per 16-element group, two groups per byte.
            for sk in 0..16 {
                let s = ((block * 5 + sk * 3) % 16) as u8;
                out[base + 74 + sk / 2] |= s << (4 * (sk % 2));
            }
        }
    }
    out
}

/// Builds a Q8_1 activation buffer: `m * (k / 32)` blocks of 36 bytes, half
/// scale at byte 0, block sum at byte 2 (unused by the kernel), 32 `i8`
/// quants at byte 4.
#[cfg(feature = "cuda")]
fn build_q8_1_activation(m: usize, k: usize) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; m * bpr * 36];
    for row in 0..m {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 36;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&0i16.to_le_bytes());
            for pos in 0..32 {
                // Offset the position stream so activation and weight quants
                // never coincide, even for the same block index.
                out[base + 4 + pos] = quant_byte(block, pos + 1000) as u8;
            }
        }
    }
    out
}

/// Repacks the per-token Q8_1 buffer into the layout the feature-major kernels
/// read.
///
/// Mirrors `quantize_f32_q8_1_mmq` in `src/quant/cuda/kernels/quant_act.cu`: a
/// 144-byte record holds 128 k-values of one token as four header words then
/// 128 int8, and records are indexed `kgroup * ntok + token`. Per 32-value
/// sub-block the header word is `half` `d` in bits 0..15 (bytes 0..1,
/// little-endian) and the int16 sum of that sub-block's 32 clamped int8 quants
/// in bits 16..31 (bytes 2..3). The sum is a raw integer, never a float:
/// |sum| <= 32 * 128 = 4096 fits int16 exactly, and the Q4_K min correction
/// needs it exact. Deriving it from the per-token bytes here, rather than
/// re-quantizing, keeps this example and the f64 reference reading one set of
/// quantized values; `d` is read from the per-token block header, which is
/// already a half value, so it round-trips exactly.
///
/// This is a CPU mirror of a GPU kernel: the two encode the same bytes and
/// MUST change together, which is what caused this function to fall out of
/// sync with `quantize_f32_q8_1_mmq` once.
#[cfg(feature = "cuda")]
fn repack_q8_1_mmq(act: &[u8], m: usize, k: usize, ntok: usize) -> Vec<u8> {
    let bpr = k / 32;
    let kgroups = bpr.div_ceil(4);
    let mut out = vec![0u8; kgroups * ntok * 144];
    for token in 0..m {
        for b in 0..bpr {
            let src = (token * bpr + b) * 36;
            let rec = ((b / 4) * ntok + token) * 144;
            let sub = b % 4;
            let d = half::f16::from_le_bytes([act[src], act[src + 1]]).to_f32();
            let sum: i32 = act[src + 4..src + 36]
                .iter()
                .map(|&byte| i32::from(byte as i8))
                .sum();
            let slot = rec + sub * 4;
            out[slot..slot + 2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
            out[slot + 2..slot + 4].copy_from_slice(&(sum as i16).to_le_bytes());
            out[rec + 16 + sub * 32..rec + 16 + sub * 32 + 32]
                .copy_from_slice(&act[src + 4..src + 36]);
        }
    }
    out
}

/// Builds a Q4_K weight buffer: `n * (k / 256)` super-blocks of 144 bytes.
///
/// Layout (authoritative source: `src/quant/cpu/kernels/quantize/q4k_q5k.rs`
/// doc comment on `quantize_q4k`, and the byte offsets its `fit_super_block`
/// writes at lines 195-215): f16 `d`@0, f16 `dmin`@2, a 12-byte packed 6-bit
/// scale/min array@4, 128-byte nibble-packed quants@16.
///
/// The 12-byte scale/min packing below is copied from `fit_super_block`
/// (lines 196-211 of that file), which is the inverse of the unpacker
/// `q4k_scale_min` in `src/quant/cuda/kernels/decode.cuh` (lines 97-107) that
/// both the dp4a and MMA MMQ kernels call.
///
/// The nibble layout for sub-block `j` (0..8) is copied from
/// `quant_mmq_q4_k_q8_1` in `src/quant/cuda/kernels/quant_gemv.cu`
/// (`blk + 16 + (j / 2) * 32`, low nibble for even `j`, high for odd) and
/// from the identical byte indexing `mmq_q4_k_stage_load` uses in
/// `src/quant/cuda/kernels/quant_mmq_mma.cu`.
#[cfg(feature = "cuda")]
fn build_q4_k_weight(n: usize, k: usize) -> Vec<u8> {
    const SUPER: usize = 256;
    const BYTES: usize = 144;
    let bpr = k / SUPER;
    let mut out = vec![0u8; n * bpr * BYTES];
    for row in 0..n {
        for sup in 0..bpr {
            let block = row * bpr + sup;
            let base = block * BYTES;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&block_scale(block * 7 + 1).to_le_bytes());

            // Eight 6-bit scales and eight 6-bit minimums, one pair per
            // 32-element sub-block.
            let mut scale = [0u8; 8];
            let mut min = [0u8; 8];
            for j in 0..8 {
                scale[j] = ((block * 11 + j * 5) % 64) as u8;
                min[j] = ((block * 13 + j * 3 + 1) % 64) as u8;
            }
            // Pack per `fit_super_block`: low four pairs are plain 6-bit
            // values in sc[0..4]/sc[4..8]; high four pairs split their low
            // nibble into sc[8..12] and their top two bits into the spare
            // high bits of sc[0..4] (scales) and sc[4..8] (minimums).
            let mut sc = [0u8; 12];
            for idx in 0..4 {
                sc[idx] = scale[idx] & 0x3F;
                sc[4 + idx] = min[idx] & 0x3F;
            }
            for idx in 0..4 {
                let j = 4 + idx;
                sc[8 + idx] = (scale[j] & 0x0F) | ((min[j] & 0x0F) << 4);
                sc[idx] |= (scale[j] >> 4) << 6;
                sc[4 + idx] |= (min[j] >> 4) << 6;
            }
            out[base + 4..base + 16].copy_from_slice(&sc);

            for j in 0..8 {
                let group = base + 16 + (j / 2) * 32;
                for pos in 0..32 {
                    let nib = quant_byte(block * 8 + j, pos) as u8 & 0x0F;
                    let byte_idx = group + pos;
                    out[byte_idx] = if j % 2 == 0 {
                        (out[byte_idx] & 0xF0) | nib
                    } else {
                        (out[byte_idx] & 0x0F) | (nib << 4)
                    };
                }
            }
        }
    }
    out
}

/// Builds a Q5_K weight buffer: `n * (k / 256)` super-blocks of 176 bytes.
///
/// Layout (authoritative source: `dequant_q5k` in
/// `src/quant/cpu/kernels/dequant_k_quants/q4k_q5k.rs`): f16 `d`@0, f16
/// `dmin`@2, a 12-byte packed 6-bit scale/min array@4, 32-byte `qh`@16, and
/// 128-byte nibble-packed low quants@48.
///
/// The scale/min packing is byte-for-byte Q4_K's, so it is written exactly as
/// [`build_q4_k_weight`] writes it. The quant packing is the inverse of that
/// dequantizer: sub-block PAIRS share one 32-byte run of `qs` (even sub-block
/// in the low nibbles, odd in the high nibbles of the same bytes), and the
/// fifth bit of element `l` of sub-block `j` is bit `j` of `qh[l]` — one `qh`
/// byte per element, carrying that element's fifth bit for all eight
/// sub-blocks, NOT a flat bitstream over the 256 values.
#[cfg(feature = "cuda")]
fn build_q5_k_weight(n: usize, k: usize) -> Vec<u8> {
    const SUPER: usize = 256;
    const BYTES: usize = 176;
    let bpr = k / SUPER;
    let mut out = vec![0u8; n * bpr * BYTES];
    for row in 0..n {
        for sup in 0..bpr {
            let block = row * bpr + sup;
            let base = block * BYTES;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&block_scale(block * 7 + 1).to_le_bytes());

            let mut scale = [0u8; 8];
            let mut min = [0u8; 8];
            for j in 0..8 {
                scale[j] = ((block * 11 + j * 5) % 64) as u8;
                min[j] = ((block * 13 + j * 3 + 1) % 64) as u8;
            }
            let mut sc = [0u8; 12];
            for idx in 0..4 {
                sc[idx] = scale[idx] & 0x3F;
                sc[4 + idx] = min[idx] & 0x3F;
            }
            for idx in 0..4 {
                let j = 4 + idx;
                sc[8 + idx] = (scale[j] & 0x0F) | ((min[j] & 0x0F) << 4);
                sc[idx] |= (scale[j] >> 4) << 6;
                sc[4 + idx] |= (min[j] >> 4) << 6;
            }
            out[base + 4..base + 16].copy_from_slice(&sc);

            for j in 0..8 {
                let group = base + 48 + (j / 2) * 32;
                for pos in 0..32 {
                    // Five bits, 0..31: four into `qs`, the fifth into `qh`.
                    let q = quant_byte(block * 8 + j, pos) as u8 & 0x1F;
                    let byte_idx = group + pos;
                    out[byte_idx] = if j % 2 == 0 {
                        (out[byte_idx] & 0xF0) | (q & 0x0F)
                    } else {
                        (out[byte_idx] & 0x0F) | ((q & 0x0F) << 4)
                    };
                    out[base + 16 + pos] |= (q >> 4) << j;
                }
            }
        }
    }
    out
}

/// Builds a Q6_K weight buffer: `n * (k / 256)` super-blocks of 210 bytes.
///
/// Layout (authoritative source: `src/quant/cpu/kernels/quantize/q6k.rs`
/// module doc, "Field order — the trap", and its `quantize_q6k_with` byte
/// offsets at lines 70-73): 128-byte `ql` (low nibbles)@0, 64-byte `qh`
/// (high bit pairs)@128, 16 signed 8-bit sub-block scales@192, f16 `d`@208.
/// `d` comes LAST, not first — this is the one GGML block whose scale isn't
/// at byte 0, per that file's warning.
///
/// The `ql`/`qh` bit-packing below is copied verbatim from that file's
/// `pack_q6k` function (lines 99-114), which both the dp4a
/// (`quant_mmq_q6_k_q8_1` in `quant_gemv.cu`) and MMA
/// (`mmq_q6_k_stage_load` in `quant_mmq_mma.cu`) kernels unpack identically.
#[cfg(feature = "cuda")]
fn build_q6_k_weight(n: usize, k: usize) -> Vec<u8> {
    const SUPER: usize = 256;
    const BYTES: usize = 210;
    let bpr = k / SUPER;
    let mut out = vec![0u8; n * bpr * BYTES];
    for row in 0..n {
        for sup in 0..bpr {
            let block = row * bpr + sup;
            let base = block * BYTES;

            // 256 biased 6-bit levels (0..63, i.e. a signed [-32, 31] value
            // biased by +32, matching the format's `d * scale * (q - 32)`).
            let mut levels = [0u8; SUPER];
            for (pos, level) in levels.iter_mut().enumerate() {
                *level = ((quant_byte(block, pos) as i32 + 128) % 64) as u8;
            }
            // 16 signed 8-bit sub-block scales, one per 16 elements.
            for ib in 0..16 {
                let scale = (((block * 17 + ib * 9) % 121) as i32 - 60) as i8;
                out[base + 192 + ib] = scale as u8;
            }
            out[base + 208..base + 210].copy_from_slice(&block_scale(block).to_le_bytes());

            for half in 0..2 {
                let hbase = half * 128;
                let ql = base + half * 64;
                let qh = base + 128 + half * 32;
                for l in 0..32 {
                    let q1 = levels[hbase + l];
                    let q2 = levels[hbase + l + 32];
                    let q3 = levels[hbase + l + 64];
                    let q4 = levels[hbase + l + 96];
                    out[ql + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
                    out[ql + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);
                    out[qh + l] =
                        (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
                }
            }
        }
    }
    out
}

/// Builds a Q3_K weight buffer: `n * (k / 256)` super-blocks of 110 bytes.
///
/// Layout (authoritative source: `dequant_q3k` in
/// `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`): 32-byte `hmask`@0,
/// 64-byte `qs`@32, 12 packed 6-bit signed scales@96, f16 `d`@108. `d` comes
/// LAST, not first.
///
/// This is the inverse of [`q3_k_reference`]'s unpack, and there is no Q3_K
/// quantizer in `src/quant/cpu/kernels/quantize/` to copy from, so the packing
/// is written against that dequantizer directly. Element `pos` contributes its
/// two low bits to `qs[32 * (pos / 128) + pos % 32]` at shift
/// `2 * ((pos % 128) / 32)` and its high bit to `hmask[pos % 32]` at bit
/// `4 * (pos / 128) + (pos % 128) / 32`. The high bit is INVERTED — it is SET
/// when the value is non-negative — so a target value `v` in [-4, 3] packs as
/// `low2 = (v + 4) & 3` with the bit set exactly when `v >= 0`.
#[cfg(feature = "cuda")]
fn build_q3_k_weight(n: usize, k: usize) -> Vec<u8> {
    const SUPER: usize = 256;
    const BYTES: usize = 110;
    let bpr = k / SUPER;
    let mut out = vec![0u8; n * bpr * BYTES];
    for row in 0..n {
        for sup in 0..bpr {
            let block = row * bpr + sup;
            let base = block * BYTES;

            for pos in 0..SUPER {
                // Target quant, spread over the format's whole [-4, 3] range.
                let v = ((i32::from(quant_byte(block, pos)) + 128) % 8) - 4;
                let nn = pos / 128;
                let t = (pos % 128) / 32;
                let r = pos % 32;
                out[base + 32 + 32 * nn + r] |= (((v + 4) & 3) as u8) << (2 * t);
                if v >= 0 {
                    out[base + r] |= 1u8 << (4 * nn + t);
                }
            }

            // 16 signed 6-bit scales, one per 16 elements, stored biased by 32.
            for j in 0..16 {
                let scale = ((block * 17 + j * 9) % 64) as i32 - 32;
                let u = (scale + 32) as u8;
                let g = j / 4;
                out[base + 96 + j % 4 + 4 * (g & 1)] |= (u & 0x0F) << (4 * (g / 2));
                out[base + 96 + 8 + j % 4] |= ((u >> 4) & 0x03) << (2 * g);
            }

            out[base + 108..base + 110].copy_from_slice(&block_scale(block).to_le_bytes());
        }
    }
    out
}

/// Builds a Q2_K weight buffer: `n * (k / 256)` super-blocks of 84 bytes.
///
/// Layout (authoritative source: `dequant_q2k` in
/// `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`): 16 scale/min bytes@0,
/// 64-byte `qs`@16, f16 `d`@80, f16 `dmin`@82. Both `d` and `dmin` come LAST.
///
/// This is the inverse of [`q2_k_reference`]'s unpack. Element `pos`
/// contributes its two UNSIGNED bits to
/// `qs[32 * (pos / 128) + 16 * ((pos % 32) / 16) + pos % 16]` at shift
/// `2 * ((pos % 128) / 32)`. Each scale byte packs the scale in its low nibble
/// and the minimum in its high nibble, both unsigned 0..15.
#[cfg(feature = "cuda")]
fn build_q2_k_weight(n: usize, k: usize) -> Vec<u8> {
    const SUPER: usize = 256;
    const BYTES: usize = 84;
    let bpr = k / SUPER;
    let mut out = vec![0u8; n * bpr * BYTES];
    for row in 0..n {
        for sup in 0..bpr {
            let block = row * bpr + sup;
            let base = block * BYTES;

            for pos in 0..SUPER {
                // Target quant, spread over the format's whole 0..3 range.
                let v = ((i32::from(quant_byte(block, pos)) + 128) % 4) as u8;
                let nn = pos / 128;
                let t = (pos % 128) / 32;
                let h = (pos % 32) / 16;
                let l = pos % 16;
                out[base + 16 + 32 * nn + 16 * h + l] |= v << (2 * t);
            }

            // 16 scale/min pairs, one per 16 elements. Both nibbles are kept
            // non-zero across the block so a dropped minimum term shows up.
            for j in 0..16 {
                let scale = ((block * 17 + j * 9) % 15 + 1) as u8;
                let minimum = ((block * 11 + j * 5) % 15 + 1) as u8;
                out[base + j] = scale | (minimum << 4);
            }

            out[base + 80..base + 82].copy_from_slice(&block_scale(block).to_le_bytes());
            // A distinct `dmin`, so a kernel that read `d` twice fails here.
            out[base + 82..base + 84].copy_from_slice(&block_scale(block + 3).to_le_bytes());
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn main() {
    if !numr::runtime::cuda::is_cuda_available() {
        println!("mmq_kernel_compare SKIPPED: CUDA is not available on this machine.");
        return;
    }

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut n, mut k, mut m) = (4096usize, 14336usize, 512usize);
    let mut format = MmqFormat::Q8_0;
    let mut force_mmq_x: Option<u32> = None;
    let mut stream_k = false;

    let mut i = 0;
    while i < argv.len() {
        let value = || argv.get(i + 1).expect("flag needs a value").clone();
        match argv[i].as_str() {
            "--n" => n = value().parse().expect("--n must be a usize"),
            "--k" => k = value().parse().expect("--k must be a usize"),
            "--m" => m = value().parse().expect("--m must be a usize"),
            "--format" => format = MmqFormat::parse(&value()),
            // Overrides the feature-major token-tile choice. The selection rule trades
            // tile efficiency against how many blocks the grid launches, and
            // the two pull opposite ways; this pins one so the trade can be
            // measured directly instead of inferred.
            "--mmq-x" => force_mmq_x = Some(value().parse().expect("--mmq-x must be a u32")),
            // Runs the stream-k pair alongside the tile-parallel kernel so the
            // two decompositions can be compared on one shape.
            "--stream-k" => {
                stream_k = true;
                i -= 1;
            }
            other => {
                panic!(
                    "unknown flag {other}, expected --n, --k, --m, --format, --mmq-x, or --stream-k"
                )
            }
        }
        i += 2;
    }

    let block_elems = format.block_elems();
    if k % block_elems != 0 {
        eprintln!(
            "--k must be a multiple of {block_elems} for --format {}, got {k}",
            format.label()
        );
        std::process::exit(1);
    }

    let device = CudaDevice::new(0);

    // `m16n8k32` needs sm_80. `caps.bf16` marks that floor, so a pre-Ampere
    // device skips here instead of failing to load the module.
    if !device.profile().caps.bf16 {
        println!(
            "mmq_kernel_compare SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k32` requires."
        );
        return;
    }

    println!(
        "mmq_kernel_compare: format={} n={n} k={k} m={m}",
        format.label()
    );

    let client = CudaRuntime::default_client(&device);
    client.synchronize();
    let device_index = device.id();

    let weight_bytes = format.build_weight(n, k);
    let act_bytes = build_q8_1_activation(m, k);

    let weight =
        Tensor::<CudaRuntime>::from_slice(&weight_bytes, &[weight_bytes.len()], &device).unwrap();
    let act = Tensor::<CudaRuntime>::from_slice(&act_bytes, &[act_bytes.len()], &device).unwrap();
    let out_dp4a = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();
    let out_mma = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();

    // Both feature-major paths use the same token tile, and the repacked
    // activation layout is strided by it, so choose it once and repack once.
    let fm_x_stride = format.feat_major_x_stride();
    let fm_scratch = format.feat_major_act_scratch();
    let fm_mmq_x = force_mmq_x.unwrap_or_else(|| {
        select_mmq_x(
            m as u32,
            mmq_x_smem_bytes(fm_x_stride, fm_scratch, 128),
            fm_x_stride,
            fm_scratch,
        )
        .expect("a feature-major variant fits")
    });
    let fm_ntok = (m as u32).div_ceil(fm_mmq_x) * fm_mmq_x;
    let packed_bytes = repack_q8_1_mmq(&act_bytes, m, k, fm_ntok as usize);
    let packed =
        Tensor::<CudaRuntime>::from_slice(&packed_bytes, &[packed_bytes.len()], &device).unwrap();

    let weight_ptr = weight.ptr();
    let act_ptr = act.ptr();
    let packed_ptr = packed.ptr();
    let out_dp4a_ptr = out_dp4a.ptr();
    let out_mma_ptr = out_mma.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let cfg = LaunchConfig {
        grid_dim: (n_u32.div_ceil(64), m_u32.div_ceil(128), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mma_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE)
            .expect("load mma module");

    // Token-major pair: the dp4a MMQ kernel and its tensor-core twin, which
    // share a grid and a per-token activation layout. Q4_0, Q5_K and Q3_K
    // compile NEITHER, so for those formats this whole comparison — both timings, both
    // reference checks, and the bit-for-bit agreement below — is skipped, and
    // only the feature-major kernels run. Resolving a symbol that is not compiled would
    // abort the tool instead.
    let token_major =
        format
            .dp4a_kernel()
            .zip(format.mma_kernel())
            .map(|(dp4a_kernel, mma_kernel)| {
                let dp4a_module =
                    kernels::get_or_load_module(client.context(), device_index, QUANT_GEMV_MODULE)
                        .expect("load dp4a module");
                let dp4a_func = kernels::get_kernel_function(&dp4a_module, dp4a_kernel)
                    .unwrap_or_else(|_| panic!("resolve {dp4a_kernel}"));
                let mma_func = kernels::get_kernel_function(&mma_module, mma_kernel)
                    .unwrap_or_else(|_| panic!("resolve {mma_kernel}"));
                (dp4a_kernel, mma_kernel, dp4a_func, mma_func)
            });

    let token_major_us = token_major.as_ref().map(|(_, _, dp4a_func, mma_func)| {
        let launch_dp4a = || unsafe {
            let mut builder = client.stream().launch_builder(dp4a_func);
            builder.arg(&act_ptr);
            builder.arg(&weight_ptr);
            builder.arg(&out_dp4a_ptr);
            builder.arg(&m_u32);
            builder.arg(&k_u32);
            builder.arg(&n_u32);
            builder.launch(cfg).expect("launch dp4a kernel");
        };
        let launch_mma = || unsafe {
            let mut builder = client.stream().launch_builder(mma_func);
            builder.arg(&act_ptr);
            builder.arg(&weight_ptr);
            builder.arg(&out_mma_ptr);
            builder.arg(&m_u32);
            builder.arg(&k_u32);
            builder.arg(&n_u32);
            builder.launch(cfg).expect("launch mma kernel");
        };

        for _ in 0..WARMUP {
            launch_dp4a();
        }
        client.synchronize();
        let started = std::time::Instant::now();
        for _ in 0..ITERS {
            launch_dp4a();
        }
        client.synchronize();
        let dp4a_us = started.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        for _ in 0..WARMUP {
            launch_mma();
        }
        client.synchronize();
        let started = std::time::Instant::now();
        for _ in 0..ITERS {
            launch_mma();
        }
        client.synchronize();
        let mma_us = started.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        (dp4a_us, mma_us)
    });

    // The feature-major kernels are the llama.cpp-geometry port: feature-major
    // tiles, weights as MMA operand A, 256 k staged per step. Their grid axes
    // are transposed against the token-major `_mma` kernel, their token tile is
    // chosen per batch size, and their shared memory is dynamic, so they need
    // their own config and an opt-in. Compiled for Q8_0, Q4_0, Q4_K, Q5_K,
    // Q6_K and Q3_K.
    let feat_major = match format.feat_major_infix() {
        Some(infix) => {
            let mmq_x = fm_mmq_x;
            assert!(
                MMQ_X_VARIANTS.contains(&mmq_x),
                "--mmq-x {mmq_x} is not a compiled variant; compiled: {MMQ_X_VARIANTS:?}"
            );
            let name = format!("quant_mmq_{infix}_q8_1_mma_x{mmq_x}");
            let func = kernels::get_kernel_function(&mma_module, &name)
                .unwrap_or_else(|_| panic!("resolve {name}"));
            func.set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                mmq_x_smem_bytes(fm_x_stride, fm_scratch, mmq_x) as i32,
            )
            .expect("opt in to dynamic shared memory for the feature-major kernel");
            let out =
                Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();
            Some((func, out, mmq_x, name))
        }
        None => None,
    };

    let feat_major_us = feat_major.as_ref().map(|(func, out, mmq_x, _)| {
        let out_ptr = out.ptr();
        let cfg_fm = LaunchConfig {
            grid_dim: (m_u32.div_ceil(*mmq_x), n_u32.div_ceil(128), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: mmq_x_smem_bytes(fm_x_stride, fm_scratch, *mmq_x),
        };
        let launch = || unsafe {
            let mut builder = client.stream().launch_builder(func);
            builder.arg(&packed_ptr);
            builder.arg(&weight_ptr);
            builder.arg(&out_ptr);
            builder.arg(&m_u32);
            builder.arg(&k_u32);
            builder.arg(&n_u32);
            builder.arg(&fm_ntok);
            builder
                .launch(cfg_fm)
                .expect("launch mma feature-major kernel");
        };
        for _ in 0..WARMUP {
            launch();
        }
        client.synchronize();
        let started = std::time::Instant::now();
        for _ in 0..ITERS {
            launch();
        }
        client.synchronize();
        started.elapsed().as_secs_f64() * 1e6 / ITERS as f64
    });

    // Stream-k: one block per SM walking a contiguous slice of the flattened
    // (feature-tile, token-tile, k-block) space, then a fixup pass folding the
    // partial tiles. It exists for the case where the tile count alone does not
    // fill the device. It reassociates the k sum across blocks, so it is checked
    // against the f64 reference, never against another kernel bit-for-bit.
    let sk = match format.feat_major_infix() {
        Some(infix) if stream_k => {
            let mmq_x = fm_mmq_x;
            let grid = device.profile().compute_units;
            let sk_name = format!("quant_mmq_{infix}_q8_1_mma_sk_x{mmq_x}");
            let fx_name = format!("quant_mmq_{infix}_q8_1_mma_fixup_x{mmq_x}");
            let sk_func = kernels::get_kernel_function(&mma_module, &sk_name)
                .unwrap_or_else(|_| panic!("resolve {sk_name}"));
            let fx_func = kernels::get_kernel_function(&mma_module, &fx_name)
                .unwrap_or_else(|_| panic!("resolve {fx_name}"));
            sk_func
                .set_attribute(
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    mmq_x_smem_bytes(fm_x_stride, fm_scratch, mmq_x) as i32,
                )
                .expect("opt in to dynamic shared memory for the stream-k kernel");
            // Never zeroed: the fixup reads only slots whose block provably
            // wrote a partial, so a memset would be pure cost.
            let ws_len = grid as usize * mmq_x as usize * 128;
            let ws =
                Tensor::<CudaRuntime>::from_slice(&vec![0f32; ws_len], &[ws_len], &device).unwrap();
            let out =
                Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();
            Some((sk_func, fx_func, out, ws, mmq_x, grid, sk_name))
        }
        _ => None,
    };

    let sk_us = sk
        .as_ref()
        .map(|(sk_func, fx_func, out, ws, mmq_x, grid, _)| {
            let out_ptr = out.ptr();
            let ws_ptr = ws.ptr();
            let cfg_sk = LaunchConfig {
                grid_dim: (*grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: mmq_x_smem_bytes(fm_x_stride, fm_scratch, *mmq_x),
            };
            let cfg_fx = LaunchConfig {
                grid_dim: (*grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let launch = || unsafe {
                let mut b = client.stream().launch_builder(sk_func);
                b.arg(&packed_ptr);
                b.arg(&weight_ptr);
                b.arg(&out_ptr);
                b.arg(&ws_ptr);
                b.arg(&m_u32);
                b.arg(&k_u32);
                b.arg(&n_u32);
                b.arg(&fm_ntok);
                b.launch(cfg_sk).expect("launch stream-k kernel");
                // Separate launch on the same stream: the fixup reads what the
                // main kernel wrote, so it must not be fused.
                let mut f = client.stream().launch_builder(fx_func);
                f.arg(&out_ptr);
                f.arg(&ws_ptr);
                f.arg(&m_u32);
                f.arg(&k_u32);
                f.arg(&n_u32);
                f.launch(cfg_fx).expect("launch stream-k fixup");
            };
            for _ in 0..WARMUP {
                launch();
            }
            client.synchronize();
            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                launch();
            }
            client.synchronize();
            started.elapsed().as_secs_f64() * 1e6 / ITERS as f64
        });

    let case = RefCase {
        weight: &weight_bytes,
        act: &act_bytes,
        m,
        n,
        k,
    };
    if let Some((_, mma_kernel, _, _)) = token_major.as_ref() {
        let dp4a_host = out_dp4a.to_vec::<f32>();
        let mma_host = out_mma.to_vec::<f32>();
        check_against_reference("dp4a", format, &dp4a_host, &case);
        check_against_reference(mma_kernel, format, &mma_host, &case);

        let mut mismatch = None;
        'outer: for row in 0..m {
            for col in 0..n {
                let idx = row * n + col;
                if dp4a_host[idx].to_bits() != mma_host[idx].to_bits() {
                    mismatch = Some((idx, dp4a_host[idx], mma_host[idx]));
                    break 'outer;
                }
            }
        }
        match mismatch {
            None => println!(
                "outputs match: dp4a and mma agree bit-for-bit at all {} elements",
                m * n
            ),
            Some((idx, a, b)) => {
                eprintln!("outputs MISMATCH at index {idx}: dp4a={a}, mma={b}");
                std::process::exit(1);
            }
        }
    } else {
        println!(
            "{}: no token-major dp4a or mma kernel is compiled, so only the \
             feature-major kernels run",
            format.label()
        );
    }
    if let Some((_, out, _, name)) = feat_major.as_ref() {
        let feat_major_host = out.to_vec::<f32>();
        check_against_reference(name, format, &feat_major_host, &case);
    }
    if let Some((_, _, out, _, _, _, name)) = sk.as_ref() {
        let sk_host = out.to_vec::<f32>();
        check_against_reference(name, format, &sk_host, &case);
    }

    if let (Some((dp4a_us, mma_us)), Some((dp4a_kernel, mma_kernel, _, _))) =
        (token_major_us, token_major.as_ref())
    {
        println!("{dp4a_kernel} (dp4a) {dp4a_us:9.2} us/call");
        println!("{mma_kernel}    {mma_us:9.2} us/call");
        println!("ratio dp4a/mma: {:.3}", dp4a_us / mma_us);
    }
    if let (Some(us), Some((_, _, _, _, mmq_x, grid, name))) = (sk_us, sk.as_ref()) {
        println!("{name} (grid {grid}, mmq_x {mmq_x}) {us:9.2} us/call");
        if let Some((_, mma_us)) = token_major_us {
            println!("ratio mma/stream-k: {:.3}", mma_us / us);
        }
    }
    if let (Some(us), Some((_, _, _, name))) = (feat_major_us, feat_major.as_ref()) {
        println!("{name} {us:9.2} us/call");
        if let Some((_, mma_us)) = token_major_us {
            println!("ratio mma/feature-major: {:.3}", mma_us / us);
        }
    }
}
