//! Launches the dp4a and tensor-core MMQ kernels back to back in one process,
//! on identical inputs, so a profiler attributes instruction counts to each
//! kernel without an A/B rebuild. Covers Q8_0, Q4_0, Q4_K, Q5_K and Q6_K via
//! `--format`.
//!
//! Q4_0 and Q5_K have no token-major kernel of either kind — neither
//! `quant_mmq_q4_0_q8_1` / `quant_mmq_q5_k_q8_1` nor their `_mma` twins exist —
//! so for those formats the tool runs and checks the feature-major kernels
//! alone and skips the token-major comparison rather than resolving a symbol
//! that is not compiled.
//!
//! ```text
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q8_0 --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q4_0 --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q4_k --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q5_k --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q6_k --n 4096 --k 14336 --m 512
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("mmq_kernel_compare needs --features cuda");
}

#[cfg(feature = "cuda")]
use boostr::quant::cuda::kernels::{self, QUANT_GEMV_MODULE, QUANT_MMQ_MMA_MODULE};
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
#[cfg(feature = "cuda")]
const fn mmq_x_smem_bytes(x_stride: u32, mmq_x: u32) -> u32 {
    4 * (128 * x_stride + mmq_x * 36)
}

/// Picks the feature-major variant that launches the fewest token tiles for
/// `m`, breaking ties toward the smaller tile because it costs fewer registers
/// and less shared memory. `smem_limit` is the device's per-block opt-in
/// maximum.
#[cfg(feature = "cuda")]
fn select_mmq_x(m: u32, smem_limit: u32, x_stride: u32) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for &mmq_x in MMQ_X_VARIANTS {
        if mmq_x_smem_bytes(x_stride, mmq_x) > smem_limit {
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
/// proportional to, so one tight bound serves Q8_0, Q4_K, and Q6_K alike
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
/// ([`q8_0_reference`], [`q4_0_reference`], [`q4_k_reference`],
/// [`q5_k_reference`], or [`q6_k_reference`]), panicking with the position and both values on the
/// first breach.
#[cfg(feature = "cuda")]
fn check_against_reference(label: &str, format: MmqFormat, got: &[f32], case: &RefCase) {
    let rtol = REFERENCE_RTOL;
    let reference: ReferenceFn = match format {
        MmqFormat::Q8_0 => q8_0_reference,
        MmqFormat::Q40 => q4_0_reference,
        MmqFormat::Q4K => q4_k_reference,
        MmqFormat::Q5K => q5_k_reference,
        MmqFormat::Q6K => q6_k_reference,
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
            other => {
                panic!("unknown --format {other}, expected one of: q8_0, q4_0, q4_k, q5_k, q6_k")
            }
        }
    }

    fn label(&self) -> &'static str {
        match self {
            MmqFormat::Q8_0 => "q8_0",
            MmqFormat::Q40 => "q4_0",
            MmqFormat::Q4K => "q4_k",
            MmqFormat::Q5K => "q5_k",
            MmqFormat::Q6K => "q6_k",
        }
    }

    /// Elements per weight block: 32 for the legacy formats Q8_0 and Q4_0, 256
    /// for the K-quant super-blocks. `k` must be a whole number of these, which
    /// mirrors the `k.is_multiple_of(...)` guards in `dispatch_matmul` and the
    /// `k_multiple` field of each `FeatMajorFormat`.
    fn block_elems(&self) -> usize {
        match self {
            MmqFormat::Q8_0 | MmqFormat::Q40 => 32,
            MmqFormat::Q4K | MmqFormat::Q5K | MmqFormat::Q6K => 256,
        }
    }

    /// Token-major dp4a MMQ kernel, or `None` for a format that has none.
    /// Q4_0 and Q5_K have no `quant_mmq_q4_0_q8_1` / `quant_mmq_q5_k_q8_1`:
    /// their only pre-feature-major GEMM path is the dequantize-then-f32
    /// kernel, which this tool does not time.
    fn dp4a_kernel(&self) -> Option<&'static str> {
        match self {
            MmqFormat::Q8_0 => Some("quant_mmq_q8_0_q8_1"),
            MmqFormat::Q40 => None,
            MmqFormat::Q4K => Some("quant_mmq_q4_k_q8_1"),
            MmqFormat::Q5K => None,
            MmqFormat::Q6K => Some("quant_mmq_q6_k_q8_1"),
        }
    }

    /// Token-major tensor-core MMQ kernel, or `None` for a format that has
    /// none. Q4_0 and Q5_K have no `_mma` twin; both went straight to the
    /// feature-major family.
    fn mma_kernel(&self) -> Option<&'static str> {
        match self {
            MmqFormat::Q8_0 => Some("quant_mmq_q8_0_q8_1_mma"),
            MmqFormat::Q40 => None,
            MmqFormat::Q4K => Some("quant_mmq_q4_k_q8_1_mma"),
            MmqFormat::Q5K => None,
            MmqFormat::Q6K => Some("quant_mmq_q6_k_q8_1_mma"),
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
    fn feat_major_x_stride(&self) -> u32 {
        match self {
            MmqFormat::Q8_0 | MmqFormat::Q40 => 76,
            MmqFormat::Q4K | MmqFormat::Q5K | MmqFormat::Q6K => 84,
        }
    }

    fn build_weight(&self, n: usize, k: usize) -> Vec<u8> {
        match self {
            MmqFormat::Q8_0 => build_q8_0_weight(n, k),
            MmqFormat::Q40 => build_q4_0_weight(n, k),
            MmqFormat::Q4K => build_q4_k_weight(n, k),
            MmqFormat::Q5K => build_q5_k_weight(n, k),
            MmqFormat::Q6K => build_q6_k_weight(n, k),
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
    let fm_mmq_x = force_mmq_x.unwrap_or_else(|| {
        select_mmq_x(m as u32, mmq_x_smem_bytes(fm_x_stride, 128), fm_x_stride)
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
    // share a grid and a per-token activation layout. Q4_0 and Q5_K compile
    // NEITHER, so for those formats this whole comparison — both timings, both
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
    // their own config and an opt-in. Compiled for Q8_0, Q4_0, Q4_K, Q5_K and
    // Q6_K.
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
                mmq_x_smem_bytes(fm_x_stride, mmq_x) as i32,
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
            shared_mem_bytes: mmq_x_smem_bytes(fm_x_stride, *mmq_x),
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
                    mmq_x_smem_bytes(fm_x_stride, mmq_x) as i32,
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
                shared_mem_bytes: mmq_x_smem_bytes(fm_x_stride, *mmq_x),
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
