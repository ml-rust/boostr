//! Vectorized element loops for TCF tile reconstruction.
//!
//! # What this is, and what it deliberately is not
//!
//! This is Section 13.0's `x_hat_i = d * f32(q_i)` and Section 13.0.1's
//! `x_hat_i = d * f32(u_i) + m`, applied eight elements at a time. It holds
//! no bit position, no plane offset, no nibble index and no field order:
//! `tcf_core::unpack` has already expanded every code into a whole byte by
//! the time a tile exists, and the effective `(d, m)` pair is resolved by
//! `tcf_core`'s own `QuantLayout::group_values` before either function here
//! is called. So this unit is a second copy of the arithmetic, never of the
//! format. The tile and group walk that feeds it lives in
//! [`crate::quant::cpu::kernels::tcf`].
//!
//! # Why it is bit-identical to the scalar definition
//!
//! `mulps` and `addps` apply the same IEEE-754 correctly-rounded operation
//! per lane that `mulss` and `addss` apply to one value, with the same
//! operand order, so no lane can differ from what the scalar expression
//! produces. `f32::from(i8)` and `f32::from(u8)` are exact over their whole
//! ranges, and so is `vcvtdq2ps` over the widened codes. There is no FMA
//! here and there must not be: the asymmetric form is one multiply and then
//! one separately rounded add, and contracting them would round once instead
//! of twice. There is no reassociation either, because every element is
//! independent of every other.
//!
//! The dot product in the fused matmul is the opposite case: lane
//! accumulation reorders a float sum, so it is NOT bit-identical. That path
//! is [`super::dot_f32`], and its callers compare against a reference within
//! tolerance rather than bit for bit.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elements one vector iteration reconstructs. Eight f32 is one AVX2
/// register, and two NEON registers.
const LANES: usize = 8;

/// Which element loop runs. Resolved once per decode call, never per group:
/// a 64-tile chunk resolves 128 groups, and a feature probe is an atomic
/// load that no branch predictor removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decoder {
    /// AVX2: widen eight codes, convert, multiply, store.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// NEON: the same eight elements, as two four-lane halves.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// One element at a time. Correct on every machine, and the definition
    /// the vector paths are tested against.
    Scalar,
}

impl Decoder {
    /// The fastest element loop this CPU supports.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            Self::Scalar
        }

        // NEON is architectural on AArch64, so it needs no runtime probe —
        // the same assumption the NEON kernels beside this one make.
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::Scalar
        }
    }

    /// Section 13.0: write `scale * f32(code)` for one symmetric group.
    ///
    /// Writes `min(codes.len(), out.len())` values and touches nothing else,
    /// so a caller that sizes both to the group width writes exactly it.
    pub fn signed(self, codes: &[i8], scale: f32, out: &mut [f32]) {
        match self {
            // SAFETY: `Avx2` is only produced by `detect` after
            // `is_x86_feature_detected!("avx2")` succeeded.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { signed_avx2(codes, scale, out) },
            // SAFETY: NEON is architectural on AArch64.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { signed_neon(codes, scale, out) },
            Self::Scalar => signed_scalar(codes, scale, out),
        }
    }

    /// Section 13.0.1: write `scale * f32(code) + min` for one asymmetric
    /// group, under the same length rule [`Self::signed`] follows.
    pub fn unsigned(self, codes: &[u8], scale: f32, min: f32, out: &mut [f32]) {
        match self {
            // SAFETY: `Avx2` is only produced by `detect` after
            // `is_x86_feature_detected!("avx2")` succeeded.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { unsigned_avx2(codes, scale, min, out) },
            // SAFETY: NEON is architectural on AArch64.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { unsigned_neon(codes, scale, min, out) },
            Self::Scalar => unsigned_scalar(codes, scale, min, out),
        }
    }
}

/// The Section 13.0 expression, one element at a time. This is the
/// definition every vector path here is asserted equal to.
fn signed_scalar(codes: &[i8], scale: f32, out: &mut [f32]) {
    for (slot, code) in out.iter_mut().zip(codes.iter()) {
        *slot = scale * f32::from(*code);
    }
}

/// The Section 13.0.1 expression, one element at a time. One multiply, then
/// one separately rounded add.
fn unsigned_scalar(codes: &[u8], scale: f32, min: f32, out: &mut [f32]) {
    for (slot, code) in out.iter_mut().zip(codes.iter()) {
        *slot = scale * f32::from(*code) + min;
    }
}

/// `scale * f32(code)` over eight signed codes per iteration.
///
/// `vpmovsxbd` sign-extends eight `i8` into eight `i32`, `vcvtdq2ps`
/// converts them exactly, and `vmulps` applies the rounding `mulss` applies,
/// per lane — with `scale` as the first operand, as in the scalar expression.
///
/// # Safety
/// - CPU must support AVX2 (enforced by `#[target_feature]` at call site).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn signed_avx2(codes: &[i8], scale: f32, out: &mut [f32]) {
    // SAFETY: AVX2 is guaranteed by `#[target_feature]`. `len` is bounded by
    // both slice lengths and every offset satisfies `offset + LANES <= len`,
    // so the 8-byte load and the 32-byte store stay inside their slices. Both
    // intrinsics accept unaligned pointers.
    unsafe {
        let len = codes.len().min(out.len());
        let chunks = len / LANES;
        let vscale = _mm256_set1_ps(scale);
        for i in 0..chunks {
            let offset = i * LANES;
            let raw = _mm_loadl_epi64(codes.as_ptr().add(offset).cast::<__m128i>());
            let floats = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_mul_ps(vscale, floats));
        }
        let tail = chunks * LANES;
        if let (Some(rest), Some(slots)) = (codes.get(tail..len), out.get_mut(tail..len)) {
            signed_scalar(rest, scale, slots);
        }
    }
}

/// `scale * f32(code) + min` over eight unsigned codes per iteration.
///
/// `vpmovzxbd` zero-extends, `vcvtdq2ps` converts exactly, then one `vmulps`
/// and one `vaddps` — never `vfmadd`, which would round once instead of
/// twice and break bit-identity with the scalar definition.
///
/// # Safety
/// - CPU must support AVX2 (enforced by `#[target_feature]` at call site).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unsigned_avx2(codes: &[u8], scale: f32, min: f32, out: &mut [f32]) {
    // SAFETY: AVX2 is guaranteed by `#[target_feature]`. `len` is bounded by
    // both slice lengths and every offset satisfies `offset + LANES <= len`,
    // so the 8-byte load and the 32-byte store stay inside their slices.
    unsafe {
        let len = codes.len().min(out.len());
        let chunks = len / LANES;
        let vscale = _mm256_set1_ps(scale);
        let vmin = _mm256_set1_ps(min);
        for i in 0..chunks {
            let offset = i * LANES;
            let raw = _mm_loadl_epi64(codes.as_ptr().add(offset).cast::<__m128i>());
            let floats = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw));
            let scaled = _mm256_mul_ps(vscale, floats);
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_add_ps(scaled, vmin));
        }
        let tail = chunks * LANES;
        if let (Some(rest), Some(slots)) = (codes.get(tail..len), out.get_mut(tail..len)) {
            unsigned_scalar(rest, scale, min, slots);
        }
    }
}

/// `scale * f32(code)` over eight signed codes per iteration, as two
/// four-lane halves.
///
/// # Safety
/// - CPU must support NEON (architectural on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn signed_neon(codes: &[i8], scale: f32, out: &mut [f32]) {
    // SAFETY: NEON is guaranteed by `#[target_feature]`. `len` is bounded by
    // both slice lengths and every offset satisfies `offset + LANES <= len`,
    // so the 8-byte load and the two 16-byte stores stay inside their slices.
    unsafe {
        let len = codes.len().min(out.len());
        let chunks = len / LANES;
        let vscale = vdupq_n_f32(scale);
        for i in 0..chunks {
            let offset = i * LANES;
            let wide = vmovl_s8(vld1_s8(codes.as_ptr().add(offset)));
            let lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide)));
            let hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide)));
            vst1q_f32(out.as_mut_ptr().add(offset), vmulq_f32(vscale, lo));
            vst1q_f32(out.as_mut_ptr().add(offset + 4), vmulq_f32(vscale, hi));
        }
        let tail = chunks * LANES;
        if let (Some(rest), Some(slots)) = (codes.get(tail..len), out.get_mut(tail..len)) {
            signed_scalar(rest, scale, slots);
        }
    }
}

/// `scale * f32(code) + min` over eight unsigned codes per iteration. One
/// multiply and one add, never `vfma`.
///
/// # Safety
/// - CPU must support NEON (architectural on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unsigned_neon(codes: &[u8], scale: f32, min: f32, out: &mut [f32]) {
    // SAFETY: NEON is guaranteed by `#[target_feature]`. `len` is bounded by
    // both slice lengths and every offset satisfies `offset + LANES <= len`,
    // so the 8-byte load and the two 16-byte stores stay inside their slices.
    unsafe {
        let len = codes.len().min(out.len());
        let chunks = len / LANES;
        let vscale = vdupq_n_f32(scale);
        let vmin = vdupq_n_f32(min);
        for i in 0..chunks {
            let offset = i * LANES;
            let wide = vmovl_u8(vld1_u8(codes.as_ptr().add(offset)));
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(wide)));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(wide)));
            vst1q_f32(
                out.as_mut_ptr().add(offset),
                vaddq_f32(vmulq_f32(vscale, lo), vmin),
            );
            vst1q_f32(
                out.as_mut_ptr().add(offset + 4),
                vaddq_f32(vmulq_f32(vscale, hi), vmin),
            );
        }
        let tail = chunks * LANES;
        if let (Some(rest), Some(slots)) = (codes.get(tail..len), out.get_mut(tail..len)) {
            unsigned_scalar(rest, scale, min, slots);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare by bits, never by `==`: bit-identity is the property under
    /// test, and `-0.0 == 0.0` would hide a difference.
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// THE correctness gate for this unit. Over the WHOLE `i8` and `u8` code
    /// ranges, with a scale and a minimum that are inexact binary fractions,
    /// the vector path must agree with the scalar expression bit for bit. A
    /// lane that rounded differently, or an FMA contraction, shows here.
    #[test]
    fn every_representable_code_matches_the_scalar_definition_bit_for_bit() {
        let signed: Vec<i8> = (i16::from(i8::MIN)..=i16::from(i8::MAX))
            .map(|v| v as i8)
            .collect();
        let mut vector = vec![0.0f32; signed.len()];
        let mut scalar = vec![0.0f32; signed.len()];
        Decoder::detect().signed(&signed, 1.0 / 3.0, &mut vector);
        signed_scalar(&signed, 1.0 / 3.0, &mut scalar);
        assert_eq!(bits(&vector), bits(&scalar));

        let unsigned: Vec<u8> = (0..=u8::MAX).collect();
        let mut vector = vec![0.0f32; unsigned.len()];
        let mut scalar = vec![0.0f32; unsigned.len()];
        Decoder::detect().unsigned(&unsigned, 1.0 / 3.0, 1.0 / 7.0, &mut vector);
        unsigned_scalar(&unsigned, 1.0 / 3.0, 1.0 / 7.0, &mut scalar);
        assert_eq!(bits(&vector), bits(&scalar));
    }

    /// A negative scale over a zero code produces negative zero, and the
    /// vector path must produce it too — which `==` would not have caught.
    #[test]
    fn a_negative_scale_over_a_zero_code_keeps_its_sign() {
        let codes: [i8; 4] = [0, 2, 1, -1];
        let mut vector = [0.0f32; 4];
        let mut scalar = [0.0f32; 4];
        Decoder::detect().signed(&codes, -0.5, &mut vector);
        signed_scalar(&codes, -0.5, &mut scalar);
        assert_eq!(bits(&vector), bits(&scalar));
    }

    /// A group shorter than one vector iteration takes the scalar tail, and
    /// a group that is not a whole multiple of eight takes both paths.
    #[test]
    fn a_partial_vector_iteration_takes_the_tail() {
        for len in [1usize, 5, 8, 13, 32] {
            let codes: Vec<i8> = (0..len).map(|i| (i as i32 - 64) as i8).collect();
            let mut vector = vec![0.0f32; len];
            let mut scalar = vec![0.0f32; len];
            Decoder::detect().signed(&codes, 0.1, &mut vector);
            signed_scalar(&codes, 0.1, &mut scalar);
            assert_eq!(bits(&vector), bits(&scalar), "len {len}");

            let codes: Vec<u8> = (0..len).map(|i| (i * 7 % 256) as u8).collect();
            let mut vector = vec![0.0f32; len];
            let mut scalar = vec![0.0f32; len];
            Decoder::detect().unsigned(&codes, 0.1, -0.3, &mut vector);
            unsigned_scalar(&codes, 0.1, -0.3, &mut scalar);
            assert_eq!(bits(&vector), bits(&scalar), "len {len}");
        }
    }

    /// Neither path writes past the values it was given.
    #[test]
    fn a_shorter_output_bounds_the_write() {
        let codes = [1i8; 16];
        let mut out = [-1.0f32; 16];
        Decoder::detect().signed(&codes[..5], 2.0, &mut out);
        assert_eq!(bits(&out[5..]), bits(&[-1.0f32; 11]));
    }
}
