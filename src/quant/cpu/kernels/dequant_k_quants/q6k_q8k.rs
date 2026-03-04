//! Q6_K and Q8_K dequantization kernels

use half::f16;

/// Dequantize Q6_K blocks to f32
///
/// Q6_K: 256 elements, 210 bytes/block
/// Layout: 128-byte ql, 64-byte qh, 16-byte scales (i8), 2-byte d
/// Processes in two halves of 128 elements each.
pub fn dequant_q6k(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        let ql = &block[0..128];
        let qh = &block[128..192];
        // SAFETY: u8 and i8 have identical size and alignment. The slice
        // block[192..208] is valid for the lifetime of `block`, and we create
        // an i8 view of exactly 16 bytes (the Q6_K scale factors are signed).
        let sc: &[i8] = unsafe { std::slice::from_raw_parts(block[192..208].as_ptr().cast(), 16) };
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        // Process in two halves of 128 elements
        for n in 0..2 {
            let y_base = n * 128;
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;

            for l in 0..32 {
                let is = l / 16;

                let q1 = ((ql[ql_base + l] & 0x0F) | ((qh[qh_base + l] & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 0x03) << 4))
                    as i8
                    - 32;
                let q3 =
                    ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 = ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 0x03) << 4))
                    as i8
                    - 32;

                out[y_base + l] = d * sc[sc_base + is] as f32 * q1 as f32;
                out[y_base + l + 32] = d * sc[sc_base + is + 2] as f32 * q2 as f32;
                out[y_base + l + 64] = d * sc[sc_base + is + 4] as f32 * q3 as f32;
                out[y_base + l + 96] = d * sc[sc_base + is + 6] as f32 * q4 as f32;
            }
        }
    }
}

/// Dequantize Q8_K blocks to f32
///
/// Q8_K: 256 elements, 292 bytes/block
/// Layout: 4-byte f32 d + 256-byte i8 qs + 32-byte i16 bsums
/// Formula: x = q * d (bsums are for dot product optimization only)
/// Note: Q8_K is unique in using f32 scale (not f16).
pub fn dequant_q8k(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 292;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..];
        // Q8K uses f32 scale, not f16
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let qs = &block[4..260];
        // block[260..292] is bsums[16] (i16), ignored for dequant
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for (out_val, &qs_val) in out.iter_mut().zip(qs.iter()) {
            *out_val = qs_val as i8 as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_q6k_zero_scales() {
        let block = [0u8; 210];
        let mut output = [0.0f32; 256];
        dequant_q6k(&block, &mut output);

        for &v in &output {
            assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q8k_known_values() {
        let mut block = [0u8; 292];
        // f32 scale = 0.5
        block[0..4].copy_from_slice(&0.5f32.to_le_bytes());
        // qs = 10 as i8 → value = 10 * 0.5 = 5.0
        block[4..260].fill(10);

        let mut output = [0.0f32; 256];
        dequant_q8k(&block, &mut output);

        for &v in &output {
            assert!((v - 5.0).abs() < 0.01, "expected 5.0, got {}", v);
        }
    }
}
