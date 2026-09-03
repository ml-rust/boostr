//! CPU dequantization kernels for IQ1 formats
//!
//! IQ1_S, IQ1_M
//!
//! Both are codebook quantizations over a SHARED 2048-point grid whose
//! components are ternary (-1, 0, 1), stored as signed bytes in
//! [`super::iq_grid::IQ1_GRID`]. Unlike the IQ2/IQ3 formats there is no sign
//! table: the sign is already part of the grid point. Instead each group
//! carries a `delta` of +/- 0.125 added to every component before scaling,
//! which is what lets a ternary codebook represent an asymmetric distribution.
use half::f16;

use super::iq_grid::IQ1_GRID;

/// The offset added to every grid component before scaling. Matches
/// llama.cpp's `IQ1S_DELTA` / `IQ1M_DELTA`.
const DELTA: f32 = 0.125;

/// Component `pos` (0..8) of grid point `idx`. The stored bytes are signed.
#[inline]
fn grid8_signed(idx: usize, pos: usize) -> f32 {
    f32::from((IQ1_GRID[idx] >> (8 * pos)) as u8 as i8)
}

/// Dequantizes IQ1_S blocks to f32
///
/// IQ1_S: 256 elements, 50 bytes/block.
/// Layout: `d:f16(2) + qs[32] + qh[16]`, with `qh` read as eight little-endian
/// `u16`, one per 32-element group. Each `u16` carries four 3-bit index-high
/// fields, a 3-bit scale at bits 12..15, and the group's delta sign in bit 15.
pub fn dequant_iq1_s(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 50;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34];
        let qh = &block[34..50];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for group in 0..8 {
            let h = u16::from_le_bytes([qh[group * 2], qh[group * 2 + 1]]);
            let dl = d * (2.0 * f32::from((h >> 12) & 7) + 1.0);
            let delta = if h & 0x8000 == 0 { DELTA } else { -DELTA };

            for sub in 0..4 {
                let idx =
                    usize::from(qs[group * 4 + sub]) | (usize::from((h >> (3 * sub)) & 7) << 8);
                for j in 0..8 {
                    out[group * 32 + sub * 8 + j] = dl * (grid8_signed(idx, j) + delta);
                }
            }
        }
    }
}

/// Dequantizes IQ1_M blocks to f32
///
/// IQ1_M: 256 elements, 56 bytes/block.
/// Layout: `qs[32] + qh[16] + scales[8]` — note there is NO leading `d` field.
/// The f16 scale is split across the top nibbles of the four little-endian
/// `u16` in `scales`, low nibble first. Those same `u16` also hold sixteen
/// 3-bit sub-scales, and each `qh` nibble supplies one grid index's high bits
/// plus that entry's delta sign.
pub fn dequant_iq1_m(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 56;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let qs = &block[0..32];
        let qh = &block[32..48];
        let scales = &block[48..56];

        let sc = |i: usize| u16::from_le_bytes([scales[i * 2], scales[i * 2 + 1]]);
        let d_bits = ((sc(0) & 0xF000) >> 12)
            | ((sc(1) & 0xF000) >> 8)
            | ((sc(2) & 0xF000) >> 4)
            | (sc(3) & 0xF000);
        let d = f16::from_bits(d_bits).to_f32();

        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for entry in 0..32 {
            let nibble = (qh[entry / 2] >> (4 * (entry % 2))) & 0x0F;
            let idx = usize::from(qs[entry]) | (usize::from(nibble & 7) << 8);
            let delta = if nibble & 8 == 0 { DELTA } else { -DELTA };

            for j in 0..8 {
                let e = entry * 8 + j;
                // One 3-bit sub-scale covers 16 elements, i.e. two grid entries.
                let k = e / 16;
                let scale = (sc(k / 4) >> (3 * (k % 4))) & 0x07;
                let dl = d * (2.0 * f32::from(scale) + 1.0);
                out[e] = dl * (grid8_signed(idx, j) + delta);
            }
        }
    }
}
