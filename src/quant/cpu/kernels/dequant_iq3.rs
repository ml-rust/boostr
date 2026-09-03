//! CPU dequantization kernels for IQ3 formats
//!
//! IQ3_XXS, IQ3_S
//!
//! Both formats are codebook quantizations: `qs` holds INDICES into a grid of
//! precomputed points, not the magnitudes themselves. The grids live in
//! [`super::iq_grid`] and are transcribed from llama.cpp's own reference.
//! Signs are separate from magnitudes throughout.
use half::f16;

use super::iq_grid::{IQ3S_GRID, IQ3XXS_GRID, KSIGNS};

/// Component `pos` (0..4) of grid point `idx`, as a magnitude.
#[inline]
fn grid4(grid: &[u32], idx: usize, pos: usize) -> f32 {
    f32::from((grid[idx] >> (8 * pos)) as u8)
}

/// `+1.0` or `-1.0` for bit `pos` of a sign byte: a set bit negates.
#[inline]
fn sign_of(sign_byte: u8, pos: usize) -> f32 {
    if (sign_byte >> pos) & 1 != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Dequantizes IQ3_XXS blocks to f32
///
/// IQ3_XXS: 256 elements, 98 bytes/block.
/// Layout: `d:f16(2) + qs[64] + scales[32]`, where `scales` is eight
/// little-endian `u32`, one per 32-element group. Each `u32` carries the
/// group's 4-bit scale in its top nibble and four 7-bit sign-table indices
/// below it. Each `qs` byte indexes a 256-point grid of 4 components, so two
/// `qs` bytes cover one 8-element sign sub-group.
pub fn dequant_iq3_xxs(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 98;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..66];
        let scales = &block[66..98];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for group in 0..8 {
            let aux = u32::from_le_bytes([
                scales[group * 4],
                scales[group * 4 + 1],
                scales[group * 4 + 2],
                scales[group * 4 + 3],
            ]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;

            for sub in 0..4 {
                let signs = KSIGNS[((aux >> (7 * sub)) & 0x7F) as usize];
                let lo = usize::from(qs[group * 8 + sub * 2]);
                let hi = usize::from(qs[group * 8 + sub * 2 + 1]);
                for j in 0..8 {
                    let mag = if j < 4 {
                        grid4(&IQ3XXS_GRID, lo, j)
                    } else {
                        grid4(&IQ3XXS_GRID, hi, j - 4)
                    };
                    out[group * 32 + sub * 8 + j] = db * mag * sign_of(signs, j);
                }
            }
        }
    }
}

/// Dequantizes IQ3_S blocks to f32
///
/// IQ3_S: 256 elements, 110 bytes/block.
/// Layout: `d:f16(2) + qs[64] + qh[8] + signs[32] + scales[4]`. Each `qs` byte
/// gets a ninth index bit from `qh`, selecting among 512 grid points of 4
/// components. Signs are explicit bits here, one per element, with no sign
/// table. `scales` holds eight 4-bit scales, one per 32-element group.
pub fn dequant_iq3_s(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 110;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..66];
        let qh = &block[66..74];
        let signs = &block[74..106];
        let scales = &block[106..110];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for (e, slot) in out.iter_mut().enumerate() {
            let entry = e / 4;
            let ninth_bit = usize::from((qh[entry / 8] >> (entry % 8)) & 1);
            let idx = usize::from(qs[entry]) | (ninth_bit << 8);

            let scale = (scales[(e / 32) / 2] >> (4 * ((e / 32) % 2))) & 0x0F;
            let db = d * (1.0 + 2.0 * f32::from(scale));

            *slot = db * grid4(&IQ3S_GRID, idx, e % 4) * sign_of(signs[e / 8], e % 8);
        }
    }
}
