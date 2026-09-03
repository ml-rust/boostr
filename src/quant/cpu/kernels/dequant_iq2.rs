//! CPU dequantization kernels for IQ2 formats
//!
//! IQ2_XXS, IQ2_XS, IQ2_S
//!
//! All three are codebook quantizations: `qs` holds INDICES into a grid of
//! precomputed 8-component points, not the magnitudes themselves. The grids
//! live in [`super::iq_grid`] and are transcribed from llama.cpp's own
//! reference. Every scale here is `d * (0.5 + s) * 0.25` for a 4-bit `s`.
use half::f16;

use super::iq_grid::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, KSIGNS};

/// Component `pos` (0..8) of grid point `idx`, as a magnitude.
#[inline]
fn grid8(grid: &[u64], idx: usize, pos: usize) -> f32 {
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

/// The 4-bit scale for grid entry `entry`, packed two per byte.
#[inline]
fn packed_scale(scales: &[u8], entry: usize) -> u8 {
    let k = entry / 2;
    (scales[k / 2] >> (4 * (k % 2))) & 0x0F
}

/// Dequantizes IQ2_XXS blocks to f32
///
/// IQ2_XXS: 256 elements, 66 bytes/block.
/// Layout: `d:f16(2) + qs[64]`, read as eight pairs of little-endian `u32`.
/// The first `u32` of a pair holds four 8-bit grid indices; the second holds
/// the group's 4-bit scale in its top nibble and four 7-bit sign-table indices
/// below it.
pub fn dequant_iq2_xxs(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 66;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..66];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for group in 0..8 {
            let read_u32 =
                |off: usize| u32::from_le_bytes([qs[off], qs[off + 1], qs[off + 2], qs[off + 3]]);
            let indices = read_u32(group * 8);
            let aux = read_u32(group * 8 + 4);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.25;

            for sub in 0..4 {
                let idx = ((indices >> (8 * sub)) & 0xFF) as usize;
                let signs = KSIGNS[((aux >> (7 * sub)) & 0x7F) as usize];
                for j in 0..8 {
                    out[group * 32 + sub * 8 + j] =
                        db * grid8(&IQ2XXS_GRID, idx, j) * sign_of(signs, j);
                }
            }
        }
    }
}

/// Dequantizes IQ2_XS blocks to f32
///
/// IQ2_XS: 256 elements, 74 bytes/block.
/// Layout: `d:f16(2) + qs[64] + scales[8]`, with `qs` read as 32 little-endian
/// `u16`. Each `u16` packs a 9-bit grid index in its low bits and a 7-bit
/// sign-table index above them. Two grid entries share one 4-bit scale.
pub fn dequant_iq2_xs(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 74;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..66];
        let scales = &block[66..74];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for entry in 0..32 {
            let q = u16::from_le_bytes([qs[entry * 2], qs[entry * 2 + 1]]);
            let idx = usize::from(q & 511);
            let signs = KSIGNS[usize::from(q >> 9)];
            let db = d * (0.5 + f32::from(packed_scale(scales, entry))) * 0.25;
            for j in 0..8 {
                out[entry * 8 + j] = db * grid8(&IQ2XS_GRID, idx, j) * sign_of(signs, j);
            }
        }
    }
}

/// Dequantizes IQ2_S blocks to f32
///
/// IQ2_S: 256 elements, 82 bytes/block.
/// Layout: `d:f16(2) + qs[32] + signs[32] + qh[8] + scales[8]`. Each `qs` byte
/// takes two more index bits from `qh`, selecting among 1024 grid points.
/// Unlike IQ2_XXS and IQ2_XS, the signs are explicit bits — one byte of eight
/// signs per grid entry — with no sign-table indirection.
pub fn dequant_iq2_s(blocks: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 82;

    let num_blocks = blocks.len() / BLOCK_BYTES;
    debug_assert_eq!(output.len(), num_blocks * BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &blocks[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34];
        let signs = &block[34..66];
        let qh = &block[66..74];
        let scales = &block[74..82];
        let out = &mut output[b * BLOCK_SIZE..][..BLOCK_SIZE];

        for entry in 0..32 {
            let high = usize::from((qh[entry / 4] >> (2 * (entry % 4))) & 0x03);
            let idx = usize::from(qs[entry]) | (high << 8);
            let db = d * (0.5 + f32::from(packed_scale(scales, entry))) * 0.25;
            for j in 0..8 {
                out[entry * 8 + j] = db * grid8(&IQ2S_GRID, idx, j) * sign_of(signs[entry], j);
            }
        }
    }
}
