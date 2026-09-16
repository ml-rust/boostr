//! Whether to launch the stream-k pair rather than the tile-parallel grid.

use super::super::formats::FeatMajorFormat;

/// Whether to launch the stream-k pair rather than the tile-parallel grid.
///
/// Stream-k splits every tile's K dimension across blocks and pays a fixup
/// pass to rejoin the partials, trading that pass for the wave a ragged tile
/// count leaves half empty. Once the tiles fill the device, tile-parallel
/// wins and needs no workspace.
///
/// A format vetoes this call through `prefers_tile_parallel`, but only once
/// the tile count passes about four thirds of the SM count: past that point
/// the split saves too little to cover the fixup pass. Below that threshold
/// the tile-parallel grid cannot fill the device, and stream-k wins for every
/// format, veto or not.
///
/// K gates it too. The partial stores and the fixup pass are a fixed cost per
/// split, paid once however short the K walk is, so a short K cannot amortise
/// them and the tile-parallel grid wins even with most SMs idle. Measured on
/// every K-quant and IQ4 format at the DiT projection shapes: below
/// `STREAM_K_MIN_K` stream-k loses for every format at every tile count
/// tried; at and above it stream-k wins.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const fn use_stream_k(
    tiles: u32,
    sms: u32,
    k: u32,
    format: &FeatMajorFormat,
) -> bool {
    sms > 0
        && k >= STREAM_K_MIN_K
        && tiles < 2 * sms
        && !(format.prefers_tile_parallel && 3 * tiles >= 4 * sms)
}

/// Shortest K the stream-k split is worth. See [`use_stream_k`].
const STREAM_K_MIN_K: u32 = 2048;

#[cfg(test)]
mod tests {
    use super::super::super::formats::{IQ3_XXS, Q4_1, Q8_0};
    use super::*;

    /// SM count the geometry cases below are written against.
    const SMS: u32 = 28;

    #[test]
    fn stream_k_only_when_the_tiles_leave_the_device_short() {
        // 32 tiles is under two waves of SMS, leaving the second wave nearly
        // empty. Q8_0's veto does not fire yet: 3 * tiles < 4 * SMS.
        assert!(use_stream_k(32, SMS, 4096, &Q8_0));
        // Two full waves already fill it, so the tile-parallel grid wins.
        assert!(!use_stream_k(56, SMS, 4096, &Q8_0));
        // No SM count reported: fall back to the tile-parallel grid.
        assert!(!use_stream_k(32, 0, 4096, &Q8_0));
    }

    #[test]
    fn a_short_k_keeps_the_tile_parallel_grid() {
        // 16 tiles is under two waves of SMS, so it would take stream-k, but a
        // K walk this short cannot amortise the partial stores and the fixup
        // pass.
        assert!(!use_stream_k(16, SMS, 1024, &Q4_1));
        assert!(!use_stream_k(16, SMS, 2047, &Q4_1));
        assert!(use_stream_k(16, SMS, 2048, &Q4_1));
    }

    #[test]
    fn a_flagged_format_vetoes_stream_k_once_the_tile_count_passes_the_threshold() {
        // 3 * tiles >= 4 * SMS: the tile count has passed about four thirds
        // of SMS, so a flagged format takes the tile-parallel grid.
        assert!(!use_stream_k(40, SMS, 4096, &Q8_0));
        // Two full waves fill the device regardless of the flag.
        assert!(!use_stream_k(56, SMS, 4096, &Q8_0));
        // An unflagged format is unaffected by the veto term at the same
        // geometry where a flagged format is vetoed.
        assert!(use_stream_k(40, SMS, 4096, &Q4_1));
    }

    #[test]
    fn the_veto_lifts_below_four_thirds_of_the_sm_count() {
        // 16 tiles leaves most of SMS with no tile at all. The tile-parallel
        // grid cannot fill the device there, so stream-k wins even for a
        // format that vetoes it once the tile count passes the threshold.
        assert!(use_stream_k(16, SMS, 4096, &IQ3_XXS));
        assert!(use_stream_k(37, SMS, 4096, &IQ3_XXS));
        // 3 * tiles just clears 4 * SMS: the veto fires just past the threshold.
        assert!(!use_stream_k(38, SMS, 4096, &IQ3_XXS));
    }
}
