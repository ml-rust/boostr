//! Arena sizing for a decode-graph capture.
//!
//! One hybrid decode step records on the order of a thousand intermediate
//! allocations, so capture goes through the arena variant of
//! `capture_graph_into_with_arena`. The first arena is sized from free device
//! memory, and a capture whose reported footprint is less than half of it is
//! retaken against a tightened arena.

use numr::runtime::cuda::CudaDevice;

use crate::error::{Error, Result};

/// Upper bound for the first capture arena.
const ARENA_CAP_BYTES: usize = 1 << 30;
/// The first arena is at most this fraction of free device memory.
const ARENA_FREE_DIVISOR: u64 = 4;
/// Recapture granularity for the tightened arena.
const ARENA_ROUND_BYTES: usize = 1 << 20;

/// Arena for the first capture: `min(free / 4, 1 GiB)`, at least 1 MiB.
///
/// # Errors
///
/// [`Error::InferenceError`] when the device memory query fails.
pub fn first_arena_bytes(device: &CudaDevice) -> Result<usize> {
    let (free, _total) = device.memory_info().map_err(|e| Error::InferenceError {
        reason: format!(
            "querying device memory to size a capture arena failed: {e}; check that the device \
             the graph captures on is still available"
        ),
    })?;
    let quarter = usize::try_from(free / ARENA_FREE_DIVISOR).unwrap_or(ARENA_CAP_BYTES);
    Ok(quarter.clamp(ARENA_ROUND_BYTES, ARENA_CAP_BYTES))
}

/// Tightened arena from a measured footprint: `used * 5 / 4`, rounded up to
/// 1 MiB.
pub fn tight_arena_bytes(used: usize) -> usize {
    let padded = (used * 5).div_ceil(4);
    padded.div_ceil(ARENA_ROUND_BYTES).max(1) * ARENA_ROUND_BYTES
}

/// True when `used` leaves more than half of `arena` unused, which is when a
/// recapture against [`tight_arena_bytes`] pays for itself.
pub fn arena_is_loose(used: usize, arena: usize) -> bool {
    used < arena / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tight_arena_pads_and_rounds_to_a_mib() {
        assert_eq!(tight_arena_bytes(0), ARENA_ROUND_BYTES);
        assert_eq!(tight_arena_bytes(1), ARENA_ROUND_BYTES);
        // 4 MiB used pads to 5 MiB, already a whole number of MiB.
        assert_eq!(tight_arena_bytes(4 << 20), 5 << 20);
        // 1 MiB + 1 byte pads past 1 MiB and rounds up to 2 MiB.
        assert_eq!(tight_arena_bytes((1 << 20) + 1), 2 << 20);
    }

    #[test]
    fn loose_arena_is_more_than_twice_the_footprint() {
        assert!(arena_is_loose(10, 100));
        assert!(!arena_is_loose(50, 100));
        assert!(!arena_is_loose(100, 100));
    }
}
