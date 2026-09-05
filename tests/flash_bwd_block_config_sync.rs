//! Guards the invariant documented on `bwd_block_config_large`/`_small` in
//! `src/ops/cuda/attention/flash/flash_block_config.rs`: those Rust tables
//! must return the exact `(BLOCK_M, BLOCK_N)` that `flash_v2_bwd.cu`
//! instantiates its `FLASH_BWD_ENTRY` kernels with for the same head_dim.
//!
//! Why this matters: `flash_v2_bwd.cu`'s backward kernel gives each thread
//! ONE K row for the whole kernel (`const int k_row = tid;`, no stride). If
//! the Rust launcher's `block_dim.x` (taken from these tables) does not
//! match the kernel's own `BLOCK_N`, K rows at or above `blockDim.x` within
//! a tile are never written by any thread. `dK`/`dV` are allocated with
//! `Tensor::empty` (not zeroed), so a mismatch does not crash — it silently
//! leaves uninitialised memory in the gradient on the live training path.
//!
//! The `.cu`-parsing half below needs neither a GPU nor the `cuda` feature:
//! it is pure text scanning, checked by `flash_bwd_entry_count_is_plausible`.
//! Comparing the parsed entries against the Rust tables does need the `cuda`
//! feature, because `bwd_block_config_{large,small}_for_test` live in a
//! `#[cfg(feature = "cuda")]` module — but it needs no GPU: both accessors
//! are pure functions with no device query.

use std::fs;
use std::path::PathBuf;

/// One parsed `FLASH_BWD_ENTRY(T, HEAD_DIM, BLOCK_M, BLOCK_N, SUFFIX)` call.
#[derive(Debug, Clone)]
struct BwdEntry {
    head_dim: usize,
    block_m: usize,
    block_n: usize,
    suffix: String,
    is_small: bool,
}

const CU_RELATIVE_PATH: &str = "src/ops/cuda/kernels/attention/flash_v2_bwd.cu";

fn read_flash_v2_bwd_cu() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CU_RELATIVE_PATH);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Parses every `FLASH_BWD_ENTRY(...)` invocation (not the `#define` line
/// itself). Entries are one per line with no nested parens, so a plain
/// comma split on the argument text is exact.
fn parse_bwd_entries(text: &str) -> Vec<BwdEntry> {
    const MACRO: &str = "FLASH_BWD_ENTRY(";
    let mut out = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find(MACRO) {
        let mpos = search_from + rel;
        let line_start = text[..mpos].rfind('\n').map(|i| i + 1).unwrap_or(0);
        search_from = mpos + MACRO.len();
        if text[line_start..].trim_start().starts_with("#define") {
            continue;
        }
        let open = mpos + MACRO.len() - 1;
        let Some(close_rel) = text[open..].find(')') else {
            continue;
        };
        let args_text = &text[open + 1..open + close_rel];
        let parts: Vec<&str> = args_text.split(',').map(|s| s.trim()).collect();
        if parts.len() != 5 {
            panic!(
                "FLASH_BWD_ENTRY invocation has {} args, expected 5: {args_text}",
                parts.len()
            );
        }
        let (head_dim, block_m, block_n, suffix) = (parts[1], parts[2], parts[3], parts[4]);
        let head_dim: usize = head_dim
            .parse()
            .unwrap_or_else(|e| panic!("bad HEAD_DIM {head_dim:?} in {args_text:?}: {e}"));
        let block_m: usize = block_m
            .parse()
            .unwrap_or_else(|e| panic!("bad BLOCK_M {block_m:?} in {args_text:?}: {e}"));
        let block_n: usize = block_n
            .parse()
            .unwrap_or_else(|e| panic!("bad BLOCK_N {block_n:?} in {args_text:?}: {e}"));
        out.push(BwdEntry {
            head_dim,
            block_m,
            block_n,
            suffix: suffix.to_string(),
            is_small: suffix.starts_with("sm_"),
        });
    }
    out
}

/// Rot detector, not a target: if the macro name or arg format changes and
/// this parser silently stops matching, the sync test below would pass
/// vacuously (zero entries to check). The real count today is 36 (6
/// head_dims x 3 dtypes x 2 variants); 24 is a floor with headroom below it.
const MIN_PLAUSIBLE_ENTRIES: usize = 24;

#[test]
fn flash_bwd_entry_count_is_plausible() {
    let text = read_flash_v2_bwd_cu();
    let entries = parse_bwd_entries(&text);
    assert!(
        entries.len() >= MIN_PLAUSIBLE_ENTRIES,
        "parsed only {} FLASH_BWD_ENTRY invocations from {CU_RELATIVE_PATH} (expected at least \
         {MIN_PLAUSIBLE_ENTRIES}). This means the parser broke, not that kernels were genuinely \
         removed — check parse_bwd_entries against the current macro invocation syntax.",
        entries.len(),
    );
}

#[test]
fn suffix_sm_prefix_is_the_only_small_variant_marker() {
    // Confirms the convention the sync test below relies on: every small
    // variant's SUFFIX starts with "sm_", and no large variant's does.
    let text = read_flash_v2_bwd_cu();
    let entries = parse_bwd_entries(&text);
    for e in &entries {
        assert_eq!(
            e.suffix.starts_with("sm_"),
            e.is_small,
            "suffix {:?} (head_dim={}) breaks the sm_-prefix convention this test assumes",
            e.suffix,
            e.head_dim,
        );
    }
}

#[cfg(feature = "cuda")]
mod cuda_sync {
    use super::*;
    use boostr::ops::cuda::attention::flash::flash_block_config::{
        bwd_block_config_large_for_test, bwd_block_config_small_for_test,
    };

    /// Head_dims are always small positive numbers (32..=256 today); this
    /// range is generous enough to catch a head_dim added to one side of the
    /// sync (Rust table or .cu file) but not the other.
    const HEAD_DIM_PROBE_RANGE: std::ops::RangeInclusive<usize> = 1..=1024;

    #[test]
    fn bwd_block_config_matches_flash_v2_bwd_cu() {
        let text = read_flash_v2_bwd_cu();
        let entries = parse_bwd_entries(&text);
        assert!(
            !entries.is_empty(),
            "no FLASH_BWD_ENTRY invocations parsed from {CU_RELATIVE_PATH}; cannot check sync"
        );

        // Every parsed entry's (BLOCK_M, BLOCK_N) must match the Rust table
        // for its head_dim and variant.
        for e in &entries {
            let table_fn: fn(usize) -> Option<(usize, usize)> = if e.is_small {
                bwd_block_config_small_for_test
            } else {
                bwd_block_config_large_for_test
            };
            let table_name = if e.is_small {
                "bwd_block_config_small"
            } else {
                "bwd_block_config_large"
            };
            let rust_dims = table_fn(e.head_dim);
            assert_eq!(
                rust_dims,
                Some((e.block_m, e.block_n)),
                "flash_v2_bwd.cu instantiates flash_attention_bwd_{}_{} with \
                 (BLOCK_M={}, BLOCK_N={}), but {table_name}({}) returns {:?}. The Rust \
                 launcher would launch with a block size that does not match the kernel's \
                 BLOCK_N, leaving dK/dV rows unwritten in memory that was never zeroed.",
                e.head_dim,
                e.suffix,
                e.block_m,
                e.block_n,
                e.head_dim,
                rust_dims,
            );
        }

        // Every head_dim on one side must exist on the other, for each
        // variant independently — a head_dim present in only the .cu file or
        // only the Rust table is exactly the drift this test guards against.
        let mut cu_large: Vec<usize> = entries
            .iter()
            .filter(|e| !e.is_small)
            .map(|e| e.head_dim)
            .collect();
        let mut cu_small: Vec<usize> = entries
            .iter()
            .filter(|e| e.is_small)
            .map(|e| e.head_dim)
            .collect();
        cu_large.sort_unstable();
        cu_large.dedup();
        cu_small.sort_unstable();
        cu_small.dedup();

        let rust_large: Vec<usize> = HEAD_DIM_PROBE_RANGE
            .clone()
            .filter(|&hd| bwd_block_config_large_for_test(hd).is_some())
            .collect();
        let rust_small: Vec<usize> = HEAD_DIM_PROBE_RANGE
            .clone()
            .filter(|&hd| bwd_block_config_small_for_test(hd).is_some())
            .collect();

        assert_eq!(
            cu_large, rust_large,
            "head_dims with a large-variant FLASH_BWD_ENTRY in {CU_RELATIVE_PATH} ({:?}) do not \
             match head_dims covered by bwd_block_config_large ({:?}). A head_dim on only one \
             side means the launcher and the kernel disagree on block size for that head_dim, \
             leaving dK/dV rows unwritten in memory that was never zeroed.",
            cu_large, rust_large,
        );
        assert_eq!(
            cu_small, rust_small,
            "head_dims with a small-variant (sm_) FLASH_BWD_ENTRY in {CU_RELATIVE_PATH} ({:?}) \
             do not match head_dims covered by bwd_block_config_small ({:?}). A head_dim on only \
             one side means the launcher and the kernel disagree on block size for that \
             head_dim, leaving dK/dV rows unwritten in memory that was never zeroed.",
            cu_small, rust_small,
        );
    }
}
