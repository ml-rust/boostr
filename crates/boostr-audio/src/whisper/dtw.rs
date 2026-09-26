//! Dynamic time warping over a Whisper alignment matrix, and the per-token
//! timestamps HuggingFace derives from its path.
//!
//! Exact port of `transformers.models.whisper.generation_whisper`:
//! `_dynamic_time_warping` for the warp itself, and the tail of
//! `_extract_token_timestamps` for the jump-time / token-timestamp derivation.
//! The tie-breaking order in the cost recurrence (diagonal, then up, then
//! left) is load-bearing: HF's `elif`/`else` chain picks `left` on every
//! three-way tie, and this port keeps that exact order so a real matrix's
//! path matches HF token-for-token, not just in total cost.

/// Runs DTW over `similarity` (row-major `[rows, cols]`, HF's alignment
/// matrix) and returns `(text_indices, time_indices)` — the row (token) and
/// column (frame) index of every step along the optimal path, in path order
/// (index 0 first).
///
/// HF runs the warp on `-matrix` (it minimizes a *cost*, and the alignment
/// matrix holds *similarity*), so this negates internally — callers pass the
/// alignment matrix as-is, matching `alignment_matrix`'s output.
///
/// Returns two empty vectors when `rows == 0` or `cols == 0`: there is no path
/// to trace, and HF itself is never called with an empty matrix on this path
/// (callers already return early on zero generated tokens).
pub fn dynamic_time_warping(
    similarity: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<usize>, Vec<usize>) {
    if rows == 0 || cols == 0 {
        return (Vec::new(), Vec::new());
    }
    debug_assert_eq!(similarity.len(), rows * cols);

    // `cost`/`trace` are (rows+1) x (cols+1), row-major, matching HF's
    // `output_length + 1` x `input_length + 1` layout exactly.
    let cost_cols = cols + 1;
    let mut cost = vec![f64::INFINITY; (rows + 1) * cost_cols];
    // Trace codes: 0 = diagonal (i-1, j-1), 1 = up (i-1, j), 2 = left (i, j-1).
    let mut trace = vec![2u8; (rows + 1) * cost_cols];

    cost[0] = 0.0;

    for j in 1..=cols {
        for i in 1..=rows {
            let c0 = cost[(i - 1) * cost_cols + (j - 1)];
            let c1 = cost[(i - 1) * cost_cols + j];
            let c2 = cost[i * cost_cols + (j - 1)];

            // Exact port of HF's `if c0 < c1 and c0 < c2: ... elif c1 < c0
            // and c1 < c2: ... else: ...` — a three-way tie (or a c0/c2 tie)
            // falls through to the `else` branch and picks `left`.
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0u8)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1u8)
            } else {
                (c2, 2u8)
            };

            let similarity_val = similarity[(i - 1) * cols + (j - 1)] as f64;
            cost[i * cost_cols + j] = -similarity_val + c;
            trace[i * cost_cols + j] = t;
        }
    }

    // HF overwrites the boundary rows/columns of `trace` AFTER filling the
    // interior, so the backtrace's forced moves at the edges are `left`
    // along row 0 and `up` along column 0, regardless of what the fill loop
    // wrote there (it never touches trace[0, *] or trace[*, 0] itself).
    trace[..cost_cols].fill(2);
    trace.iter_mut().step_by(cost_cols).for_each(|t| *t = 1);

    let mut text_indices = Vec::new();
    let mut time_indices = Vec::new();
    let mut i = rows;
    let mut j = cols;
    while i > 0 || j > 0 {
        text_indices.push(i - 1);
        time_indices.push(j - 1);
        match trace[i * cost_cols + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    text_indices.reverse();
    time_indices.reverse();
    (text_indices, time_indices)
}

/// Per-token timestamps (seconds) for `rows` generated tokens, HF's
/// `_extract_token_timestamps` tail:
///
/// 1. DTW over `alignment` gives `(text_indices, time_indices)`.
/// 2. `jump_times[k]` is `time_indices` at the first path step that reaches
///    text row `k`, times `time_precision` — one entry per row of `alignment`.
/// 3. The full array is `num_input_ids` zeros (the forced prefix has no
///    meaningful timestamp), then `jump_times`, then `jump_times`'s last
///    value duplicated once more (the final generated token is never fed
///    back into the decoder, so it has no row of its own and HF reuses the
///    previous jump time rather than leaving it undefined).
///
/// Length: `num_input_ids + rows + 1`. Empty when `rows == 0` (matching HF's
/// early return for zero DTW rows, which this crate's callers also check for
/// before calling this at all, since there is no path to derive `words` from
/// either way).
pub fn token_timestamps(
    alignment: &[f32],
    rows: usize,
    cols: usize,
    time_precision: f32,
    num_input_ids: usize,
) -> Vec<f32> {
    if rows == 0 {
        return Vec::new();
    }
    let (text_indices, time_indices) = dynamic_time_warping(alignment, rows, cols);

    let mut jump_times: Vec<f32> = Vec::with_capacity(rows);
    let mut prev_text: Option<usize> = None;
    for (step, &text_idx) in text_indices.iter().enumerate() {
        let is_jump = prev_text != Some(text_idx);
        if is_jump {
            jump_times.push(time_indices[step] as f32 * time_precision);
        }
        prev_text = Some(text_idx);
    }

    let mut out = Vec::with_capacity(num_input_ids + rows + 1);
    out.resize(num_input_ids, 0.0);
    out.extend_from_slice(&jump_times);
    if let Some(&last) = jump_times.last() {
        out.push(last);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Similarity matrix and expected DTW path pasted from a direct run of
    /// HF's `_dynamic_time_warping(-matrix)` in Python: 3 tokens, 5 frames.
    #[rustfmt::skip]
    const SIM_3X5: [f32; 15] = [
        0.10, 0.20, 0.90, 0.10, 0.05,
        0.05, 0.10, 0.30, 0.80, 0.10,
        0.05, 0.05, 0.10, 0.20, 0.90,
    ];

    #[test]
    fn matches_hf_reference_path() {
        let (text_indices, time_indices) = dynamic_time_warping(&SIM_3X5, 3, 5);
        assert_eq!(text_indices, vec![0, 0, 0, 1, 1, 2, 2]);
        assert_eq!(time_indices, vec![0, 1, 2, 2, 3, 3, 4]);
    }

    #[test]
    fn jump_times_match_hf_reference() {
        let ts = token_timestamps(&SIM_3X5, 3, 5, 0.02, 0);
        // jump_times = [0.0, 0.04, 0.06] pasted from the same Python run;
        // num_input_ids = 0, so the array is jump_times + one duplicate.
        let expected = [0.0f32, 0.04, 0.06, 0.06];
        assert_eq!(ts.len(), expected.len());
        for (got, want) in ts.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn token_timestamps_prepends_zeros_for_prefix() {
        let ts = token_timestamps(&SIM_3X5, 3, 5, 0.02, 4);
        assert_eq!(&ts[..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(ts.len(), 4 + 3 + 1);
        let expected_tail = [0.0f32, 0.04, 0.06, 0.06];
        for (got, want) in ts[4..].iter().zip(expected_tail.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    /// A uniform (all-zero) matrix ties every step, exercising HF's
    /// `else`-falls-to-`left` tie order. Expected path pasted from the same
    /// Python run: 2 tokens, 3 frames, flat similarity.
    #[test]
    fn ties_prefer_left_matching_hf() {
        let flat = [0.0f32; 6];
        let (text_indices, time_indices) = dynamic_time_warping(&flat, 2, 3);
        assert_eq!(text_indices, vec![0, 1, 1, 1]);
        assert_eq!(time_indices, vec![0, 0, 1, 2]);
    }

    #[test]
    fn empty_matrix_yields_empty_path() {
        let (text_indices, time_indices) = dynamic_time_warping(&[], 0, 5);
        assert!(text_indices.is_empty());
        assert!(time_indices.is_empty());
        assert!(token_timestamps(&[], 0, 5, 0.02, 4).is_empty());
    }
}
