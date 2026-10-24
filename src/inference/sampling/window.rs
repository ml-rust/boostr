//! The penalty window: distinct token ids of the recent history and their counts.

use std::collections::BTreeMap;

/// The distinct ids of the penalty window and how often each occurs, ascending by id.
///
/// The window is the last `last_n` tokens of `recent`. `last_n == 0`, or a
/// value at or past `recent.len()`, takes all of `recent`. This is the window
/// of [`SampleRule::penalty_last_n`](crate::inference::speculative::SampleRule).
///
/// The two vectors have the same length. They are the `token_ids` (I64) and
/// `token_counts` (I32) inputs of
/// [`SamplingOps::apply_sampling_penalties`](crate::ops::traits::inference::SamplingOps).
/// An id at or past the vocab stays in the result. The penalty op skips it.
pub fn penalty_window(recent: &[u32], last_n: usize) -> (Vec<i64>, Vec<i32>) {
    let start = if last_n == 0 {
        0
    } else {
        recent.len().saturating_sub(last_n)
    };
    let mut counts = BTreeMap::<u32, i32>::new();
    for &token in &recent[start..] {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(token, count)| (i64::from(token), count))
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::traits::inference::SamplingOps;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use std::collections::HashMap;

    #[test]
    fn empty_history_gives_an_empty_window() {
        assert_eq!(penalty_window(&[], 0), (vec![], vec![]));
        assert_eq!(penalty_window(&[], 8), (vec![], vec![]));
    }

    #[test]
    fn zero_and_long_windows_take_the_whole_history() {
        let recent = [3, 1, 4, 1, 5];
        let all = (vec![1, 3, 4, 5], vec![2, 1, 1, 1]);
        assert_eq!(penalty_window(&recent, 0), all);
        assert_eq!(penalty_window(&recent, 5), all);
        assert_eq!(penalty_window(&recent, 64), all);
    }

    #[test]
    fn a_short_window_takes_the_tail_sorted_by_id() {
        let recent = [9, 9, 7, 2, 7, 2, 2];
        assert_eq!(penalty_window(&recent, 4), (vec![2, 7], vec![3, 1]));
        assert_eq!(penalty_window(&recent, 1), (vec![2], vec![1]));
        let (ids, _) = penalty_window(&[50, 3, 1000, 3, 0], 0);
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "{ids:?}");
    }

    /// The penalties of a sorted window and of the same window in hash order.
    #[test]
    fn penalties_ignore_the_window_order() -> crate::error::Result<()> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let recent = [3u32, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 40];
        let row: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) * 0.5).collect();

        let mut hashed = HashMap::<u32, i32>::new();
        for &t in &recent {
            *hashed.entry(t).or_insert(0) += 1;
        }
        let (hash_ids, hash_counts): (Vec<i64>, Vec<i32>) =
            hashed.into_iter().map(|(t, c)| (i64::from(t), c)).unzip();
        let (ids, counts) = penalty_window(&recent, 0);

        let mut out = Vec::new();
        for (ids, counts) in [(ids, counts), (hash_ids, hash_counts)] {
            let logits = Tensor::<CpuRuntime>::from_slice(&row, &[row.len()], &device)?;
            let ids_t = Tensor::<CpuRuntime>::from_slice(&ids, &[ids.len()], &device)?;
            let counts_t = Tensor::<CpuRuntime>::from_slice(&counts, &[counts.len()], &device)?;
            client.apply_sampling_penalties(&logits, &ids_t, &counts_t, 1.3, 0.2, 0.1)?;
            let bits: Vec<u32> = logits.to_vec::<f32>().iter().map(|v| v.to_bits()).collect();
            out.push(bits);
        }
        assert_eq!(out[0], out[1]);
        Ok(())
    }
}
