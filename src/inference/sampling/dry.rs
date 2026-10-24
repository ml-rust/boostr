//! The DRY (Don't Repeat Yourself) penalty over host logits.

use crate::error::{Error, Result};

/// The config of [`apply_dry_penalty`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DryParams {
    /// Subtracted from a logit once per matched token. 0 disables the penalty.
    pub multiplier: f32,
    /// The shortest repeated n-gram the penalty counts. Its last token is the candidate.
    pub base: usize,
    /// Tokens of history the scan reads, counted back from the end. 0 reads all of it.
    pub allowed_length: usize,
}

impl DryParams {
    /// Check every field against its range.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] naming the first field out of range.
    pub fn check(&self) -> Result<()> {
        if !self.multiplier.is_finite() || self.multiplier < 0.0 {
            return Err(Error::InvalidArgument {
                arg: "multiplier",
                reason: format!(
                    "dry: got {}; pass a finite value of at least 0, or 0.0 to disable it",
                    self.multiplier
                ),
            });
        }
        if self.base == 0 {
            return Err(Error::InvalidArgument {
                arg: "base",
                reason: "dry: got 0; pass an n-gram length of at least 1".to_string(),
            });
        }
        Ok(())
    }
}

/// Penalize each token that extends an n-gram already in `history`.
///
/// The suffix is the last `base - 1` tokens of the scanned history. Each
/// earlier match of the suffix extends backward while the tokens agree. The
/// token after that match loses `multiplier * match_length` from its logit.
/// A token that follows several matches loses each penalty in turn.
///
/// A scanned history shorter than `base` changes nothing. An id at or past
/// `logits.len()` gets no penalty. `base == 0` changes nothing: call
/// [`DryParams::check`] first to reject it.
pub fn apply_dry_penalty(logits: &mut [f32], history: &[u32], params: &DryParams) {
    let DryParams {
        multiplier,
        base,
        allowed_length,
    } = *params;
    if base == 0 {
        return;
    }
    let history = if allowed_length > 0 && allowed_length < history.len() {
        &history[history.len() - allowed_length..]
    } else {
        history
    };
    if history.len() < base {
        return;
    }

    let suffix_start = history.len().saturating_sub(base - 1);
    let suffix = &history[suffix_start..];
    for start in 0..suffix_start {
        let end = start + suffix.len();
        if end >= history.len() {
            break;
        }
        if history[start..end] != *suffix {
            continue;
        }
        let mut match_len = suffix.len();
        let (mut s, mut e) = (start, suffix_start);
        while s > 0 && e > 0 && history[s - 1] == history[e - 1] {
            match_len += 1;
            s -= 1;
            e -= 1;
        }
        let next = history[end] as usize;
        if next < logits.len() {
            logits[next] -= multiplier * match_len as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(multiplier: f32, base: usize, allowed_length: usize) -> DryParams {
        DryParams {
            multiplier,
            base,
            allowed_length,
        }
    }

    fn penalized(history: &[u32], p: &DryParams) -> Vec<f32> {
        let mut logits = vec![0.0f32; 8];
        apply_dry_penalty(&mut logits, history, p);
        logits
    }

    #[test]
    fn no_repeat_leaves_the_logits_unchanged() {
        assert_eq!(
            penalized(&[0, 1, 2, 3, 4], &params(1.0, 2, 0)),
            vec![0.0; 8]
        );
    }

    #[test]
    fn the_follower_of_a_repeated_token_is_penalized() {
        // Suffix [2] occurred at index 1, followed by 3.
        let got = penalized(&[1, 2, 3, 2], &params(0.5, 2, 0));
        let mut want = vec![0.0f32; 8];
        want[3] = -0.5;
        assert_eq!(got, want);
    }

    #[test]
    fn the_penalty_scales_with_the_match_length() {
        // Suffix [2] matches at index 1; [1, 2] extends it to length 2.
        let long = penalized(&[1, 2, 3, 1, 2], &params(1.0, 2, 0));
        assert_eq!(long[3], -2.0);
        // Only [2] matches when the earlier token differs.
        let short = penalized(&[1, 2, 3, 4, 2], &params(1.0, 2, 0));
        assert_eq!(short[3], -1.0);
    }

    #[test]
    fn a_history_shorter_than_base_changes_nothing() {
        assert_eq!(penalized(&[1, 2], &params(1.0, 3, 0)), vec![0.0; 8]);
        assert_eq!(penalized(&[1, 2, 1, 2], &params(1.0, 0, 0)), vec![0.0; 8]);
    }

    #[test]
    fn allowed_length_limits_the_scan_to_the_tail() {
        let history = [2, 5, 0, 0, 0, 2];
        assert_eq!(penalized(&history, &params(1.0, 2, 0))[5], -1.0);
        assert_eq!(penalized(&history, &params(1.0, 2, 4)), vec![0.0; 8]);
    }

    #[test]
    fn an_id_past_the_vocab_gets_no_penalty() {
        assert_eq!(penalized(&[2, 40, 2], &params(1.0, 2, 0)), vec![0.0; 8]);
    }

    #[test]
    fn check_names_the_bad_field() {
        params(0.8, 2, 0).check().expect("valid");
        params(0.0, 1, 7).check().expect("disabled");
        for (p, arg) in [
            (params(f32::NAN, 2, 0), "multiplier"),
            (params(f32::INFINITY, 2, 0), "multiplier"),
            (params(-0.1, 2, 0), "multiplier"),
            (params(1.0, 0, 0), "base"),
        ] {
            match p.check() {
                Err(Error::InvalidArgument { arg: got, .. }) => assert_eq!(got, arg, "{p:?}"),
                other => panic!("{p:?}: expected InvalidArgument, got {other:?}"),
            }
        }
    }
}
