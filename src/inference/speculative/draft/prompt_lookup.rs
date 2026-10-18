//! Prompt lookup: draft by copying what followed an earlier n-gram.

use super::drafter::Drafter;
use crate::error::{Error, Result};

/// Tokens of context a [`PromptLookup`] scans by default.
pub const DEFAULT_LOOKUP_WINDOW: usize = 4096;

/// Drafts from the context itself, with no model.
///
/// Take the trailing n-gram of the context, for `n` from `max_ngram` down to
/// `min_ngram`. Find its most recent earlier occurrence. Propose the tokens
/// that followed it, up to `k` and up to the context end. An earlier
/// occurrence can overlap the trailing n-gram. A longer n-gram match wins
/// over a more recent shorter one. No match gives an empty proposal.
///
/// The scan reads only the last `window` tokens of the context. A proposal
/// costs `O(window * max_ngram)` comparisons per tried `n` at worst. It
/// allocates only the returned proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptLookup {
    min_ngram: usize,
    max_ngram: usize,
    window: usize,
}

impl PromptLookup {
    /// A drafter that tries n-grams of `max_ngram` down to `min_ngram` tokens
    /// within the last `window` tokens.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `min_ngram` is 0, `min_ngram` exceeds
    /// `max_ngram`, or `window` is not larger than `max_ngram`.
    pub fn new(min_ngram: usize, max_ngram: usize, window: usize) -> Result<Self> {
        if min_ngram == 0 {
            return Err(Error::InvalidArgument {
                arg: "min_ngram",
                reason: "got 0; pass an n-gram length of at least 1".into(),
            });
        }
        if min_ngram > max_ngram {
            return Err(Error::InvalidArgument {
                arg: "max_ngram",
                reason: format!(
                    "got {max_ngram}, below min_ngram {min_ngram}; pass at least {min_ngram}"
                ),
            });
        }
        if window <= max_ngram {
            return Err(Error::InvalidArgument {
                arg: "window",
                reason: format!(
                    "got {window}; a match of {max_ngram} tokens and its next token need a \
                     window of at least {}",
                    max_ngram + 1
                ),
            });
        }
        Ok(Self {
            min_ngram,
            max_ngram,
            window,
        })
    }

    /// Shortest n-gram tried.
    pub fn min_ngram(&self) -> usize {
        self.min_ngram
    }

    /// Longest n-gram tried.
    pub fn max_ngram(&self) -> usize {
        self.max_ngram
    }

    /// Tokens at the context end the scan reads.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Start of the tokens after the most recent earlier occurrence of the
    /// trailing `n` tokens, within the window.
    fn continuation_start(&self, context: &[u32], n: usize) -> Option<usize> {
        let len = context.len();
        if n >= len {
            return None;
        }
        let tail = &context[len - n..];
        let lowest = len.saturating_sub(self.window);
        let highest = len - n - 1;
        if highest < lowest {
            return None;
        }
        (lowest..=highest)
            .rev()
            .find(|&s| &context[s..s + n] == tail)
            .map(|s| s + n)
    }
}

impl Default for PromptLookup {
    /// N-grams of 3 down to 1 token in the last [`DEFAULT_LOOKUP_WINDOW`] tokens.
    fn default() -> Self {
        Self {
            min_ngram: 1,
            max_ngram: 3,
            window: DEFAULT_LOOKUP_WINDOW,
        }
    }
}

impl Drafter for PromptLookup {
    fn propose(&mut self, context: &[u32], k: usize) -> Vec<u32> {
        if k == 0 {
            return Vec::new();
        }
        for n in (self.min_ngram..=self.max_ngram).rev() {
            if let Some(start) = self.continuation_start(context, n) {
                let end = (start + k).min(context.len());
                return context[start..end].to_vec();
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(min: usize, max: usize) -> PromptLookup {
        PromptLookup::new(min, max, DEFAULT_LOOKUP_WINDOW).expect("valid")
    }

    #[test]
    fn no_match_gives_nothing() {
        let mut d = PromptLookup::default();
        assert!(d.propose(&[1, 2, 3, 4], 5).is_empty());
        assert!(d.propose(&[7], 5).is_empty());
        assert!(d.propose(&[], 5).is_empty());
    }

    #[test]
    fn zero_k_gives_nothing() {
        let mut d = PromptLookup::default();
        assert!(d.propose(&[1, 1, 1], 0).is_empty());
    }

    #[test]
    fn most_recent_match_wins() {
        let mut d = lookup(1, 1);
        assert_eq!(d.propose(&[1, 2, 1, 3, 1], 2), vec![3, 1]);
    }

    #[test]
    fn overlapping_match_stops_at_context_end() {
        let mut d = lookup(1, 3);
        assert_eq!(d.propose(&[5, 5, 5, 5], 3), vec![5]);
        assert_eq!(d.propose(&[4, 5, 4, 5, 4], 4), vec![5, 4]);
    }

    #[test]
    fn proposal_respects_k() {
        let mut d = lookup(1, 1);
        assert_eq!(d.propose(&[1, 2, 3, 4, 5, 1], 2), vec![2, 3]);
        assert_eq!(d.propose(&[1, 2, 3, 4, 5, 1], 9), vec![2, 3, 4, 5, 1]);
    }

    #[test]
    fn longer_ngram_wins_over_a_more_recent_shorter_one() {
        let mut d = lookup(1, 2);
        assert_eq!(d.propose(&[7, 8, 1, 9, 8, 2, 7, 8], 2), vec![1, 9]);
        let mut d = lookup(1, 1);
        assert_eq!(d.propose(&[7, 8, 1, 9, 8, 2, 7, 8], 2), vec![2, 7]);
    }

    #[test]
    fn falls_back_to_a_shorter_ngram() {
        let mut d = lookup(1, 3);
        assert_eq!(d.propose(&[1, 2, 9, 3, 2], 2), vec![9, 3]);
        let mut d = lookup(2, 3);
        assert!(d.propose(&[1, 2, 9, 3, 2], 2).is_empty());
    }

    #[test]
    fn window_caps_the_scan() {
        let mut context = vec![1, 2];
        context.extend(10..20);
        context.push(1);
        let mut wide = PromptLookup::new(1, 1, context.len()).expect("valid");
        assert_eq!(wide.propose(&context, 2), vec![2, 10]);
        let mut narrow = PromptLookup::new(1, 1, context.len() - 1).expect("valid");
        assert!(narrow.propose(&context, 2).is_empty());
    }

    #[test]
    fn construction_checks_its_arguments() {
        let arg = |r: Result<PromptLookup>| match r {
            Err(Error::InvalidArgument { arg, .. }) => arg,
            other => panic!("expected InvalidArgument, got {other:?}"),
        };
        assert_eq!(arg(PromptLookup::new(0, 3, 10)), "min_ngram");
        assert_eq!(arg(PromptLookup::new(3, 2, 10)), "max_ngram");
        assert_eq!(arg(PromptLookup::new(1, 3, 3)), "window");
        let d = PromptLookup::new(2, 4, 5).expect("valid");
        assert_eq!((d.min_ngram(), d.max_ngram(), d.window()), (2, 4, 5));
    }
}
