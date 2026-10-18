//! Limits that automaton construction enforces on an expanded expression.

use super::error::{GrammarError, GrammarErrorKind};
use super::limits::ExpandLimits;

/// Byte classes in the widest DFA row.
const MAX_CLASSES: usize = 256;

/// Limits that NFA and DFA construction enforce.
///
/// Each check runs while the automaton grows, so an exceeded limit stops
/// the work early. Each field states the worst-case memory it implies.
/// The sum of those figures bounds memory for any input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutomatonLimits {
    /// Maximum NFA state count.
    ///
    /// Each state holds two vectors, so the worst case is about 50 bytes per state.
    pub max_nfa_states: usize,
    /// Maximum NFA edge count: byte-range edges plus epsilon edges.
    ///
    /// An edge takes about 8 bytes, so the worst case is 16 MB at the default.
    pub max_nfa_edges: usize,
    /// Maximum DFA state count during subset construction, before minimization.
    ///
    /// The final `ByteDfa` table holds 1 KiB per state, 16 MiB at the default.
    /// The value must stay at or below `AutomatonLimits::max_dfa_states_ceiling`.
    pub max_dfa_states: usize,
    /// Maximum subset construction work: NFA states and edges visited, summed over all DFA states.
    ///
    /// Each state set element takes 4 bytes, so the worst case is 40 MB at the default.
    pub max_subset_work: u64,
}

impl Default for AutomatonLimits {
    fn default() -> Self {
        Self {
            max_nfa_states: 200_000,
            max_nfa_edges: 2_000_000,
            max_dfa_states: 16_384,
            max_subset_work: 10_000_000,
        }
    }
}

impl AutomatonLimits {
    /// The largest `max_dfa_states` that minimization can index with `u32` offsets.
    ///
    /// Minimization adds one sink state and keeps one row per byte class.
    pub fn max_dfa_states_ceiling() -> usize {
        usize::try_from(u32::MAX / 256)
            .unwrap_or(0)
            .saturating_sub(1)
    }

    /// Check that the limits fit the automaton's `u32` indexes.
    ///
    /// A `max_dfa_states` above `max_dfa_states_ceiling` is a `LimitExceeded` error.
    pub fn check(&self) -> Result<(), GrammarError> {
        let rows = self
            .max_dfa_states
            .checked_add(1)
            .and_then(|states| states.checked_mul(MAX_CLASSES));
        if rows.is_some_and(|rows| u32::try_from(rows).is_ok()) {
            return Ok(());
        }
        Err(GrammarError::new(
            GrammarErrorKind::LimitExceeded,
            None,
            format!(
                "max_dfa_states {} is above the maximum {}. Lower max_dfa_states.",
                self.max_dfa_states,
                Self::max_dfa_states_ceiling()
            ),
        ))
    }
}

/// Limits for the whole path from grammar text to `ByteDfa`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileLimits {
    /// Limits for parsing and rule expansion.
    pub expand: ExpandLimits,
    /// Limits for NFA and DFA construction.
    pub automaton: AutomatonLimits,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_match_the_documented_values() {
        let limits = AutomatonLimits::default();
        assert_eq!(limits.max_nfa_states, 200_000);
        assert_eq!(limits.max_nfa_edges, 2_000_000);
        assert_eq!(limits.max_dfa_states, 16_384);
        assert_eq!(limits.max_subset_work, 10_000_000);
        assert!(limits.check().is_ok());
        let all = CompileLimits::default();
        assert_eq!(all.expand, ExpandLimits::default());
        assert_eq!(all.automaton, limits);
    }

    fn with_dfa_states(max_dfa_states: usize) -> AutomatonLimits {
        AutomatonLimits {
            max_dfa_states,
            ..AutomatonLimits::default()
        }
    }

    #[test]
    fn check_rejects_a_dfa_state_limit_that_overflows_u32_offsets() {
        assert!(with_dfa_states(16_384).check().is_ok());
        let err = with_dfa_states(17_000_000).check().err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::LimitExceeded)
        );
        assert_eq!(
            err.map(|e| e.message().to_string()).unwrap_or_default(),
            "max_dfa_states 17000000 is above the maximum 16777214. Lower max_dfa_states."
        );
        assert!(with_dfa_states(usize::MAX).check().is_err());
    }

    #[test]
    fn check_accepts_the_ceiling_and_rejects_one_more() {
        let ceiling = AutomatonLimits::max_dfa_states_ceiling();
        assert_eq!(ceiling, 16_777_214);
        assert!(with_dfa_states(ceiling).check().is_ok());
        assert!(with_dfa_states(ceiling + 1).check().is_err());
    }
}
