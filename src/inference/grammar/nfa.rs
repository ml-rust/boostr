//! Thompson-style NFA over bytes, built from an expanded `Expr`.
//!
//! Construction runs on an explicit task stack, so tree depth never grows the
//! call stack. A task asks for the paths of one expression between two states.
//! Each task only adds edges that leave its `from` state, enter its `to` state,
//! or touch states it creates. Each loop gets a fresh state, so sibling
//! constructions never share a cycle.

use super::ast::Expr;
use super::automaton_limits::AutomatonLimits;
use super::error::{GrammarError, GrammarErrorKind, excerpt};
use super::utf8::{any_scalar, class_scalars, utf8_sequences};

/// One NFA state: epsilon edges and byte-range edges.
#[derive(Clone, Debug, Default)]
pub(super) struct NfaState {
    /// Targets reached without input.
    pub eps: Vec<u32>,
    /// `(lo, hi, target)`: an edge on each byte in `lo..=hi`.
    pub bytes: Vec<(u8, u8, u32)>,
}

/// A byte NFA with one start state and one accepting state.
#[derive(Clone, Debug)]
pub(super) struct Nfa {
    /// Every state, by id.
    pub states: Vec<NfaState>,
    /// Start state id.
    pub start: u32,
    /// Accepting state id.
    pub accept: u32,
}

impl Nfa {
    /// Build the NFA of `expr`. The state and edge counts must stay within their limits.
    pub(super) fn build(expr: &Expr, limits: &AutomatonLimits) -> Result<Nfa, GrammarError> {
        let mut b = Builder {
            states: Vec::new(),
            limit: limits.max_nfa_states,
            edges: 0,
            edge_limit: limits.max_nfa_edges,
        };
        let start = b.state()?;
        let accept = b.state()?;
        let mut tasks: Vec<(&Expr, u32, u32)> = vec![(expr, start, accept)];
        while let Some((expr, from, to)) = tasks.pop() {
            match expr {
                Expr::Empty => b.eps(from, to)?,
                Expr::Literal(text) => b.chain(from, to, text.bytes().map(|x| (x, x)))?,
                Expr::Class { negated, ranges } => {
                    b.scalars(from, to, &class_scalars(*negated, ranges))?;
                }
                Expr::Any => b.scalars(from, to, &any_scalar())?,
                Expr::Ref(name) => return Err(reference(name)),
                Expr::Seq(items) => {
                    let mut at = from;
                    let count = items.len();
                    for (i, item) in items.iter().enumerate() {
                        let next = if i.saturating_add(1) == count {
                            to
                        } else {
                            b.state()?
                        };
                        tasks.push((item, at, next));
                        at = next;
                    }
                    if count == 0 {
                        b.eps(from, to)?;
                    }
                }
                Expr::Alt(items) => tasks.extend(items.iter().map(|item| (item, from, to))),
                Expr::Repeat {
                    expr: body,
                    min,
                    max,
                } => {
                    let body: &Expr = body;
                    // A maximum below the minimum matches nothing.
                    if max.is_some_and(|max| max < *min) {
                        continue;
                    }
                    let mut at = from;
                    for _ in 0..*min {
                        let next = b.state()?;
                        tasks.push((body, at, next));
                        at = next;
                    }
                    match max {
                        None => {
                            let lp = b.state()?;
                            b.eps(at, lp)?;
                            tasks.push((body, lp, lp));
                            b.eps(lp, to)?;
                        }
                        Some(max) => {
                            for _ in *min..*max {
                                let next = b.state()?;
                                b.eps(at, to)?;
                                tasks.push((body, at, next));
                                at = next;
                            }
                            b.eps(at, to)?;
                        }
                    }
                }
            }
        }
        Ok(Nfa {
            states: b.states,
            start,
            accept,
        })
    }
}

/// The error for a rule reference left in an expression.
fn reference(name: &str) -> GrammarError {
    GrammarError::new(
        GrammarErrorKind::Unsupported,
        None,
        format!(
            "expression holds rule reference `{}`. \
             Call `Grammar::expand` before you build an automaton.",
            excerpt(name)
        ),
    )
}

struct Builder {
    states: Vec<NfaState>,
    limit: usize,
    /// Edges added so far, byte-range and epsilon.
    edges: usize,
    edge_limit: usize,
}

impl Builder {
    /// A new state. Stops once the count passes `max_nfa_states`.
    fn state(&mut self) -> Result<u32, GrammarError> {
        let id = self.states.len();
        let id32 = u32::try_from(id).ok().filter(|_| id < self.limit);
        let Some(id32) = id32 else {
            return Err(GrammarError::new(
                GrammarErrorKind::LimitExceeded,
                None,
                format!(
                    "NFA reaches {} states, above max_nfa_states {}. \
                     Lower the repeat counts or raise max_nfa_states.",
                    id.saturating_add(1),
                    self.limit
                ),
            ));
        };
        self.states.push(NfaState::default());
        Ok(id32)
    }

    /// Count one more edge. Stops once the count passes `max_nfa_edges`.
    fn count_edge(&mut self) -> Result<(), GrammarError> {
        let count = self.edges.saturating_add(1);
        if count > self.edge_limit {
            return Err(GrammarError::new(
                GrammarErrorKind::LimitExceeded,
                None,
                format!(
                    "NFA reaches {count} edges, above max_nfa_edges {}. \
                     Lower the repeat counts, use fewer class ranges or raise max_nfa_edges.",
                    self.edge_limit
                ),
            ));
        }
        self.edges = count;
        Ok(())
    }

    fn eps(&mut self, from: u32, to: u32) -> Result<(), GrammarError> {
        self.count_edge()?;
        if let Some(state) = self.states.get_mut(from as usize) {
            state.eps.push(to);
        }
        Ok(())
    }

    fn edge(&mut self, from: u32, lo: u8, hi: u8, to: u32) -> Result<(), GrammarError> {
        self.count_edge()?;
        if let Some(state) = self.states.get_mut(from as usize) {
            state.bytes.push((lo, hi, to));
        }
        Ok(())
    }

    /// A path from `from` to `to` through one fresh state per inner step.
    fn chain(
        &mut self,
        from: u32,
        to: u32,
        steps: impl ExactSizeIterator<Item = (u8, u8)>,
    ) -> Result<(), GrammarError> {
        let count = steps.len();
        if count == 0 {
            return self.eps(from, to);
        }
        let mut at = from;
        for (i, (lo, hi)) in steps.enumerate() {
            let next = if i.saturating_add(1) == count {
                to
            } else {
                self.state()?
            };
            self.edge(at, lo, hi, next)?;
            at = next;
        }
        Ok(())
    }

    /// One path per UTF-8 sequence of the scalar ranges. No ranges add no path.
    fn scalars(&mut self, from: u32, to: u32, ranges: &[(u32, u32)]) -> Result<(), GrammarError> {
        for &(lo, hi) in ranges {
            for seq in utf8_sequences(lo, hi) {
                self.chain(from, to, seq.iter().copied())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(expr: &Expr) -> Result<Nfa, GrammarError> {
        Nfa::build(expr, &AutomatonLimits::default())
    }

    #[test]
    fn literal_builds_one_state_per_inner_byte() {
        let nfa = build(&Expr::Literal("é!".into())).ok();
        // Start, accept and two inner states for three bytes.
        assert_eq!(nfa.map(|n| n.states.len()), Some(4));
    }

    #[test]
    fn empty_alt_adds_no_edge() {
        let nfa = build(&Expr::Alt(vec![]));
        let edges = nfa.map(|n| {
            n.states
                .iter()
                .map(|s| s.eps.len() + s.bytes.len())
                .sum::<usize>()
        });
        assert_eq!(edges.ok(), Some(0));
    }

    #[test]
    fn references_are_unsupported() {
        let err = build(&Expr::Ref("value".into())).err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::Unsupported)
        );
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(msg.contains("`value`"), "{msg}");
    }

    #[test]
    fn state_limit_stops_construction() {
        let limits = AutomatonLimits {
            max_nfa_states: 50,
            ..AutomatonLimits::default()
        };
        let expr = Expr::Repeat {
            expr: Box::new(Expr::Literal("ab".into())),
            min: 100,
            max: Some(100),
        };
        let err = Nfa::build(&expr, &limits).err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::LimitExceeded)
        );
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(
            msg.contains("NFA reaches 51 states, above max_nfa_states 50"),
            "{msg}"
        );
    }

    fn wide_class() -> Expr {
        // 64 disjoint single-byte ranges: 0x01, 0x03, ... 0x7F.
        Expr::Class {
            negated: false,
            ranges: (0..64u8)
                .map(|i| (char::from(2 * i + 1), char::from(2 * i + 1)))
                .collect(),
        }
    }

    #[test]
    fn edge_limit_stops_construction_and_reports_the_counts() {
        let limits = AutomatonLimits {
            max_nfa_edges: 10_000,
            ..AutomatonLimits::default()
        };
        let expr = Expr::Repeat {
            expr: Box::new(wide_class()),
            min: 1_000,
            max: Some(1_000),
        };
        let err = Nfa::build(&expr, &limits).err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::LimitExceeded)
        );
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(
            msg.contains("NFA reaches 10001 edges, above max_nfa_edges 10000"),
            "{msg}"
        );
        // States stay far below their limit: only edges grew.
        assert!(Nfa::build(&expr, &AutomatonLimits::default()).is_ok());
    }

    #[test]
    fn epsilon_edges_count_toward_the_edge_limit() {
        let limits = AutomatonLimits {
            max_nfa_edges: 5,
            ..AutomatonLimits::default()
        };
        let expr = Expr::Repeat {
            expr: Box::new(Expr::Empty),
            min: 0,
            max: Some(20),
        };
        let msg = Nfa::build(&expr, &limits)
            .err()
            .map(|e| e.message().to_string())
            .unwrap_or_default();
        assert!(msg.contains("above max_nfa_edges 5"), "{msg}");
    }

    #[test]
    fn huge_hand_built_repeat_counts_stop_at_the_limit() {
        let expr = Expr::Repeat {
            expr: Box::new(Expr::Empty),
            min: u32::MAX,
            max: None,
        };
        let kind = build(&expr).err().map(|e| e.kind());
        assert_eq!(kind, Some(GrammarErrorKind::LimitExceeded));
    }
}
