//! Subset construction from a byte NFA to a DFA over byte classes.
//!
//! The edge ranges of the NFA cut `0..=255` into byte classes. Bytes in one
//! class behave the same in every state, so the construction steps once per
//! class. A DFA state is the sorted set of NFA states in an epsilon closure.
//! The set keeps only states with byte edges and the accepting state. An empty
//! set is the dead state, which the DFA never holds as a state.

use std::collections::HashMap;
use std::rc::Rc;

use super::automaton_limits::AutomatonLimits;
use super::error::{GrammarError, GrammarErrorKind};
use super::nfa::Nfa;

/// Transition sentinel for the dead state.
pub(super) const DEAD: u32 = u32::MAX;

/// A partial DFA over byte classes.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ClassDfa {
    /// Class id of each byte.
    pub classes: [u8; 256],
    /// Number of byte classes, from 1 to 256.
    pub num_classes: usize,
    /// `trans[state * num_classes + class]` is the next state, or `DEAD`.
    pub trans: Vec<u32>,
    /// Accepting flag of each state.
    pub accepting: Vec<bool>,
    /// Start state, or `DEAD` for the empty language.
    pub start: u32,
}

impl ClassDfa {
    /// Number of states.
    pub(super) fn num_states(&self) -> usize {
        self.accepting.len()
    }

    /// The next state from `state` on byte class `class`, or `DEAD`.
    pub(super) fn next(&self, state: u32, class: usize) -> u32 {
        let at = (state as usize)
            .saturating_mul(self.num_classes)
            .saturating_add(class);
        self.trans.get(at).copied().unwrap_or(DEAD)
    }
}

/// The class id of each byte and the class count for the edges of `nfa`.
pub(super) fn byte_classes(nfa: &Nfa) -> ([u8; 256], usize) {
    let mut cut = [false; 257];
    for state in &nfa.states {
        for &(lo, hi, _) in &state.bytes {
            for at in [usize::from(lo), usize::from(hi).saturating_add(1)] {
                if let Some(slot) = cut.get_mut(at) {
                    *slot = true;
                }
            }
        }
    }
    let mut classes = [0u8; 256];
    let mut id = 0u8;
    for (b, slot) in classes.iter_mut().enumerate() {
        if b > 0 && cut.get(b).copied().unwrap_or(false) {
            id = id.saturating_add(1);
        }
        *slot = id;
    }
    (classes, usize::from(id).saturating_add(1))
}

/// Determinize `nfa`. Stops early at `max_dfa_states` or `max_subset_work`.
pub(super) fn determinize(nfa: &Nfa, limits: &AutomatonLimits) -> Result<ClassDfa, GrammarError> {
    let (classes, num_classes) = byte_classes(nfa);
    let mut run = Run {
        nfa,
        limits,
        work: 0,
        stamp: vec![0; nfa.states.len()],
        generation: 0,
        ids: HashMap::new(),
        sets: Vec::new(),
        dfa: ClassDfa {
            classes,
            num_classes,
            trans: Vec::new(),
            accepting: Vec::new(),
            start: DEAD,
        },
    };
    let start = run.closure(&[nfa.start])?;
    run.dfa.start = run.intern(start)?;
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_classes];
    let mut index = 0usize;
    while let Some(set) = run.sets.get(index).cloned() {
        for &s in set.iter() {
            let Some(state) = nfa.states.get(s as usize) else {
                continue;
            };
            for &(lo, hi, target) in &state.bytes {
                let first = usize::from(classes.get(usize::from(lo)).copied().unwrap_or(0));
                let last = usize::from(classes.get(usize::from(hi)).copied().unwrap_or(0));
                run.charge(last.saturating_sub(first).saturating_add(1))?;
                for bucket in buckets.iter_mut().take(last.saturating_add(1)).skip(first) {
                    bucket.push(target);
                }
            }
        }
        let row = index.saturating_mul(num_classes);
        for (class, bucket) in buckets.iter_mut().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let next = run.closure(bucket.as_slice())?;
            bucket.clear();
            let id = run.intern(next)?;
            if let Some(slot) = run.dfa.trans.get_mut(row.saturating_add(class)) {
                *slot = id;
            }
        }
        index = index.saturating_add(1);
    }
    Ok(run.dfa)
}

struct Run<'n> {
    nfa: &'n Nfa,
    limits: &'n AutomatonLimits,
    /// Work units spent so far.
    work: u64,
    /// Generation stamp of each NFA state, for closure deduplication.
    stamp: Vec<u32>,
    generation: u32,
    /// DFA state id of each interned set.
    ids: HashMap<Rc<[u32]>, u32>,
    /// Interned sets, by DFA state id.
    sets: Vec<Rc<[u32]>>,
    dfa: ClassDfa,
}

impl Run<'_> {
    /// Spend `units` of work. Stops once the total passes `max_subset_work`.
    fn charge(&mut self, units: usize) -> Result<(), GrammarError> {
        self.work = self
            .work
            .saturating_add(u64::try_from(units).unwrap_or(u64::MAX));
        if self.work > self.limits.max_subset_work {
            return Err(GrammarError::new(
                GrammarErrorKind::LimitExceeded,
                None,
                format!(
                    "subset construction reaches {} work units, above max_subset_work {}. \
                     Simplify the grammar or raise max_subset_work.",
                    self.work, self.limits.max_subset_work
                ),
            ));
        }
        Ok(())
    }

    /// The sorted epsilon closure of `seeds`, kept to states with byte edges or acceptance.
    fn closure(&mut self, seeds: &[u32]) -> Result<Vec<u32>, GrammarError> {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamp.fill(0);
            self.generation = 1;
        }
        let generation = self.generation;
        let mut stack = Vec::with_capacity(seeds.len());
        for &seed in seeds {
            if let Some(mark) = self.stamp.get_mut(seed as usize)
                && *mark != generation
            {
                *mark = generation;
                stack.push(seed);
            }
        }
        let mut out = Vec::new();
        while let Some(s) = stack.pop() {
            let Some(state) = self.nfa.states.get(s as usize) else {
                continue;
            };
            self.work = self.work.saturating_add(1);
            if !state.bytes.is_empty() || s == self.nfa.accept {
                out.push(s);
            }
            for &next in &state.eps {
                if let Some(mark) = self.stamp.get_mut(next as usize)
                    && *mark != generation
                {
                    *mark = generation;
                    stack.push(next);
                }
            }
            self.charge(state.eps.len())?;
        }
        out.sort_unstable();
        Ok(out)
    }

    /// The DFA state id of `set`. A new set adds a state. An empty set is `DEAD`.
    fn intern(&mut self, set: Vec<u32>) -> Result<u32, GrammarError> {
        if set.is_empty() {
            return Ok(DEAD);
        }
        if let Some(&id) = self.ids.get(set.as_slice()) {
            return Ok(id);
        }
        let count = self.sets.len();
        let id = u32::try_from(count)
            .ok()
            .filter(|&id| count < self.limits.max_dfa_states && id != DEAD);
        let Some(id) = id else {
            return Err(GrammarError::new(
                GrammarErrorKind::LimitExceeded,
                None,
                format!(
                    "DFA reaches {} states, above max_dfa_states {}. \
                     Simplify the grammar or raise max_dfa_states.",
                    count.saturating_add(1),
                    self.limits.max_dfa_states
                ),
            ));
        };
        self.charge(set.len())?;
        let accepting = set.binary_search(&self.nfa.accept).is_ok();
        let set: Rc<[u32]> = Rc::from(set);
        self.ids.insert(Rc::clone(&set), id);
        self.sets.push(set);
        self.dfa.accepting.push(accepting);
        let width = self.dfa.num_classes;
        self.dfa.trans.extend(std::iter::repeat_n(DEAD, width));
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::ast::Expr;

    fn dfa(expr: &Expr) -> ClassDfa {
        let limits = AutomatonLimits::default();
        let nfa = Nfa::build(expr, &limits);
        let dfa = nfa.and_then(|n| determinize(&n, &limits));
        match dfa {
            Ok(d) => d,
            Err(err) => panic!("{expr:?}: {err}"),
        }
    }

    fn class(lo: char, hi: char) -> Expr {
        Expr::Class {
            negated: false,
            ranges: vec![(lo, hi)],
        }
    }

    #[test]
    fn byte_classes_split_at_every_edge_bound() {
        let d = dfa(&Expr::Alt(vec![class('a', 'c'), class('b', 'd')]));
        // Classes: below a, a, b-c, d, above d.
        assert_eq!(d.num_classes, 5);
        let c = |b: u8| d.classes.get(usize::from(b)).copied();
        assert_eq!(c(b'a'), Some(1));
        assert_eq!(c(b'b'), c(b'c'));
        assert_ne!(c(b'c'), c(b'd'));
        assert_eq!(c(0), Some(0));
        assert_eq!(c(255), Some(4));
    }

    #[test]
    fn no_edges_give_one_class() {
        let d = dfa(&Expr::Empty);
        assert_eq!(d.num_classes, 1);
        assert_eq!(d.num_states(), 1);
        assert_eq!(d.accepting, vec![true]);
        assert_eq!(d.next(0, 0), DEAD);
    }

    #[test]
    fn a_dead_end_set_is_not_a_state() {
        let d = dfa(&Expr::Seq(vec![
            Expr::Literal("a".into()),
            Expr::Alt(vec![]),
        ]));
        assert_eq!(d.num_states(), 1);
        let a = usize::from(d.classes.get(usize::from(b'a')).copied().unwrap_or(0));
        assert_eq!(d.next(d.start, a), DEAD);
        assert_eq!(dfa(&Expr::Alt(vec![])).start, DEAD);
    }

    #[test]
    fn state_and_work_limits_stop_early() {
        let expr = Expr::Repeat {
            expr: Box::new(class('a', 'z')),
            min: 0,
            max: Some(40),
        };
        let nfa = Nfa::build(&expr, &AutomatonLimits::default());
        let few = AutomatonLimits {
            max_dfa_states: 10,
            ..AutomatonLimits::default()
        };
        let err = nfa.as_ref().ok().and_then(|n| determinize(n, &few).err());
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(
            msg.contains("DFA reaches 11 states, above max_dfa_states 10"),
            "{msg}"
        );
        let cheap = AutomatonLimits {
            max_subset_work: 100,
            ..AutomatonLimits::default()
        };
        let err = nfa.as_ref().ok().and_then(|n| determinize(n, &cheap).err());
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(msg.contains("above max_subset_work 100"), "{msg}");
    }
}
