//! `ByteDfa`: a trimmed byte-level DFA with a dense transition table.
//!
//! Every state reaches an accepting state, and state 0 is the start. The dead
//! state is not a state: a missing transition is `ByteDfa::NO_TRANSITION`.

use std::collections::VecDeque;

use super::error::{GrammarError, GrammarErrorKind};
use super::subset::{ClassDfa, DEAD};

/// A deterministic automaton over bytes for one grammar.
///
/// `transitions()[state * 256 + byte]` is the next state or `NO_TRANSITION`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteDfa {
    transitions: Vec<u32>,
    accepting: Vec<bool>,
}

impl ByteDfa {
    /// Table entry for a byte the grammar forbids in that state.
    pub const NO_TRANSITION: u32 = u32::MAX;

    /// Number of states. It is at least 1.
    pub fn num_states(&self) -> usize {
        self.accepting.len()
    }

    /// The start state. It is always 0.
    pub fn start(&self) -> u32 {
        0
    }

    /// True when `state` exists and accepts.
    pub fn is_accepting(&self, state: u32) -> bool {
        self.accepting.get(state as usize).copied().unwrap_or(false)
    }

    /// The state after `byte` from `state`, or `None` when the grammar forbids it.
    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        let at = (state as usize)
            .checked_mul(256)?
            .checked_add(usize::from(byte))?;
        self.transitions
            .get(at)
            .copied()
            .filter(|&next| next != Self::NO_TRANSITION)
    }

    /// The state after every byte of `bytes` from `state`, or `None` at the first forbidden byte.
    pub fn walk(&self, state: u32, bytes: &[u8]) -> Option<u32> {
        if !self.is_viable(state) {
            return None;
        }
        bytes
            .iter()
            .try_fold(state, |at, &byte| self.step(at, byte))
    }

    /// True when the grammar matches exactly `bytes`.
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        self.walk(self.start(), bytes)
            .is_some_and(|state| self.is_accepting(state))
    }

    /// True when `state` exists. Every state reaches an accepting state.
    pub fn is_viable(&self, state: u32) -> bool {
        (state as usize) < self.num_states()
    }

    /// The dense table: 256 entries per state, `NO_TRANSITION` for a forbidden byte.
    pub fn transitions(&self) -> &[u32] {
        &self.transitions
    }

    /// The accepting flag of each state.
    pub fn accepting(&self) -> &[bool] {
        &self.accepting
    }

    /// Trim `dfa`, number its states from 0 in breadth-first order, and expand its classes.
    ///
    /// A start state that reaches no accepting state is an `EmptyLanguage` error.
    pub(super) fn from_class_dfa(dfa: &ClassDfa) -> Result<ByteDfa, GrammarError> {
        let viable = viable_states(dfa);
        let live = |s: u32| s != DEAD && viable.get(s as usize).copied().unwrap_or(false);
        if !live(dfa.start) {
            return Err(GrammarError::new(
                GrammarErrorKind::EmptyLanguage,
                None,
                "grammar matches no string. \
                 Remove the class or alternative that matches nothing.",
            ));
        }
        let k = dfa.num_classes;
        let mut new_id = vec![DEAD; dfa.num_states()];
        let mut order = Vec::new();
        let mut queue = VecDeque::from([dfa.start]);
        if let Some(slot) = new_id.get_mut(dfa.start as usize) {
            *slot = 0;
        }
        while let Some(s) = queue.pop_front() {
            order.push(s);
            for c in 0..k {
                let t = dfa.next(s, c);
                if !live(t) {
                    continue;
                }
                if let Some(slot) = new_id.get_mut(t as usize)
                    && *slot == DEAD
                {
                    *slot = u32::try_from(order.len().saturating_add(queue.len())).unwrap_or(DEAD);
                    queue.push_back(t);
                }
            }
        }
        let mut transitions = Vec::with_capacity(order.len().saturating_mul(256));
        let mut accepting = Vec::with_capacity(order.len());
        for &s in &order {
            transitions.extend(dfa.classes.iter().map(|&class| {
                let t = dfa.next(s, usize::from(class));
                if live(t) {
                    new_id.get(t as usize).copied().unwrap_or(DEAD)
                } else {
                    Self::NO_TRANSITION
                }
            }));
            accepting.push(dfa.accepting.get(s as usize).copied().unwrap_or(false));
        }
        Ok(ByteDfa {
            transitions,
            accepting,
        })
    }
}

/// True for each state of `dfa` that reaches an accepting state.
fn viable_states(dfa: &ClassDfa) -> Vec<bool> {
    let n = dfa.num_states();
    // Reverse edges in compressed rows, one row per target.
    let mut offsets = vec![0usize; n.saturating_add(1)];
    for (_, t) in edges(dfa) {
        if let Some(slot) = offsets.get_mut((t as usize).saturating_add(1)) {
            *slot = slot.saturating_add(1);
        }
    }
    for i in 1..offsets.len() {
        let prev = offsets.get(i - 1).copied().unwrap_or(0);
        if let Some(slot) = offsets.get_mut(i) {
            *slot = slot.saturating_add(prev);
        }
    }
    let mut fill = offsets.clone();
    let mut sources = vec![0u32; offsets.last().copied().unwrap_or(0)];
    for (s, t) in edges(dfa) {
        if let Some(at) = fill.get_mut(t as usize) {
            if let Some(slot) = sources.get_mut(*at) {
                *slot = s;
            }
            *at = at.saturating_add(1);
        }
    }
    let mut viable = dfa.accepting.clone();
    let mut stack: Vec<u32> = (0..u32::try_from(n).unwrap_or(0))
        .filter(|&s| dfa.accepting.get(s as usize).copied().unwrap_or(false))
        .collect();
    while let Some(t) = stack.pop() {
        let lo = offsets.get(t as usize).copied().unwrap_or(0);
        let hi = offsets
            .get((t as usize).saturating_add(1))
            .copied()
            .unwrap_or(0);
        for &s in sources.get(lo..hi).unwrap_or_default() {
            if let Some(flag) = viable.get_mut(s as usize)
                && !*flag
            {
                *flag = true;
                stack.push(s);
            }
        }
    }
    viable
}

/// Every `(source, target)` edge of `dfa` that enters a state.
fn edges(dfa: &ClassDfa) -> impl Iterator<Item = (u32, u32)> + '_ {
    let n = dfa.num_states();
    let k = dfa.num_classes;
    (0..u32::try_from(n).unwrap_or(0))
        .flat_map(move |s| (0..k).map(move |c| (s, dfa.next(s, c))))
        .filter(move |&(_, t)| (t as usize) < n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A class DFA over one class per byte: `edges` are `(from, byte, to)`.
    fn class_dfa(n: usize, edges: &[(u32, u8, u32)], accepting: &[u32], start: u32) -> ClassDfa {
        let mut classes = [0u8; 256];
        for (b, slot) in classes.iter_mut().enumerate() {
            *slot = u8::try_from(b).unwrap_or(0);
        }
        let mut trans = vec![DEAD; n * 256];
        for &(from, byte, to) in edges {
            trans[from as usize * 256 + usize::from(byte)] = to;
        }
        ClassDfa {
            classes,
            num_classes: 256,
            trans,
            accepting: (0..n as u32).map(|s| accepting.contains(&s)).collect(),
            start,
        }
    }

    #[test]
    fn trimming_drops_dead_ends_and_numbers_from_the_start() {
        // 2 is the start. 0 is a dead end. 3 is unreachable.
        let raw = class_dfa(4, &[(2, b'a', 1), (2, b'b', 0), (3, b'c', 1)], &[1], 2);
        let dfa = ByteDfa::from_class_dfa(&raw).ok();
        let dfa = dfa.as_ref();
        assert_eq!(dfa.map(ByteDfa::num_states), Some(2));
        assert_eq!(dfa.and_then(|d| d.step(0, b'a')), Some(1));
        assert_eq!(dfa.and_then(|d| d.step(0, b'b')), None);
        assert_eq!(dfa.map(|d| d.accepts(b"a")), Some(true));
        assert_eq!(dfa.map(|d| d.accepts(b"b")), Some(false));
        assert_eq!(dfa.map(|d| d.transitions().len()), Some(512));
        assert_eq!(dfa.map(|d| d.accepting().to_vec()), Some(vec![false, true]));
        assert_eq!(dfa.map(|d| d.is_viable(1)), Some(true));
        assert_eq!(dfa.map(|d| d.is_viable(2)), Some(false));
        assert_eq!(dfa.and_then(|d| d.walk(2, b"")), None);
        assert_eq!(dfa.and_then(|d| d.step(7, b'a')), None);
        assert_eq!(dfa.map(|d| d.is_accepting(9)), Some(false));
    }

    #[test]
    fn a_start_without_an_accepting_path_is_an_empty_language() {
        let raw = class_dfa(2, &[(0, b'a', 1)], &[], 0);
        let kind = ByteDfa::from_class_dfa(&raw).err().map(|e| e.kind());
        assert_eq!(kind, Some(GrammarErrorKind::EmptyLanguage));
        let none = class_dfa(0, &[], &[], DEAD);
        let err = ByteDfa::from_class_dfa(&none).err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::EmptyLanguage)
        );
        let msg = err.map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.starts_with("grammar error: grammar matches no string."),
            "{msg}"
        );
    }

    #[test]
    fn breadth_first_numbering_is_stable() {
        let raw = class_dfa(
            3,
            &[(0, b'b', 2), (0, b'a', 1), (1, b'x', 2), (2, b'y', 2)],
            &[2],
            0,
        );
        let dfa = ByteDfa::from_class_dfa(&raw).ok();
        // Byte order visits `a` before `b`, so old state 1 stays 1.
        assert_eq!(dfa.as_ref().and_then(|d| d.walk(0, b"a")), Some(1));
        assert_eq!(dfa.as_ref().and_then(|d| d.walk(0, b"b")), Some(2));
        assert_eq!(dfa.as_ref().and_then(|d| d.walk(0, b"axyy")), Some(2));
        assert_eq!(dfa.map(|d| d.accepts(b"byyy")), Some(true));
    }
}
