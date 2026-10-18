//! Hopcroft minimization of a partial class DFA.
//!
//! The minimizer adds the dead state as one explicit sink, so the DFA is
//! complete. Hopcroft refinement then splits blocks by inverse transitions.
//! Every state that reaches no accepting state lands in the block of the
//! sink. The output drops that block, so a transition into it is `DEAD`.
//!
//! All indexes are `u32`. `AutomatonLimits::check` keeps `(states + 1) * classes`
//! within `u32`. Every step here checks its arithmetic again and returns an error.

use super::error::GrammarError;
use super::partition::{Partition, bad, index};
use super::subset::{ClassDfa, DEAD};

/// The minimal DFA for the language of `dfa`.
///
/// The output has no state that reaches no accepting state.
/// An empty language gives zero states and a `DEAD` start.
pub(super) fn minimize(dfa: &ClassDfa) -> Result<ClassDfa, GrammarError> {
    let k = dfa.num_classes;
    let n = dfa.num_states();
    let empty = ClassDfa {
        classes: dfa.classes,
        num_classes: k,
        trans: Vec::new(),
        accepting: Vec::new(),
        start: DEAD,
    };
    if dfa.start == DEAD {
        return Ok(empty);
    }
    if dfa.start as usize >= n {
        return Err(GrammarError::internal("start state is out of range"));
    }
    let sink = index(n)?;
    let total = n.checked_add(1).ok_or_else(bad("state count overflows"))?;
    // The successor of `s` on `c` in the complete DFA.
    let succ = |s: u32, c: usize| match dfa.next(s, c) {
        DEAD => sink,
        t => t,
    };
    let inverse = Inverse::new(total, k, &succ)?;
    // The sink is the last state and does not accept.
    let mut flags = dfa.accepting.clone();
    flags.push(false);
    let mut part = Partition::new(&flags)?;
    let mut work: Vec<u32> = (0..part.count).collect();
    let mut in_work = vec![true; work.len()];
    let mut members = Vec::new();
    let mut touched = Vec::new();
    while let Some(a) = work.pop() {
        set_flag(&mut in_work, a, false);
        members.clear();
        members.extend_from_slice(part.members(a));
        for c in 0..k {
            touched.clear();
            for &t in &members {
                for &s in inverse.sources(c, t)? {
                    if let Some(b) = part.mark(s) {
                        touched.push(b);
                    }
                }
            }
            for &b in &touched {
                let Some(nb) = part.split(b)? else { continue };
                in_work.push(false);
                // A queued block keeps its place, so both halves get queued.
                // Otherwise the smaller half is enough.
                let queued = in_work.get(b as usize).copied().unwrap_or(false);
                let pick = if queued || part.size(nb) <= part.size(b) {
                    nb
                } else {
                    b
                };
                set_flag(&mut in_work, pick, true);
                work.push(pick);
            }
        }
    }
    quotient(dfa, &part, sink, &succ)
}

/// The DFA over blocks, without the block of `sink`.
fn quotient(
    dfa: &ClassDfa,
    part: &Partition,
    sink: u32,
    succ: &impl Fn(u32, usize) -> u32,
) -> Result<ClassDfa, GrammarError> {
    let k = dfa.num_classes;
    let dead_block = part.block_of(sink);
    let mut new_id = vec![DEAD; part.count as usize];
    let mut reps = Vec::new();
    for b in 0..part.count {
        if b == dead_block {
            continue;
        }
        if let (Some(&rep), Some(slot)) = (part.members(b).first(), new_id.get_mut(b as usize)) {
            *slot = index(reps.len())?;
            reps.push(rep);
        }
    }
    let id_of = |s: u32| {
        new_id
            .get(part.block_of(s) as usize)
            .copied()
            .ok_or_else(bad("block is out of range"))
    };
    let cells = reps
        .len()
        .checked_mul(k)
        .ok_or_else(bad("table size overflows"))?;
    let mut trans = Vec::with_capacity(cells);
    let mut accepting = Vec::with_capacity(reps.len());
    for &rep in &reps {
        for c in 0..k {
            trans.push(id_of(succ(rep, c))?);
        }
        let flag = dfa.accepting.get(rep as usize).copied();
        accepting.push(flag.ok_or_else(bad("representative is out of range"))?);
    }
    Ok(ClassDfa {
        classes: dfa.classes,
        num_classes: k,
        trans,
        accepting,
        start: id_of(dfa.start)?,
    })
}

fn set_flag(flags: &mut [bool], i: u32, value: bool) {
    if let Some(flag) = flags.get_mut(i as usize) {
        *flag = value;
    }
}

/// Inverse transitions in compressed rows: the sources of each `(class, target)`.
struct Inverse {
    total: usize,
    offsets: Vec<u32>,
    sources: Vec<u32>,
}

impl Inverse {
    /// The row of `(c, t)`: `c * total + t`. A target at or above `total` is an error.
    fn row(total: usize, c: usize, t: u32) -> Result<usize, GrammarError> {
        let t = t as usize;
        c.checked_mul(total)
            .and_then(|base| base.checked_add(t))
            .filter(|_| t < total)
            .ok_or_else(bad("transition target is out of range"))
    }

    fn new(
        total: usize,
        k: usize,
        succ: &impl Fn(u32, usize) -> u32,
    ) -> Result<Self, GrammarError> {
        let rows = total
            .checked_mul(k)
            .and_then(|rows| rows.checked_add(1))
            .ok_or_else(bad("row count overflows"))?;
        let states = 0..index(total)?;
        let mut offsets = vec![0u32; rows];
        for s in states.clone() {
            for c in 0..k {
                let at = Self::row(total, c, succ(s, c))?.saturating_add(1);
                let slot = offsets.get_mut(at).ok_or_else(bad("row is out of range"))?;
                *slot = slot.checked_add(1).ok_or_else(bad("offset above u32"))?;
            }
        }
        let mut sum = 0u32;
        for slot in &mut offsets {
            sum = sum.checked_add(*slot).ok_or_else(bad("offset above u32"))?;
            *slot = sum;
        }
        let mut fill: Vec<u32> = offsets.clone();
        let mut sources = vec![0u32; sum as usize];
        for s in states {
            for c in 0..k {
                let row = Self::row(total, c, succ(s, c))?;
                let at = fill.get_mut(row).ok_or_else(bad("row is out of range"))?;
                let slot = sources
                    .get_mut(*at as usize)
                    .ok_or_else(bad("offset is out of range"))?;
                *slot = s;
                *at = at.checked_add(1).ok_or_else(bad("offset above u32"))?;
            }
        }
        Ok(Self {
            total,
            offsets,
            sources,
        })
    }

    fn sources(&self, c: usize, t: u32) -> Result<&[u32], GrammarError> {
        let row = Self::row(self.total, c, t)?;
        let end = row.checked_add(1).ok_or_else(bad("row overflows"))?;
        let (Some(&lo), Some(&hi)) = (self.offsets.get(row), self.offsets.get(end)) else {
            return Err(GrammarError::internal("row is out of range"));
        };
        self.sources
            .get(lo as usize..hi as usize)
            .ok_or_else(bad("offset is out of range"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::ast::Expr;
    use crate::inference::grammar::automaton_limits::AutomatonLimits;
    use crate::inference::grammar::nfa::Nfa;
    use crate::inference::grammar::subset::determinize;

    fn class_dfa(expr: &Expr) -> ClassDfa {
        let limits = AutomatonLimits::default();
        match Nfa::build(expr, &limits).and_then(|n| determinize(&n, &limits)) {
            Ok(d) => d,
            Err(err) => panic!("{expr:?}: {err}"),
        }
    }

    fn min(dfa: &ClassDfa) -> ClassDfa {
        match minimize(dfa) {
            Ok(d) => d,
            Err(err) => panic!("{err}"),
        }
    }

    fn lit(s: &str) -> Expr {
        Expr::Literal(s.into())
    }

    fn star(e: Expr) -> Expr {
        Expr::Repeat {
            expr: Box::new(e),
            min: 0,
            max: None,
        }
    }

    #[test]
    fn equal_suffixes_merge() {
        let raw = class_dfa(&Expr::Alt(vec![lit("ax"), lit("bx")]));
        assert_eq!(raw.num_states(), 4);
        assert_eq!(min(&raw).num_states(), 3);
    }

    #[test]
    fn a_duplicate_alternative_minimizes_like_one() {
        let dup = min(&class_dfa(&star(Expr::Alt(vec![lit("a"), lit("a")]))));
        let one = min(&class_dfa(&star(lit("a"))));
        assert_eq!(dup.num_states(), 1);
        assert_eq!(one.num_states(), 1);
    }

    #[test]
    fn dead_ends_drop_and_empty_languages_have_no_state() {
        let d = min(&class_dfa(&Expr::Seq(vec![lit("a"), Expr::Alt(vec![])])));
        assert_eq!(d.num_states(), 0);
        assert_eq!(d.start, DEAD);
        let d = min(&class_dfa(&Expr::Alt(vec![
            lit("ab"),
            Expr::Seq(vec![lit("c"), Expr::Alt(vec![])]),
        ])));
        assert_eq!(d.num_states(), 3);
    }

    #[test]
    fn suffix_window_needs_two_to_the_n_states() {
        let ab = Expr::Class {
            negated: false,
            ranges: vec![('a', 'b')],
        };
        for n in 1..=5u32 {
            let expr = Expr::Seq(vec![
                star(ab.clone()),
                lit("a"),
                Expr::Repeat {
                    expr: Box::new(ab.clone()),
                    min: n,
                    max: Some(n),
                },
            ]);
            assert_eq!(min(&class_dfa(&expr)).num_states(), 1 << (n + 1), "n = {n}");
        }
    }

    fn message(err: Option<GrammarError>) -> String {
        err.map(|e| e.message().to_string()).unwrap_or_default()
    }

    #[test]
    fn a_transition_target_out_of_range_is_an_internal_error() {
        let err = Inverse::new(3, 2, &|_, _| 9).err();
        let msg = message(err);
        assert!(msg.starts_with("internal automaton error: "), "{msg}");
        assert!(msg.contains("transition target is out of range"), "{msg}");
    }

    #[test]
    fn a_start_state_out_of_range_is_an_internal_error() {
        let mut raw = class_dfa(&lit("a"));
        raw.start = 7;
        let msg = message(minimize(&raw).err());
        assert!(msg.starts_with("internal automaton error: "), "{msg}");
    }

    #[test]
    fn inverse_rows_list_the_sources_of_each_target() {
        // Two classes over three states: only class 0 from state 0 reaches state 1.
        let succ = |s: u32, c: usize| if (s, c) == (0, 0) { 1 } else { 2 };
        let inv = match Inverse::new(3, 2, &succ) {
            Ok(inv) => inv,
            Err(err) => panic!("{err}"),
        };
        assert_eq!(inv.sources(0, 1).ok(), Some(&[0u32][..]));
        assert_eq!(inv.sources(0, 2).ok(), Some(&[1u32, 2][..]));
        assert_eq!(inv.sources(1, 0).ok(), Some(&[0u32; 0][..]));
        assert!(inv.sources(0, 3).is_err());
    }
}
