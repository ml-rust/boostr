//! Block partition of DFA states for Hopcroft refinement.

use super::error::GrammarError;

/// A closure that builds the internal error for `what`.
pub(super) fn bad(what: &'static str) -> impl FnOnce() -> GrammarError {
    move || GrammarError::internal(what)
}

/// The `u32` form of `n`, or an internal error.
pub(super) fn index(n: usize) -> Result<u32, GrammarError> {
    u32::try_from(n).map_err(|_| GrammarError::internal("count above u32"))
}

/// A partition of states into blocks, each a contiguous range of `elems`.
///
/// `take` and `put` index by a state below the checked state count or by a
/// block below `count`, so every index is in range.
pub(super) struct Partition {
    elems: Vec<u32>,
    pos: Vec<u32>,
    block: Vec<u32>,
    start: Vec<u32>,
    end: Vec<u32>,
    /// Marked states of each block, held at the front of its range.
    marked: Vec<u32>,
    /// Block count.
    pub count: u32,
}

impl Partition {
    /// Two initial blocks: accepting states, then the rest. An empty block is skipped.
    pub(super) fn new(accepting: &[bool]) -> Result<Self, GrammarError> {
        let total = accepting.len();
        let (yes, no): (Vec<u32>, Vec<u32>) =
            (0..index(total)?).partition(|&s| accepting.get(s as usize).copied().unwrap_or(false));
        let mut part = Self {
            elems: Vec::with_capacity(total),
            pos: vec![0; total],
            block: vec![0; total],
            start: Vec::new(),
            end: Vec::new(),
            marked: Vec::new(),
            count: 0,
        };
        for group in [yes, no] {
            if group.is_empty() {
                continue;
            }
            let id = part.count;
            let lo = index(part.elems.len())?;
            for s in group {
                put(&mut part.pos, s, index(part.elems.len())?);
                put(&mut part.block, s, id);
                part.elems.push(s);
            }
            part.start.push(lo);
            part.end.push(index(part.elems.len())?);
            part.marked.push(0);
            part.count = id.checked_add(1).ok_or_else(bad("block count above u32"))?;
        }
        Ok(part)
    }

    pub(super) fn block_of(&self, s: u32) -> u32 {
        take(&self.block, s)
    }

    pub(super) fn size(&self, b: u32) -> u32 {
        take(&self.end, b).saturating_sub(take(&self.start, b))
    }

    pub(super) fn members(&self, b: u32) -> &[u32] {
        let range = take(&self.start, b) as usize..take(&self.end, b) as usize;
        self.elems.get(range).unwrap_or_default()
    }

    /// Mark `s`. Returns its block when `s` is the first mark there.
    pub(super) fn mark(&mut self, s: u32) -> Option<u32> {
        let b = take(&self.block, s);
        let marked = take(&self.marked, b);
        let front = take(&self.start, b).saturating_add(marked);
        let at = take(&self.pos, s);
        if at < front {
            return None;
        }
        let other = take(&self.elems, front);
        put(&mut self.elems, front, s);
        put(&mut self.elems, at, other);
        put(&mut self.pos, s, front);
        put(&mut self.pos, other, at);
        put(&mut self.marked, b, marked.saturating_add(1));
        (marked == 0).then_some(b)
    }

    /// Split the marked states of `b` into a new block. Clears the marks.
    pub(super) fn split(&mut self, b: u32) -> Result<Option<u32>, GrammarError> {
        let marked = take(&self.marked, b);
        put(&mut self.marked, b, 0);
        if marked == 0 || marked == self.size(b) {
            return Ok(None);
        }
        let nb = self.count;
        self.count = nb.checked_add(1).ok_or_else(bad("block count above u32"))?;
        let lo = take(&self.start, b);
        let hi = lo.saturating_add(marked);
        self.start.push(lo);
        self.end.push(hi);
        self.marked.push(0);
        put(&mut self.start, b, hi);
        for i in lo..hi {
            let s = take(&self.elems, i);
            put(&mut self.block, s, nb);
        }
        Ok(Some(nb))
    }
}

fn take(v: &[u32], i: u32) -> u32 {
    v.get(i as usize).copied().unwrap_or(0)
}

fn put(v: &mut [u32], i: u32, x: u32) {
    if let Some(slot) = v.get_mut(i as usize) {
        *slot = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn partition(accepting: &[bool]) -> Partition {
        match Partition::new(accepting) {
            Ok(p) => p,
            Err(err) => panic!("{err}"),
        }
    }

    #[test]
    fn empty_initial_blocks_are_skipped() {
        assert_eq!(partition(&[false, false]).count, 1);
        assert_eq!(partition(&[true, false, true]).count, 2);
    }

    #[test]
    fn split_moves_the_marked_states_to_a_new_block() {
        let mut part = partition(&[false, false, false]);
        assert_eq!(part.mark(1), Some(0));
        assert_eq!(part.mark(2), None);
        assert_eq!(part.split(0).ok().flatten(), Some(1));
        assert_eq!(part.count, 2);
        assert_eq!(part.block_of(1), 1);
        assert_eq!(part.block_of(0), 0);
        assert_eq!(part.size(1), 2);
        // A block with every state marked does not split.
        part.mark(1);
        part.mark(2);
        assert_eq!(part.split(1).ok().flatten(), None);
    }
}
