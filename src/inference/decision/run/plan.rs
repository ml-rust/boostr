//! The host description of one decision request: a shared prefix, one
//! suffix per row, and the positions whose hidden rows are read.

use crate::error::{Error, Result};

/// The input every row continues.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrefixInput {
    /// Token ids. An empty list is no prefix.
    Tokens(Vec<u32>),
}

impl PrefixInput {
    /// The prefix token ids.
    pub fn tokens(&self) -> &[u32] {
        match self {
            Self::Tokens(t) => t,
        }
    }

    /// The prefix length in positions.
    pub fn len(&self) -> usize {
        self.tokens().len()
    }

    /// `true` when there is no prefix.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One hidden row to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gather {
    /// The row the position belongs to. `None` reads the prefix.
    pub row: Option<usize>,
    /// The position in the concatenated sequence `prefix + row`. A prefix
    /// gather needs `pos < prefix_len`. A row gather needs `prefix_len <=
    /// pos < prefix_len + row_len`: a row reads only its own tokens.
    pub pos: usize,
}

/// One decision request: the prefix, every row's suffix tokens, and the
/// gathers in the order the head reads them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionPlan {
    /// The input every row continues.
    pub prefix: PrefixInput,
    /// The suffix tokens of each row. Every row is non-empty.
    pub rows: Vec<Vec<u32>>,
    /// The hidden rows to read, in the order the head receives them.
    pub gathers: Vec<Gather>,
}

fn invalid(reason: String) -> Error {
    Error::InvalidArgument {
        arg: "plan",
        reason,
    }
}

impl DecisionPlan {
    /// The prefix length in positions.
    pub fn prefix_len(&self) -> usize {
        self.prefix.len()
    }

    /// The positions of every prefix gather, sorted, without repeats.
    pub fn prefix_positions(&self) -> Vec<usize> {
        let mut pos: Vec<usize> = self
            .gathers
            .iter()
            .filter(|g| g.row.is_none())
            .map(|g| g.pos)
            .collect();
        pos.sort_unstable();
        pos.dedup();
        pos
    }

    /// Checks the plan: at least one gather, no empty row, and every
    /// gather inside the prefix or inside its own row's real tokens.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] naming the first gather or row that fails.
    pub fn validate(&self) -> Result<()> {
        if self.gathers.is_empty() {
            return Err(invalid("no gathers: a plan reads at least one row".into()));
        }
        if let Some(r) = self.rows.iter().position(Vec::is_empty) {
            return Err(invalid(format!("row {r} has no tokens")));
        }
        let p = self.prefix_len();
        for (i, g) in self.gathers.iter().enumerate() {
            match g.row {
                None if g.pos >= p => {
                    return Err(invalid(format!(
                        "gather {i} reads prefix position {} but the prefix has {p} positions",
                        g.pos
                    )));
                }
                None => {}
                Some(r) if r >= self.rows.len() => {
                    return Err(invalid(format!(
                        "gather {i} names row {r} but the plan has {} rows",
                        self.rows.len()
                    )));
                }
                Some(r) => {
                    let end = p + self.rows[r].len();
                    if g.pos < p {
                        return Err(invalid(format!(
                            "gather {i} reads position {} of row {r}, inside the prefix of \
                             {p} positions; read it with row None",
                            g.pos
                        )));
                    }
                    if g.pos >= end {
                        return Err(invalid(format!(
                            "gather {i} reads position {} of row {r}, past its last real \
                             token at {}; that position is padding or missing",
                            g.pos,
                            end - 1
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(prefix: Vec<u32>, rows: Vec<Vec<u32>>, gathers: Vec<Gather>) -> DecisionPlan {
        DecisionPlan {
            prefix: PrefixInput::Tokens(prefix),
            rows,
            gathers,
        }
    }

    fn at(row: Option<usize>, pos: usize) -> Gather {
        Gather { row, pos }
    }

    fn err(p: &DecisionPlan) -> String {
        p.validate().unwrap_err().to_string()
    }

    #[test]
    fn valid_plans_pass() {
        let p = plan(
            vec![1, 2, 3],
            vec![vec![4], vec![5, 6]],
            vec![at(Some(0), 3), at(Some(1), 4), at(None, 0), at(None, 2)],
        );
        p.validate().unwrap();
        assert_eq!(p.prefix_len(), 3);
        let q = plan(
            vec![],
            vec![vec![1, 2]],
            vec![at(Some(0), 0), at(Some(0), 1)],
        );
        q.validate().unwrap();
        let r = plan(
            vec![1, 2],
            vec![],
            vec![at(None, 1), at(None, 0), at(None, 1)],
        );
        r.validate().unwrap();
        assert_eq!(r.prefix_positions(), vec![0, 1]);
    }

    #[test]
    fn pad_position_is_refused_with_its_value() {
        let p = plan(vec![1, 2], vec![vec![3, 4], vec![5]], vec![at(Some(1), 3)]);
        let e = err(&p);
        assert!(e.contains("gather 0 reads position 3 of row 1"), "{e}");
        assert!(e.contains("last real token at 2"), "{e}");
    }

    #[test]
    fn out_of_range_and_empty_are_refused() {
        let e = err(&plan(vec![1], vec![vec![2]], vec![at(Some(4), 1)]));
        assert!(e.contains("names row 4 but the plan has 1 rows"), "{e}");
        let e = err(&plan(vec![1], vec![vec![2]], vec![at(None, 1)]));
        assert!(e.contains("prefix position 1 but the prefix has 1"), "{e}");
        let e = err(&plan(vec![1, 2], vec![vec![2]], vec![at(Some(0), 1)]));
        assert!(e.contains("inside the prefix of 2 positions"), "{e}");
        let e = err(&plan(vec![1], vec![vec![2], vec![]], vec![at(Some(0), 1)]));
        assert!(e.contains("row 1 has no tokens"), "{e}");
        let e = err(&plan(vec![1], vec![vec![2]], vec![]));
        assert!(e.contains("no gathers"), "{e}");
    }
}
