//! GBNF syntax tree over Unicode scalar values.
//!
//! The `Expr` constructors simplify trivial structure. They never change the
//! language an expression matches.

mod expr;
mod grammar;
mod tree;

pub use expr::Expr;
pub(crate) use expr::repeat_text;
pub use grammar::{Grammar, ROOT_RULE, Reference, Rule, Span};
