//! GBNF grammar front end and byte-level automaton compiler.
//!
//! `Grammar::parse` reads GBNF text. `Grammar::expand` inlines every rule
//! reference and returns the `root` expression. `Grammar::compile` lowers
//! that expression to a minimal `ByteDfa`.

pub mod ast;
pub mod automaton_limits;
mod compile;
mod dfa;
pub mod error;
mod expand;
mod lexer;
pub mod limits;
mod literal;
mod minimize;
mod nfa;
mod parser;
mod partition;
#[cfg(test)]
mod reference;
mod repeat;
mod subset;
mod utf8;

pub use ast::{Expr, Grammar, ROOT_RULE, Reference, Rule, Span};
pub use automaton_limits::{AutomatonLimits, CompileLimits};
pub use compile::compile;
pub use dfa::ByteDfa;
pub use error::{GrammarError, GrammarErrorKind};
pub use limits::ExpandLimits;
