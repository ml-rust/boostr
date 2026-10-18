//! GBNF grammar front end: lexer, parser, syntax tree and rule expansion.
//!
//! `Grammar::parse` reads GBNF text. `Grammar::expand` inlines every rule
//! reference and returns the `root` expression.

pub mod ast;
pub mod error;
mod expand;
mod lexer;
pub mod limits;
mod literal;
mod parser;
mod repeat;

pub use ast::{Expr, Grammar, ROOT_RULE, Reference, Rule, Span};
pub use error::{GrammarError, GrammarErrorKind};
pub use limits::ExpandLimits;
