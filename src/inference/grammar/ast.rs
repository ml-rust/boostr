//! GBNF syntax tree over Unicode scalar values.
//!
//! The `Expr` constructors simplify trivial structure. They never change the
//! language an expression matches.

use std::fmt;

/// Name of the rule a grammar starts from.
pub const ROOT_RULE: &str = "root";

/// A 1-based source position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    /// Line number, from 1.
    pub line: u32,
    /// Column number in code points, from 1.
    pub column: u32,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A parsed GBNF grammar. Rules keep their source order.
#[derive(Clone, Debug, PartialEq)]
pub struct Grammar {
    /// Every rule definition, in source order.
    pub rules: Vec<Rule>,
}

/// One `name ::= alternatives` definition.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    /// Rule name.
    pub name: String,
    /// Rule body.
    pub expr: Expr,
    /// Position of the rule name.
    pub span: Span,
    /// Every rule reference in the body, in source order.
    pub references: Vec<Reference>,
}

/// A rule reference site inside a rule body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference {
    /// Name of the referenced rule.
    pub name: String,
    /// Position of the reference.
    pub span: Span,
}

/// A GBNF expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// Matches the empty string.
    Empty,
    /// Matches this exact non-empty text.
    Literal(String),
    /// Matches one code point inside (or, if `negated`, outside) the inclusive ranges.
    Class {
        /// True for `[^...]`.
        negated: bool,
        /// Inclusive code point ranges, `lo <= hi`, in source order.
        ranges: Vec<(char, char)>,
    },
    /// Matches any one Unicode scalar value.
    Any,
    /// Matches the named rule.
    Ref(String),
    /// Matches each item in order. An empty `Seq` matches the empty string.
    Seq(Vec<Expr>),
    /// Matches any one item. An empty `Alt` matches nothing.
    Alt(Vec<Expr>),
    /// Matches `expr` from `min` to `max` times. `None` means no upper bound.
    Repeat {
        /// Repeated expression.
        expr: Box<Expr>,
        /// Minimum count.
        min: u32,
        /// Maximum count, or `None` for no bound.
        max: Option<u32>,
    },
}

impl Grammar {
    /// The rule named `name`, if defined.
    pub fn rule(&self, name: &str) -> Option<&Rule> {
        self.rules.iter().find(|rule| rule.name == name)
    }

    /// The `root` rule, if defined.
    pub fn root(&self) -> Option<&Rule> {
        self.rule(ROOT_RULE)
    }
}

impl Expr {
    /// A literal. The empty string gives `Empty`.
    pub fn literal(text: impl Into<String>) -> Expr {
        let text = text.into();
        if text.is_empty() {
            Expr::Empty
        } else {
            Expr::Literal(text)
        }
    }

    /// A sequence. Nested sequences flatten and `Empty` items drop.
    /// No items give `Empty`. One item gives that item.
    pub fn seq(items: Vec<Expr>) -> Expr {
        let mut flat = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Expr::Empty => {}
                Expr::Seq(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        match flat.len() {
            0 => Expr::Empty,
            1 => flat.pop().unwrap_or(Expr::Empty),
            _ => Expr::Seq(flat),
        }
    }

    /// An alternation. Nested alternations flatten. One item gives that item.
    pub fn alt(items: Vec<Expr>) -> Expr {
        let mut flat = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Expr::Alt(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        if flat.len() == 1 {
            flat.pop().unwrap_or(Expr::Alt(Vec::new()))
        } else {
            Expr::Alt(flat)
        }
    }

    /// A repetition. `{1,1}` gives `expr`. `{0,0}` and a repeated `Empty` give `Empty`.
    pub fn repeat(expr: Expr, min: u32, max: Option<u32>) -> Expr {
        match (min, max) {
            (_, Some(0)) => Expr::Empty,
            _ if expr == Expr::Empty => Expr::Empty,
            (1, Some(1)) => expr,
            _ => Expr::Repeat {
                expr: Box::new(expr),
                min,
                max,
            },
        }
    }

    /// Weight of this node alone, without its children.
    ///
    /// A node weighs 1. A literal adds 1 per started 16 bytes of text.
    /// A class adds 1 per range. One unit stands for at most 16 heap bytes
    /// beside the node, so a weighted count bounds the memory a tree holds.
    pub fn weight(&self) -> usize {
        let payload = match self {
            Expr::Literal(text) => text.len().div_ceil(LITERAL_BYTES_PER_UNIT),
            Expr::Class { ranges, .. } => ranges.len(),
            _ => 0,
        };
        payload.saturating_add(1)
    }

    /// Weighted node count of this expression tree, this node included.
    ///
    /// Each node adds its `weight`, so a large literal or class counts by size.
    pub fn node_count(&self) -> usize {
        let children = match self {
            Expr::Empty | Expr::Literal(_) | Expr::Class { .. } | Expr::Any | Expr::Ref(_) => 0,
            Expr::Seq(items) | Expr::Alt(items) => items
                .iter()
                .fold(0usize, |sum, item| sum.saturating_add(item.node_count())),
            Expr::Repeat { expr, .. } => expr.node_count(),
        };
        children.saturating_add(self.weight())
    }
}

/// Literal bytes that one weight unit covers.
const LITERAL_BYTES_PER_UNIT: usize = 16;

/// The source form of a repeat count: `{n}`, `{n,}` or `{n,m}`.
pub(crate) fn repeat_text(min: u32, max: Option<u32>) -> String {
    match max {
        Some(max) if max == min => format!("{{{min}}}"),
        Some(max) => format!("{{{min},{max}}}"),
        None => format!("{{{min},}}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(s: &str) -> Expr {
        Expr::Literal(s.to_string())
    }

    #[test]
    fn literal_of_empty_text_is_empty() {
        assert_eq!(Expr::literal(""), Expr::Empty);
        assert_eq!(Expr::literal("a"), lit("a"));
    }

    #[test]
    fn seq_flattens_and_drops_empty() {
        let inner = Expr::Seq(vec![lit("b"), lit("c")]);
        let got = Expr::seq(vec![lit("a"), Expr::Empty, inner]);
        assert_eq!(got, Expr::Seq(vec![lit("a"), lit("b"), lit("c")]));
        assert_eq!(Expr::seq(vec![Expr::Empty, lit("a")]), lit("a"));
        assert_eq!(Expr::seq(vec![]), Expr::Empty);
    }

    #[test]
    fn alt_flattens_and_keeps_empty() {
        let inner = Expr::Alt(vec![lit("b"), lit("c")]);
        let got = Expr::alt(vec![lit("a"), inner, Expr::Empty]);
        assert_eq!(
            got,
            Expr::Alt(vec![lit("a"), lit("b"), lit("c"), Expr::Empty])
        );
        assert_eq!(Expr::alt(vec![lit("a")]), lit("a"));
    }

    #[test]
    fn repeat_simplifies_only_trivial_counts() {
        assert_eq!(Expr::repeat(lit("a"), 1, Some(1)), lit("a"));
        assert_eq!(Expr::repeat(lit("a"), 0, Some(0)), Expr::Empty);
        assert_eq!(Expr::repeat(Expr::Empty, 2, None), Expr::Empty);
        assert_eq!(
            Expr::repeat(lit("a"), 0, Some(1)),
            Expr::Repeat {
                expr: Box::new(lit("a")),
                min: 0,
                max: Some(1)
            }
        );
    }

    #[test]
    fn node_count_counts_every_node() {
        let e = Expr::Seq(vec![lit("a"), Expr::repeat(Expr::Any, 0, None)]);
        assert_eq!(e.node_count(), 5);
    }

    #[test]
    fn weight_grows_with_payload() {
        assert_eq!(Expr::Any.weight(), 1);
        assert_eq!(Expr::Empty.weight(), 1);
        assert_eq!(lit("a").weight(), 2);
        assert_eq!(lit(&"x".repeat(16)).weight(), 2);
        assert_eq!(lit(&"x".repeat(17)).weight(), 3);
        assert_eq!(lit("é").weight(), 2);
        assert_eq!(lit(&"x".repeat(100_000)).weight(), 6_251);
        let class = |n: u32| Expr::Class {
            negated: false,
            ranges: (0..n).map(|_| ('a', 'z')).collect(),
        };
        assert_eq!(class(0).weight(), 1);
        assert_eq!(class(3).weight(), 4);
        let rep = Expr::repeat(class(3), 0, None);
        assert_eq!(rep.weight(), 1);
        assert_eq!(rep.node_count(), 5);
    }

    #[test]
    fn repeat_text_matches_source_forms() {
        assert_eq!(repeat_text(2, Some(2)), "{2}");
        assert_eq!(repeat_text(2, None), "{2,}");
        assert_eq!(repeat_text(2, Some(4)), "{2,4}");
    }

    #[test]
    fn span_displays_line_and_column() {
        assert_eq!(Span { line: 3, column: 7 }.to_string(), "3:7");
    }
}
