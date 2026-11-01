//! `Clone`, `PartialEq`, `Debug` and `Drop` for `Expr` over explicit heap stacks.
//!
//! Derived forms recurse once per tree level, so a deep tree can exhaust the
//! call stack. These walks keep their pending work on the heap at any depth.
//! `Debug` writes the same text as the derived form, in both `{:?}` and `{:#?}`.

use std::fmt::{self, Write as _};
use std::mem;

use super::expr::Expr;

/// One level of `{:#?}` indentation.
const PAD: &str = "    ";

impl Expr {
    /// True for a node with children: `Seq`, `Alt` or `Repeat`.
    fn is_branch(&self) -> bool {
        matches!(self, Expr::Seq(_) | Expr::Alt(_) | Expr::Repeat { .. })
    }

    /// Move every child that has children of its own onto `stack`.
    fn take_branches(&mut self, stack: &mut Vec<Expr>) {
        match self {
            Expr::Seq(items) | Expr::Alt(items) => {
                for item in items.iter_mut().filter(|item| item.is_branch()) {
                    stack.push(mem::replace(item, Expr::Empty));
                }
            }
            Expr::Repeat { expr, .. } if expr.is_branch() => {
                stack.push(mem::replace(&mut **expr, Expr::Empty));
            }
            _ => {}
        }
    }

    /// A copy of a leaf node. A branch gives `Empty`, so callers pass leaves only.
    fn clone_leaf(&self) -> Expr {
        match self {
            Expr::Literal(text) => Expr::Literal(text.clone()),
            Expr::Class { negated, ranges } => Expr::Class {
                negated: *negated,
                ranges: ranges.clone(),
            },
            Expr::Ref(name) => Expr::Ref(name.clone()),
            Expr::Any => Expr::Any,
            Expr::Empty | Expr::Seq(_) | Expr::Alt(_) | Expr::Repeat { .. } => Expr::Empty,
        }
    }

    /// The `Debug` pieces of this node at nesting `indent`, in output order.
    ///
    /// Children stay `Piece::Node` entries, so the caller formats them later.
    fn debug_pieces<'a>(&'a self, indent: usize, pretty: bool, out: &mut Vec<Piece<'a>>) {
        let inner = indent.saturating_add(1);
        match self {
            Expr::Empty => out.push(Piece::Text("Empty")),
            Expr::Any => out.push(Piece::Text("Any")),
            Expr::Literal(text) => {
                out.push(Piece::Text("Literal("));
                let value = Piece::Owned(payload(text, inner, pretty));
                tuple_field(out, value, indent, pretty);
            }
            Expr::Ref(name) => {
                out.push(Piece::Text("Ref("));
                let value = Piece::Owned(payload(name, inner, pretty));
                tuple_field(out, value, indent, pretty);
            }
            Expr::Seq(items) | Expr::Alt(items) => {
                let alt = matches!(self, Expr::Alt(_));
                out.push(Piece::Text(if alt { "Alt(" } else { "Seq(" }));
                if pretty {
                    out.push(Piece::Line(inner));
                }
                list(out, items, inner, pretty);
                if pretty {
                    out.push(Piece::Text(","));
                    out.push(Piece::Line(indent));
                }
                out.push(Piece::Text(")"));
            }
            Expr::Class { negated, ranges } => {
                out.push(Piece::Text("Class {"));
                field(out, "negated: ", true, inner, pretty);
                out.push(Piece::Owned(payload(negated, inner, pretty)));
                field(out, "ranges: ", false, inner, pretty);
                out.push(Piece::Owned(payload(ranges, inner, pretty)));
                struct_end(out, indent, pretty);
            }
            Expr::Repeat { expr, min, max } => {
                out.push(Piece::Text("Repeat {"));
                field(out, "expr: ", true, inner, pretty);
                out.push(Piece::Node(expr, inner));
                field(out, "min: ", false, inner, pretty);
                out.push(Piece::Owned(payload(min, inner, pretty)));
                field(out, "max: ", false, inner, pretty);
                out.push(Piece::Owned(payload(max, inner, pretty)));
                struct_end(out, indent, pretty);
            }
        }
    }
}

impl Drop for Expr {
    /// Each node on the stack loses its branch children before it drops.
    /// Its own drop then frees leaves only, so no drop nests.
    fn drop(&mut self) {
        let mut stack = Vec::new();
        self.take_branches(&mut stack);
        while let Some(mut node) = stack.pop() {
            node.take_branches(&mut stack);
        }
    }
}

/// One pending step of an `Expr` clone.
enum CloneStep<'a> {
    /// Copy this node. A leaf leaves its copy on the value stack at once.
    Node(&'a Expr),
    /// Join the last `count` values into an `Alt`, or a `Seq` when `alt` is false.
    Items { alt: bool, count: usize },
    /// Wrap the last value in a repeat.
    Repeat { min: u32, max: Option<u32> },
}

impl Clone for Expr {
    /// The copy keeps the exact tree shape. It never flattens or simplifies.
    fn clone(&self) -> Self {
        if !self.is_branch() {
            return self.clone_leaf();
        }
        let mut steps = vec![CloneStep::Node(self)];
        let mut values: Vec<Expr> = Vec::new();
        while let Some(step) = steps.pop() {
            match step {
                CloneStep::Node(node) => match node {
                    Expr::Seq(items) | Expr::Alt(items) => {
                        steps.push(CloneStep::Items {
                            alt: matches!(node, Expr::Alt(_)),
                            count: items.len(),
                        });
                        steps.extend(items.iter().rev().map(CloneStep::Node));
                    }
                    Expr::Repeat { expr, min, max } => {
                        steps.push(CloneStep::Repeat {
                            min: *min,
                            max: *max,
                        });
                        steps.push(CloneStep::Node(expr));
                    }
                    leaf => values.push(leaf.clone_leaf()),
                },
                CloneStep::Items { alt, count } => {
                    let items = values.split_off(values.len().saturating_sub(count));
                    values.push(if alt {
                        Expr::Alt(items)
                    } else {
                        Expr::Seq(items)
                    });
                }
                CloneStep::Repeat { min, max } => {
                    let expr = Box::new(values.pop().unwrap_or(Expr::Empty));
                    values.push(Expr::Repeat { expr, min, max });
                }
            }
        }
        values.pop().unwrap_or(Expr::Empty)
    }
}

impl PartialEq for Expr {
    /// Structural equality, the same relation the derived form gives.
    fn eq(&self, other: &Self) -> bool {
        let mut stack: Vec<(&Expr, &Expr)> = Vec::new();
        let mut next = Some((self, other));
        while let Some(pair) = next.take().or_else(|| stack.pop()) {
            match pair {
                (Expr::Empty, Expr::Empty) | (Expr::Any, Expr::Any) => {}
                (Expr::Literal(a), Expr::Literal(b)) | (Expr::Ref(a), Expr::Ref(b)) => {
                    if a != b {
                        return false;
                    }
                }
                (
                    Expr::Class {
                        negated: a_negated,
                        ranges: a_ranges,
                    },
                    Expr::Class {
                        negated: b_negated,
                        ranges: b_ranges,
                    },
                ) => {
                    if a_negated != b_negated || a_ranges != b_ranges {
                        return false;
                    }
                }
                (Expr::Seq(a), Expr::Seq(b)) | (Expr::Alt(a), Expr::Alt(b)) => {
                    if a.len() != b.len() {
                        return false;
                    }
                    stack.extend(a.iter().zip(b.iter()));
                }
                (
                    Expr::Repeat {
                        expr: a,
                        min: a_min,
                        max: a_max,
                    },
                    Expr::Repeat {
                        expr: b,
                        min: b_min,
                        max: b_max,
                    },
                ) => {
                    if a_min != b_min || a_max != b_max {
                        return false;
                    }
                    next = Some((&**a, &**b));
                }
                _ => return false,
            }
        }
        true
    }
}

impl Eq for Expr {}

/// One pending piece of `Debug` output.
enum Piece<'a> {
    /// Format this node at this nesting.
    Node(&'a Expr, usize),
    /// Write fixed text.
    Text(&'static str),
    /// Write formatted text.
    Owned(String),
    /// Write a line break and this many levels of indentation.
    Line(usize),
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pretty = f.alternate();
        let mut stack = vec![Piece::Node(self, 0)];
        let mut pieces = Vec::new();
        while let Some(piece) = stack.pop() {
            match piece {
                Piece::Node(node, indent) => {
                    node.debug_pieces(indent, pretty, &mut pieces);
                    stack.extend(pieces.drain(..).rev());
                }
                Piece::Text(text) => f.write_str(text)?,
                Piece::Owned(text) => f.write_str(&text)?,
                Piece::Line(indent) => {
                    f.write_char('\n')?;
                    for _ in 0..indent {
                        f.write_str(PAD)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// `value` as `Debug` text. The `{:#?}` form indents its inner lines by `indent` levels.
fn payload(value: &dyn fmt::Debug, indent: usize, pretty: bool) -> String {
    if !pretty {
        return format!("{value:?}");
    }
    let text = format!("{value:#?}");
    text.replace('\n', &format!("\n{}", PAD.repeat(indent)))
}

/// The one field of a tuple variant and its closing parenthesis. The `(` is already out.
fn tuple_field<'a>(out: &mut Vec<Piece<'a>>, value: Piece<'a>, indent: usize, pretty: bool) {
    if pretty {
        out.push(Piece::Line(indent.saturating_add(1)));
        out.push(value);
        out.push(Piece::Text(","));
        out.push(Piece::Line(indent));
    } else {
        out.push(value);
    }
    out.push(Piece::Text(")"));
}

/// The separator and name before a struct field, at nesting `inner`.
fn field(out: &mut Vec<Piece<'_>>, name: &'static str, first: bool, inner: usize, pretty: bool) {
    if pretty {
        if !first {
            out.push(Piece::Text(","));
        }
        out.push(Piece::Line(inner));
    } else {
        out.push(Piece::Text(if first { " " } else { ", " }));
    }
    out.push(Piece::Text(name));
}

/// The end of a struct variant after its last field.
fn struct_end(out: &mut Vec<Piece<'_>>, indent: usize, pretty: bool) {
    if pretty {
        out.push(Piece::Text(","));
        out.push(Piece::Line(indent));
        out.push(Piece::Text("}"));
    } else {
        out.push(Piece::Text(" }"));
    }
}

/// `items` as a `Debug` list at nesting `level`.
fn list<'a>(out: &mut Vec<Piece<'a>>, items: &'a [Expr], level: usize, pretty: bool) {
    if items.is_empty() {
        out.push(Piece::Text("[]"));
        return;
    }
    out.push(Piece::Text("["));
    let inner = level.saturating_add(1);
    for (i, item) in items.iter().enumerate() {
        if pretty {
            out.push(Piece::Line(inner));
        } else if i > 0 {
            out.push(Piece::Text(", "));
        }
        out.push(Piece::Node(item, inner));
        if pretty {
            out.push(Piece::Text(","));
        }
    }
    if pretty {
        out.push(Piece::Line(level));
    }
    out.push(Piece::Text("]"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(s: &str) -> Expr {
        Expr::Literal(s.to_string())
    }

    /// Every variant once, with an empty list and a nested struct.
    fn sample() -> Expr {
        Expr::Seq(vec![
            lit("a"),
            Expr::Alt(vec![]),
            Expr::Repeat {
                expr: Box::new(Expr::Class {
                    negated: true,
                    ranges: vec![('a', 'z')],
                }),
                min: 0,
                max: Some(1),
            },
            Expr::Ref("r".into()),
            Expr::Any,
            Expr::Empty,
        ])
    }

    /// `depth` levels of `Seq([Literal, ...])` around a `Repeat` chain of `depth` levels.
    fn deep(depth: usize) -> Expr {
        let mut expr = Expr::Any;
        for _ in 0..depth {
            expr = Expr::Repeat {
                expr: Box::new(expr),
                min: 0,
                max: None,
            };
        }
        for _ in 0..depth {
            expr = Expr::Seq(vec![lit("x"), expr]);
        }
        expr
    }

    #[test]
    fn debug_matches_the_derived_text() {
        let compact = "Seq([Literal(\"a\"), Alt([]), Repeat { expr: Class { negated: true, \
                       ranges: [('a', 'z')] }, min: 0, max: Some(1) }, Ref(\"r\"), Any, Empty])";
        assert_eq!(format!("{:?}", sample()), compact);
        let pretty = r#"Seq(
    [
        Literal(
            "a",
        ),
        Alt(
            [],
        ),
        Repeat {
            expr: Class {
                negated: true,
                ranges: [
                    (
                        'a',
                        'z',
                    ),
                ],
            },
            min: 0,
            max: Some(
                1,
            ),
        },
        Ref(
            "r",
        ),
        Any,
        Empty,
    ],
)"#;
        assert_eq!(format!("{:#?}", sample()), pretty);
        let nested = format!("{:#?}", Some(lit("b")));
        assert_eq!(nested, "Some(\n    Literal(\n        \"b\",\n    ),\n)");
    }

    #[test]
    fn clone_keeps_the_exact_shape() {
        let original = Expr::Seq(vec![Expr::Seq(vec![lit("a")]), Expr::Alt(vec![lit("b")])]);
        let copy = original.clone();
        assert_eq!(copy, original);
        assert!(matches!(&copy, Expr::Seq(items) if matches!(&items[0], Expr::Seq(_))));
        assert_eq!(sample().clone(), sample());
        assert_eq!(lit("z").clone(), lit("z"));
    }

    #[test]
    fn equality_tells_every_field_apart() {
        let base = sample();
        let mut other = sample();
        if let Expr::Seq(items) = &mut other {
            items[2] = Expr::Repeat {
                expr: Box::new(Expr::Class {
                    negated: true,
                    ranges: vec![('a', 'y')],
                }),
                min: 0,
                max: Some(1),
            };
        }
        assert_ne!(base, other);
        assert_ne!(Expr::Seq(vec![lit("a")]), Expr::Alt(vec![lit("a")]));
        assert_ne!(
            Expr::Seq(vec![lit("a")]),
            Expr::Seq(vec![lit("a"), lit("a")])
        );
        assert_ne!(
            Expr::repeat(lit("a"), 0, None),
            Expr::repeat(lit("a"), 1, None)
        );
        assert_ne!(lit("a"), Expr::Ref("a".into()));
        assert_eq!(Expr::Empty, Expr::Empty);
    }

    #[test]
    fn deep_trees_clone_compare_format_and_drop_on_the_heap() {
        let depth = 100_000;
        let tree = deep(depth);
        let copy = tree.clone();
        assert_eq!(copy, tree);
        assert_eq!(copy.node_count(), tree.node_count());
        let changed = Expr::Seq(vec![lit("y"), deep(depth - 1)]);
        assert_ne!(changed, tree);
        drop(changed);
        let text = format!("{:?}", deep(1_000));
        assert!(text.starts_with("Seq([Literal(\"x\"), Seq(["));
        assert!(text.contains("Repeat { expr: Any, min: 0, max: None }"));
        assert!(text.ends_with(&format!("min: 0, max: None }}{}", "])".repeat(1_000))));
        assert_eq!(text.matches("Seq([").count(), 1_000);
        assert_eq!(text.matches("Repeat {").count(), 1_000);
        drop(copy);
        drop(tree);
    }
}
