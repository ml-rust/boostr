//! Grammars, rules, rule references and source positions.

use std::fmt;

use super::expr::Expr;

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

impl Grammar {
    /// The rule named `name`, if defined.
    pub fn rule(&self, name: &str) -> Option<&Rule> {
        self.rules.iter().find(|rule| rule.name == name)
    }

    /// The `root` rule, if defined.
    pub fn root(&self) -> Option<&Rule> {
        self.rule(ROOT_RULE)
    }

    /// A grammar from named rule bodies, kept in order.
    ///
    /// Rule `i` gets the span `i + 1:1`. Its references are the `Expr::Ref`
    /// nodes of its body in source order, each at the rule span.
    /// `expand` checks names, so an undefined or duplicate name is an error there.
    pub fn from_rules(rules: Vec<(String, Expr)>) -> Grammar {
        let rules = rules
            .into_iter()
            .enumerate()
            .map(|(i, (name, expr))| {
                let line = u32::try_from(i.saturating_add(1)).unwrap_or(u32::MAX);
                let span = Span { line, column: 1 };
                let references = references(&expr, span);
                Rule {
                    name,
                    expr,
                    span,
                    references,
                }
            })
            .collect();
        Grammar { rules }
    }
}

/// Every `Expr::Ref` in `expr`, in source order, each at `span`.
///
/// The walk runs on an explicit stack, so tree depth never grows the call stack.
fn references(expr: &Expr, span: Span) -> Vec<Reference> {
    let mut out = Vec::new();
    let mut stack = vec![expr];
    while let Some(node) = stack.pop() {
        match node {
            Expr::Ref(name) => out.push(Reference {
                name: name.clone(),
                span,
            }),
            Expr::Seq(items) | Expr::Alt(items) => stack.extend(items.iter().rev()),
            Expr::Repeat { expr, .. } => stack.push(expr),
            Expr::Empty | Expr::Literal(_) | Expr::Class { .. } | Expr::Any => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(s: &str) -> Expr {
        Expr::Literal(s.to_string())
    }

    #[test]
    fn from_rules_lists_references_in_source_order() {
        let body = Expr::Seq(vec![
            Expr::Ref("a".into()),
            Expr::Alt(vec![lit("x"), Expr::repeat(Expr::Ref("b".into()), 0, None)]),
            Expr::Ref("a".into()),
        ]);
        let grammar = Grammar::from_rules(vec![
            ("root".into(), body.clone()),
            ("a".into(), lit("a")),
            ("b".into(), Expr::Ref("a".into())),
        ]);
        assert_eq!(grammar.rules.len(), 3);
        let root = grammar.root().map(|r| (r.expr.clone(), r.span));
        assert_eq!(root, Some((body, Span { line: 1, column: 1 })));
        let names: Vec<&str> = grammar.rules[0]
            .references
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(names, ["a", "b", "a"]);
        assert!(grammar.rules[0].references.iter().all(|r| r.span.line == 1));
        assert!(grammar.rules[1].references.is_empty());
        assert_eq!(grammar.rules[2].span, Span { line: 3, column: 1 });
        assert_eq!(grammar.rules[2].references[0].name, "a");
    }

    #[test]
    fn from_rules_expands_like_parsed_text() {
        let built = Grammar::from_rules(vec![
            (
                "root".into(),
                Expr::seq(vec![Expr::Ref("d".into()), lit("!")]),
            ),
            ("d".into(), lit("hi")),
        ]);
        let parsed = Grammar::parse("root ::= d \"!\"\nd ::= \"hi\"");
        let limits = crate::inference::grammar::ExpandLimits::default();
        assert_eq!(
            built.expand(&limits).ok(),
            parsed.ok().and_then(|g| g.expand(&limits).ok())
        );
        let missing = Grammar::from_rules(vec![("root".into(), Expr::Ref("nope".into()))]);
        let kind = missing.expand(&limits).err().map(|e| e.kind());
        assert_eq!(
            kind,
            Some(crate::inference::grammar::GrammarErrorKind::UndefinedRule)
        );
    }

    #[test]
    fn span_displays_line_and_column() {
        assert_eq!(Span { line: 3, column: 7 }.to_string(), "3:7");
    }
}
