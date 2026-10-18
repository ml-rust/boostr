//! The schema walker: rule registry, walk limits and per-node dispatch.

use std::collections::HashSet;

use serde_json::{Map, Value};

use super::keywords::{self, COMBINATORS, TYPE_KEYWORDS};
use super::options::SchemaOptions;
use super::refs::{self, RefState, child};
use super::{generic, scalar, union};
use crate::inference::grammar::ast::{Expr, Grammar, ROOT_RULE};
use crate::inference::grammar::error::{GrammarError, GrammarErrorKind};

/// Deepest schema nesting the walker follows, `$ref` targets included.
pub(super) const MAX_SCHEMA_DEPTH: u32 = 128;

/// Most sub-schemas one walk visits, `$ref` expansions included.
pub(super) const MAX_SCHEMA_VISITS: usize = 100_000;

/// Largest expression that a builder copies inline. A larger one becomes a rule.
const INLINE_NODES: usize = 8;

/// The language of a sub-schema.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Lang {
    /// The JSON texts this expression matches.
    Expr(Expr),
    /// No JSON text. The string says why, for the error message.
    Empty(String),
}

/// Walk state for one schema document.
pub(super) struct Walker<'s> {
    /// The schema document that `$ref` pointers resolve against.
    pub(super) root: &'s Value,
    /// Compiler config.
    pub(super) opts: &'s SchemaOptions,
    /// Recursion state of `$ref` expansion.
    pub(super) refs: RefState,
    rules: Vec<(String, Expr)>,
    defined: HashSet<String>,
    fresh: usize,
    depth: u32,
    visits: usize,
}

impl<'s> Walker<'s> {
    /// A walker over the document `root`.
    pub(super) fn new(root: &'s Value, opts: &'s SchemaOptions) -> Self {
        Self {
            root,
            opts,
            refs: RefState::new(root),
            rules: Vec::new(),
            defined: HashSet::new(),
            fresh: 0,
            depth: 0,
            visits: 0,
        }
    }

    /// A reference to the shared rule `name`. `build` runs on first use only.
    pub(super) fn rule(&mut self, name: &str, build: impl FnOnce(&mut Self) -> Expr) -> Expr {
        if self.defined.insert(name.to_string()) {
            let expr = build(self);
            self.rules.push((name.to_string(), expr));
        }
        Expr::Ref(name.to_string())
    }

    /// `expr` as a new rule named `prefix-N`, and a reference to it.
    pub(super) fn fresh_rule(&mut self, prefix: &str, expr: Expr) -> Expr {
        self.fresh = self.fresh.saturating_add(1);
        let name = format!("{prefix}-{}", self.fresh);
        self.defined.insert(name.clone());
        self.rules.push((name.clone(), expr));
        Expr::Ref(name)
    }

    /// `expr` for use at two or more sites: a new rule when it is large.
    ///
    /// Expansion still copies a rule at every use. The rule keeps the grammar small.
    pub(super) fn share(&mut self, expr: Expr) -> Expr {
        if expr.node_count() <= INLINE_NODES {
            expr
        } else {
            self.fresh_rule("part", expr)
        }
    }

    /// Whitespace between tokens: at most `max_whitespace` of space, `\t` and `\n`.
    pub(super) fn ws(&mut self) -> Expr {
        let max = self.opts.max_whitespace;
        if max == 0 {
            return Expr::Empty;
        }
        self.rule("ws", |_| {
            let class = Expr::Class {
                negated: false,
                ranges: vec![(' ', ' '), ('\t', '\t'), ('\n', '\n')],
            };
            Expr::repeat(class, 0, Some(max))
        })
    }

    /// `ws , ws`: the separator between two members.
    pub(super) fn comma(&mut self) -> Expr {
        let ws = self.ws();
        Expr::seq(vec![ws.clone(), Expr::literal(","), ws])
    }

    /// The language of the sub-schema `node` at the JSON pointer `path`.
    pub(super) fn schema(&mut self, node: &'s Value, path: &str) -> Result<Lang, GrammarError> {
        if self.depth >= MAX_SCHEMA_DEPTH {
            return Err(limit(format!(
                "schema nesting at `{path}` exceeds {MAX_SCHEMA_DEPTH} levels. \
                 Flatten the schema or lower max_depth."
            )));
        }
        self.visits = self.visits.saturating_add(1);
        if self.visits > MAX_SCHEMA_VISITS {
            return Err(limit(format!(
                "schema walk visits more than {MAX_SCHEMA_VISITS} sub-schemas at `{path}`. \
                 Reuse fewer `$ref` targets or lower max_depth."
            )));
        }
        self.depth = self.depth.saturating_add(1);
        let out = match node {
            Value::Bool(true) => Ok(Lang::Expr(generic::any_value(self))),
            Value::Bool(false) => Ok(Lang::Empty(format!("the schema at `{path}` is `false`"))),
            Value::Object(map) => self.node(map, path),
            other => Err(invalid(
                path,
                &format!("is {}, not a schema object or boolean", kind(other)),
            )),
        };
        self.depth = self.depth.saturating_sub(1);
        out
    }

    /// The language of a schema object.
    fn node(&mut self, map: &'s Map<String, Value>, path: &str) -> Result<Lang, GrammarError> {
        keywords::check(map, path, self.opts.strict_keywords)?;
        let base = match self.combinator(map, path)? {
            Some(("$ref", value)) => refs::reference(self, value, path)?,
            Some(("allOf", value)) => union::all_of(self, value, &child(path, "allOf"))?,
            Some((name, value)) => union::any_of(self, value, &child(path, name))?,
            None if map.contains_key("enum") || map.contains_key("const") => {
                union::enum_const(map, path)?
            }
            None => union::typed(self, map, path)?,
        };
        let nullable = match map.get("nullable") {
            None => false,
            Some(Value::Bool(flag)) => *flag,
            Some(_) => return Err(invalid(&child(path, "nullable"), "must be a boolean")),
        };
        Ok(if nullable {
            union::union(vec![base, Lang::Expr(scalar::null())])
        } else {
            base
        })
    }

    /// The first combinator keyword of `map` and its value.
    ///
    /// In strict mode a second combinator or a type keyword beside it is an error.
    fn combinator(
        &self,
        map: &'s Map<String, Value>,
        path: &str,
    ) -> Result<Option<(&'static str, &'s Value)>, GrammarError> {
        let mut found = COMBINATORS
            .iter()
            .filter_map(|&name| map.get(name).map(|value| (name, value)));
        let Some((first, value)) = found.next() else {
            return Ok(None);
        };
        if self.opts.strict_keywords {
            let other = found.next().map(|(name, _)| name).or_else(|| {
                TYPE_KEYWORDS
                    .iter()
                    .copied()
                    .find(|name| map.contains_key(*name))
            });
            if let Some(other) = other {
                return Err(keywords::combined(path, first, other));
            }
        }
        Ok(Some((first, value)))
    }

    /// The grammar whose `root` matches `root` with whitespace at both ends.
    ///
    /// An empty `root` is an `EmptyLanguage` error with its reason.
    pub(super) fn finish(mut self, root: Lang) -> Result<Grammar, GrammarError> {
        let root = match root {
            Lang::Expr(expr) => expr,
            Lang::Empty(reason) => {
                return Err(GrammarError::new(
                    GrammarErrorKind::EmptyLanguage,
                    None,
                    format!(
                        "schema matches no JSON text: {reason}. \
                         Relax the schema so that at least one value is valid."
                    ),
                ));
            }
        };
        let ws = self.ws();
        let mut rules = Vec::with_capacity(self.rules.len().saturating_add(1));
        rules.push((ROOT_RULE.to_string(), Expr::seq(vec![ws.clone(), root, ws])));
        rules.append(&mut self.rules);
        Ok(Grammar::from_rules(rules))
    }
}

/// The count keyword `key` of `map`, such as `minLength`, if present.
///
/// The value must be a non-negative integer that fits `u32`.
pub(super) fn count(
    map: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<Option<u32>, GrammarError> {
    let Some(value) = map.get(key) else {
        return Ok(None);
    };
    let whole = value.as_u64().or_else(|| {
        value
            .as_f64()
            .filter(|f| f.fract() == 0.0 && (0.0..18_446_744_073_709_551_616.0).contains(f))
            .map(|f| f as u64)
    });
    let Some(n) = whole else {
        return Err(invalid(&child(path, key), "must be a non-negative integer"));
    };
    u32::try_from(n).map(Some).map_err(|_| {
        limit(format!(
            "`{key}` {n} at `{path}` is above {}. Lower `{key}`.",
            u32::MAX
        ))
    })
}

/// The compact JSON text of `value`, the form `serde_json::to_string` writes.
pub(super) fn compact(value: &Value, path: &str) -> Result<String, GrammarError> {
    serde_json::to_string(value).map_err(|err| {
        invalid(
            path,
            &format!("holds a value that does not serialize as JSON ({err})"),
        )
    })
}

/// A `Syntax` error: the schema at `path` breaks the JSON Schema shape.
pub(super) fn invalid(path: &str, problem: &str) -> GrammarError {
    GrammarError::new(
        GrammarErrorKind::Syntax,
        None,
        format!("schema at `{path}` {problem}. Correct the schema."),
    )
}

/// A `LimitExceeded` error with `message`.
pub(super) fn limit(message: String) -> GrammarError {
    GrammarError::new(GrammarErrorKind::LimitExceeded, None, message)
}

/// The JSON type of `value`, with an article, for messages.
pub(super) fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::json_schema::fixture::{error, opts};
    use serde_json::json;

    #[test]
    fn count_reads_whole_numbers_and_rejects_the_rest() {
        let map = json!({"a": 3, "b": 4.0, "c": -1, "d": 1.5, "e": "2", "f": 5_000_000_000u64});
        let map = map.as_object().cloned().unwrap_or_default();
        assert_eq!(count(&map, "a", "#").ok(), Some(Some(3)));
        assert_eq!(count(&map, "b", "#").ok(), Some(Some(4)));
        assert_eq!(count(&map, "z", "#").ok(), Some(None));
        for key in ["c", "d", "e"] {
            let err = count(&map, key, "#").err().map(|e| e.kind());
            assert_eq!(err, Some(GrammarErrorKind::Syntax), "{key}");
        }
        let err = count(&map, "f", "#").err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::LimitExceeded)
        );
    }

    #[test]
    fn a_non_schema_value_is_a_syntax_error_with_its_pointer() {
        let err = error(r#"{"properties": {"a": 5}}"#, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::Syntax);
        assert!(
            err.message().contains("`#/properties/a`"),
            "{}",
            err.message()
        );
        let err = error(r#"{"type": "string", "nullable": 1}"#, &opts());
        assert!(err.message().contains("`#/nullable`"), "{}", err.message());
    }

    #[test]
    fn false_schema_is_an_empty_language_error() {
        let err = error("false", &opts());
        assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage);
        assert!(
            err.message().contains("`#` is `false`"),
            "{}",
            err.message()
        );
    }

    #[test]
    fn a_combinator_beside_a_type_keyword_is_an_error_in_strict_mode_only() {
        let schema =
            r##"{"$ref": "#/$defs/a", "type": "object", "$defs": {"a": {"type": "object"}}}"##;
        let err = error(schema, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::Unsupported);
        assert!(err.message().contains("`$ref`"), "{}", err.message());
        assert!(err.message().contains("`type`"), "{}", err.message());
        let loose = SchemaOptions {
            strict_keywords: false,
            ..opts()
        };
        let d = crate::inference::grammar::json_schema::fixture::dfa(schema, &loose);
        assert!(d.accepts(b"{}"));
    }
}
