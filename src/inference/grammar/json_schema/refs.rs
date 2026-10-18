//! `$ref` resolution inside one document, recursion depth, and JSON pointers.
//!
//! A `$ref` target expands as a rule. A target reached again while it is
//! still expanding is recursive. Each recursive level is a new rule, and the
//! level at `max_depth` leaves the language.

use std::collections::HashMap;

use serde_json::Value;

use super::generic;
use super::keywords::unsupported;
use super::walk::{Lang, Walker, invalid};
use crate::inference::grammar::error::{GrammarError, GrammarErrorKind, excerpt};

/// The `$ref` targets being expanded and the rules built so far.
pub(super) struct RefState {
    /// Address of each target on the current resolution path, outermost first.
    /// The document root is always the first entry.
    stack: Vec<usize>,
    /// The language of each target, by target address and resolution path.
    memo: HashMap<(usize, Vec<usize>), Lang>,
}

impl RefState {
    /// The state for a walk that starts at `root`.
    pub(super) fn new(root: &Value) -> Self {
        Self {
            stack: vec![address(root)],
            memo: HashMap::new(),
        }
    }
}

/// The identity of a schema node inside the borrowed document.
fn address(node: &Value) -> usize {
    std::ptr::from_ref(node) as usize
}

/// The language of the `$ref` keyword with value `value`, in the schema at `path`.
pub(super) fn reference<'s>(
    w: &mut Walker<'s>,
    value: &'s Value,
    path: &str,
) -> Result<Lang, GrammarError> {
    let at = child(path, "$ref");
    let Value::String(text) = value else {
        return Err(invalid(&at, "must be a string"));
    };
    let shown = excerpt(text);
    let fragment = match text.strip_prefix('#') {
        Some(fragment) if fragment.is_empty() || fragment.starts_with('/') => fragment,
        _ => {
            if w.opts.strict_keywords {
                let what = format!("`$ref` `{shown}` outside this document or by anchor");
                return Err(unsupported(&at, &what));
            }
            return Ok(Lang::Expr(generic::any_value(w)));
        }
    };
    let Some(target) = resolve(w.root, fragment) else {
        return Err(GrammarError::new(
            GrammarErrorKind::UndefinedRule,
            None,
            format!(
                "`$ref` `{shown}` at `{at}` names no schema in this document. \
                 Correct the pointer or add the definition."
            ),
        ));
    };
    let key = address(target);
    let level = w.refs.stack.iter().filter(|&&entry| entry == key).count();
    let max_depth = usize::try_from(w.opts.max_depth).unwrap_or(usize::MAX);
    if level > 0 && level >= max_depth {
        return Ok(Lang::Empty(format!(
            "recursive `$ref` `{shown}` at `{at}` reaches max_depth {} on every branch. \
             Raise max_depth or make the recursion optional",
            w.opts.max_depth
        )));
    }
    let memo_key = (key, w.refs.stack.clone());
    if let Some(found) = w.refs.memo.get(&memo_key) {
        return Ok(found.clone());
    }
    w.refs.stack.push(key);
    let out = w.schema(target, &shown);
    w.refs.stack.pop();
    let lang = match out? {
        Lang::Expr(expr) => Lang::Expr(w.fresh_rule("ref", expr)),
        empty => empty,
    };
    w.refs.memo.insert(memo_key, lang.clone());
    Ok(lang)
}

/// The node at the JSON pointer `fragment` inside `root`.
///
/// `fragment` is empty or starts with `/`. Each token is percent-decoded,
/// then `~1` reads as `/` and `~0` as `~`.
pub(super) fn resolve<'s>(root: &'s Value, fragment: &str) -> Option<&'s Value> {
    let mut node = root;
    if fragment.is_empty() {
        return Some(node);
    }
    for raw in fragment.split('/').skip(1) {
        let token = percent_decode(raw)?.replace("~1", "/").replace("~0", "~");
        node = match node {
            Value::Object(map) => map.get(&token)?,
            Value::Array(items) => items.get(token.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(node)
}

/// `text` with each `%XX` escape decoded. A bad escape or bad UTF-8 gives `None`.
fn percent_decode(text: &str) -> Option<String> {
    if !text.contains('%') {
        return Some(text.to_string());
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while let Some(&byte) = bytes.get(i) {
        if byte == b'%' {
            let hex = bytes.get(i.saturating_add(1)..i.saturating_add(3))?;
            if !hex.iter().all(u8::is_ascii_hexdigit) {
                return None;
            }
            let hex = std::str::from_utf8(hex).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i = i.saturating_add(3);
        } else {
            out.push(byte);
            i = i.saturating_add(1);
        }
    }
    String::from_utf8(out).ok()
}

/// The JSON pointer of member `token` under `path`, for messages.
///
/// The token is escaped per RFC 6901 and cut to an excerpt.
pub(super) fn child(path: &str, token: &str) -> String {
    let escaped = excerpt(token).replace('~', "~0").replace('/', "~1");
    format!("{path}/{escaped}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::json_schema::SchemaOptions;
    use crate::inference::grammar::json_schema::fixture::{check, dfa, error, opts};
    use serde_json::json;

    #[test]
    fn pointers_resolve_with_escapes_and_indexes() {
        let doc = json!({"$defs": {"a/b": {"x": 1}, "m~n": [10, {"y": 2}], "sp ace": 3}});
        assert_eq!(resolve(&doc, ""), Some(&doc));
        assert_eq!(resolve(&doc, "/$defs/a~1b/x"), Some(&json!(1)));
        assert_eq!(resolve(&doc, "/$defs/m~0n/1/y"), Some(&json!(2)));
        assert_eq!(resolve(&doc, "/$defs/sp%20ace"), Some(&json!(3)));
        assert_eq!(resolve(&doc, "/$defs/missing"), None);
        assert_eq!(resolve(&doc, "/$defs/m~0n/9"), None);
        assert_eq!(resolve(&doc, "/$defs/bad%2"), None);
        assert_eq!(child("#", "a/b~c"), "#/a~1b~0c");
    }

    #[test]
    fn a_shared_definition_expands_at_each_use() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "home": {"$ref": "#/$defs/point"},
                "work": {"$ref": "#/definitions/point"}
            },
            "required": ["home", "work"],
            "$defs": {"point": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}},
            "definitions": {"point": {"$ref": "#/$defs/point"}}
        }"##;
        let d = dfa(schema, &opts());
        check(
            &d,
            &[r#"{"home": {"x": 1}, "work": {"x": -2}}"#],
            &[
                r#"{"home": {"x": 1}}"#,
                r#"{"home": {"x": 1}, "work": {"y": 2}}"#,
                r#"{"home": {"x": 1.5}, "work": {"x": 2}}"#,
            ],
        );
    }

    /// A tree `{"v": 0, "kids": [...]}` nested `depth` nodes deep.
    fn tree(depth: u32) -> String {
        let mut text = r#"{"v": 0}"#.to_string();
        for _ in 1..depth {
            text = format!(r#"{{"v": 1, "kids": [{text}, {{"v": 2}}]}}"#);
        }
        text
    }

    const TREE: &str = r##"{
        "type": "object",
        "properties": {"v": {"type": "integer"}, "kids": {"type": "array", "items": {"$ref": "#"}}},
        "required": ["v"]
    }"##;

    #[test]
    fn a_recursive_ref_nests_up_to_max_depth() {
        for max_depth in [1, 3, 6] {
            let o = SchemaOptions {
                max_depth,
                ..opts()
            };
            let d = dfa(TREE, &o);
            for depth in 1..=max_depth {
                assert!(d.accepts(tree(depth).as_bytes()), "{max_depth} {depth}");
            }
            let deeper = tree(max_depth.saturating_add(1));
            assert!(!d.accepts(deeper.as_bytes()), "{max_depth}");
        }
    }

    #[test]
    fn a_required_recursion_with_no_exit_is_an_empty_language_error() {
        let schema =
            r##"{"type": "object", "properties": {"next": {"$ref": "#"}}, "required": ["next"]}"##;
        let err = error(schema, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage);
        assert!(err.message().contains("`$ref` `#`"), "{}", err.message());
        assert!(err.message().contains("max_depth 6"), "{}", err.message());
    }

    #[test]
    fn a_missing_target_is_an_undefined_rule_error() {
        let err = error(r##"{"$ref": "#/$defs/none"}"##, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::UndefinedRule);
        assert!(
            err.message().contains("`#/$defs/none`"),
            "{}",
            err.message()
        );
        assert!(err.message().contains("at `#/$ref`"), "{}", err.message());
        let err = error(r##"{"$ref": 3}"##, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::Syntax);
    }

    #[test]
    fn external_and_anchor_refs_are_unsupported_in_strict_mode() {
        for target in ["https://example.com/s.json", "#anchor"] {
            let err = error(&format!(r#"{{"$ref": "{target}"}}"#), &opts());
            assert_eq!(err.kind(), GrammarErrorKind::Unsupported, "{target}");
            assert!(err.message().contains("`#/$ref`"), "{}", err.message());
        }
    }

    #[test]
    fn percent_escapes_need_two_hex_digits() {
        let doc = json!({"a/b": 1});
        assert_eq!(resolve(&doc, "/a%2Fb"), Some(&json!(1)));
        assert_eq!(percent_decode("%2F"), Some("/".to_string()));
        assert_eq!(percent_decode("%+f"), None);
        assert_eq!(percent_decode("%f+"), None);
        assert_eq!(percent_decode("%-1"), None);
    }
}
