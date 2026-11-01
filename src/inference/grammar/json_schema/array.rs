//! Array grammars: `items`, tuples from `prefixItems`, and item count bounds.
//!
//! Item `i` matches `prefixItems[i]` while `i` is inside the tuple, then
//! `items`. An absent `items` allows any value after the tuple, and `false`
//! allows none. A draft-7 `items` array reads as the tuple, with
//! `additionalItems` in the role of `items`. A tuple can stop early unless
//! `minItems` asks for more items.

use serde_json::{Map, Value};

use super::generic;
use super::limits::{BLOCK, blocks, check_members};
use super::refs::child;
use super::walk::{Lang, Walker, count, invalid};
use crate::inference::grammar::ast::Expr;
use crate::inference::grammar::error::{GrammarError, GrammarErrorKind};

/// The array schema `map` at `path`.
pub(super) fn schema<'s>(
    w: &mut Walker<'s>,
    map: &'s Map<String, Value>,
    path: &str,
) -> Result<Lang, GrammarError> {
    let min = count(map, "minItems", path)?.unwrap_or(0);
    let max = count(map, "maxItems", path)?;
    if let Some(max) = max
        && min > max
    {
        return Err(GrammarError::new(
            GrammarErrorKind::EmptyLanguage,
            None,
            format!(
                "`minItems` {min} is above `maxItems` {max} at `{path}`. \
                 Set minItems to at most maxItems."
            ),
        ));
    }
    let (tuple, tuple_key, rest_key) = match (map.get("prefixItems"), map.get("items")) {
        (Some(tuple), _) => (Some(tuple), "prefixItems", "items"),
        (None, Some(tuple @ Value::Array(_))) => (Some(tuple), "items", "additionalItems"),
        _ => (None, "prefixItems", "items"),
    };
    let mut elems = Vec::new();
    let mut cap = None;
    if let Some(tuple) = tuple {
        let at = child(path, tuple_key);
        let Value::Array(schemas) = tuple else {
            return Err(invalid(&at, "must be an array of schemas"));
        };
        check_members(tuple_key, &at, schemas.len())?;
        for (i, sub) in schemas.iter().enumerate() {
            match w.schema(sub, &child(&at, &i.to_string()))? {
                Lang::Expr(expr) => elems.push(expr),
                Lang::Empty(_) => {
                    cap = Some(index(i));
                    break;
                }
            }
        }
    }
    let rest = match (cap, map.get(rest_key)) {
        (Some(_), _) | (None, Some(Value::Bool(false))) => None,
        (None, None) => Some(generic::any_value(w)),
        (None, Some(sub)) => match w.schema(sub, &child(path, rest_key))? {
            Lang::Expr(expr) => Some(expr),
            Lang::Empty(_) => None,
        },
    };
    let n = index(elems.len());
    let cap = match (cap, &rest) {
        (Some(cap), _) => Some(cap),
        (None, None) => Some(n),
        (None, Some(_)) => None,
    };
    let hi = match (max, cap) {
        (Some(max), Some(cap)) => Some(max.min(cap)),
        (max, cap) => max.or(cap),
    };
    if let Some(hi) = hi
        && min > hi
    {
        return Ok(Lang::Empty(format!(
            "the array at `{path}` needs at least {min} items, \
             and its item schemas allow at most {hi}"
        )));
    }
    Ok(Lang::Expr(list(w, elems, rest, min, hi)))
}

/// `[ ws ( ] | items ws ] )` with between `min` and `hi` items.
fn list(
    w: &mut Walker<'_>,
    mut elems: Vec<Expr>,
    rest: Option<Expr>,
    min: u32,
    hi: Option<u32>,
) -> Expr {
    let ws = w.ws();
    let close = Expr::literal("]");
    if let Some(hi) = hi {
        elems.truncate(usize::try_from(hi).unwrap_or(usize::MAX));
    }
    let n = index(elems.len());
    let body = if elems.is_empty() {
        match (rest, hi) {
            (_, Some(0)) | (None, _) => None,
            (Some(item), hi) => Some(uniform(w, item, min.max(1), hi)),
        }
    } else {
        Some(tuple(
            w,
            elems,
            rest.filter(|_| hi.is_none_or(|hi| hi > n)),
            min,
            hi,
        ))
    };
    let Some(body) = body else {
        return Expr::seq(vec![Expr::literal("["), ws, close]);
    };
    let full = Expr::seq(vec![body, ws.clone(), close.clone()]);
    let rest = if min == 0 {
        Expr::alt(vec![close, full])
    } else {
        full
    };
    Expr::seq(vec![Expr::literal("["), ws, rest])
}

/// `( item ws , ws ){lo-1,hi-1} item`: from `lo` to `hi` items, `lo >= 1`.
///
/// The loop copies and the last copy start at the same input positions,
/// so determinization tracks them in one state set.
fn uniform(w: &mut Walker<'_>, item: Expr, lo: u32, hi: Option<u32>) -> Expr {
    let item = w.share(item);
    let comma = w.comma();
    let head = Expr::seq(vec![item.clone(), comma]);
    let more = Expr::repeat(
        head,
        lo.saturating_sub(1),
        hi.map(|hi| hi.saturating_sub(1)),
    );
    Expr::seq(vec![more, item])
}

/// The tuple `elems` with each later item optional once `min` items are present,
/// then the `rest` items up to `hi` in all.
///
/// The first item and the items before `min` form a flat sequence.
/// `optional_tail` holds the items after them.
fn tuple(
    w: &mut Walker<'_>,
    mut elems: Vec<Expr>,
    rest: Option<Expr>,
    min: u32,
    hi: Option<u32>,
) -> Expr {
    let n = index(elems.len());
    let tail = match rest {
        Some(item) => {
            let comma = w.comma();
            Expr::repeat(
                Expr::seq(vec![comma, item]),
                min.saturating_sub(n),
                hi.map(|hi| hi.saturating_sub(n)),
            )
        }
        None => Expr::Empty,
    };
    let required = usize::try_from(min)
        .unwrap_or(usize::MAX)
        .max(1)
        .min(elems.len());
    let optional = elems.split_off(required);
    let mut items = Vec::with_capacity(required.saturating_mul(2).saturating_add(1));
    for (i, elem) in elems.into_iter().enumerate() {
        if i > 0 {
            items.push(w.comma());
        }
        items.push(elem);
    }
    items.push(optional_tail(w, optional, tail));
    Expr::seq(items)
}

/// The optional items `elems`, each as `ws , ws elem`, then `cont` after the last.
///
/// An item can appear only after the one before it. Up to `BLOCK` items nest
/// to the right as `( , e0 ( , e1 ... cont )? )?`, so each element appears once.
/// Each item nests the tree two levels deeper. So past `BLOCK` items, block
/// `k` becomes `X(k) = full(k) X(k + 1) | prefix(k)`. `full(k)` is the whole
/// block, flat. `prefix(k)` is any shorter run of it, the empty run included.
/// The last block nests over `cont` as above.
fn optional_tail(w: &mut Walker<'_>, elems: Vec<Expr>, cont: Expr) -> Expr {
    let units: Vec<Expr> = elems
        .into_iter()
        .map(|elem| {
            let comma = w.comma();
            Expr::seq(vec![comma, elem])
        })
        .collect();
    if units.len() <= BLOCK {
        return nest(units, cont);
    }
    let mut chunks = blocks(units);
    let mut tail = match chunks.pop() {
        Some(last) => nest(last, cont),
        None => cont,
    };
    for block in chunks.into_iter().rev() {
        // `full` and `prefix` both hold every unit but the last, so share them.
        let mut block: Vec<Expr> = block.into_iter().map(|unit| w.share(unit)).collect();
        let mut full = block.clone();
        block.pop();
        let prefix = nest(block, Expr::Empty);
        full.push(tail);
        tail = Expr::alt(vec![Expr::seq(full), prefix]);
    }
    tail
}

/// `( u0 ( u1 ... ( un cont )? ... )? )?`: each unit only after the one before it.
fn nest(units: Vec<Expr>, cont: Expr) -> Expr {
    let mut tail = cont;
    for unit in units.into_iter().rev() {
        tail = Expr::repeat(Expr::seq(vec![unit, tail]), 0, Some(1));
    }
    tail
}

/// `i` as a `u32` item count. A count above `u32::MAX` saturates.
fn index(i: usize) -> u32 {
    u32::try_from(i).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::json_schema::fixture::{check, dfa, error, opts};

    #[test]
    fn items_arrays_accept_any_count() {
        let d = dfa(
            r#"{"type": "array", "items": {"type": "integer"}}"#,
            &opts(),
        );
        check(
            &d,
            &["[]", "[ ]", "[1]", "[1,2,3]", "[1, 2, 3]", "[\n1,\n2\n]"],
            &[
                "[1,]", "[,1]", "[1 2]", "[1.5]", "[\"1\"]", "[", "1", "[[1]]",
            ],
        );
    }

    #[test]
    fn min_and_max_items_bound_the_count() {
        let d = dfa(
            r#"{"type": "array", "items": {"type": "boolean"}, "minItems": 2, "maxItems": 4}"#,
            &opts(),
        );
        check(
            &d,
            &[
                "[true,false]",
                "[true, true, true]",
                "[false,false,false,false]",
            ],
            &["[]", "[true]", "[true,true,true,true,true]", "[true,,true]"],
        );
        let d = dfa(r#"{"items": {"type": "null"}, "maxItems": 1}"#, &opts());
        check(&d, &["[]", "[null]"], &["[null,null]"]);
        let d = dfa(
            r#"{"type": "array", "minItems": 1, "items": {"type": "null"}}"#,
            &opts(),
        );
        check(&d, &["[null]", "[null,null,null]"], &["[]"]);
        let err = error(
            r#"{"type": "array", "minItems": 3, "maxItems": 2}"#,
            &opts(),
        );
        assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage);
    }

    #[test]
    fn prefix_items_form_a_tuple() {
        let closed = r#"{"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "items": false}"#;
        let d = dfa(closed, &opts());
        check(
            &d,
            &["[]", r#"["a"]"#, r#"["a", 1]"#],
            &[r#"[1]"#, r#"["a", "b"]"#, r#"["a", 1, 2]"#, r#"[1, "a"]"#],
        );
        let open =
            r#"{"prefixItems": [{"type": "string"}], "items": {"type": "integer"}, "minItems": 2}"#;
        check(
            &dfa(open, &opts()),
            &[r#"["a", 1]"#, r#"["a", 1, 2, 3]"#],
            &[r#"["a"]"#, r#"["a", "b"]"#, "[]"],
        );
        let draft7 =
            r#"{"items": [{"type": "null"}, {"type": "boolean"}], "additionalItems": false}"#;
        check(
            &dfa(draft7, &opts()),
            &["[null, true]", "[null]"],
            &["[null, true, 1]", "[true]"],
        );
        let capped = r#"{"prefixItems": [{"type": "null"}, false], "minItems": 2}"#;
        let err = error(capped, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage);
        assert!(err.message().contains("at most 1"), "{}", err.message());
    }

    #[test]
    fn an_absent_items_allows_any_value() {
        let o = crate::inference::grammar::json_schema::SchemaOptions {
            max_value_depth: 2,
            ..opts()
        };
        let d = dfa(r#"{"type": "array"}"#, &o);
        check(
            &d,
            &["[]", r#"[1, "a", [null], {"k": true}]"#],
            &["[[[[1]]]]", "{}"],
        );
    }

    #[test]
    fn objects_nest_in_arrays_in_objects() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
                        "required": ["id"]
                    }
                }
            },
            "required": ["rows"]
        }"#;
        let d = dfa(schema, &opts());
        check(
            &d,
            &[
                r#"{"rows": []}"#,
                r#"{"rows": [{"id": 1}, {"id": 2, "tags": ["a", "b"]}]}"#,
                "{\n\"rows\": [\n{\"id\": 1, \"tags\": []}\n]\n}",
            ],
            &[
                r#"{}"#,
                r#"{"rows": [{"tags": []}]}"#,
                r#"{"rows": [{"id": 1, "tags": ["a", "b", "c"]}]}"#,
                r#"{"rows": [{"id": 1}],}"#,
                r#"{"rows": {"id": 1}}"#,
            ],
        );
    }

    /// The array text of `n` nulls, then `tail` items.
    fn nulls(n: usize, tail: &[&str]) -> String {
        let mut items = vec!["null"; n];
        items.extend_from_slice(tail);
        format!("[{}]", items.join(","))
    }

    #[test]
    fn long_tuples_split_into_blocks_and_keep_their_language() {
        let compact = crate::inference::grammar::json_schema::SchemaOptions {
            max_whitespace: 0,
            ..opts()
        };
        let items = vec![serde_json::json!({"type": "null"}); 2 * BLOCK + 6];
        let n = items.len();
        let closed = serde_json::json!({"prefixItems": items, "items": false});
        let d = dfa(&closed.to_string(), &compact);
        for k in 0..=n {
            assert!(d.accepts(nulls(k, &[]).as_bytes()), "{k} items");
        }
        let longer = nulls(n + 1, &[]);
        check(
            &d,
            &[],
            &[longer.as_str(), "[null,]", "[,null]", "[null null]"],
        );
        let mut open = closed.clone();
        open["items"] = serde_json::json!({"type": "integer"});
        open["minItems"] = serde_json::json!(BLOCK + 3);
        let d = dfa(&open.to_string(), &compact);
        let accept = [
            nulls(BLOCK + 3, &[]),
            nulls(BLOCK + 4, &[]),
            nulls(n, &[]),
            nulls(n, &["1", "2"]),
        ];
        let reject = [
            nulls(BLOCK + 2, &[]),
            nulls(n - 1, &["1"]),
            "[]".to_string(),
        ];
        for text in &accept {
            assert!(d.accepts(text.as_bytes()), "must accept {text}");
        }
        for text in &reject {
            assert!(!d.accepts(text.as_bytes()), "must reject {text}");
        }
    }

    #[test]
    fn tuple_length_is_limited() {
        use crate::inference::grammar::ExpandLimits;
        use crate::inference::grammar::json_schema::fixture::{limit_message, lowered};
        let items = |n: usize| vec![serde_json::json!({"type": "integer"}); n];
        let grammar = lowered(&serde_json::json!({"prefixItems": items(1_000)}));
        let expanded = grammar.expand(&ExpandLimits::default());
        assert!(expanded.is_ok(), "{:?}", expanded.err());
        let msg = limit_message(&serde_json::json!({"prefixItems": items(1_001)}));
        assert!(msg.contains("`prefixItems` at `#/prefixItems`"), "{msg}");
        let msg = limit_message(&serde_json::json!({"items": items(1_001)}));
        assert!(msg.contains("`items` at `#/items`"), "{msg}");
    }
}
