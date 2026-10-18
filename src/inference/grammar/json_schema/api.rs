//! Entry points: schema to grammar, schema to DFA, and the `json_object` grammar.

use serde_json::Value;

use super::generic;
use super::options::SchemaOptions;
use super::walk::{Lang, Walker};
use crate::inference::grammar::{
    ByteDfa, CompileLimits, Grammar, GrammarError, GrammarErrorKind, Span,
};

/// Keys that mark a value as a schema, not as an OpenAI `response_format` wrapper.
const SCHEMA_KEYS: &[&str] = &[
    "type",
    "properties",
    "$ref",
    "enum",
    "const",
    "anyOf",
    "oneOf",
];

/// A grammar whose language is the JSON texts valid for `schema`.
///
/// Property order follows the key order of the input `serde_json::Value`.
/// The application must enable the `preserve_order` feature of `serde_json`
/// to keep declared order. Without it the order is alphabetical. Whitespace runs between tokens and at both ends stay within
/// `max_whitespace`. A schema that matches nothing is an `EmptyLanguage` error.
/// A keyword the grammar cannot enforce is an `Unsupported` error in strict mode.
pub fn json_schema_to_grammar(
    schema: &Value,
    opts: &SchemaOptions,
) -> Result<Grammar, GrammarError> {
    let mut walker = Walker::new(schema, opts);
    let root = walker.schema(schema, "#")?;
    walker.finish(root)
}

/// `json_schema_to_grammar` over schema text.
///
/// Text that is not JSON is a `Syntax` error at its line and column.
/// The caller must cap the text length first. Property order follows the key
/// order of the parsed text when the application enables the `preserve_order`
/// feature of `serde_json`. Without it the order is alphabetical.
pub fn json_schema_text_to_grammar(
    text: &str,
    opts: &SchemaOptions,
) -> Result<Grammar, GrammarError> {
    let schema: Value = serde_json::from_str(text).map_err(|err| {
        let span = Span {
            line: u32::try_from(err.line()).unwrap_or(u32::MAX),
            column: u32::try_from(err.column()).unwrap_or(u32::MAX),
        };
        GrammarError::at(
            GrammarErrorKind::Syntax,
            span,
            format!("schema text is not valid JSON: {err}. Correct the JSON text."),
        )
    })?;
    json_schema_to_grammar(&schema, opts)
}

/// The grammar of any JSON object, for `json_object` mode.
///
/// Values nest at most `max_value_depth` containers deep, the object included.
/// A `max_value_depth` of 0 allows no object and is an `EmptyLanguage` error.
pub fn json_object_grammar(opts: &SchemaOptions) -> Result<Grammar, GrammarError> {
    if opts.max_value_depth == 0 {
        return Err(GrammarError::new(
            GrammarErrorKind::EmptyLanguage,
            None,
            "json_object mode needs max_value_depth 1 or more, and it is 0. \
             Raise max_value_depth.",
        ));
    }
    let root = Value::Null;
    let mut walker = Walker::new(&root, opts);
    let object = generic::object(&mut walker, opts.max_value_depth);
    walker.finish(Lang::Expr(object))
}

/// `json_schema_to_grammar`, then `Grammar::compile` under `limits`.
///
/// Property order follows the key order of the input `serde_json::Value`.
/// The application must enable the `preserve_order` feature of `serde_json`
/// to keep declared order. Without it the order is alphabetical.
pub fn compile_json_schema(
    schema: &Value,
    opts: &SchemaOptions,
    limits: &CompileLimits,
) -> Result<ByteDfa, GrammarError> {
    limits.automaton.check()?;
    json_schema_to_grammar(schema, opts)?.compile(limits)
}

/// The schema inside an OpenAI `json_schema` wrapper, or `value` itself.
///
/// The wrapper is `{"name": .., "schema": {..}, "strict": ..}`. A value is a
/// wrapper when its `schema` member is an object and it has none of `type`,
/// `properties`, `$ref`, `enum`, `const`, `anyOf` and `oneOf`.
pub fn unwrap_response_format(value: &Value) -> &Value {
    let Value::Object(map) = value else {
        return value;
    };
    if SCHEMA_KEYS.iter().any(|key| map.contains_key(*key)) {
        return value;
    }
    match map.get("schema") {
        Some(inner @ Value::Object(_)) => inner,
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::json_schema::fixture::{check, compiled, dfa, opts};
    use crate::inference::grammar::json_schema::instances::{self, Inst};
    use crate::inference::grammar::reference::Lcg;
    use serde::Serialize;
    use serde_json::{Map, json};

    /// `inst` with every whitespace run at most one character.
    fn pretty(inst: &Inst) -> String {
        if instances::has_container_atom(inst) {
            return instances::write(inst, &mut || "\n".to_string());
        }
        let mut out = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(b"");
        let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
        instances::to_value(inst).serialize(&mut ser).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn random_schemas_accept_valid_and_reject_mutated_instances() {
        let o = SchemaOptions {
            max_value_depth: 1,
            ..opts()
        };
        let mut rng = Lcg(0x5eed_1234);
        let mut cases = 0;
        for _ in 0..160 {
            let schema = instances::schema(&mut rng, 3);
            let d = compiled(compile_json_schema(&schema, &o, &CompileLimits::default()));
            for _ in 0..2 {
                let inst = instances::instance(&mut rng, &schema);
                let mut spaces = Lcg(rng.next_u64());
                let mut run = || {
                    let n = spaces.below(3);
                    (0..n)
                        .map(|_| spaces.pick(&[' ', '\t', '\n']))
                        .collect::<String>()
                };
                let texts = [
                    serde_json::to_string(&instances::to_value(&inst)).unwrap(),
                    instances::write(&inst, &mut run),
                    pretty(&inst),
                ];
                for text in &texts {
                    assert!(d.accepts(text.as_bytes()), "{schema}\nmust accept {text:?}");
                }
                let bad = instances::mutate(&mut rng, &schema, &inst);
                assert!(!d.accepts(bad.as_bytes()), "{schema}\nmust reject {bad:?}");
                cases += 1;
            }
        }
        assert!(cases >= 300, "{cases}");
    }

    #[test]
    fn thirty_optional_properties_of_mixed_types_stay_small() {
        let kinds = [
            r#"{"type": "string"}"#,
            r#"{"type": "integer"}"#,
            r#"{"type": "number"}"#,
            r#"{"type": "boolean"}"#,
            r#"{"type": "null"}"#,
            r#"{"enum": ["red", "green"]}"#,
            r#"{"type": "array", "items": {"type": "integer"}}"#,
        ];
        let props: Vec<String> = (0..30)
            .map(|i| format!(r#""field_{i:02}": {}"#, kinds[i % kinds.len()]))
            .collect();
        let schema = format!(
            r#"{{"type": "object", "properties": {{{}}}}}"#,
            props.join(", ")
        );
        let d = dfa(&schema, &opts());
        assert!(d.num_states() < 20_000, "{} states", d.num_states());
        check(
            &d,
            &[
                "{}",
                r#"{"field_00": "a", "field_05": "red", "field_24": true, "field_29": -3}"#,
                r#"{"field_13": [1, 2], "field_14": "x"}"#,
            ],
            &[
                r#"{"field_24": true, "field_00": "a"}"#,
                r#"{"field_30": 1}"#,
                r#"{"field_29": 1.5}"#,
            ],
        );
    }

    #[test]
    fn twelve_required_strings_stay_small() {
        let props: Vec<String> = (0..12)
            .map(|i| format!(r#""name_{i:02}": {{"type": "string"}}"#))
            .collect();
        let names: Vec<String> = (0..12).map(|i| format!(r#""name_{i:02}""#)).collect();
        let schema = format!(
            r#"{{"type": "object", "properties": {{{}}}, "required": [{}]}}"#,
            props.join(", "),
            names.join(", ")
        );
        let d = dfa(&schema, &opts());
        assert!(d.num_states() < 2_000, "{} states", d.num_states());
    }

    /// Drop a deep schema one level at a time, so drop never recurses deep.
    fn dismantle(mut value: Value) {
        loop {
            let next = value
                .as_object_mut()
                .and_then(|map| map.remove("properties"))
                .and_then(|mut props| props.as_object_mut().and_then(|p| p.remove("a")));
            match next {
                Some(inner) => value = inner,
                None => break,
            }
        }
    }

    #[test]
    fn a_hostile_nesting_depth_is_a_limit_error_without_stack_overflow() {
        let worker = std::thread::spawn(|| {
            let mut schema = json!({"type": "string"});
            for _ in 0..10_000 {
                let mut props = Map::new();
                props.insert("a".into(), schema);
                let mut map = Map::new();
                map.insert("type".into(), json!("object"));
                map.insert("properties".into(), Value::Object(props));
                schema = Value::Object(map);
            }
            let err = compile_json_schema(&schema, &opts(), &CompileLimits::default()).err();
            dismantle(schema);
            err.map(|e| (e.kind(), e.message().to_string()))
        });
        let err = worker.join().ok().flatten();
        let (kind, message) = err.unwrap_or_else(|| panic!("deep schema must fail"));
        assert_eq!(kind, GrammarErrorKind::LimitExceeded);
        assert!(message.contains("128 levels"), "{message}");
    }

    #[test]
    fn exponential_ref_reuse_stops_at_the_node_limit() {
        let mut defs = Map::new();
        for i in 0..40 {
            let next = format!("#/$defs/d{}", i + 1);
            defs.insert(
                format!("d{i}"),
                json!({"type": "array", "prefixItems": [{"$ref": next}, {"$ref": next}], "items": false}),
            );
        }
        defs.insert("d40".into(), json!({"type": "integer"}));
        let schema = json!({"$ref": "#/$defs/d0", "$defs": defs});
        let err = compile_json_schema(&schema, &opts(), &CompileLimits::default()).err();
        assert_eq!(err.map(|e| e.kind()), Some(GrammarErrorKind::LimitExceeded));
    }

    #[test]
    fn schema_text_parses_or_reports_its_position() {
        let grammar = json_schema_text_to_grammar(r#"{"type": "boolean"}"#, &opts());
        let d = compiled(grammar.and_then(|g| g.compile(&CompileLimits::default())));
        assert!(d.accepts(b"false"));
        let err = json_schema_text_to_grammar("{\n  \"type\": }", &opts()).err();
        let err = err.unwrap_or_else(|| panic!("bad JSON must fail"));
        assert_eq!(err.kind(), GrammarErrorKind::Syntax);
        assert_eq!(err.span().map(|s| s.line), Some(2));
    }

    #[test]
    fn response_format_wrappers_unwrap_to_their_schema() {
        let inner = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        let wrapper = json!({"name": "answer", "schema": inner, "strict": true});
        assert_eq!(unwrap_response_format(&wrapper), &inner);
        assert_eq!(unwrap_response_format(&inner), &inner);
        let property_named_schema =
            json!({"properties": {"schema": {}}, "schema": {"type": "null"}});
        assert_eq!(
            unwrap_response_format(&property_named_schema),
            &property_named_schema
        );
        let not_object = json!({"schema": "text"});
        assert_eq!(unwrap_response_format(&not_object), &not_object);
        assert_eq!(unwrap_response_format(&json!(true)), &json!(true));
    }

    #[test]
    fn the_grammar_root_wraps_the_schema_in_whitespace() {
        let grammar = json_schema_to_grammar(&json!({"type": "null"}), &opts());
        let grammar = grammar.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(grammar.rules[0].name, "root");
        assert!(grammar.rule("ws").is_some());
        let compact = SchemaOptions {
            max_whitespace: 0,
            ..opts()
        };
        let grammar = json_schema_to_grammar(&json!({"type": "null"}), &compact);
        assert!(grammar.ok().and_then(|g| g.rule("ws").cloned()).is_none());
    }
}
