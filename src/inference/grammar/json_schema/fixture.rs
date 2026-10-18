//! Shared test helpers: compile a schema and check tables of instances.

use super::{SchemaOptions, compile_json_schema};
use crate::inference::grammar::{ByteDfa, CompileLimits, GrammarError};

/// The default config.
pub(super) fn opts() -> SchemaOptions {
    SchemaOptions::default()
}

/// The DFA in `result`. An error fails the test with its message.
pub(super) fn compiled(result: Result<ByteDfa, GrammarError>) -> ByteDfa {
    match result {
        Ok(dfa) => dfa,
        Err(err) => panic!("compile error: {err}"),
    }
}

/// The schema text `schema` parsed as JSON.
pub(super) fn parse(schema: &str) -> serde_json::Value {
    match serde_json::from_str(schema) {
        Ok(value) => value,
        Err(err) => panic!("test schema is not JSON: {err}: {schema}"),
    }
}

/// The DFA of the schema text `schema` under the default `CompileLimits`.
pub(super) fn dfa(schema: &str, opts: &SchemaOptions) -> ByteDfa {
    match compile_json_schema(&parse(schema), opts, &CompileLimits::default()) {
        Ok(dfa) => dfa,
        Err(err) => panic!("{err}: {schema}"),
    }
}

/// The compile error of the schema text `schema`. Success fails the test.
pub(super) fn error(schema: &str, opts: &SchemaOptions) -> GrammarError {
    match compile_json_schema(&parse(schema), opts, &CompileLimits::default()) {
        Ok(dfa) => panic!("{schema} must fail, got {} states", dfa.num_states()),
        Err(err) => err,
    }
}

/// Every text in `accept` must match, and every text in `reject` must not.
pub(super) fn check(dfa: &ByteDfa, accept: &[&str], reject: &[&str]) {
    for text in accept {
        assert!(dfa.accepts(text.as_bytes()), "must accept {text:?}");
    }
    for text in reject {
        assert!(!dfa.accepts(text.as_bytes()), "must reject {text:?}");
    }
}

/// The message of the `LimitExceeded` error that lowering `schema` returns.
pub(super) fn limit_message(schema: &serde_json::Value) -> String {
    match super::json_schema_to_grammar(schema, &opts()) {
        Ok(_) => panic!("the schema must exceed a limit"),
        Err(err) => {
            assert_eq!(
                err.kind(),
                crate::inference::grammar::GrammarErrorKind::LimitExceeded
            );
            err.message().to_string()
        }
    }
}

/// The grammar of `schema`. An error fails the test with its message.
pub(super) fn lowered(schema: &serde_json::Value) -> crate::inference::grammar::Grammar {
    match super::json_schema_to_grammar(schema, &opts()) {
        Ok(grammar) => grammar,
        Err(err) => panic!("lowering error: {err}"),
    }
}
