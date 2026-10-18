//! JSON Schema to grammar compiler for constrained JSON generation.
//!
//! `json_schema_to_grammar` builds a `Grammar` AST whose language is the JSON
//! texts valid for a schema. `compile_json_schema` lowers it to a `ByteDfa`.
//!
//! Property order follows the key order of the input `serde_json::Value`. The
//! application must enable the `preserve_order` feature of `serde_json` to keep
//! declared order. Without it the order is alphabetical.

mod api;
mod array;
#[cfg(test)]
mod fixture;
mod generic;
#[cfg(test)]
mod instances;
mod keywords;
mod limits;
mod object;
mod options;
mod refs;
mod scalar;
mod string;
mod union;
mod walk;

pub use api::{
    compile_json_schema, json_object_grammar, json_schema_text_to_grammar, json_schema_to_grammar,
    unwrap_response_format,
};
pub use keywords::{ANNOTATION_KEYWORDS, UNSUPPORTED_KEYWORDS};
pub use options::SchemaOptions;
