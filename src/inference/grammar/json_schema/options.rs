//! Config for the JSON-schema compiler.

/// Config for `json_schema_to_grammar` and `json_object_grammar`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchemaOptions {
    /// Maximum nesting of one recursive `$ref` target. Default 6.
    ///
    /// The schema root counts as one level of `#`. A branch that nests deeper
    /// leaves the language. A `$ref` that is not on its own resolution path
    /// costs no depth.
    pub max_depth: u32,
    /// Maximum container nesting of an any-value position. Default 4.
    ///
    /// It bounds `{}`, `true`, a schema with no type keyword, an
    /// `additionalProperties` of `true`, and `json_object_grammar`. The
    /// innermost level allows only scalars. The automaton grows about 4 times
    /// per level, so a value far above the default exceeds the default
    /// `CompileLimits`.
    pub max_value_depth: u32,
    /// Maximum whitespace run between tokens and at both ends of the text.
    ///
    /// A run holds space, `\t` and `\n` only. 0 gives compact JSON. Default 2.
    pub max_whitespace: u32,
    /// Policy for a validation keyword that a grammar cannot enforce. Default true.
    ///
    /// True makes such a keyword an `Unsupported` error. False ignores it,
    /// so the language becomes a superset of the schema.
    pub strict_keywords: bool,
    /// Extra keys for an object schema with no `additionalProperties`. Default false.
    ///
    /// False allows no extra key. True allows extra keys with any value after
    /// the declared properties.
    pub additional_properties_default: bool,
}

impl Default for SchemaOptions {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_value_depth: 4,
            max_whitespace: 2,
            strict_keywords: true,
            additional_properties_default: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_match_the_documented_values() {
        let opts = SchemaOptions::default();
        assert_eq!(opts.max_depth, 6);
        assert_eq!(opts.max_value_depth, 4);
        assert_eq!(opts.max_whitespace, 2);
        assert!(opts.strict_keywords);
        assert!(!opts.additional_properties_default);
    }
}
