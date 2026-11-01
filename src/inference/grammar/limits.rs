//! Limits that parsing and expansion enforce on untrusted grammar text.

/// Limits that parsing and expansion enforce.
///
/// The limits bound memory and work for any input. A caller that reads a
/// grammar from a request must still cap the request body before it parses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpandLimits {
    /// Maximum length of the grammar text in bytes.
    /// Parsing checks it before it allocates anything.
    pub max_input_bytes: usize,
    /// Maximum weighted node count of the expanded root expression.
    ///
    /// Each node counts its `Expr::weight`, so a large literal or class
    /// counts by size. Every clone of a rule counts again. One unit stands
    /// for one tree node plus at most 16 heap bytes.
    pub max_nodes: usize,
    /// Maximum `n` or `m` in a `{n,m}` repeat.
    pub max_repeat: u32,
    /// Maximum repeat product: the unrolled size of one repeat in context.
    ///
    /// A repeat has the factor `m` for `{n,m}`, `n` for `{n,}`, and 1 for
    /// `*`, `+` and `?`. A factor of 0 counts as 1. The product of a repeat
    /// is its factor times the factors of every enclosing repeat, across rule
    /// references. Its unrolled body size counts every node by `Expr::weight`
    /// and multiplies each nested repeat body by that repeat's factor.
    /// Expansion stops when a product above 1 times the unrolled body size
    /// exceeds this limit.
    ///
    /// After a successful expansion, every repeat with a product above 1
    /// unrolls, with all its copies in context, to at most this many units.
    /// A repeat with product 1 copies nothing, and `max_nodes` bounds it.
    pub max_repeat_product: u64,
    /// Maximum nesting of `( ... )` groups in the source.
    /// A value far above the default can exhaust the stack.
    pub max_group_depth: u32,
    /// Maximum expansion depth: nested expressions plus rule references.
    ///
    /// Expansion keeps its walk on the heap at any depth. The expanded tree
    /// can nest this deep. `Expr` clones, compares, formats and drops on the
    /// heap, and NFA construction does too. So this limit bounds work, not stack.
    pub max_expand_depth: u32,
}

impl Default for ExpandLimits {
    /// `max_nodes` 500 000 lets a large JSON-schema grammar pass.
    /// Short literals and classes weigh 2 to 5 units in such a grammar.
    fn default() -> Self {
        Self {
            max_input_bytes: 1 << 20,
            max_nodes: 500_000,
            max_repeat: 1_000,
            max_repeat_product: 100_000,
            max_group_depth: 256,
            max_expand_depth: 1_024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_match_the_documented_values() {
        let limits = ExpandLimits::default();
        assert_eq!(limits.max_input_bytes, 1_048_576);
        assert_eq!(limits.max_nodes, 500_000);
        assert_eq!(limits.max_repeat, 1_000);
        assert_eq!(limits.max_repeat_product, 100_000);
        assert_eq!(limits.max_group_depth, 256);
        assert_eq!(limits.max_expand_depth, 1_024);
    }
}
