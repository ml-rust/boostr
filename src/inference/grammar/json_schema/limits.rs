//! Member-count limit for schema keywords that list members.

use super::walk::limit;
use crate::inference::grammar::error::GrammarError;

/// Most members one keyword can list: `properties`, `required`, `prefixItems`,
/// an `items` array, `enum`, `anyOf` and `oneOf`.
///
/// Optional properties and optional tuple items nest one `Expr` level each,
/// and `Expr` clone, drop and `node_count` recurse. This limit keeps that
/// depth far below the stack limit. It also bounds the quadratic `Seq`
/// flattening in `object::body` and `array::tuple` to about 1e6 moves.
pub(super) const MAX_MEMBERS: usize = 1_000;

/// An error when the keyword `keyword` at the pointer `at` lists more than `MAX_MEMBERS`.
pub(super) fn check_members(keyword: &str, at: &str, count: usize) -> Result<(), GrammarError> {
    if count > MAX_MEMBERS {
        return Err(limit(format!(
            "`{keyword}` at `{at}` lists {count} members, and the limit is {MAX_MEMBERS}. \
             Split the schema or remove members."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_limit_is_inclusive_and_the_message_names_keyword_and_path() {
        assert!(check_members("enum", "#/enum", MAX_MEMBERS).is_ok());
        let err = check_members("enum", "#/enum", MAX_MEMBERS + 1).err();
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(msg.contains("`enum` at `#/enum`"), "{msg}");
        assert!(msg.contains("1001 members"), "{msg}");
        assert!(msg.contains("limit is 1000"), "{msg}");
    }
}
