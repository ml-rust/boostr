//! Member-count limit for schema keywords that list members, and the block
//! size that keeps long optional runs shallow.

use super::walk::limit;
use crate::inference::grammar::error::GrammarError;

/// Most members one keyword can list: `properties`, `required`, `prefixItems`,
/// an `items` array, `enum`, `anyOf` and `oneOf`.
///
/// This limit bounds the grammar size of one keyword.
pub(super) const MAX_MEMBERS: usize = 1_000;

/// Most optional members that one nested run holds.
///
/// Each optional property and each optional tuple item nests the tree two
/// `Expr` levels deeper. Past this count the builders split the members into
/// blocks and join the blocks in a second run. Tree depth then grows with the
/// block size plus the block count, not with the member count. At
/// `MAX_MEMBERS` both stay at most 32, so the expanded tree stays far below
/// the default `ExpandLimits::max_expand_depth`.
pub(super) const BLOCK: usize = 32;

const _: () = assert!(MAX_MEMBERS.div_ceil(BLOCK) <= BLOCK);

/// `items` in order, split into blocks of `BLOCK`. Only the last block can be shorter.
pub(super) fn blocks<T>(items: Vec<T>) -> Vec<Vec<T>> {
    let mut out = Vec::with_capacity(items.len().div_ceil(BLOCK));
    let mut block = Vec::with_capacity(BLOCK.min(items.len()));
    for item in items {
        block.push(item);
        if block.len() == BLOCK {
            out.push(std::mem::take(&mut block));
        }
    }
    if !block.is_empty() {
        out.push(block);
    }
    out
}

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

    #[test]
    fn blocks_keep_order_and_only_the_last_is_short() {
        let sizes =
            |n: usize| -> Vec<usize> { blocks((0..n).collect()).iter().map(Vec::len).collect() };
        assert!(sizes(0).is_empty());
        assert_eq!(sizes(BLOCK), [BLOCK]);
        assert_eq!(sizes(2 * BLOCK + 1), [BLOCK, BLOCK, 1]);
        assert_eq!(sizes(MAX_MEMBERS).len(), MAX_MEMBERS.div_ceil(BLOCK));
        let flat: Vec<usize> = blocks((0..70).collect()).into_iter().flatten().collect();
        assert_eq!(flat, (0..70).collect::<Vec<_>>());
    }
}
