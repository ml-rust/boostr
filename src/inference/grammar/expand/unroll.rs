//! Repeat expansion: bound checks and the repeat product.
//!
//! A repeat has a factor: `m` for `{n,m}`, `n` for `{n,}`, and 1 for `*`,
//! `+` and `?`. The product of a repeat multiplies its factor with the factors
//! of every enclosing repeat. A product above 1 times the unrolled body size
//! must stay within `max_repeat_product`.

use crate::inference::grammar::ast::{Expr, repeat_text};
use crate::inference::grammar::error::{GrammarError, excerpt};

use super::expander::{Expanded, Expander};

impl Expander<'_> {
    /// Expand `body{min,max}` in rule `owner` and check its repeat product.
    ///
    /// The check runs once before the body, on the product alone, and once after it.
    pub(super) fn repeat(
        &mut self,
        body: &Expr,
        min: u32,
        max: Option<u32>,
        owner: usize,
    ) -> Result<Expanded, GrammarError> {
        self.check_repeat(min, max, owner)?;
        let factor = u64::from(max.unwrap_or(min).max(1));
        let outer = self.product;
        let product = outer.saturating_mul(factor);
        // The body weighs at least 1, so the product alone can exceed the limit.
        self.check_product(product, product, (min, max), owner)?;
        self.product = product;
        let (body, size) = self.expr(body, owner)?;
        self.product = outer;
        self.check_product(product, product.saturating_mul(size), (min, max), owner)?;
        let unrolled = factor.saturating_mul(size).saturating_add(1);
        Ok((Expr::repeat(body, min, max), unrolled))
    }

    /// Check `{min,max}` for order and against `max_repeat`.
    fn check_repeat(&self, min: u32, max: Option<u32>, owner: usize) -> Result<(), GrammarError> {
        let text = repeat_text(min, max);
        if max.is_some_and(|max| max < min) {
            return Err(GrammarError::syntax(
                self.owner_span(owner),
                format!(
                    "rule `{}` has repeat `{text}` with a maximum below its minimum. \
                     Write `{{n,m}}` with n <= m.",
                    excerpt(self.owner_name(owner))
                ),
            ));
        }
        let limit = self.limits.max_repeat;
        if min > limit || max.is_some_and(|max| max > limit) {
            return Err(Self::limit(
                self.owner_span(owner),
                format!(
                    "rule `{}` has repeat `{text}` above max_repeat {limit}. \
                     Use `*` or `+` for an unbounded count.",
                    excerpt(self.owner_name(owner))
                ),
            ));
        }
        Ok(())
    }

    /// Stop when a repeat with `product` above 1 unrolls to more units than the limit.
    fn check_product(
        &self,
        product: u64,
        units: u64,
        (min, max): (u32, Option<u32>),
        owner: usize,
    ) -> Result<(), GrammarError> {
        let limit = self.limits.max_repeat_product;
        if product <= 1 || units <= limit {
            return Ok(());
        }
        Err(Self::limit(
            self.owner_span(owner),
            format!(
                "rule `{}` has repeat `{}` that unrolls to at least {units} units in context, \
                 above max_repeat_product {limit}. Lower the nested repeat counts or use `*`.",
                excerpt(self.owner_name(owner)),
                repeat_text(min, max)
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::grammar::{ExpandLimits, Expr, Grammar, GrammarError, GrammarErrorKind};

    fn expand_with(src: &str, limits: &ExpandLimits) -> Result<Expr, GrammarError> {
        Grammar::parse(src)?.expand(limits)
    }

    fn product_limit(src: &str, max_repeat_product: u64) -> Result<Expr, GrammarError> {
        let limits = ExpandLimits {
            max_repeat_product,
            ..ExpandLimits::default()
        };
        expand_with(src, &limits)
    }

    #[test]
    fn nested_repeat_counts_multiply_into_the_product() {
        let src = "root ::= ((\"a\"{1000}){1000}){1000}";
        let err = expand_with(src, &ExpandLimits::default()).err();
        let kind = err.as_ref().map(GrammarError::kind);
        assert_eq!(kind, Some(GrammarErrorKind::LimitExceeded));
        let span = err.as_ref().and_then(GrammarError::span);
        assert_eq!(span.map(|s| (s.line, s.column)), Some((1, 1)));
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(msg.contains("max_repeat_product 100000"), "{msg}");
        assert!(msg.contains("`{1000}`"), "{msg}");
        let ok = expand_with("root ::= ([0-9]{1,3}){1,10}", &ExpandLimits::default());
        assert!(ok.is_ok(), "{ok:?}");
    }

    /// Check that `units` is the smallest `max_repeat_product` that accepts `src`.
    fn assert_product(src: &str, units: u64) {
        assert!(product_limit(src, units).is_ok(), "{src} at {units}");
        let err = product_limit(src, units - 1).err();
        let kind = err.as_ref().map(GrammarError::kind);
        assert_eq!(kind, Some(GrammarErrorKind::LimitExceeded), "{src}");
        let want = format!("max_repeat_product {}", units - 1);
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(msg.contains(&want), "{src}: {msg}");
    }

    #[test]
    fn repeat_product_counts_the_unrolled_body() {
        // `"a"` weighs 2, so ten copies weigh 20.
        assert_product("root ::= \"a\"{10}", 20);
        assert_product("root ::= \"a\"{3,}", 6);
        assert_product("root ::= \"a\"{2,5}", 10);
        // Body sizes 2, 1 + 2 * 2 and 1 + 3 * 5 under products 24, 12 and 4.
        assert_product("root ::= ((\"a\"{2}){3}){4}", 64);
        // `d` unrolls to 1 + 10 * 2, and its clone under `{10}` still counts.
        assert_product("root ::= d (d){10}\nd ::= \"a\"{10}", 210);
    }

    #[test]
    fn a_factor_of_one_copies_nothing_and_passes_any_product_limit() {
        for src in [
            "root ::= (\"abc\" [a-z])?",
            "root ::= (\"abc\")*",
            "root ::= [a-z]+",
            "root ::= \"a\"{0,1}",
            "root ::= \"a\"{0}",
        ] {
            assert!(product_limit(src, 0).is_ok(), "{src}");
        }
    }

    #[test]
    fn repeat_limit_applies_at_expansion() {
        let limits = ExpandLimits {
            max_repeat: 10,
            ..ExpandLimits::default()
        };
        let err = expand_with("root ::= \"a\"{20}", &limits).err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::LimitExceeded)
        );
        let span = err.as_ref().and_then(GrammarError::span);
        assert_eq!(span.map(|s| (s.line, s.column)), Some((1, 1)));
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(msg.contains("`{20}`"), "{msg}");
    }
}
