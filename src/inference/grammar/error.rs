//! Errors from parsing and expanding a GBNF grammar.

use std::borrow::Cow;
use std::fmt;

use super::ast::{ROOT_RULE, Span};

/// The class of a grammar error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GrammarErrorKind {
    /// The text does not follow GBNF syntax.
    Syntax,
    /// A rule reference names no defined rule.
    UndefinedRule,
    /// Two rules share one name.
    DuplicateRule,
    /// The grammar defines no `root` rule.
    MissingRoot,
    /// A rule refers back to itself, directly or through other rules.
    Cycle,
    /// The text uses GBNF syntax outside the accepted subset.
    Unsupported,
    /// The grammar exceeds a node, depth or repeat limit.
    LimitExceeded,
}

/// An error in a GBNF grammar, with its source position when known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarError {
    kind: GrammarErrorKind,
    message: String,
    span: Option<Span>,
}

impl GrammarError {
    /// An error with an optional position.
    pub(crate) fn new(
        kind: GrammarErrorKind,
        span: Option<Span>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            span,
        }
    }

    /// An error at a known position.
    pub(crate) fn at(kind: GrammarErrorKind, span: Span, message: impl Into<String>) -> Self {
        Self::new(kind, Some(span), message)
    }

    /// A syntax error at a known position.
    pub(crate) fn syntax(span: Span, message: impl Into<String>) -> Self {
        Self::at(GrammarErrorKind::Syntax, span, message)
    }

    /// The grammar defines no `root` rule.
    pub(crate) fn missing_root() -> Self {
        Self::new(
            GrammarErrorKind::MissingRoot,
            None,
            format!(
                "grammar has no `{ROOT_RULE}` rule. \
                 Define `{ROOT_RULE} ::= ...` as the start rule."
            ),
        )
    }

    /// Rule `name` at `span` repeats the definition at `first`.
    pub(crate) fn duplicate_rule(name: &str, span: Span, first: Span) -> Self {
        Self::at(
            GrammarErrorKind::DuplicateRule,
            span,
            format!(
                "rule `{}` is already defined at {first}. Merge both bodies with `|`.",
                excerpt(name)
            ),
        )
    }

    /// Rule `from` refers at `span` to `name`, which no rule defines.
    pub(crate) fn undefined(name: &str, from: &str, span: Span) -> Self {
        let (name, from) = (excerpt(name), excerpt(from));
        Self::at(
            GrammarErrorKind::UndefinedRule,
            span,
            format!(
                "rule `{name}` is not defined. Rule `{from}` refers to it. \
                 Define `{name} ::= ...` or correct the name."
            ),
        )
    }

    /// The rules in `chain` refer to each other in a loop, closed at `span`.
    pub(crate) fn cycle(chain: &[&str], span: Span) -> Self {
        Self::at(
            GrammarErrorKind::Cycle,
            span,
            format!(
                "rule cycle: {}. Recursive rules are outside the regular subset. \
                 Write repetition with `*`, `+` or `{{n,m}}`.",
                cycle_path(chain)
            ),
        )
    }

    /// The error class.
    pub fn kind(&self) -> GrammarErrorKind {
        self.kind
    }

    /// The message: what is wrong, the offending input and the next action.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The source position, when the error has one.
    pub fn span(&self) -> Option<Span> {
        self.span
    }
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.span {
            Some(span) => write!(f, "grammar error at {span}: {}", self.message),
            None => write!(f, "grammar error: {}", self.message),
        }
    }
}

impl std::error::Error for GrammarError {}

/// Longest run of characters a message quotes from the input.
pub(crate) const EXCERPT_CHARS: usize = 64;

/// Most rule names a cycle message lists.
pub(crate) const CYCLE_NAMES: usize = 16;

/// `text` cut to `EXCERPT_CHARS` characters, with `...` after a cut.
///
/// Every message that quotes input text quotes it through this function.
/// A message then stays short for any input size.
pub(crate) fn excerpt(text: &str) -> Cow<'_, str> {
    match text.char_indices().nth(EXCERPT_CHARS) {
        Some((end, _)) => Cow::Owned(format!("{}...", text.get(..end).unwrap_or_default())),
        None => Cow::Borrowed(text),
    }
}

/// The first `CYCLE_NAMES` names of `chain` joined by ` -> `, then `... (N more)`.
fn cycle_path(chain: &[&str]) -> String {
    let shown: Vec<Cow<'_, str>> = chain.iter().take(CYCLE_NAMES).map(|n| excerpt(n)).collect();
    let path = shown.join(" -> ");
    match chain.len().saturating_sub(CYCLE_NAMES) {
        0 => path,
        rest => format!("{path} -> ... ({rest} more)"),
    }
}

impl From<GrammarError> for crate::error::Error {
    fn from(err: GrammarError) -> Self {
        crate::error::Error::InvalidArgument {
            arg: "grammar",
            reason: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_shows_position_and_message() {
        let err = GrammarError::syntax(Span { line: 2, column: 5 }, "unexpected `)`.");
        assert_eq!(err.to_string(), "grammar error at 2:5: unexpected `)`.");
        let err = GrammarError::new(GrammarErrorKind::MissingRoot, None, "no root.");
        assert_eq!(err.to_string(), "grammar error: no root.");
    }

    #[test]
    fn excerpt_cuts_long_text_at_a_char_boundary() {
        assert_eq!(excerpt("abc"), "abc");
        let exact = "x".repeat(EXCERPT_CHARS);
        assert_eq!(excerpt(&exact), exact.as_str());
        let long = "é".repeat(1_000);
        let cut = excerpt(&long);
        assert_eq!(cut.chars().count(), EXCERPT_CHARS + 3);
        assert!(cut.ends_with("é..."), "{cut}");
    }

    #[test]
    fn cycle_path_lists_the_first_names_then_a_count() {
        assert_eq!(cycle_path(&["a", "b", "a"]), "a -> b -> a");
        let names: Vec<String> = (0..20).map(|i| format!("r{i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let path = cycle_path(&refs);
        assert!(path.starts_with("r0 -> r1 -> "), "{path}");
        assert!(path.ends_with("r15 -> ... (4 more)"), "{path}");
        let exact = cycle_path(refs.get(..CYCLE_NAMES).unwrap_or_default());
        assert!(exact.ends_with("r14 -> r15"), "{exact}");
    }

    #[test]
    fn messages_quote_at_most_an_excerpt_of_a_name() {
        let name = "n".repeat(1_000_000);
        let at = Span { line: 1, column: 1 };
        for err in [
            GrammarError::undefined(&name, &name, at),
            GrammarError::duplicate_rule(&name, at, at),
            GrammarError::cycle(&[name.as_str(); 5_000], at),
        ] {
            assert!(err.message().len() < 2_000, "{}", err.message().len());
        }
    }

    #[test]
    fn converts_to_crate_error() {
        let err = GrammarError::syntax(Span { line: 1, column: 1 }, "bad.");
        let converted: crate::error::Error = err.into();
        match converted {
            crate::error::Error::InvalidArgument { arg, reason } => {
                assert_eq!(arg, "grammar");
                assert_eq!(reason, "grammar error at 1:1: bad.");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }
}
