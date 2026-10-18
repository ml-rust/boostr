//! The lowering pipeline: `Expr` to NFA, DFA, minimal DFA, then `ByteDfa`.

use super::ast::{Expr, Grammar};
use super::automaton_limits::{AutomatonLimits, CompileLimits};
use super::dfa::ByteDfa;
use super::error::GrammarError;
use super::minimize::minimize;
use super::nfa::Nfa;
use super::subset::determinize;

impl ByteDfa {
    /// The minimal trimmed DFA for the UTF-8 encodings of the strings `expr` matches.
    ///
    /// `expr` must hold no rule reference. A grammar that matches no string
    /// is an `EmptyLanguage` error. `limits` bounds every construction step.
    /// A `max_dfa_states` above `AutomatonLimits::max_dfa_states_ceiling` is a `LimitExceeded` error.
    pub fn from_expr(expr: &Expr, limits: &AutomatonLimits) -> Result<ByteDfa, GrammarError> {
        limits.check()?;
        let nfa = Nfa::build(expr, limits)?;
        let dfa = determinize(&nfa, limits)?;
        drop(nfa);
        ByteDfa::from_class_dfa(&minimize(&dfa)?)
    }
}

impl Grammar {
    /// Expand the grammar and lower its root to a minimal `ByteDfa`.
    pub fn compile(&self, limits: &CompileLimits) -> Result<ByteDfa, GrammarError> {
        limits.automaton.check()?;
        let root = self.expand(&limits.expand)?;
        ByteDfa::from_expr(&root, &limits.automaton)
    }
}

/// Parse GBNF text, expand it and lower it to a minimal `ByteDfa`.
///
/// Any input returns a DFA or an error, never a panic. `limits` bounds memory and work.
pub fn compile(src: &str, limits: &CompileLimits) -> Result<ByteDfa, GrammarError> {
    limits.automaton.check()?;
    Grammar::parse_with_limits(src, &limits.expand)?.compile(limits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::reference::{Lcg, matches, random_expr, test_strings};
    use crate::inference::grammar::{ExpandLimits, GrammarErrorKind};

    fn dfa(src: &str) -> ByteDfa {
        match compile(src, &CompileLimits::default()) {
            Ok(d) => d,
            Err(err) => panic!("{src}: {err}"),
        }
    }

    fn with(automaton: AutomatonLimits) -> CompileLimits {
        CompileLimits {
            automaton,
            ..CompileLimits::default()
        }
    }

    fn error(src: &str, limits: &CompileLimits) -> GrammarError {
        match compile(src, limits) {
            Ok(d) => panic!("{src} must fail, got {} states", d.num_states()),
            Err(err) => err,
        }
    }

    fn check(d: &ByteDfa, yes: &[&str], no: &[&str]) {
        for s in yes {
            assert!(d.accepts(s.as_bytes()), "must accept {s:?}");
        }
        for s in no {
            assert!(!d.accepts(s.as_bytes()), "must reject {s:?}");
        }
    }

    fn unminimized(expr: &Expr) -> Result<ByteDfa, GrammarError> {
        let limits = AutomatonLimits::default();
        let nfa = Nfa::build(expr, &limits)?;
        ByteDfa::from_class_dfa(&determinize(&nfa, &limits)?)
    }

    /// True when some byte string leads from `state` to an accepting state.
    fn reaches_accepting(d: &ByteDfa, state: u32) -> bool {
        let mut seen = vec![false; d.num_states()];
        let mut stack = vec![state];
        while let Some(s) = stack.pop() {
            if d.is_accepting(s) {
                return true;
            }
            for b in 0..=255u8 {
                if let Some(t) = d.step(s, b)
                    && !seen[t as usize]
                {
                    seen[t as usize] = true;
                    stack.push(t);
                }
            }
        }
        false
    }

    fn utf8(s: &[char]) -> Vec<u8> {
        s.iter().collect::<String>().into_bytes()
    }

    #[test]
    fn fixed_classes_compile_to_a_chain() {
        let d = dfa("root ::= [12][34][56]");
        assert_eq!(d.num_states(), 4);
        check(
            &d,
            &["135", "246", "145", "136"],
            &["", "13", "1356", "315", "137"],
        );
        assert_eq!(d.start(), 0);
        assert_eq!(d.walk(0, b"13"), d.step(d.step(0, b'1').unwrap_or(0), b'3'));
    }

    #[test]
    fn number_grammar_matches_json_numbers() {
        let d = dfa(r#"root ::= "-"? ("0" | [1-9][0-9]*) ("." [0-9]+)?"#);
        check(
            &d,
            &["0", "-0", "7", "120", "-3.25", "0.5", "10.0"],
            &["", "-", "01", "1.", ".5", "1.2.3", "--1", "+1", "1e5"],
        );
    }

    #[test]
    fn json_string_grammar_handles_escapes() {
        let d = dfa(r#"root ::= "\"" ([^"\\\x00-\x1f] | "\\" ["\\/bfnrt])* "\"""#);
        check(
            &d,
            &[
                r#""""#,
                r#""abc""#,
                r#""a\"b""#,
                r#""\\\/\n\t""#,
                "\"é世\u{1F600}\"",
            ],
            &["", "\"", r#""\x""#, "\"a\nb\"", r#""a"b""#, r#""\""#, "abc"],
        );
        assert!(!d.accepts(&[b'"', 0xC0, 0x80, b'"']));
        assert!(!d.accepts(&[b'"', 0xED, 0xA0, 0x80, b'"']));
    }

    #[test]
    fn any_star_accepts_all_utf8_and_nothing_else() {
        let d = dfa("root ::= .*");
        check(&d, &["", "abc", "é世\u{10FFFF}"], &[]);
        let bad: [&[u8]; 5] = [
            &[0x80],
            &[0xC3],
            &[0xF4, 0x90, 0x80, 0x80],
            &[0xFF],
            &[0xE0, 0x80, 0x80],
        ];
        for bad in bad {
            assert!(!d.accepts(bad), "{bad:x?}");
        }
        // The start, 1 to 3 pending continuation bytes, and the second bytes after E0, ED, F0 and F4.
        assert_eq!(d.num_states(), 8);
    }

    #[test]
    fn unicode_classes_and_bounded_repeats() {
        let d = dfa("root ::= [α-ω]+ [^a-z] [\\U0001F600-\\U0001F64F]{2,4}");
        check(
            &d,
            &["αβ1😀😃", "ω世😀😀😀😀"],
            &["αβa😀😀", "α1😀", "α1😀😀😀😀😀", "1😀😀"],
        );
        let d = dfa(r#"root ::= "ab"{2,4}"#);
        check(
            &d,
            &["abab", "ababab", "abababab"],
            &["ab", "ababababab", ""],
        );
    }

    #[test]
    fn empty_string_grammar_accepts_only_empty() {
        let d = dfa(r#"root ::= """#);
        assert_eq!(d.num_states(), 1);
        assert!(d.is_accepting(0));
        check(&d, &[""], &["a"]);
        assert!(d.transitions().iter().all(|&t| t == ByteDfa::NO_TRANSITION));
    }

    #[test]
    fn empty_language_is_an_error() {
        for src in [
            r"root ::= [^\x00-\U0010FFFF]",
            r#"root ::= "a" [^\x00-\U0010FFFF] | "b" [^\x00-\U0010FFFF]"#,
        ] {
            let err = error(src, &CompileLimits::default());
            assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage, "{src}");
            assert!(err.message().contains("matches no string"), "{err}");
            let converted: crate::error::Error = err.into();
            assert!(
                converted.to_string().contains("matches no string"),
                "{converted}"
            );
        }
        let kind = ByteDfa::from_expr(&Expr::Alt(vec![]), &AutomatonLimits::default())
            .err()
            .map(|e| e.kind());
        assert_eq!(kind, Some(GrammarErrorKind::EmptyLanguage));
    }

    #[test]
    fn dead_states_are_trimmed() {
        let d = dfa(r#"root ::= "ab" | "ac" | "x" [^\x00-\U0010FFFF]"#);
        assert_eq!(d.num_states(), 3);
        assert_eq!(d.step(0, b'x'), None);
        assert_eq!(d.step(0, b'b'), None);
        let mid = d.step(0, b'a');
        assert_eq!(mid.and_then(|s| d.step(s, b'd')), None);
        for s in 0..d.num_states() as u32 {
            assert!(d.is_viable(s));
            assert!(reaches_accepting(&d, s), "state {s}");
        }
        assert!(
            d.transitions()
                .iter()
                .all(|&t| t == ByteDfa::NO_TRANSITION || (t as usize) < d.num_states())
        );
    }

    #[test]
    fn duplicate_alternatives_give_the_minimal_size() {
        assert_eq!(
            dfa(r#"root ::= ("a" | "a")*"#).num_states(),
            dfa(r#"root ::= "a"*"#).num_states()
        );
        assert_eq!(dfa(r#"root ::= "a"*"#).num_states(), 1);
        assert_eq!(dfa(r#"root ::= ("ab")* | "ab" ("ab")*"#).num_states(), 2);
        assert_eq!(
            dfa("root ::= (\"a\"|\"b\")* \"a\" (\"a\"|\"b\"){3}").num_states(),
            16
        );
    }

    #[test]
    fn random_expressions_match_the_reference() {
        let mut rng = Lcg(0x5EED);
        for round in 0..420 {
            let expr = random_expr(&mut rng, 4);
            let strings = test_strings(&mut rng, 60);
            let minimal = ByteDfa::from_expr(&expr, &AutomatonLimits::default());
            let raw = unminimized(&expr);
            let (minimal, raw) = match (minimal, raw) {
                (Ok(m), Ok(r)) => (m, r),
                (Err(a), Err(b)) => {
                    assert_eq!(a.kind(), GrammarErrorKind::EmptyLanguage, "{expr:?}");
                    assert_eq!(b.kind(), GrammarErrorKind::EmptyLanguage, "{expr:?}");
                    assert!(strings.iter().all(|s| !matches(&expr, s)), "{expr:?}");
                    continue;
                }
                (m, r) => panic!("round {round}: {expr:?}: {:?} vs {:?}", m.err(), r.err()),
            };
            assert!(minimal.num_states() <= raw.num_states(), "{expr:?}");
            for s in &strings {
                let want = matches(&expr, s);
                let bytes = utf8(s);
                assert_eq!(
                    minimal.accepts(&bytes),
                    want,
                    "round {round}: {expr:?} on {s:?}"
                );
                assert_eq!(
                    raw.accepts(&bytes),
                    want,
                    "round {round}: unminimized {expr:?} on {s:?}"
                );
            }
        }
    }

    #[test]
    fn a_large_dfa_stops_at_max_dfa_states() {
        let limits = with(AutomatonLimits {
            max_dfa_states: 1_000,
            ..AutomatonLimits::default()
        });
        let err = error(r#"root ::= ("a" | "b")* "a" ("a" | "b"){14}"#, &limits);
        assert_eq!(err.kind(), GrammarErrorKind::LimitExceeded);
        assert!(
            err.message()
                .contains("DFA reaches 1001 states, above max_dfa_states 1000"),
            "{err}"
        );
    }

    #[test]
    fn a_large_nfa_stops_at_max_nfa_states() {
        let limits = with(AutomatonLimits {
            max_nfa_states: 10_000,
            ..AutomatonLimits::default()
        });
        let err = error(r#"root ::= ("abcdefghij"{100}){100}"#, &limits);
        assert_eq!(err.kind(), GrammarErrorKind::LimitExceeded);
        assert!(
            err.message()
                .contains("NFA reaches 10001 states, above max_nfa_states 10000"),
            "{err}"
        );
    }

    #[test]
    fn a_large_edge_count_stops_at_max_nfa_edges() {
        let wide = Expr::Class {
            negated: false,
            ranges: (0..64u8)
                .map(|i| (char::from(2 * i + 1), char::from(2 * i + 1)))
                .collect(),
        };
        let expr = Expr::Repeat {
            expr: Box::new(wide),
            min: 1_000,
            max: Some(1_000),
        };
        let limits = AutomatonLimits {
            max_nfa_edges: 20_000,
            ..AutomatonLimits::default()
        };
        let err = ByteDfa::from_expr(&expr, &limits).err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::LimitExceeded)
        );
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(
            msg.contains("NFA reaches 20001 edges, above max_nfa_edges 20000"),
            "{msg}"
        );
    }

    #[test]
    fn a_dfa_state_limit_above_the_offset_range_is_rejected_up_front() {
        let too_many = AutomatonLimits {
            max_dfa_states: 17_000_000,
            ..AutomatonLimits::default()
        };
        let expected = "max_dfa_states 17000000 is above the maximum 16777214";
        let err = ByteDfa::from_expr(&Expr::Literal("a".into()), &too_many).err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::LimitExceeded)
        );
        assert!(err.is_some_and(|e| e.message().contains(expected)));
        let err = error("root ::= \"a\"", &with(too_many));
        assert!(err.message().contains(expected), "{err}");
        let grammar = Grammar::parse_with_limits("root ::= \"a\"", &ExpandLimits::default());
        let err = grammar.and_then(|g| g.compile(&with(too_many))).err();
        assert!(err.is_some_and(|e| e.message().contains(expected)));
        let fits = AutomatonLimits {
            max_dfa_states: 16_384,
            ..AutomatonLimits::default()
        };
        assert!(compile("root ::= \"a\"", &with(fits)).is_ok());
    }

    #[test]
    fn subset_work_stops_at_max_subset_work() {
        let limits = with(AutomatonLimits {
            max_subset_work: 200,
            ..AutomatonLimits::default()
        });
        let err = error("root ::= [a-z]+ [0-9]* .{3}", &limits);
        assert_eq!(err.kind(), GrammarErrorKind::LimitExceeded);
        assert!(err.message().contains("above max_subset_work 200"), "{err}");
    }

    #[test]
    fn nested_repeats_within_the_product_limit_compile() {
        let d = dfa(r#"root ::= ((("ab"{3}){4}){5}){6}"#);
        assert!(d.accepts("ab".repeat(360).as_bytes()));
        assert!(!d.accepts("ab".repeat(359).as_bytes()));
        assert_eq!(d.num_states(), 721);
        let d = dfa(r#"root ::= (((("a"{1,5}){1,5}){1,5}){1,5})"#);
        assert!(d.accepts(b"a") && d.accepts(&[b'a'; 625]) && !d.accepts(&[b'a'; 626]));
    }

    #[test]
    fn deep_trees_build_without_recursion() {
        let depth = 100_000;
        let mut expr = Expr::Literal("a".into());
        for _ in 0..depth {
            expr = Expr::Repeat {
                expr: Box::new(expr),
                min: 0,
                max: Some(1),
            };
        }
        let d = ByteDfa::from_expr(&expr, &AutomatonLimits::default());
        assert!(
            d.as_ref()
                .is_ok_and(|d| d.accepts(b"") && d.accepts(b"a") && !d.accepts(b"aa"))
        );
        // Take the tree apart in a loop: the derived drop recurses per level.
        while let Expr::Repeat { expr: inner, .. } = expr {
            expr = *inner;
        }
    }

    #[test]
    fn references_left_in_an_expr_are_unsupported() {
        let expr = Expr::Seq(vec![Expr::Literal("a".into()), Expr::Ref("rest".into())]);
        let kind = ByteDfa::from_expr(&expr, &AutomatonLimits::default())
            .err()
            .map(|e| e.kind());
        assert_eq!(kind, Some(GrammarErrorKind::Unsupported));
    }

    #[test]
    fn hostile_inputs_return_errors() {
        for src in [
            "",
            "root ::=",
            "root ::= (",
            "root ::= a\na ::= root",
            "root ::= [^]{100}",
        ] {
            let _ = compile(src, &CompileLimits::default());
        }
        assert!(compile("root ::= [^]{3}", &CompileLimits::default()).is_ok());
    }
}
