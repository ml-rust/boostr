//! Recursive-descent GBNF parser.
//!
//! A rule body runs until a later line starts with `name ::=`, or to the end
//! of input. Newlines inside a body are whitespace.

use std::collections::HashMap;

use super::ast::{Expr, Grammar, ROOT_RULE, Reference, Rule, Span};
use super::error::{GrammarError, GrammarErrorKind, excerpt};
use super::lexer::{Cursor, describe, is_name_char};
use super::limits::ExpandLimits;
use super::repeat::postfix;

impl Grammar {
    /// Parse GBNF text with the default limits.
    pub fn parse(src: &str) -> Result<Self, GrammarError> {
        Self::parse_with_limits(src, &ExpandLimits::default())
    }

    /// Parse GBNF text. `limits` bounds input length, group nesting and repeat counts.
    ///
    /// Text over `max_input_bytes` is an error before any allocation.
    /// A caller that reads text from a request must still cap the request body.
    pub fn parse_with_limits(src: &str, limits: &ExpandLimits) -> Result<Self, GrammarError> {
        if src.len() > limits.max_input_bytes {
            return Err(GrammarError::new(
                GrammarErrorKind::LimitExceeded,
                None,
                format!(
                    "grammar text is {} bytes, above max_input_bytes {}. \
                     Send a shorter grammar or raise max_input_bytes.",
                    src.len(),
                    limits.max_input_bytes
                ),
            ));
        }
        Parser {
            cur: Cursor::new(src),
            limits,
            rule: String::new(),
            refs: Vec::new(),
        }
        .grammar()
    }
}

struct Parser<'l> {
    cur: Cursor,
    limits: &'l ExpandLimits,
    /// Name of the rule being parsed.
    rule: String,
    /// References found in the rule being parsed.
    refs: Vec<Reference>,
}

impl Parser<'_> {
    fn grammar(mut self) -> Result<Grammar, GrammarError> {
        let mut rules = Vec::new();
        let mut defined: HashMap<String, Span> = HashMap::new();
        loop {
            self.cur.skip_trivia();
            if self.cur.peek().is_none() {
                break;
            }
            let rule = self.rule()?;
            if let Some(&first) = defined.get(&rule.name) {
                return Err(GrammarError::duplicate_rule(&rule.name, rule.span, first));
            }
            defined.insert(rule.name.clone(), rule.span);
            rules.push(rule);
        }
        if !defined.contains_key(ROOT_RULE) {
            return Err(GrammarError::missing_root());
        }
        Ok(Grammar { rules })
    }

    fn rule(&mut self) -> Result<Rule, GrammarError> {
        let span = self.cur.span();
        let name = self.cur.name();
        if name.is_empty() {
            return Err(GrammarError::syntax(
                span,
                format!(
                    "expected a rule name, found {}. A rule starts with `name ::=`. \
                     Names use ASCII letters, digits, `-` and `_`.",
                    describe(self.cur.peek())
                ),
            ));
        }
        self.cur.skip_inline_space();
        if !self.cur.eat_str("::=") {
            return Err(GrammarError::syntax(
                self.cur.span(),
                format!(
                    "expected `::=` after rule name `{}`, found {}. \
                     Names use ASCII letters, digits, `-` and `_`.",
                    excerpt(&name),
                    describe(self.cur.peek())
                ),
            ));
        }
        self.rule.clone_from(&name);
        self.refs.clear();
        let expr = self.alternatives(0, None, span)?;
        Ok(Rule {
            name,
            expr,
            span,
            references: std::mem::take(&mut self.refs),
        })
    }

    /// Parse `seq | seq | ...`. `group` holds the position of the open `(`.
    fn alternatives(
        &mut self,
        depth: u32,
        group: Option<Span>,
        rule_span: Span,
    ) -> Result<Expr, GrammarError> {
        let mut alts = Vec::new();
        loop {
            let (items, stop) = self.sequence(depth, group)?;
            if items.is_empty() {
                return Err(self.empty_error(alts.is_empty(), group, rule_span, stop));
            }
            alts.push(Expr::seq(items));
            if !self.cur.eat('|') {
                return Ok(Expr::alt(alts));
            }
        }
    }

    fn empty_error(
        &self,
        first: bool,
        group: Option<Span>,
        rule_span: Span,
        stop: Span,
    ) -> GrammarError {
        let empty = "Write `\"\"` to match the empty string.";
        let rule = excerpt(&self.rule);
        match (first, group) {
            (true, None) => GrammarError::syntax(
                rule_span,
                format!("rule `{rule}` has an empty body. {empty}"),
            ),
            (true, Some(open)) => {
                GrammarError::syntax(open, format!("group `()` is empty. {empty}"))
            }
            (false, _) => GrammarError::syntax(
                stop,
                format!(
                    "rule `{rule}` has an empty alternative before {}. {empty}",
                    describe(self.cur.peek())
                ),
            ),
        }
    }

    /// Parse elements up to `|`, `)`, the next rule or the end of input.
    /// Returns the items and the position where the sequence stops.
    fn sequence(
        &mut self,
        depth: u32,
        group: Option<Span>,
    ) -> Result<(Vec<Expr>, Span), GrammarError> {
        let mut items: Vec<Expr> = Vec::new();
        let mut postfixed = false;
        loop {
            let crossed = self.cur.skip_trivia();
            let at = self.cur.span();
            let next = self.cur.peek();
            let Some(c) = next.filter(|_| !(crossed && self.cur.at_rule_start())) else {
                if let Some(open) = group {
                    let before = match next {
                        Some(_) => "the next rule".to_string(),
                        None => describe(None),
                    };
                    return Err(GrammarError::syntax(
                        open,
                        format!(
                            "group has no closing `)` before {before}. Close every `(` with `)`."
                        ),
                    ));
                }
                return Ok((items, at));
            };
            match c {
                '|' => return Ok((items, at)),
                ')' if group.is_some() => return Ok((items, at)),
                ')' => {
                    return Err(GrammarError::syntax(
                        at,
                        "unexpected `)` with no open `(`. Remove it or open a group.",
                    ));
                }
                '*' | '+' | '?' | '{' => {
                    let Some(last) = items.pop() else {
                        return Err(GrammarError::syntax(
                            at,
                            format!(
                                "operator `{c}` has no element before it. \
                                 Put it after a literal, class, `.`, rule name or group."
                            ),
                        ));
                    };
                    if postfixed {
                        return Err(GrammarError::syntax(
                            at,
                            format!(
                                "operator `{c}` follows another operator. \
                                 Wrap the repeated element in a group, as in `(a*)?`."
                            ),
                        ));
                    }
                    let (min, max) = postfix(&mut self.cur, c, self.limits.max_repeat)?;
                    items.push(Expr::repeat(last, min, max));
                    postfixed = true;
                }
                _ => {
                    items.push(self.element(c, at, depth)?);
                    postfixed = false;
                }
            }
        }
    }

    /// Parse one element that starts with `c` at `at`.
    fn element(&mut self, c: char, at: Span, depth: u32) -> Result<Expr, GrammarError> {
        match c {
            '"' => self.cur.literal().map(Expr::literal),
            '[' => self.cur.class(),
            '.' => {
                self.cur.bump();
                Ok(Expr::Any)
            }
            '(' => self.group(at, depth),
            '<' | '!' if c == '<' || self.cur.peek_at(1) == Some('<') => Err(GrammarError::at(
                GrammarErrorKind::Unsupported,
                at,
                "token references are not supported, as in `<[123]>`, `!<[123]>` or `<think>`. \
                 Write the token text as a literal.",
            )),
            c if is_name_char(c) => {
                let name = self.cur.name();
                self.refs.push(Reference {
                    name: name.clone(),
                    span: at,
                });
                Ok(Expr::Ref(name))
            }
            ':' => Err(GrammarError::syntax(
                at,
                "unexpected `:`. A rule definition `name ::=` must start its own line.",
            )),
            other => Err(GrammarError::syntax(
                at,
                format!(
                    "unexpected {}. An element is a literal `\"...\"`, a class `[...]`, `.`, \
                     a rule name or a group `( ... )`.",
                    describe(Some(other))
                ),
            )),
        }
    }

    fn group(&mut self, open: Span, depth: u32) -> Result<Expr, GrammarError> {
        if depth >= self.limits.max_group_depth {
            return Err(GrammarError::at(
                GrammarErrorKind::LimitExceeded,
                open,
                format!(
                    "group nesting exceeds max_group_depth {}. \
                     Flatten the groups or move them into rules.",
                    self.limits.max_group_depth
                ),
            ));
        }
        self.cur.bump();
        let inner = self.alternatives(depth.saturating_add(1), Some(open), open)?;
        if !self.cur.eat(')') {
            return Err(GrammarError::syntax(
                open,
                format!(
                    "group has no closing `)` before {}.",
                    describe(self.cur.peek())
                ),
            ));
        }
        Ok(inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(s: &str) -> Expr {
        Expr::Literal(s.to_string())
    }

    fn rf(s: &str) -> Expr {
        Expr::Ref(s.to_string())
    }

    fn rep(e: Expr, min: u32, max: Option<u32>) -> Expr {
        Expr::Repeat {
            expr: Box::new(e),
            min,
            max,
        }
    }

    fn class(negated: bool, ranges: &[(char, char)]) -> Expr {
        Expr::Class {
            negated,
            ranges: ranges.to_vec(),
        }
    }

    fn root(src: &str) -> Expr {
        match Grammar::parse(src) {
            Ok(g) => g.root().map(|r| r.expr.clone()).unwrap_or(Expr::Empty),
            Err(e) => panic!("{src:?} must parse: {e}"),
        }
    }

    #[test]
    fn every_syntax_form_parses_to_its_ast() {
        let cases: Vec<(&str, Expr)> = vec![
            (r#"root ::= "abc""#, lit("abc")),
            (r#"root ::= "a\n\t\\\"\x41\u00e9""#, lit("a\n\t\\\"Aé")),
            (
                "root ::= [a-z0-9_]",
                class(false, &[('a', 'z'), ('0', '9'), ('_', '_')]),
            ),
            (
                r#"root ::= [^"\\]"#,
                class(true, &[('"', '"'), ('\\', '\\')]),
            ),
            (
                r"root ::= [\u0000-\u001f]",
                class(false, &[('\0', '\u{1f}')]),
            ),
            (
                r"root ::= [\]\-\^]",
                class(false, &[(']', ']'), ('-', '-'), ('^', '^')]),
            ),
            ("root ::= [α-ω]", class(false, &[('α', 'ω')])),
            ("root ::= [^]", class(true, &[])),
            ("root ::= .", Expr::Any),
            ("root ::= .*", rep(Expr::Any, 0, None)),
            (r#"root ::= "a"{2}"#, rep(lit("a"), 2, Some(2))),
            (r#"root ::= "a"{2,}"#, rep(lit("a"), 2, None)),
            (r#"root ::= "a"{2,4}"#, rep(lit("a"), 2, Some(4))),
            (r#"root ::= "a"{,3}"#, rep(lit("a"), 0, Some(3))),
            (r#"root ::= "a"{1}"#, lit("a")),
            (r#"root ::= "a"{0}"#, Expr::Empty),
            (r#"root ::= "a" *"#, rep(lit("a"), 0, None)),
            (
                "root ::= x+ y? z*",
                Expr::Seq(vec![
                    rep(rf("x"), 1, None),
                    rep(rf("y"), 0, Some(1)),
                    rep(rf("z"), 0, None),
                ]),
            ),
            (
                r#"root ::= ("a" | "b")+"#,
                rep(Expr::Alt(vec![lit("a"), lit("b")]), 1, None),
            ),
            (
                r#"root ::= ("a" ("b" | "c")?)*"#,
                rep(
                    Expr::Seq(vec![
                        lit("a"),
                        rep(Expr::Alt(vec![lit("b"), lit("c")]), 0, Some(1)),
                    ]),
                    0,
                    None,
                ),
            ),
            (r#"root ::= (("a"))"#, lit("a")),
            (r#"root ::= """#, Expr::Empty),
            (
                r#"root ::= "a" | """#,
                Expr::Alt(vec![lit("a"), Expr::Empty]),
            ),
            (
                "root ::=\n  \"a\"\n  \"b\"\nnext ::= \"c\"",
                Expr::Seq(vec![lit("a"), lit("b")]),
            ),
            (
                "root ::= \"a\"\n  | \"b\"\n  | \"c\"\n",
                Expr::Alt(vec![lit("a"), lit("b"), lit("c")]),
            ),
            (
                "# head\nroot ::= \"a\" # tail\n  | \"b\" # more\n# end\n",
                Expr::Alt(vec![lit("a"), lit("b")]),
            ),
            (
                "root ::= (\n  \"a\"\n  | \"b\"\n)\n",
                Expr::Alt(vec![lit("a"), lit("b")]),
            ),
            (
                "root ::= \"a\"\r\n  | \"b\"\r\n",
                Expr::Alt(vec![lit("a"), lit("b")]),
            ),
            (r#"root ::= "|" "#, lit("|")),
            (r##"root ::= "# not a comment""##, lit("# not a comment")),
        ];
        for (src, want) in cases {
            assert_eq!(root(src), want, "{src:?}");
        }
    }

    #[test]
    fn rules_split_at_lines_that_start_a_rule() {
        let src = "root ::= a b\n\nb ::= \"x\"\n  a ::= [0-9]\n";
        let g = Grammar::parse(src).map_err(|e| e.to_string());
        let names: Vec<String> = g
            .iter()
            .flat_map(|g| g.rules.iter().map(|r| r.name.clone()))
            .collect();
        assert_eq!(names, ["root", "b", "a"]);
        let g = g.unwrap_or(Grammar { rules: vec![] });
        let spans: Vec<Span> = g.rules.iter().map(|r| r.span).collect();
        assert_eq!(
            spans,
            [
                Span { line: 1, column: 1 },
                Span { line: 3, column: 1 },
                Span { line: 4, column: 3 }
            ]
        );
        let refs = g.root().map(|r| r.references.clone()).unwrap_or_default();
        assert_eq!(
            refs,
            [
                Reference {
                    name: "a".into(),
                    span: Span {
                        line: 1,
                        column: 10
                    }
                },
                Reference {
                    name: "b".into(),
                    span: Span {
                        line: 1,
                        column: 12
                    }
                },
            ]
        );
    }

    /// Parse and expand with default limits.
    fn compile(src: &str) -> Result<Expr, GrammarError> {
        Grammar::parse(src)?.expand(&ExpandLimits::default())
    }

    #[test]
    fn invalid_grammars_report_kind_and_position() {
        use GrammarErrorKind::*;
        /// Source, expected kind, expected `(line, column)`.
        type Case<'a> = (&'a str, GrammarErrorKind, Option<(u32, u32)>);
        let cases: Vec<Case> = vec![
            (r#"a ::= "x""#, MissingRoot, None),
            ("", MissingRoot, None),
            ("root ::= foo", UndefinedRule, Some((1, 10))),
            (r#"root ::= "abc"#, Syntax, Some((1, 10))),
            ("root ::= [a-z", Syntax, Some((1, 10))),
            (r#"root ::= ("a""#, Syntax, Some((1, 10))),
            ("root ::= (\"a\"\nb ::= \"c\"", Syntax, Some((1, 10))),
            (r#"root ::= "\q""#, Syntax, Some((1, 11))),
            (r#"root ::= "\x1""#, Syntax, Some((1, 11))),
            (r#"root ::= "\uD800""#, Syntax, Some((1, 11))),
            (r#"root ::= "a"{3,2}"#, Syntax, Some((1, 13))),
            (r#"root ::= ("a")*?"#, Syntax, Some((1, 16))),
            (r#"root ::= "a"**"#, Syntax, Some((1, 14))),
            (r#"root ::= "a"{2}{3}"#, Syntax, Some((1, 16))),
            ("root ::= <[123]>", Unsupported, Some((1, 10))),
            ("root ::= !<[123]>", Unsupported, Some((1, 10))),
            ("root ::= <think>", Unsupported, Some((1, 10))),
            (r#"ro.ot ::= "a""#, Syntax, Some((1, 3))),
            (r#""root" ::= "a""#, Syntax, Some((1, 1))),
            (r#"é ::= "a""#, Syntax, Some((1, 1))),
            ("root ::=", Syntax, Some((1, 1))),
            ("root ::=\nnext ::= \"a\"", Syntax, Some((1, 1))),
            (r#"root ::= "a" |"#, Syntax, Some((1, 15))),
            (r#"root ::= "a" | | "b""#, Syntax, Some((1, 16))),
            ("root ::= ()", Syntax, Some((1, 10))),
            (
                "root ::= \"a\"\nroot ::= \"b\"",
                DuplicateRule,
                Some((2, 1)),
            ),
            ("root ::= *", Syntax, Some((1, 10))),
            (r#"root ::= "a")"#, Syntax, Some((1, 13))),
            (r#"root ::= "a" b ::= "c""#, Syntax, Some((1, 16))),
            (r#"root ::= "a"{1001}"#, LimitExceeded, Some((1, 14))),
            (
                r#"root ::= "a"{99999999999999999999}"#,
                LimitExceeded,
                Some((1, 14)),
            ),
            ("root ::= [z-a]", Syntax, Some((1, 11))),
            ("root ::= []", Syntax, Some((1, 10))),
            ("root ::= root", Cycle, Some((1, 10))),
            ("root ::= a @", Syntax, Some((1, 12))),
        ];
        for (src, kind, pos) in cases {
            let err = match compile(src) {
                Ok(e) => panic!("{src:?} must fail, got {e:?}"),
                Err(err) => err,
            };
            assert_eq!(err.kind(), kind, "{src:?}: {err}");
            let got = err.span().map(|s| (s.line, s.column));
            assert_eq!(got, pos, "{src:?}: {err}");
        }
    }

    #[test]
    fn messages_quote_the_offending_input() {
        for (src, needle) in [
            ("root ::= a @", "`@`"),
            ("root ::= foo", "`foo`"),
            (r#"root ::= "\q""#, r"`\q`"),
            ("root ::= <[1]>", "token references are not supported"),
            (r#"a ::= "x""#, "`root`"),
            ("root ::=", "`root`"),
        ] {
            let msg = compile(src)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(msg.contains(needle), "{src:?}: {msg}");
        }
    }

    #[test]
    fn every_ascii_punctuation_returns() {
        for p in (0u8..128)
            .map(char::from)
            .filter(char::is_ascii_punctuation)
        {
            for src in [
                p.to_string(),
                format!("root ::= {p}"),
                format!("root ::= a{p}"),
                format!("root ::= ({p}"),
                format!("root ::= [{p}"),
                format!("root ::= \"{p}"),
                format!("root ::= a {p}\n"),
            ] {
                let _ = compile(&src);
            }
        }
    }

    #[test]
    fn truncated_inputs_are_errors() {
        for t in ["\"abc", "[a-", "(", "a{", "a{2", "a{2,", "\\x1", "\\u12"] {
            assert!(Grammar::parse(t).is_err(), "{t:?}");
            let src = format!("root ::= {t}");
            assert!(Grammar::parse(&src).is_err(), "{src:?}");
        }
        for t in ["\\x1", "\\u12", "\\U0001F6", "\\"] {
            let quoted = format!("root ::= \"{t}");
            assert!(Grammar::parse(&quoted).is_err(), "{quoted:?}");
            let class = format!("root ::= [{t}");
            assert!(Grammar::parse(&class).is_err(), "{class:?}");
        }
    }

    #[test]
    fn deep_nesting_stops_at_the_group_limit() {
        let src = format!("root ::= {}\"a\"{}", "(".repeat(10_000), ")".repeat(10_000));
        let err = Grammar::parse(&src).err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::LimitExceeded)
        );
        assert_eq!(
            err.and_then(|e| e.span()),
            Some(Span {
                line: 1,
                column: 266
            })
        );
        let src = format!("root ::= {}\"a\"{}", "(".repeat(256), ")".repeat(256));
        assert_eq!(compile(&src), Ok(lit("a")));
    }

    #[test]
    fn input_over_max_input_bytes_is_rejected_first() {
        let src = "root ::= \"abc\"";
        let limits = |max_input_bytes| ExpandLimits {
            max_input_bytes,
            ..ExpandLimits::default()
        };
        assert!(Grammar::parse_with_limits(src, &limits(src.len())).is_ok());
        let err = Grammar::parse_with_limits(src, &limits(src.len() - 1)).err();
        assert_eq!(
            err.as_ref().map(GrammarError::kind),
            Some(GrammarErrorKind::LimitExceeded)
        );
        assert_eq!(err.as_ref().and_then(GrammarError::span), None);
        let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
        assert!(msg.contains("14 bytes, above max_input_bytes 13"), "{msg}");
        let unclosed = "root ::= (".repeat(8);
        let kind = Grammar::parse_with_limits(&unclosed, &limits(8)).err();
        assert_eq!(
            kind.map(|e| e.kind()),
            Some(GrammarErrorKind::LimitExceeded)
        );
    }

    #[test]
    fn messages_stay_short_for_a_one_megabyte_rule_name() {
        let name = "n".repeat(1_000_000);
        let limits = ExpandLimits {
            max_input_bytes: 4 << 20,
            ..ExpandLimits::default()
        };
        for src in [
            format!("root ::= {name}"),
            format!("{name} ::= \"a\"\n{name} ::= \"b\""),
            format!("{name} ::="),
            format!("{name} ::= \"a\" | | \"b\""),
            format!("{name} \"a\""),
        ] {
            let err = Grammar::parse_with_limits(&src, &limits).and_then(|g| g.expand(&limits));
            let msg = err.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(!msg.is_empty(), "{}", src.len());
            assert!(msg.len() < 400, "{}: {msg}", msg.len());
            assert!(msg.contains(&format!("`{}...`", "n".repeat(64))), "{msg}");
        }
    }

    #[test]
    fn random_garbage_always_returns() {
        let alphabet: Vec<char> = "\"[]()|*+?{},.:=#\\-^<>!_ \n\t\raz09xuU:::=root"
            .chars()
            .collect();
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize
        };
        for i in 0..2_000 {
            let len = next() % 48;
            let body: String = (0..len)
                .map(|_| alphabet[next() % alphabet.len()])
                .collect();
            let src = if i % 2 == 0 {
                format!("root ::= {body}")
            } else {
                body
            };
            let _ = compile(&src);
        }
    }
}
