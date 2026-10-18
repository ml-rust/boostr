//! The expansion walk over rules, references, sequences and alternations.
//!
//! Expansion finds cycles with a depth-first walk over an explicit path stack.
//! A rule expands once. Each later use clones that result and counts its weight.
//! Each expanded expression also carries its unrolled size for the repeat product.

use std::collections::HashMap;

use crate::inference::grammar::ast::{Expr, Grammar, ROOT_RULE, Span};
use crate::inference::grammar::error::{GrammarError, GrammarErrorKind};
use crate::inference::grammar::limits::ExpandLimits;

/// Position used when a rule has no span.
const START: Span = Span { line: 1, column: 1 };

/// An expanded expression and its unrolled size in weight units.
///
/// The unrolled size counts each node by `Expr::weight`. A repeat counts
/// 1 plus its factor times the unrolled size of its body.
pub(super) type Expanded = (Expr, u64);

/// First reference span of each name in one rule body.
type Sites<'g> = HashMap<&'g str, Span>;

/// One expanded rule.
#[derive(Clone)]
struct Memo {
    expr: Expr,
    /// Weighted node count of `expr`.
    nodes: usize,
    /// Unrolled size of `expr`.
    unrolled: u64,
}

impl Grammar {
    /// Inline every rule reference reachable from `root` and return the root expression.
    ///
    /// Every reference in every rule must name a defined rule. A recursive rule is an error.
    /// `limits` bounds the weighted node count and the repeat product of the result.
    pub fn expand(&self, limits: &ExpandLimits) -> Result<Expr, GrammarError> {
        let mut index = HashMap::with_capacity(self.rules.len());
        for (i, rule) in self.rules.iter().enumerate() {
            if let Some(first) = index.insert(rule.name.as_str(), i) {
                let first = self.rules.get(first).map_or(rule.span, |r| r.span);
                return Err(GrammarError::duplicate_rule(&rule.name, rule.span, first));
            }
        }
        let sites = self.sites(&index)?;
        let Some(&root) = index.get(ROOT_RULE) else {
            return Err(GrammarError::missing_root());
        };
        let count = self.rules.len();
        let root_span = self.rules.get(root).map_or(START, |r| r.span);
        Expander {
            grammar: self,
            limits,
            index,
            sites,
            memo: vec![None; count],
            on_path: vec![false; count],
            path: Vec::new(),
            nodes: 0,
            depth: 0,
            product: 1,
        }
        .rule(root, root_span)
        .map(|(expr, _)| expr)
    }

    /// The reference sites of every rule. A reference to an undefined rule is an error.
    fn sites(&self, index: &HashMap<&str, usize>) -> Result<Vec<Sites<'_>>, GrammarError> {
        let mut all = Vec::with_capacity(self.rules.len());
        for rule in &self.rules {
            let mut first = Sites::with_capacity(rule.references.len());
            for site in &rule.references {
                if !index.contains_key(site.name.as_str()) {
                    return Err(GrammarError::undefined(&site.name, &rule.name, site.span));
                }
                first.entry(site.name.as_str()).or_insert(site.span);
            }
            all.push(first);
        }
        Ok(all)
    }
}

pub(super) struct Expander<'g> {
    grammar: &'g Grammar,
    pub(super) limits: &'g ExpandLimits,
    index: HashMap<&'g str, usize>,
    /// Reference sites of each rule, by rule index.
    sites: Vec<Sites<'g>>,
    /// Expanded rules.
    memo: Vec<Option<Memo>>,
    /// True for each rule on the current reference path.
    on_path: Vec<bool>,
    /// Rules being expanded, outermost first.
    path: Vec<usize>,
    /// Weight units produced so far.
    nodes: usize,
    /// Current recursion depth.
    depth: u32,
    /// Product of the factors of every enclosing repeat.
    pub(super) product: u64,
}

impl Expander<'_> {
    /// Expand rule `idx`, referenced at `site`.
    fn rule(&mut self, idx: usize, site: Span) -> Result<Expanded, GrammarError> {
        let cached = self
            .memo
            .get(idx)
            .and_then(Option::as_ref)
            .map(|m| (m.nodes, m.unrolled));
        if let Some((nodes, unrolled)) = cached {
            self.add(nodes, site)?;
            if let Some(Some(memo)) = self.memo.get(idx) {
                return Ok((memo.expr.clone(), unrolled));
            }
        }
        if self.on_path.get(idx).copied().unwrap_or(false) {
            return Err(self.cycle(idx, site));
        }
        let grammar = self.grammar;
        let Some(rule) = grammar.rules.get(idx) else {
            return Err(GrammarError::at(
                GrammarErrorKind::UndefinedRule,
                site,
                format!("rule index {idx} names no rule in this grammar."),
            ));
        };
        if let Some(slot) = self.on_path.get_mut(idx) {
            *slot = true;
        }
        self.path.push(idx);
        let (expr, unrolled) = self.expr(&rule.expr, idx)?;
        self.path.pop();
        if let Some(slot) = self.on_path.get_mut(idx) {
            *slot = false;
        }
        if let Some(slot) = self.memo.get_mut(idx) {
            let nodes = expr.node_count();
            *slot = Some(Memo {
                expr: expr.clone(),
                nodes,
                unrolled,
            });
        }
        Ok((expr, unrolled))
    }

    /// Expand `expr`, which sits in the body of rule `owner`.
    pub(super) fn expr(&mut self, expr: &Expr, owner: usize) -> Result<Expanded, GrammarError> {
        self.depth = self.depth.saturating_add(1);
        if self.depth > self.limits.max_expand_depth {
            return Err(Self::limit(
                self.owner_span(owner),
                format!(
                    "expansion depth exceeds max_expand_depth {}. \
                     Shorten the chain of rule references or nested groups.",
                    self.limits.max_expand_depth
                ),
            ));
        }
        let weight = expr.weight();
        let out = match expr {
            Expr::Empty | Expr::Class { .. } | Expr::Any => (expr.clone(), units(weight)),
            Expr::Literal(text) => (Expr::literal(text.clone()), units(weight)),
            Expr::Ref(name) => self.reference(name, owner)?,
            Expr::Seq(items) => {
                let (items, size) = self.items(items, owner)?;
                (Expr::seq(items), size)
            }
            Expr::Alt(items) => {
                let (items, size) = self.items(items, owner)?;
                (Expr::alt(items), size)
            }
            Expr::Repeat { expr, min, max } => self.repeat(expr, *min, *max, owner)?,
        };
        if !matches!(expr, Expr::Ref(_)) {
            self.add(weight, self.owner_span(owner))?;
        }
        self.depth = self.depth.saturating_sub(1);
        Ok(out)
    }

    /// Expand `items`. The size is their unrolled sum plus 1 for the parent node.
    fn items(&mut self, items: &[Expr], owner: usize) -> Result<(Vec<Expr>, u64), GrammarError> {
        let mut out = Vec::with_capacity(items.len());
        let mut size = 1u64;
        for item in items {
            let (expr, n) = self.expr(item, owner)?;
            out.push(expr);
            size = size.saturating_add(n);
        }
        Ok((out, size))
    }

    fn reference(&mut self, name: &str, owner: usize) -> Result<Expanded, GrammarError> {
        let site = self
            .sites
            .get(owner)
            .and_then(|sites| sites.get(name))
            .copied()
            .unwrap_or_else(|| self.owner_span(owner));
        match self.index.get(name) {
            Some(&target) => self.rule(target, site),
            None => Err(GrammarError::undefined(name, self.owner_name(owner), site)),
        }
    }

    /// Count `n` new weight units, produced at `span`. Stop once the total passes `max_nodes`.
    fn add(&mut self, n: usize, span: Span) -> Result<(), GrammarError> {
        self.nodes = self.nodes.saturating_add(n);
        if self.nodes > self.limits.max_nodes {
            return Err(Self::limit(
                span,
                format!(
                    "expansion reaches {} weighted nodes, above max_nodes {}. \
                     Reuse fewer large rules or raise max_nodes.",
                    self.nodes, self.limits.max_nodes
                ),
            ));
        }
        Ok(())
    }

    fn cycle(&self, idx: usize, site: Span) -> GrammarError {
        let start = self.path.iter().position(|&p| p == idx).unwrap_or(0);
        let chain: Vec<&str> = self
            .path
            .get(start..)
            .unwrap_or_default()
            .iter()
            .chain(std::iter::once(&idx))
            .map(|&p| self.owner_name(p))
            .collect();
        GrammarError::cycle(&chain, site)
    }

    pub(super) fn limit(span: Span, message: String) -> GrammarError {
        GrammarError::at(GrammarErrorKind::LimitExceeded, span, message)
    }

    pub(super) fn owner_name(&self, owner: usize) -> &str {
        self.grammar
            .rules
            .get(owner)
            .map_or("?", |r| r.name.as_str())
    }

    pub(super) fn owner_span(&self, owner: usize) -> Span {
        self.grammar.rules.get(owner).map_or(START, |r| r.span)
    }
}

/// A weight as unrolled-size units.
fn units(weight: usize) -> u64 {
    u64::try_from(weight).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::ast::Rule;

    fn lit(s: &str) -> Expr {
        Expr::Literal(s.to_string())
    }

    fn rep(e: Expr, min: u32, max: Option<u32>) -> Expr {
        Expr::Repeat {
            expr: Box::new(e),
            min,
            max,
        }
    }

    fn expand_with(src: &str, limits: &ExpandLimits) -> Result<Expr, GrammarError> {
        Grammar::parse(src)?.expand(limits)
    }

    fn expand(src: &str) -> Result<Expr, GrammarError> {
        expand_with(src, &ExpandLimits::default())
    }

    fn error(src: &str) -> GrammarError {
        match expand(src) {
            Ok(e) => panic!("{src:?} must fail, got {e:?}"),
            Err(err) => err,
        }
    }

    fn at(line: u32, column: u32) -> Option<Span> {
        Some(Span { line, column })
    }

    #[test]
    fn inlining_matches_the_hand_written_expr() {
        let src = "root ::= greeting \" \" name\n\
                   greeting ::= \"hi\" | \"hello\"\n\
                   name ::= [a-z]+";
        let want = Expr::Seq(vec![
            Expr::Alt(vec![lit("hi"), lit("hello")]),
            lit(" "),
            rep(
                Expr::Class {
                    negated: false,
                    ranges: vec![('a', 'z')],
                },
                1,
                None,
            ),
        ]);
        assert_eq!(expand(src), Ok(want));
    }

    #[test]
    fn nested_sequences_flatten_after_inlining() {
        let src = "root ::= ab \"c\"\nab ::= \"a\" \"b\"";
        assert_eq!(
            expand(src),
            Ok(Expr::Seq(vec![lit("a"), lit("b"), lit("c")]))
        );
    }

    #[test]
    fn a_rule_used_three_times_expands_the_same_each_time() {
        let src = "root ::= d \"-\" d \"-\" (d | \"x\")\nd ::= [0-9]{2}";
        let d = rep(
            Expr::Class {
                negated: false,
                ranges: vec![('0', '9')],
            },
            2,
            Some(2),
        );
        let want = Expr::Seq(vec![
            d.clone(),
            lit("-"),
            d.clone(),
            lit("-"),
            Expr::Alt(vec![d, lit("x")]),
        ]);
        assert_eq!(expand(src), Ok(want));
    }

    #[test]
    fn self_cycles_name_the_rule() {
        let err = error("root ::= a\na ::= \"x\" a | \"y\"");
        assert_eq!(err.kind(), GrammarErrorKind::Cycle);
        assert!(err.message().starts_with("rule cycle: a -> a. "), "{err}");
        assert_eq!(err.span(), at(2, 11));
        let err = error("root ::= a\na ::= (\"x\" a)*");
        assert!(err.message().starts_with("rule cycle: a -> a. "), "{err}");
        let err = error("root ::= \"x\" root?");
        assert!(
            err.message().starts_with("rule cycle: root -> root. "),
            "{err}"
        );
    }

    #[test]
    fn three_rule_cycle_lists_the_rules_in_order() {
        let src = "root ::= a\na ::= \"1\" b\nb ::= \"2\" c\nc ::= (\"3\" a)?";
        let err = error(src);
        assert_eq!(err.kind(), GrammarErrorKind::Cycle);
        assert!(
            err.message().starts_with("rule cycle: a -> b -> c -> a. "),
            "{err}"
        );
        assert_eq!(err.span(), at(4, 12));
    }

    #[test]
    fn undefined_reference_inside_a_group_names_it() {
        let err = error("root ::= (\"a\" | missing)");
        assert_eq!(err.kind(), GrammarErrorKind::UndefinedRule);
        assert_eq!(err.span(), at(1, 17));
        assert!(err.message().contains("`missing`"), "{err}");
        assert!(err.message().contains("`root`"), "{err}");
    }

    #[test]
    fn undefined_reference_in_an_unused_rule_is_an_error() {
        let err = error("root ::= \"a\"\nunused ::= nope");
        assert_eq!(err.kind(), GrammarErrorKind::UndefinedRule);
        assert_eq!(err.span(), at(2, 12));
    }

    #[test]
    fn node_limit_counts_every_cloned_use() {
        let mut src = String::from("root ::= a20\na0 ::= \"xy\"\n");
        for i in 1..=20 {
            src.push_str(&format!("a{i} ::= a{} a{}\n", i - 1, i - 1));
        }
        let err = error(&src);
        assert_eq!(err.kind(), GrammarErrorKind::LimitExceeded);
        assert!(err.message().contains("max_nodes 500000"), "{err}");

        // Each one-byte literal weighs 2 and the sequence weighs 1.
        let src = "root ::= \"a\" \"b\" \"c\"";
        let limits = |max_nodes| ExpandLimits {
            max_nodes,
            ..ExpandLimits::default()
        };
        assert!(expand_with(src, &limits(7)).is_ok());
        let err = expand_with(src, &limits(6)).err().map(|e| e.kind());
        assert_eq!(err, Some(GrammarErrorKind::LimitExceeded));
    }

    /// `big ::= body` and a root that refers to `big` `uses` times.
    fn many_uses(body: &str, uses: usize) -> String {
        let mut src = format!("big ::= {body}\nroot ::=");
        for _ in 0..uses {
            src.push_str(" big");
        }
        src
    }

    #[test]
    fn large_payloads_cloned_many_times_stop_at_max_nodes() {
        let literal = format!("\"{}\"", "x".repeat(100_000));
        let class = format!("[{}]", "a".repeat(50_000));
        for (body, uses) in [(literal, 10_000), (class, 1_000)] {
            let err = error(&many_uses(&body, uses));
            assert_eq!(err.kind(), GrammarErrorKind::LimitExceeded);
            assert!(err.message().contains("max_nodes 500000"), "{err}");
        }
        let small = expand(&many_uses("\"x\"", 10_000));
        assert!(
            matches!(&small, Ok(Expr::Seq(items)) if items.len() == 10_000),
            "{:?}",
            small.err()
        );
    }

    #[test]
    fn twenty_thousand_single_use_rules_expand() {
        let count = 20_000;
        let mut src = String::from("root ::=");
        for i in 0..count {
            src.push_str(&format!(" r{i}"));
        }
        for i in 0..count {
            src.push_str(&format!("\nr{i} ::= \"x\""));
        }
        let got = expand(&src);
        assert!(
            matches!(&got, Ok(Expr::Seq(items)) if items.len() == count),
            "{:?}",
            got.err()
        );
    }

    #[test]
    fn a_reference_reports_the_first_site_of_its_name() {
        // `b{0}` drops at parse time, and its site still names `b` first.
        let err = error("root ::= b\nb ::= b{0} \"x\" b");
        assert_eq!(err.kind(), GrammarErrorKind::Cycle);
        assert_eq!(err.span(), at(2, 7));
    }

    #[test]
    fn a_five_thousand_rule_cycle_message_stays_short() {
        let mut src = String::from("root ::= r0\n");
        for i in 0..5_000 {
            src.push_str(&format!("r{i} ::= r{}\n", (i + 1) % 5_000));
        }
        let limits = ExpandLimits {
            max_expand_depth: 20_000,
            ..ExpandLimits::default()
        };
        // The 5 000-level walk needs more stack than a test thread has.
        let run = move || expand_with(&src, &limits);
        let joined = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(run)
            .map(|handle| handle.join());
        let err = match joined {
            Ok(Ok(Err(err))) => err,
            other => panic!("expected a cycle error, got {other:?}"),
        };
        assert_eq!(err.kind(), GrammarErrorKind::Cycle);
        let msg = err.message();
        assert!(msg.starts_with("rule cycle: r0 -> r1 -> "), "{msg}");
        assert!(msg.contains("r15 -> ... (4985 more)"), "{msg}");
        assert!(msg.len() < 400, "{}: {msg}", msg.len());
    }

    fn hand_built(name: &str, expr: Expr) -> Grammar {
        Grammar {
            rules: vec![Rule {
                name: name.to_string(),
                expr,
                span: Span { line: 1, column: 1 },
                references: vec![],
            }],
        }
    }

    #[test]
    fn hand_built_grammars_are_checked() {
        let limits = ExpandLimits::default();
        let reversed = hand_built(ROOT_RULE, rep(lit("a"), 3, Some(2)));
        let kind = reversed.expand(&limits).err().map(|e| e.kind());
        assert_eq!(kind, Some(GrammarErrorKind::Syntax));
        let no_root = hand_built("start", lit("a"));
        let kind = no_root.expand(&limits).err().map(|e| e.kind());
        assert_eq!(kind, Some(GrammarErrorKind::MissingRoot));
        let dangling = hand_built(ROOT_RULE, Expr::Ref("gone".into()));
        let kind = dangling.expand(&limits).err().map(|e| e.kind());
        assert_eq!(kind, Some(GrammarErrorKind::UndefinedRule));
        let mut twice = hand_built(ROOT_RULE, lit("a"));
        twice.rules.extend(hand_built(ROOT_RULE, lit("b")).rules);
        let kind = twice.expand(&limits).err().map(|e| e.kind());
        assert_eq!(kind, Some(GrammarErrorKind::DuplicateRule));
    }

    fn chain(len: usize) -> String {
        let mut src = String::from("root ::= r0\n");
        for i in 0..len {
            src.push_str(&format!("r{i} ::= r{}\n", i + 1));
        }
        src.push_str(&format!("r{len} ::= \"z\"\n"));
        src
    }

    #[test]
    fn long_reference_chains_stop_at_the_depth_limit() {
        assert_eq!(expand(&chain(100)), Ok(lit("z")));
        let err = error(&chain(2_000));
        assert_eq!(err.kind(), GrammarErrorKind::LimitExceeded);
        assert!(err.message().contains("max_expand_depth 1024"), "{err}");
    }
}
