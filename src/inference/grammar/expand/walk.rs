//! The expansion walk: a depth-first walk over an explicit task stack.
//!
//! Rule chains and nested groups grow the task and value stacks on the heap,
//! never the call stack. A start task checks the depth limit and pushes the
//! tasks of its children. An end task joins their values into one node.

use crate::inference::grammar::ast::{Expr, Grammar, Span};
use crate::inference::grammar::error::{GrammarError, GrammarErrorKind};

use super::expander::{Expanded, Expander, Memo};
use super::unroll::OpenRepeat;

/// One pending step of the expansion walk.
enum Task<'g> {
    /// Expand `expr` in the body of rule `owner`.
    Expr { expr: &'g Expr, owner: usize },
    /// Expand rule `idx`, referenced at `site`.
    Rule { idx: usize, site: Span },
    /// Take rule `idx` off the path and keep its expanded body.
    EndRule { idx: usize },
    /// Close a rule reference. The rule counts its own weight.
    EndRef,
    /// Join the last `count` values into one sequence or alternation.
    EndItems {
        alt: bool,
        count: usize,
        weight: usize,
        owner: usize,
    },
    /// Wrap the last value in an open repeat.
    EndRepeat {
        open: OpenRepeat,
        weight: usize,
        owner: usize,
    },
}

/// Expanded values of finished tasks, innermost last.
type Values = Vec<Expanded>;

impl<'g> Expander<'g> {
    /// Expand rule `root`, referenced at `site`. Each `Expr` task leaves one value.
    pub(super) fn run(&mut self, root: usize, site: Span) -> Result<Expanded, GrammarError> {
        let mut tasks = vec![Task::Rule { idx: root, site }];
        let mut values = Values::new();
        while let Some(task) = tasks.pop() {
            match task {
                Task::Expr { expr, owner } => self.enter(expr, owner, &mut tasks, &mut values)?,
                Task::Rule { idx, site } => self.rule(idx, site, &mut tasks, &mut values)?,
                Task::EndRule { idx } => {
                    self.path.pop();
                    if let Some(slot) = self.on_path.get_mut(idx) {
                        *slot = false;
                    }
                    // The root rule ends the walk, so no later use reads its memo.
                    if !tasks.is_empty() {
                        self.remember(idx, values.last());
                    }
                }
                Task::EndRef => self.depth = self.depth.saturating_sub(1),
                Task::EndItems {
                    alt,
                    count,
                    weight,
                    owner,
                } => {
                    let start = values.len().saturating_sub(count);
                    let mut items = Vec::with_capacity(count);
                    let mut size = 1u64;
                    for (expr, n) in values.drain(start..) {
                        items.push(expr);
                        size = size.saturating_add(n);
                    }
                    let expr = if alt {
                        Expr::alt(items)
                    } else {
                        Expr::seq(items)
                    };
                    self.leave(weight, owner)?;
                    values.push((expr, size));
                }
                Task::EndRepeat {
                    open,
                    weight,
                    owner,
                } => {
                    let body = take(&mut values)?;
                    let out = self.close_repeat(open, body, owner)?;
                    self.leave(weight, owner)?;
                    values.push(out);
                }
            }
        }
        take(&mut values)
    }

    /// Start rule `idx`, referenced at `site`. A memo hit leaves its value at once.
    fn rule(
        &mut self,
        idx: usize,
        site: Span,
        tasks: &mut Vec<Task<'g>>,
        values: &mut Values,
    ) -> Result<(), GrammarError> {
        let cached = self.memo.get(idx).and_then(Option::as_ref).map(|m| m.nodes);
        if let Some(nodes) = cached {
            self.add(nodes, site)?;
            if let Some(Some(memo)) = self.memo.get(idx) {
                values.push((memo.expr.clone(), memo.unrolled));
                return Ok(());
            }
        }
        if self.on_path.get(idx).copied().unwrap_or(false) {
            return Err(self.cycle(idx, site));
        }
        let grammar: &'g Grammar = self.grammar;
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
        tasks.push(Task::EndRule { idx });
        tasks.push(Task::Expr {
            expr: &rule.expr,
            owner: idx,
        });
        Ok(())
    }

    /// Keep the expanded body of rule `idx` for its later uses.
    fn remember(&mut self, idx: usize, value: Option<&Expanded>) {
        if let (Some(slot), Some((expr, unrolled))) = (self.memo.get_mut(idx), value) {
            *slot = Some(Memo {
                expr: expr.clone(),
                nodes: expr.node_count(),
                unrolled: *unrolled,
            });
        }
    }

    /// Start `expr`, which sits in the body of rule `owner`. A leaf leaves its value at once.
    fn enter(
        &mut self,
        expr: &'g Expr,
        owner: usize,
        tasks: &mut Vec<Task<'g>>,
        values: &mut Values,
    ) -> Result<(), GrammarError> {
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
        match expr {
            Expr::Empty | Expr::Class { .. } | Expr::Any => {
                self.leave(weight, owner)?;
                values.push((expr.clone(), units(weight)));
            }
            Expr::Literal(text) => {
                self.leave(weight, owner)?;
                values.push((Expr::literal(text.clone()), units(weight)));
            }
            Expr::Ref(name) => {
                let (idx, site) = self.target(name, owner)?;
                tasks.push(Task::EndRef);
                tasks.push(Task::Rule { idx, site });
            }
            Expr::Seq(items) | Expr::Alt(items) => {
                tasks.push(Task::EndItems {
                    alt: matches!(expr, Expr::Alt(_)),
                    count: items.len(),
                    weight,
                    owner,
                });
                tasks.extend(items.iter().rev().map(|expr| Task::Expr { expr, owner }));
            }
            Expr::Repeat {
                expr: body,
                min,
                max,
            } => {
                let open = self.open_repeat(*min, *max, owner)?;
                tasks.push(Task::EndRepeat {
                    open,
                    weight,
                    owner,
                });
                tasks.push(Task::Expr { expr: body, owner });
            }
        }
        Ok(())
    }

    /// Finish one non-reference node of rule `owner`: count its weight and leave its depth.
    fn leave(&mut self, weight: usize, owner: usize) -> Result<(), GrammarError> {
        self.add(weight, self.owner_span(owner))?;
        self.depth = self.depth.saturating_sub(1);
        Ok(())
    }

    /// The rule that `name` names and its first reference site in rule `owner`.
    fn target(&self, name: &str, owner: usize) -> Result<(usize, Span), GrammarError> {
        let site = self
            .sites
            .get(owner)
            .and_then(|sites| sites.get(name))
            .copied()
            .unwrap_or_else(|| self.owner_span(owner));
        match self.index.get(name) {
            Some(&target) => Ok((target, site)),
            None => Err(GrammarError::undefined(name, self.owner_name(owner), site)),
        }
    }
}

/// The last finished value. Every `Expr` task leaves one, so an empty stack is a bug.
fn take(values: &mut Values) -> Result<Expanded, GrammarError> {
    values
        .pop()
        .ok_or_else(|| GrammarError::internal("the expansion value stack is empty"))
}

/// A weight as unrolled-size units.
fn units(weight: usize) -> u64 {
    u64::try_from(weight).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use crate::inference::grammar::{ExpandLimits, Expr, Grammar, GrammarError, GrammarErrorKind};

    fn expand_with(src: &str, limits: &ExpandLimits) -> Result<Expr, GrammarError> {
        Grammar::parse(src)?.expand(limits)
    }

    fn expand(src: &str) -> Result<Expr, GrammarError> {
        expand_with(src, &ExpandLimits::default())
    }

    #[test]
    fn trees_near_the_depth_limit_expand_clone_and_drop() {
        // Each rule opens an alternation, a sequence and a reference: 3 levels.
        let len = 330;
        let mut src = String::from("root ::= r0\n");
        for i in 0..len {
            src.push_str(&format!("r{i} ::= \"a\" r{} | \"b\"\n", i + 1));
        }
        src.push_str(&format!("r{len} ::= \"z\"\n"));
        let got = expand(&src);
        let Ok(expr) = &got else {
            panic!("{:?}", got.err());
        };
        // Per rule: alt 1, seq 1, two literals 2 each.
        assert_eq!(expr.node_count(), len * 6 + 2);
        assert_eq!(expr.clone(), *expr);
        drop(got);
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
        let err = match expand_with(&src, &limits) {
            Err(err) => err,
            Ok(expr) => panic!("expected a cycle error, got {expr:?}"),
        };
        assert_eq!(err.kind(), GrammarErrorKind::Cycle);
        let msg = err.message();
        assert!(msg.starts_with("rule cycle: r0 -> r1 -> "), "{msg}");
        assert!(msg.contains("r15 -> ... (4985 more)"), "{msg}");
        assert!(msg.len() < 400, "{}: {msg}", msg.len());
    }
}
