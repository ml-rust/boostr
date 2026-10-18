//! Reference matcher over `Expr` and random expression generator, for tests only.
//!
//! The matcher computes every input position an expression can end at, with
//! a memo per node and start position. It is slow and plainly correct, so
//! automaton tests compare against it.

use std::collections::{BTreeSet, HashMap};

use super::ast::Expr;

/// True when `expr` matches all of `input`.
pub(super) fn matches(expr: &Expr, input: &[char]) -> bool {
    let mut matcher = Matcher {
        input,
        memo: HashMap::new(),
    };
    matcher.ends(expr, 0).contains(&input.len())
}

struct Matcher<'a> {
    input: &'a [char],
    memo: HashMap<(*const Expr, usize), BTreeSet<usize>>,
}

impl Matcher<'_> {
    /// Every position where a match of `expr` that starts at `at` can end.
    fn ends(&mut self, expr: &Expr, at: usize) -> BTreeSet<usize> {
        let key = (std::ptr::from_ref(expr), at);
        if let Some(found) = self.memo.get(&key) {
            return found.clone();
        }
        let out = self.compute(expr, at);
        self.memo.insert(key, out.clone());
        out
    }

    /// The end positions of each start in `from`.
    fn step(&mut self, expr: &Expr, from: &BTreeSet<usize>) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        for &p in from {
            out.extend(self.ends(expr, p));
        }
        out
    }

    fn compute(&mut self, expr: &Expr, at: usize) -> BTreeSet<usize> {
        let input = self.input;
        let mut out = BTreeSet::new();
        match expr {
            Expr::Empty => {
                out.insert(at);
            }
            Expr::Literal(text) => {
                let want: Vec<char> = text.chars().collect();
                if input.get(at..at + want.len()) == Some(want.as_slice()) {
                    out.insert(at + want.len());
                }
            }
            Expr::Class { negated, ranges } => {
                if let Some(c) = input.get(at) {
                    let inside = ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(c));
                    if inside != *negated {
                        out.insert(at + 1);
                    }
                }
            }
            Expr::Any => {
                if at < input.len() {
                    out.insert(at + 1);
                }
            }
            Expr::Ref(_) => {}
            Expr::Seq(items) => {
                out.insert(at);
                for item in items {
                    out = self.step(item, &out);
                }
            }
            Expr::Alt(items) => {
                for item in items {
                    out.extend(self.ends(item, at));
                }
            }
            Expr::Repeat { expr, min, max } => {
                if max.is_some_and(|max| max < *min) {
                    return out;
                }
                let mut current = BTreeSet::from([at]);
                for _ in 0..*min {
                    current = self.step(expr, &current);
                }
                out.extend(current.iter().copied());
                match max {
                    Some(max) => {
                        for _ in *min..*max {
                            current = self.step(expr, &current);
                            out.extend(current.iter().copied());
                        }
                    }
                    None => loop {
                        let next = self.step(expr, &current);
                        let fresh: BTreeSet<usize> = next.difference(&out).copied().collect();
                        if fresh.is_empty() {
                            break;
                        }
                        out.extend(fresh.iter().copied());
                        current = fresh;
                    },
                }
            }
        }
        out
    }
}

/// A seeded linear congruential generator.
pub(super) struct Lcg(pub u64);

impl Lcg {
    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// A value in `0..n`.
    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }

    pub(super) fn pick<T: Clone>(&mut self, items: &[T]) -> T {
        let i = self.below(items.len() as u64) as usize;
        items[i].clone()
    }
}

/// The alphabet of random expressions.
pub(super) const ALPHABET: [char; 5] = ['a', 'b', 'c', 'é', '世'];

/// Characters outside the alphabet that random strings also use.
pub(super) const EXTRA: [char; 4] = ['z', '\u{7f}', '\u{10000}', '\u{10ffff}'];

/// A random expression of depth at most `depth`.
pub(super) fn random_expr(rng: &mut Lcg, depth: u32) -> Expr {
    let leaf = depth == 0 || rng.below(3) == 0;
    if leaf {
        return match rng.below(8) {
            0 => Expr::Empty,
            1 => Expr::Any,
            2 => Expr::Alt(vec![]),
            3 | 4 => {
                let len = 1 + rng.below(3);
                Expr::Literal((0..len).map(|_| rng.pick(&ALPHABET)).collect())
            }
            _ => {
                let pairs = [('a', 'b'), ('c', 'c'), ('é', '世'), ('a', 'é'), ('b', 'b')];
                let count = 1 + rng.below(2);
                Expr::Class {
                    negated: rng.below(3) == 0,
                    ranges: (0..count).map(|_| rng.pick(&pairs)).collect(),
                }
            }
        };
    }
    let children = |rng: &mut Lcg, n: u64| -> Vec<Expr> {
        (0..n).map(|_| random_expr(rng, depth - 1)).collect()
    };
    match rng.below(3) {
        0 => {
            let n = 2 + rng.below(2);
            Expr::Seq(children(rng, n))
        }
        1 => {
            let n = 1 + rng.below(3);
            Expr::Alt(children(rng, n))
        }
        _ => {
            let min = rng.below(3) as u32;
            let max = match rng.below(3) {
                0 => None,
                _ => Some(min + rng.below(3) as u32),
            };
            Expr::Repeat {
                expr: Box::new(random_expr(rng, depth - 1)),
                min,
                max,
            }
        }
    }
}

/// Every string over `ALPHABET` up to length 4, then `random` strings up to length 8.
pub(super) fn test_strings(rng: &mut Lcg, random: usize) -> Vec<Vec<char>> {
    let mut out: Vec<Vec<char>> = vec![vec![]];
    let mut layer: Vec<Vec<char>> = vec![vec![]];
    for _ in 0..4 {
        layer = layer
            .iter()
            .flat_map(|s| {
                ALPHABET.iter().map(move |&c| {
                    let mut t = s.clone();
                    t.push(c);
                    t
                })
            })
            .collect();
        out.extend(layer.iter().cloned());
    }
    let all: Vec<char> = ALPHABET.iter().chain(EXTRA.iter()).copied().collect();
    for _ in 0..random {
        let len = rng.below(9) as usize;
        out.push((0..len).map(|_| rng.pick(&all)).collect());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(s: &str) -> Expr {
        Expr::Literal(s.into())
    }

    fn rep(e: Expr, min: u32, max: Option<u32>) -> Expr {
        Expr::Repeat {
            expr: Box::new(e),
            min,
            max,
        }
    }

    fn m(expr: &Expr, s: &str) -> bool {
        matches(expr, &s.chars().collect::<Vec<_>>())
    }

    #[test]
    fn hand_computed_cases_match() {
        let ab = Expr::Class {
            negated: false,
            ranges: vec![('a', 'b')],
        };
        let not_a = Expr::Class {
            negated: true,
            ranges: vec![('a', 'a')],
        };
        assert!(m(&Expr::Empty, ""));
        assert!(!m(&Expr::Empty, "a"));
        assert!(m(&lit("é世"), "é世"));
        assert!(!m(&lit("é世"), "é"));
        assert!(m(&ab, "b") && !m(&ab, "c") && !m(&ab, ""));
        assert!(m(&not_a, "世") && !m(&not_a, "a"));
        assert!(m(&Expr::Any, "\u{10ffff}") && !m(&Expr::Any, "ab"));
        assert!(!m(&Expr::Ref("x".into()), ""));
        assert!(!m(&Expr::Alt(vec![]), ""));
        assert!(m(&Expr::Seq(vec![]), ""));
        let seq = Expr::Seq(vec![lit("a"), rep(lit("b"), 0, None), lit("c")]);
        assert!(m(&seq, "ac") && m(&seq, "abbbc") && !m(&seq, "abca"));
        let bounded = rep(lit("ab"), 2, Some(3));
        let got: Vec<bool> = ["ab", "abab", "ababab", "abababab"]
            .iter()
            .map(|s| m(&bounded, s))
            .collect();
        assert_eq!(got, vec![false, true, true, false]);
        let nested = rep(rep(lit("a"), 0, Some(1)), 0, None);
        assert!(m(&nested, "") && m(&nested, "aaaa") && !m(&nested, "b"));
        let alt = rep(Expr::Alt(vec![lit("a"), lit("bc")]), 1, None);
        assert!(m(&alt, "abca") && !m(&alt, "") && !m(&alt, "ab"));
        assert!(!m(&rep(lit("a"), 3, Some(2)), "aaa"));
        let exact = rep(ab, 3, Some(3));
        assert!(m(&exact, "aba") && !m(&exact, "ab") && !m(&exact, "abab"));
    }

    #[test]
    fn generator_is_deterministic_and_bounded() {
        let (mut r1, mut r2) = (Lcg(7), Lcg(7));
        let a: Vec<Expr> = (0..20).map(|_| random_expr(&mut r1, 4)).collect();
        let b: Vec<Expr> = (0..20).map(|_| random_expr(&mut r2, 4)).collect();
        assert_eq!(a, b);
        assert!(a.windows(2).any(|w| w[0] != w[1]));
        let strings = test_strings(&mut Lcg(1), 60);
        assert_eq!(strings.len(), 1 + 5 + 25 + 125 + 625 + 60);
    }
}
