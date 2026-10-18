//! CPU implementation of GrammarDfaOps.
//!
//! Each vocabulary token walks the DFA from the current state over its bytes.
//! The rules match the CUDA and WebGPU kernels.

use crate::error::Result;
use crate::inference::grammar::device::check_logits;
use crate::ops::traits::inference::grammar::{DeviceGrammarDfa, GrammarDfaOps};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl GrammarDfaOps<CpuRuntime> for CpuClient {
    fn grammar_dfa_mask_logits(
        &self,
        logits: &Tensor<CpuRuntime>,
        grammar: &DeviceGrammarDfa<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        let vocab_size = grammar.vocab_size();
        check_logits(logits, vocab_size)?;
        let mut data: Vec<f32> = logits.try_to_vec()?;
        let start = data.len().saturating_sub(vocab_size);
        let view = DfaView::new(grammar);
        let state = grammar.current_state();
        if let Some(row) = data.get_mut(start..) {
            for (token, logit) in row.iter_mut().enumerate() {
                if !view.allows(state, token) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Tensor::from_slice(&data, logits.shape(), logits.device())?)
    }
}

/// Host views of the grammar tables. Every value is an F32 integer.
struct DfaView<'a> {
    table: &'a [f32],
    accepting: &'a [f32],
    bytes: &'a [f32],
    offsets: &'a [f32],
    eos: &'a [f32],
    num_states: usize,
}

impl<'a> DfaView<'a> {
    fn new(grammar: &'a DeviceGrammarDfa<CpuRuntime>) -> Self {
        let vocab = grammar.vocab();
        // SAFETY: CPU storage pointers are host pointers. `from_byte_dfa` builds
        // each table as one whole F32 tensor, and `grammar` keeps it alive for 'a.
        let (table, accepting, bytes, offsets, eos) = unsafe {
            (
                grammar.transition_table().storage().as_host_slice::<f32>(),
                grammar.accepting_mask().storage().as_host_slice::<f32>(),
                vocab.bytes().storage().as_host_slice::<f32>(),
                vocab.offsets().storage().as_host_slice::<f32>(),
                grammar.eos_tokens().storage().as_host_slice::<f32>(),
            )
        };
        Self {
            table,
            accepting,
            bytes: bytes.get(..vocab.total_bytes()).unwrap_or_default(),
            offsets,
            eos: eos.get(..grammar.num_eos()).unwrap_or_default(),
            num_states: grammar.num_states(),
        }
    }

    /// True when the grammar allows `token` from `state`.
    fn allows(&self, state: u32, token: usize) -> bool {
        let cur = state as usize;
        if cur >= self.num_states {
            return false;
        }
        if self.eos.contains(&(token as f32)) {
            return self.accepting.get(cur).is_some_and(|&a| a != 0.0);
        }
        let start = self.offsets.get(token).copied();
        let end = self.offsets.get(token.saturating_add(1)).copied();
        let (Some(start), Some(end)) = (start, end) else {
            return false;
        };
        // A zero-length token makes no progress, so it is never allowed.
        if end <= start {
            return false;
        }
        let Some(bytes) = self.bytes.get(start as usize..end as usize) else {
            return false;
        };
        let states = 0.0..self.num_states as f32;
        let mut at = cur;
        for &byte in bytes {
            if !(0.0..=255.0).contains(&byte) {
                return false;
            }
            let Some(&next) = self.table.get(at * 256 + byte as usize) else {
                return false;
            };
            if !states.contains(&next) {
                return false;
            }
            at = next as usize;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::{ByteDfa, CompileLimits, VocabBytes, compile};
    use numr::runtime::cpu::CpuDevice;

    const NUMBER: &str = concat!(
        r#"root ::= "-"? ("0" | [1-9] [0-9]*) "#,
        r#"("." [0-9]+)? ([eE] ("-" | "+")? [0-9]+)?"#
    );

    fn dfa(src: &str) -> ByteDfa {
        compile(src, &CompileLimits::default()).unwrap()
    }

    fn grammar(d: &ByteDfa, vocab: &VocabBytes, eos: &[u32]) -> DeviceGrammarDfa<CpuRuntime> {
        DeviceGrammarDfa::from_byte_dfa(d, vocab, eos, &CpuDevice::new()).unwrap()
    }

    /// Masked `logits` from `state`.
    fn masked(g: &mut DeviceGrammarDfa<CpuRuntime>, state: u32, logits: &[f32]) -> Vec<f32> {
        g.set_current_state(state).unwrap();
        let device = CpuDevice::new();
        let t = Tensor::<CpuRuntime>::from_slice(logits, &[logits.len()], &device).unwrap();
        let client = CpuClient::new(device);
        client.grammar_dfa_mask_logits(&t, g).unwrap().to_vec()
    }

    /// The ids the mask allows from `state`.
    fn allowed(g: &mut DeviceGrammarDfa<CpuRuntime>, state: u32) -> Vec<usize> {
        let zeros = vec![0.0; g.vocab_size()];
        let out = masked(g, state, &zeros);
        (0..out.len()).filter(|&i| out[i].is_finite()).collect()
    }

    /// The ids the rules allow, from the host DFA.
    fn expected(d: &ByteDfa, vocab: &VocabBytes, eos: &[u32], state: u32) -> Vec<usize> {
        (0..vocab.len())
            .filter(|&i| {
                let bytes = vocab.token_bytes(i as u32).unwrap_or_default();
                if eos.contains(&(i as u32)) {
                    d.is_accepting(state)
                } else {
                    !bytes.is_empty() && d.walk(state, bytes).is_some()
                }
            })
            .collect()
    }

    fn at(d: &ByteDfa, prefix: &[u8]) -> u32 {
        d.walk(d.start(), prefix).unwrap()
    }

    #[test]
    fn three_digit_grammar_allows_exactly_the_expected_tokens() {
        let d = dfa("root ::= [12][34][56]");
        // 9 is EOS. Its piece "1" is valid text at the start, so a walked piece would leak.
        let tokens: [&[u8]; 12] = [
            b"1",
            b"2",
            b"3",
            b"5",
            b"13",
            b"1356",
            b"35",
            &[0xFF],
            b"",
            b"1",
            b"7",
            b"246",
        ];
        let vocab = VocabBytes::new(tokens).unwrap();
        let eos = [9];
        let mut g = grammar(&d, &vocab, &eos);
        let cases: [(&[u8], &[usize]); 4] = [
            // "1356" has four characters, so it never fits the three-character grammar.
            (b"", &[0, 1, 4, 11]),
            (b"1", &[2, 6]),
            (b"13", &[3]),
            (b"135", &[9]),
        ];
        for (prefix, want) in cases {
            let state = at(&d, prefix);
            assert_eq!(allowed(&mut g, state), want, "after {prefix:?}");
            assert_eq!(allowed(&mut g, state), expected(&d, &vocab, &eos, state));
        }
    }

    #[test]
    fn json_number_masks_match_the_host_walk() {
        let d = dfa(NUMBER);
        // Ids 10.. after the digits: 19 is "00", 23 is empty, 24 is EOS.
        let extra: [&[u8]; 16] = [
            b"-",
            b".",
            b"e",
            b"E",
            b"+",
            b"12",
            b"1.5",
            b"e+3",
            b"0.0",
            b"00",
            b"--",
            &[0xC3, 0xA9],
            &[0x80],
            b"",
            b"</s>",
            b" ",
        ];
        let mut tokens: Vec<Vec<u8>> = (b'0'..=b'9').map(|b| vec![b]).collect();
        tokens.extend(extra.iter().map(|t| t.to_vec()));
        let eos_id = tokens.len() as u32 - 2;
        let vocab = VocabBytes::new(&tokens).unwrap();
        let eos = [eos_id];
        let mut g = grammar(&d, &vocab, &eos);
        let cases: [(&[u8], bool); 9] = [
            (b"", false),
            (b"-", false),
            (b"0", true),
            (b"12", true),
            (b"1.", false),
            (b"1.5", true),
            (b"1e", false),
            (b"1e+", false),
            (b"-0.25E-7", true),
        ];
        for (prefix, accepts) in cases {
            let state = at(&d, prefix);
            let got = allowed(&mut g, state);
            assert_eq!(got, expected(&d, &vocab, &eos, state), "after {prefix:?}");
            assert_eq!(
                got.contains(&(eos_id as usize)),
                accepts,
                "EOS after {prefix:?}"
            );
            assert!(
                !got.contains(&(tokens.len() - 3)),
                "empty token after {prefix:?}"
            );
        }
        assert!(
            !allowed(&mut g, at(&d, b"")).contains(&19),
            "\"00\" at the start"
        );
    }

    #[test]
    fn eos_follows_the_accepting_flag_only() {
        let d = dfa("root ::= \"ab\"");
        let vocab = VocabBytes::new([&b"a"[..], b"b", b"", b"ab"]).unwrap();
        let mut g = grammar(&d, &vocab, &[3]);
        assert_eq!(allowed(&mut g, at(&d, b"")), vec![0]);
        assert_eq!(allowed(&mut g, at(&d, b"a")), vec![1]);
        assert_eq!(allowed(&mut g, at(&d, b"ab")), vec![3]);

        let optional = dfa("root ::= \"a\"?");
        let mut g = grammar(&optional, &vocab, &[3]);
        assert_eq!(allowed(&mut g, optional.start()), vec![0, 3]);
        assert_eq!(allowed(&mut g, at(&optional, b"a")), vec![3]);
    }

    #[test]
    fn no_eos_ids_and_empty_tokens_are_never_allowed() {
        let d = dfa("root ::= \"a\"?");
        let vocab = VocabBytes::new([&b""[..], b"a", b""]).unwrap();
        let mut g = grammar(&d, &vocab, &[]);
        assert_eq!(allowed(&mut g, d.start()), vec![1]);
        assert_eq!(allowed(&mut g, at(&d, b"a")), Vec::<usize>::new());
    }

    #[test]
    fn allowed_logits_keep_their_values_on_the_last_row() {
        let d = dfa("root ::= \"ab\"");
        let vocab = VocabBytes::new([&b"a"[..], b"b"]).unwrap();
        let g = grammar(&d, &vocab, &[]);
        let device = CpuDevice::new();
        let data = [9.0f32, 8.0, 1.5, -2.0];
        let t = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &device).unwrap();
        let out: Vec<f32> = CpuClient::new(device)
            .grammar_dfa_mask_logits(&t, &g)
            .unwrap()
            .to_vec();
        assert_eq!(&out[..3], &data[..3]);
        assert_eq!(out[3], f32::NEG_INFINITY);
        assert_eq!(t.to_vec::<f32>(), data);
    }

    /// Greedy decoding over logits that prefer EOS, then invalid tokens.
    fn greedy(d: &ByteDfa, tokens: &[Vec<u8>], eos_id: u32, logits: &[f32]) -> Vec<u8> {
        let vocab = VocabBytes::new(tokens).unwrap();
        let mut g = grammar(d, &vocab, &[eos_id]);
        let mut state = d.start();
        let mut out = Vec::new();
        for _ in 0..64 {
            let row = masked(&mut g, state, logits);
            let best = (0..row.len())
                .filter(|&i| row[i].is_finite())
                .fold(None, |b: Option<usize>, i| match b {
                    Some(j) if row[j] >= row[i] => Some(j),
                    _ => Some(i),
                })
                .expect("the grammar leaves no token");
            if best as u32 == eos_id {
                assert!(d.is_accepting(state), "EOS from a non-accepting state");
                return out;
            }
            let bytes = vocab.token_bytes(best as u32).unwrap();
            state = d.walk(state, bytes).expect("an allowed token walks");
            out.extend_from_slice(bytes);
        }
        panic!("no EOS after 64 steps: {out:?}");
    }

    /// Logits from a fixed LCG, with EOS first, then the invalid tokens.
    fn hostile_logits(n: usize, eos: usize, invalid: &[usize]) -> Vec<f32> {
        let mut seed = 0x2545_F491u32;
        let mut logits: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        logits[eos] = 100.0;
        for (rank, &i) in invalid.iter().enumerate() {
            logits[i] = 50.0 - rank as f32;
        }
        logits
    }

    #[test]
    fn greedy_decoding_ends_at_eos_in_an_accepting_state() {
        let d = dfa("root ::= [12][34][56]");
        let pieces: [&[u8]; 10] = [
            b"1",
            b"2",
            b"3",
            b"5",
            b"13",
            b"1356",
            b"7",
            &[0xFF],
            b"",
            b"</s>",
        ];
        let tokens: Vec<Vec<u8>> = pieces.iter().map(|t| t.to_vec()).collect();
        let logits = hostile_logits(tokens.len(), 9, &[8, 7, 6]);
        let out = greedy(&d, &tokens, 9, &logits);
        assert!(d.accepts(&out), "{out:?}");

        let d = dfa(NUMBER);
        let mut tokens: Vec<Vec<u8>> = (b'0'..=b'9').map(|b| vec![b]).collect();
        let extra: [&[u8]; 7] = [b"-", b".", b"e", b"x", b"", &[0xFE], b"</s>"];
        tokens.extend(extra.iter().map(|t| t.to_vec()));
        let eos = tokens.len() - 1;
        let logits = hostile_logits(tokens.len(), eos, &[eos - 1, eos - 2, eos - 3, 11]);
        let out = greedy(&d, &tokens, eos as u32, &logits);
        assert!(d.accepts(&out), "{out:?}");
        assert!(!out.is_empty());
    }
}
