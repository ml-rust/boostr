//! `DeviceGrammarDfa`: one grammar's DFA on a device, ready for logit masking.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::dfa::ByteDfa;
use super::vocab::{DeviceVocab, F32_EXACT_LIMIT, VocabBytes};
use crate::error::{Error, Result};

/// Device table entry for a byte the grammar forbids in that state.
pub const INVALID_STATE: i32 = -1;

/// Most distinct EOS token ids one `DeviceGrammarDfa` takes.
///
/// Each mask thread scans the EOS list, so the list stays short.
pub const MAX_EOS_TOKENS: usize = 64;

/// One grammar's DFA on a device, with a shared vocabulary upload.
///
/// The mask op allows a token from the current state by these rules:
/// - An EOS token is allowed iff the current state accepts. Its bytes are never walked.
/// - A non-EOS token is allowed iff each of its bytes has a transition.
/// - A non-EOS token with zero bytes is never allowed. It makes no progress.
/// - No token is allowed from a state outside the table.
///
/// Every table holds F32 values that are exact integers:
/// - `transition_table`: `[num_states * 256]`, the next state or `INVALID_STATE`.
/// - `accepting_mask`: `[num_states]`, 1 for an accepting state, else 0.
/// - `eos_tokens`: `[max(num_eos, 1)]`, sorted ids. One `-1` pads an empty list.
///
/// The table is per grammar: 1 KiB per state, so 16 MiB at 16,384 states.
/// The vocabulary tensors belong to the `VocabBytes` and are shared. A clone
/// shares every tensor.
pub struct DeviceGrammarDfa<R: Runtime> {
    transition_table: Tensor<R>,
    accepting_mask: Tensor<R>,
    vocab: DeviceVocab<R>,
    eos_tokens: Tensor<R>,
    num_eos: usize,
    num_states: usize,
    vocab_size: usize,
    current_state: u32,
}

impl<R: Runtime> Clone for DeviceGrammarDfa<R> {
    fn clone(&self) -> Self {
        Self {
            transition_table: self.transition_table.clone(),
            accepting_mask: self.accepting_mask.clone(),
            vocab: self.vocab.clone(),
            eos_tokens: self.eos_tokens.clone(),
            num_eos: self.num_eos,
            num_states: self.num_states,
            vocab_size: self.vocab_size,
            current_state: self.current_state,
        }
    }
}

impl<R: Runtime<DType = DType>> DeviceGrammarDfa<R> {
    /// Upload `dfa` to `device`, starting at its start state.
    ///
    /// `vocab` uploads on its first use per device, and later grammars reuse
    /// that copy. `eos_token_ids` can repeat ids and can be empty. With no
    /// EOS id, the grammar can never end generation by itself.
    ///
    /// An EOS id past the vocabulary, more than `MAX_EOS_TOKENS` distinct ids,
    /// and more than `F32_EXACT_LIMIT` states are errors.
    pub fn from_byte_dfa(
        dfa: &ByteDfa,
        vocab: &VocabBytes,
        eos_token_ids: &[u32],
        device: &R::Device,
    ) -> Result<Self> {
        let num_states = dfa.num_states();
        if num_states > F32_EXACT_LIMIT {
            return Err(Error::InvalidArgument {
                arg: "dfa",
                reason: format!(
                    "DeviceGrammarDfa::from_byte_dfa: the DFA has {num_states} states, above the \
                     limit of {F32_EXACT_LIMIT}. Lower AutomatonLimits::max_dfa_states"
                ),
            });
        }
        let transitions = dfa.transitions();
        if Some(transitions.len()) != num_states.checked_mul(256) {
            return Err(Error::InvalidArgument {
                arg: "dfa",
                reason: format!(
                    "DeviceGrammarDfa::from_byte_dfa: the table holds {} entries for {num_states} \
                     states, not 256 per state",
                    transitions.len()
                ),
            });
        }
        let eos = eos_list(eos_token_ids, vocab.len())?;
        // Every kept id is below num_states <= F32_EXACT_LIMIT, so the F32 value is exact.
        let table: Vec<f32> = transitions
            .iter()
            .map(|&next| {
                if (next as usize) < num_states {
                    next as f32
                } else {
                    INVALID_STATE as f32
                }
            })
            .collect();
        let accepting: Vec<f32> = dfa
            .accepting()
            .iter()
            .map(|&a| if a { 1.0 } else { 0.0 })
            .collect();
        let mut eos_f32: Vec<f32> = eos.iter().map(|&id| id as f32).collect();
        if eos_f32.is_empty() {
            eos_f32.push(INVALID_STATE as f32);
        }
        Ok(Self {
            transition_table: Tensor::<R>::from_slice(&table, &[table.len()], device)?,
            accepting_mask: Tensor::<R>::from_slice(&accepting, &[accepting.len()], device)?,
            vocab: vocab.device::<R>(device)?,
            eos_tokens: Tensor::<R>::from_slice(&eos_f32, &[eos_f32.len()], device)?,
            num_eos: eos.len(),
            num_states,
            vocab_size: vocab.len(),
            current_state: dfa.start(),
        })
    }
}

impl<R: Runtime> DeviceGrammarDfa<R> {
    /// The DFA state the next mask starts from.
    pub fn current_state(&self) -> u32 {
        self.current_state
    }

    /// Set the state the next mask starts from, after the sampler advances its host DFA.
    ///
    /// A state outside the table is an error, and the state stays unchanged.
    pub fn set_current_state(&mut self, state: u32) -> Result<()> {
        if (state as usize) >= self.num_states {
            return Err(Error::InvalidArgument {
                arg: "state",
                reason: format!(
                    "DeviceGrammarDfa::set_current_state: state {state} is outside the DFA of {} \
                     states",
                    self.num_states
                ),
            });
        }
        self.current_state = state;
        Ok(())
    }

    /// Number of DFA states.
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Number of token ids the mask covers.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of distinct EOS ids. `eos_tokens` holds them first.
    pub fn num_eos(&self) -> usize {
        self.num_eos
    }

    /// `[num_states * 256]` F32 transition table.
    pub fn transition_table(&self) -> &Tensor<R> {
        &self.transition_table
    }

    /// `[num_states]` F32 accepting flags.
    pub fn accepting_mask(&self) -> &Tensor<R> {
        &self.accepting_mask
    }

    /// `[max(num_eos, 1)]` F32 EOS ids.
    pub fn eos_tokens(&self) -> &Tensor<R> {
        &self.eos_tokens
    }

    /// The shared vocabulary upload.
    pub fn vocab(&self) -> &DeviceVocab<R> {
        &self.vocab
    }
}

/// Sorted distinct EOS ids, each inside the vocabulary.
fn eos_list(ids: &[u32], vocab_size: usize) -> Result<Vec<u32>> {
    let mut eos = ids.to_vec();
    eos.sort_unstable();
    eos.dedup();
    if let Some(&bad) = eos.iter().find(|&&id| (id as usize) >= vocab_size) {
        return Err(Error::InvalidArgument {
            arg: "eos_token_ids",
            reason: format!(
                "DeviceGrammarDfa::from_byte_dfa: EOS id {bad} is outside the vocabulary of \
                 {vocab_size} tokens"
            ),
        });
    }
    if eos.len() > MAX_EOS_TOKENS {
        return Err(Error::InvalidArgument {
            arg: "eos_token_ids",
            reason: format!(
                "DeviceGrammarDfa::from_byte_dfa: {} distinct EOS ids, above the limit of \
                 {MAX_EOS_TOKENS}",
                eos.len()
            ),
        });
    }
    Ok(eos)
}

/// Check that `logits` fit the mask: F32, F16 or BF16, contiguous, and at least `vocab_size` values.
///
/// F16 and BF16 need the `f16` feature. Any other dtype is an error that names it.
/// The mask covers the last `vocab_size` values.
pub(crate) fn check_logits<R: Runtime<DType = DType>>(
    logits: &Tensor<R>,
    vocab_size: usize,
) -> Result<()> {
    match logits.dtype() {
        DType::F32 => {}
        DType::F16 | DType::BF16 if cfg!(feature = "f16") => {}
        dtype @ (DType::F16 | DType::BF16) => {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "grammar_dfa_mask_logits: {dtype:?} logits need the `f16` feature. Build \
                     boostr with `f16` or pass F32 logits"
                ),
            });
        }
        dtype => {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "grammar_dfa_mask_logits: logits must be F32, F16 or BF16, got {dtype:?}"
                ),
            });
        }
    }
    if !logits.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "grammar_dfa_mask_logits: logits are not contiguous (shape {:?}, strides {:?}). \
                 Pass a contiguous tensor",
                logits.shape(),
                logits.strides()
            ),
        });
    }
    let total = logits.numel();
    if total < vocab_size {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "grammar_dfa_mask_logits: logits hold {total} values, fewer than the grammar \
                 vocab of {vocab_size}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::{CompileLimits, compile};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn dfa(src: &str) -> ByteDfa {
        compile(src, &CompileLimits::default()).unwrap()
    }

    fn try_build(
        d: &ByteDfa,
        vocab: &VocabBytes,
        eos: &[u32],
    ) -> Result<DeviceGrammarDfa<CpuRuntime>> {
        DeviceGrammarDfa::<CpuRuntime>::from_byte_dfa(d, vocab, eos, &CpuDevice::new())
    }

    fn build(src: &str, vocab: &VocabBytes, eos: &[u32]) -> DeviceGrammarDfa<CpuRuntime> {
        try_build(&dfa(src), vocab, eos).unwrap()
    }

    #[test]
    fn tables_map_forbidden_bytes_to_the_sentinel() {
        let dfa = dfa("root ::= \"ab\"");
        let vocab = VocabBytes::new([&b"a"[..], b"b"]).unwrap();
        let g = try_build(&dfa, &vocab, &[1, 1]).unwrap();
        assert_eq!(g.num_states(), dfa.num_states());
        assert_eq!(g.vocab_size(), 2);
        assert_eq!(g.current_state(), 0);
        assert_eq!(g.num_eos(), 1);
        assert_eq!(g.eos_tokens().to_vec::<f32>(), vec![1.0]);
        let table = g.transition_table().to_vec::<f32>();
        assert_eq!(table.len(), dfa.num_states() * 256);
        for (i, &next) in dfa.transitions().iter().enumerate() {
            let want = if next == ByteDfa::NO_TRANSITION {
                -1.0
            } else {
                next as f32
            };
            assert_eq!(table[i], want, "entry {i}");
        }
        let accepting: Vec<f32> = dfa
            .accepting()
            .iter()
            .map(|&a| if a { 1.0 } else { 0.0 })
            .collect();
        assert_eq!(g.accepting_mask().to_vec::<f32>(), accepting);
    }

    #[test]
    fn an_empty_eos_list_pads_one_entry() {
        let vocab = VocabBytes::new([&b"a"[..]]).unwrap();
        let g = build("root ::= \"a\"", &vocab, &[]);
        assert_eq!(g.num_eos(), 0);
        assert_eq!(g.eos_tokens().to_vec::<f32>(), vec![-1.0]);
    }

    #[test]
    fn bad_eos_ids_are_errors() {
        let d = dfa("root ::= \"a\"");
        let vocab = VocabBytes::new((0..100u8).map(|b| vec![b])).unwrap();
        let past = try_build(&d, &vocab, &[100]).err();
        assert!(past.is_some_and(|e| e.to_string().contains("EOS id 100")));
        let many: Vec<u32> = (0..=MAX_EOS_TOKENS as u32).collect();
        let over = try_build(&d, &vocab, &many).err();
        assert!(over.is_some_and(|e| e.to_string().contains("distinct EOS ids")));
    }

    #[test]
    fn set_current_state_refuses_a_state_outside_the_table() {
        let d = dfa("root ::= \"ab\"");
        let vocab = VocabBytes::new([&b"a"[..]]).unwrap();
        let mut g = try_build(&d, &vocab, &[]).unwrap();
        let last = d.num_states() as u32 - 1;
        assert!(g.set_current_state(last).is_ok());
        assert_eq!(g.current_state(), last);
        assert!(g.set_current_state(last + 1).is_err());
        assert_eq!(g.current_state(), last);
    }

    #[test]
    fn grammars_built_from_one_vocab_share_its_upload() {
        let vocab = VocabBytes::new([&b"1"[..], b"2", b"12"]).unwrap();
        let a = build("root ::= [12]+", &vocab, &[]);
        let b = build("root ::= \"21\"", &vocab.clone(), &[2]);
        assert!(a.vocab().bytes().shares_storage_with(b.vocab().bytes()));
        assert!(a.vocab().offsets().shares_storage_with(b.vocab().offsets()));
        assert!(
            !a.transition_table()
                .shares_storage_with(b.transition_table())
        );
    }

    #[test]
    fn check_logits_refuses_short_and_non_float_logits() {
        let device = CpuDevice::new();
        let short = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2], &[2], &device).unwrap();
        assert!(check_logits(&short, 3).is_err());
        assert!(check_logits(&short, 2).is_ok());
        let ints = Tensor::<CpuRuntime>::from_slice(&[0i32; 4], &[4], &device).unwrap();
        let err = check_logits(&ints, 2).unwrap_err().to_string();
        assert!(err.contains("I32"), "{err}");
        let wide = Tensor::<CpuRuntime>::from_slice(&[0f64; 4], &[4], &device).unwrap();
        let err = check_logits(&wide, 2).unwrap_err().to_string();
        assert!(err.contains("F64"), "{err}");
    }
}
