//! Backend parity tests for GrammarDfaOps.
//!
//! The CPU mask is the reference, and every backend must match it exactly.
//! Each case builds its grammar with `DeviceGrammarDfa::from_byte_dfa`.

#![cfg_attr(
    not(any(feature = "wgpu", feature = "cuda")),
    allow(dead_code, unused_imports)
)]

use super::helpers::*;
use boostr::inference::grammar::json_schema::{SchemaOptions, compile_json_schema};
use boostr::inference::grammar::{ByteDfa, CompileLimits, VocabBytes, compile};
use boostr::ops::traits::inference::grammar::{DeviceGrammarDfa, GrammarDfaOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// The synthetic vocabulary size of the JSON-schema case.
const LCG_VOCAB: usize = 20_000;

/// One parity case: a DFA, a vocabulary, its EOS ids and the states to mask from.
struct Case {
    name: &'static str,
    dfa: ByteDfa,
    vocab: VocabBytes,
    eos: Vec<u32>,
    states: Vec<u32>,
}

fn every_state(dfa: &ByteDfa) -> Vec<u32> {
    (0..dfa.num_states() as u32).collect()
}

fn small_case() -> Case {
    let dfa = compile(r#"root ::= "a" ("b" "a")*"#, &CompileLimits::default()).unwrap();
    let tokens: [&[u8]; 7] = [b"a", b"b", b"ab", b"", b"ba", b"aa", b"</s>"];
    Case {
        name: "a(ba)*",
        states: every_state(&dfa),
        dfa,
        vocab: VocabBytes::new(tokens).unwrap(),
        eos: vec![6],
    }
}

fn digits_case() -> Case {
    let dfa = compile("root ::= [12][34][56]", &CompileLimits::default()).unwrap();
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
    Case {
        name: "[12][34][56]",
        states: every_state(&dfa),
        dfa,
        vocab: VocabBytes::new(tokens).unwrap(),
        eos: vec![9],
    }
}

/// Deterministic LCG step.
fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed >> 8
}

/// Every single byte first, then LCG pieces of 0 to 6 bytes, mostly JSON characters.
fn lcg_vocab(n: usize) -> Vec<Vec<u8>> {
    const JSON: &[u8] = b"{}[]\":,.-+0123456789aeghmnstr \n";
    let mut seed = 0x1234_5678u32;
    let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let pieces: [&[u8]; 8] = [
        b"{\"",
        b"\"name\"",
        b"\":",
        b"\":\"",
        b"\",\"",
        b"age",
        b"12",
        b"]}",
    ];
    tokens.extend(pieces.iter().map(|p| p.to_vec()));
    while tokens.len() < n {
        let len = lcg(&mut seed) % 7;
        let token = (0..len)
            .map(|_| {
                let pick = lcg(&mut seed);
                if pick.is_multiple_of(4) {
                    (pick >> 4) as u8
                } else {
                    JSON[(pick >> 4) as usize % JSON.len()]
                }
            })
            .collect();
        tokens.push(token);
    }
    tokens
}

fn json_case() -> Case {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age"]
    });
    let opts = SchemaOptions::default();
    let dfa = compile_json_schema(&schema, &opts, &CompileLimits::default()).unwrap();
    let text = br#"{"name":"x","age":12,"tags":["a"]}"#;
    let mut states: Vec<u32> = (0..=text.len())
        .map_while(|end| dfa.walk(dfa.start(), &text[..end]))
        .collect();
    let stride = (dfa.num_states() / 16).max(1);
    states.extend((0..dfa.num_states()).step_by(stride).map(|s| s as u32));
    states.sort_unstable();
    states.dedup();
    assert!(
        states.len() > 4,
        "the sample reaches few states: {states:?}"
    );
    let last = LCG_VOCAB as u32 - 1;
    Case {
        name: "json schema, LCG vocab",
        dfa,
        vocab: VocabBytes::new(lcg_vocab(LCG_VOCAB)).unwrap(),
        eos: vec![last, last - 1],
        states,
    }
}

fn cases() -> Vec<Case> {
    vec![small_case(), digits_case(), json_case()]
}

/// Deterministic logits with distinct values.
fn logits_data(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.37 + 0.5).sin() * 4.0)
        .collect()
}

/// The mask of the last row of `[2, vocab]` logits on one backend, per state.
fn masks_on<R: Runtime<DType = numr::dtype::DType>>(
    client: &R::Client,
    device: &R::Device,
    case: &Case,
) -> Vec<Vec<f32>>
where
    R::Client: GrammarDfaOps<R>,
{
    let vocab_size = case.vocab.len();
    let data = logits_data(2 * vocab_size);
    let logits = Tensor::<R>::from_slice(&data, &[2, vocab_size], device).unwrap();
    let mut grammar =
        DeviceGrammarDfa::<R>::from_byte_dfa(&case.dfa, &case.vocab, &case.eos, device).unwrap();
    let again =
        DeviceGrammarDfa::<R>::from_byte_dfa(&case.dfa, &case.vocab, &case.eos, device).unwrap();
    assert!(
        grammar
            .vocab()
            .bytes()
            .shares_storage_with(again.vocab().bytes()),
        "{}: one vocabulary upload per device",
        case.name
    );
    assert!(
        grammar
            .vocab()
            .offsets()
            .shares_storage_with(again.vocab().offsets()),
        "{}: one offsets upload per device",
        case.name
    );
    let masks = case
        .states
        .iter()
        .map(|&state| {
            grammar.set_current_state(state).unwrap();
            client
                .grammar_dfa_mask_logits(&logits, &grammar)
                .unwrap()
                .to_vec::<f32>()
        })
        .collect();
    assert_eq!(
        logits.to_vec::<f32>(),
        data,
        "{}: the input logits stay unchanged",
        case.name
    );
    masks
}

/// The ids the rules allow from `state`, from the host DFA.
fn expected(case: &Case, state: u32) -> Vec<usize> {
    (0..case.vocab.len())
        .filter(|&id| {
            if case.eos.contains(&(id as u32)) {
                return case.dfa.is_accepting(state);
            }
            let bytes = case.vocab.token_bytes(id as u32).unwrap_or_default();
            !bytes.is_empty() && case.dfa.walk(state, bytes).is_some()
        })
        .collect()
}

#[test]
fn test_grammar_dfa_mask_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    for case in cases() {
        let cpu = masks_on::<CpuRuntime>(&cpu_client, &cpu_device, &case);
        let vocab_size = case.vocab.len();
        for (mask, &state) in cpu.iter().zip(&case.states) {
            let row = &mask[vocab_size..];
            let allowed: Vec<usize> = (0..vocab_size).filter(|&i| row[i].is_finite()).collect();
            assert_eq!(
                allowed,
                expected(&case, state),
                "{} from state {state}",
                case.name
            );
            assert!(
                mask[..vocab_size].iter().all(|v| v.is_finite()),
                "row 0 stays unmasked"
            );
        }

        #[cfg(feature = "cuda")]
        with_cuda_backend(|client, device| {
            let gpu = masks_on::<numr::runtime::cuda::CudaRuntime>(&client, &device, &case);
            for ((g, c), state) in gpu.iter().zip(&cpu).zip(&case.states) {
                assert_eq!(g, c, "cuda {} from state {state}", case.name);
            }
        });

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|client, device| {
            let gpu = masks_on::<numr::runtime::wgpu::WgpuRuntime>(&client, &device, &case);
            for ((g, c), state) in gpu.iter().zip(&cpu).zip(&case.states) {
                assert_eq!(g, c, "wgpu {} from state {state}", case.name);
            }
        });
    }
}

/// A half-precision logits dtype.
#[cfg(feature = "f16")]
#[derive(Clone, Copy, Debug)]
enum Half {
    F16,
    Bf16,
}

#[cfg(feature = "f16")]
impl Half {
    const ALL: [Half; 2] = [Half::F16, Half::Bf16];

    fn dtype(self) -> numr::dtype::DType {
        match self {
            Half::F16 => numr::dtype::DType::F16,
            Half::Bf16 => numr::dtype::DType::BF16,
        }
    }

    fn tensor<R: Runtime<DType = numr::dtype::DType>>(
        self,
        data: &[f32],
        shape: &[usize],
        device: &R::Device,
    ) -> Tensor<R> {
        match self {
            Half::F16 => {
                let h: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
                Tensor::<R>::from_slice(&h, shape, device).unwrap()
            }
            Half::Bf16 => {
                let h: Vec<half::bf16> = data.iter().map(|&x| half::bf16::from_f32(x)).collect();
                Tensor::<R>::from_slice(&h, shape, device).unwrap()
            }
        }
    }

    fn bits<R: Runtime<DType = numr::dtype::DType>>(self, t: &Tensor<R>) -> Vec<u16> {
        match self {
            Half::F16 => t
                .to_vec::<half::f16>()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
            Half::Bf16 => t
                .to_vec::<half::bf16>()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
        }
    }
}

/// The half-precision mask bits of `[2, vocab]` logits on one backend, per state.
///
/// Each mask keeps the input dtype, and the input logits stay unchanged.
#[cfg(feature = "f16")]
fn half_masks_on<R: Runtime<DType = numr::dtype::DType>>(
    client: &R::Client,
    device: &R::Device,
    case: &Case,
    half: Half,
) -> Vec<Vec<u16>>
where
    R::Client: GrammarDfaOps<R>,
{
    let vocab_size = case.vocab.len();
    let data = logits_data(2 * vocab_size);
    let logits = half.tensor::<R>(&data, &[2, vocab_size], device);
    let before = half.bits(&logits);
    let mut grammar =
        DeviceGrammarDfa::<R>::from_byte_dfa(&case.dfa, &case.vocab, &case.eos, device).unwrap();
    let masks = case
        .states
        .iter()
        .map(|&state| {
            grammar.set_current_state(state).unwrap();
            let out = client.grammar_dfa_mask_logits(&logits, &grammar).unwrap();
            assert_eq!(out.dtype(), half.dtype(), "{} {half:?}", case.name);
            half.bits(&out)
        })
        .collect();
    assert_eq!(half.bits(&logits), before, "{}: input unchanged", case.name);
    masks
}

#[cfg(feature = "f16")]
#[test]
fn test_grammar_dfa_mask_half_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    for case in cases() {
        let vocab_size = case.vocab.len();
        let data = logits_data(2 * vocab_size);
        for half in Half::ALL {
            let cpu = half_masks_on::<CpuRuntime>(&cpu_client, &cpu_device, &case, half);
            let input = half.tensor::<CpuRuntime>(&data, &[2, vocab_size], &cpu_device);
            let input_bits = half.bits(&input);
            let infinity = half.tensor::<CpuRuntime>(&[f32::NEG_INFINITY], &[1], &cpu_device);
            let neg_inf = half.bits(&infinity)[0];
            for (mask, &state) in cpu.iter().zip(&case.states) {
                let row = &mask[vocab_size..];
                let allowed: Vec<usize> = (0..vocab_size).filter(|&i| row[i] != neg_inf).collect();
                let want = expected(&case, state);
                assert_eq!(allowed, want, "{} {half:?} from state {state}", case.name);
                for i in 0..vocab_size {
                    if want.contains(&i) {
                        assert_eq!(row[i], input_bits[vocab_size + i], "{} bits", case.name);
                    }
                }
                assert_eq!(&mask[..vocab_size], &input_bits[..vocab_size]);
            }

            #[cfg(feature = "cuda")]
            with_cuda_backend(|client, device| {
                let gpu = half_masks_on::<numr::runtime::cuda::CudaRuntime>(
                    &client, &device, &case, half,
                );
                for ((g, c), state) in gpu.iter().zip(&cpu).zip(&case.states) {
                    assert_eq!(g, c, "cuda {} {half:?} from state {state}", case.name);
                }
            });
        }
    }
}

#[test]
fn test_grammar_dfa_mask_refuses_f64_and_i32_logits() {
    let (cpu_client, cpu_device) = setup_cpu();
    let case = small_case();
    let grammar = DeviceGrammarDfa::<CpuRuntime>::from_byte_dfa(
        &case.dfa,
        &case.vocab,
        &case.eos,
        &cpu_device,
    )
    .unwrap();
    let n = case.vocab.len();
    let wide = Tensor::<CpuRuntime>::from_slice(&vec![0.0f64; n], &[n], &cpu_device).unwrap();
    let err = cpu_client
        .grammar_dfa_mask_logits(&wide, &grammar)
        .unwrap_err();
    assert!(err.to_string().contains("F64"), "{err}");
    let ints = Tensor::<CpuRuntime>::from_slice(&vec![0i32; n], &[n], &cpu_device).unwrap();
    let err = cpu_client
        .grammar_dfa_mask_logits(&ints, &grammar)
        .unwrap_err();
    assert!(err.to_string().contains("I32"), "{err}");
}
