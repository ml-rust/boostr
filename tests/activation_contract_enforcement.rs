//! What a container that carries an activation contract prevents, and what one
//! that cannot carry it lets through.
//!
//! # The thing that goes wrong
//!
//! A quantized weight is not prepared in a vacuum. Whoever produced it chose a
//! rounding grid against an assumed activation: exact f32 activations for one
//! kernel family, activations themselves quantized to 8-bit codes for another.
//! Those are two different arithmetic contracts, and the SAME weight bytes run
//! under either one. The dot product does not stop, does not warn, and does not
//! produce obvious garbage — it produces a slightly different answer, every
//! layer, compounding through the model.
//!
//! # Why GGUF cannot catch it
//!
//! A GGUF tensor's type is a `ggml_type` enum: `Q6_K` says how the weight's
//! bits are laid out and nothing else. There is no field in it for an
//! activation contract, so a GGUF file cannot state one and a GGUF reader has
//! nothing to check. That is a limit of the format, not a defect in this
//! crate's GGUF code — no GGUF implementation, ours or anyone's, can detect
//! this, because the information was never written down. Our own dispatcher
//! therefore picks between the two contracts on tensor SHAPE alone
//! (`k.is_multiple_of(256)`), which is the correct behaviour available to it.
//!
//! # What TCF adds
//!
//! TCF makes the contract part of the dispatch key (SPECIFICATION.md Section 9):
//! every tensor carries a `ContractRecord`, the loader attaches it to the
//! weight, and the point where a kernel is selected checks the kernel's own
//! declared contract against it. A mismatch is
//! `Error::ActivationContractMismatch` and the operation stops. Section 9
//! defines no float fallback, so a refusal is never quietly rerouted onto
//! whatever kernel happens to be nearby.
//!
//! # Reading this file
//!
//! Half 1 (`gguf_*`) runs one weight under both contracts and measures the gap.
//! Half 2 (`tcf_*`) offers a contracted weight to a kernel that does not
//! satisfy it and shows the refusal. The last test states the difference in one
//! assertion: only the TCF weight can answer "which contract were you prepared
//! for?".
//!
//! Everything here is CPU-only and runs on any machine.

use boostr::QuantMatmulOps;
use boostr::error::Error;
use boostr::quant::cpu::kernels::quantize::quantize_q6k;
use boostr::quant::cpu::kernels::simd::fused_q6k_dot::fused_dot_q6k;
use boostr::quant::cpu::kernels::simd::fused_q6k_q8k_dot::fused_dot_q6k_q8k;
use boostr::quant::cpu::kernels::simd::quantize_act_q8k::{Q8K_BLOCK_BYTES, quantize_f32_to_q8k};
use boostr::quant::{ActivationContract, KernelContract, QuantFormat, QuantTensor, TcfEncoding};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use tcf_core::{
    ContractFlags, ContractRecord, DotAccumulator, ExecutionRole, InputRepresentation, MathMode,
    NativeEncoding, OutputDtype, QuantAxis, RoundingMode, ScaleComputeDtype, pack, quantize,
};

/// Rows of the weight matrix — the output width.
const N: usize = 8;
/// Columns of the weight matrix — the reduction width. A multiple of 256 so
/// the Q6_K super-block divides it evenly.
const K: usize = 512;

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// A deterministic signal with sign changes, a flat run and a spike, so the
/// quantization grid is exercised rather than a smooth ramp that any grid fits.
fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let x = (i + seed) as f32;
            match (i + seed) % 7 {
                0 => 0.75,
                2 => -(x * 0.011).sin() * 2.5,
                4 => (x * 0.037).cos() * 1.5,
                6 => (x * 0.005).sin() * 3.5,
                _ => (x * 0.023).sin() * 1.1 - 0.2,
            }
        })
        .collect()
}

/// The weight, quantized to GGUF `Q6_K`: `[N, K]` row-major, blocks along K.
fn q6k_weight_bytes() -> Vec<u8> {
    let blocks_per_row = K / QuantFormat::Q6K.block_size();
    let row_bytes = blocks_per_row * QuantFormat::Q6K.block_bytes();
    let mut bytes = vec![0u8; N * row_bytes];
    for row in 0..N {
        let source = values(K, row * 13 + 1);
        quantize_q6k(&source, &mut bytes[row * row_bytes..(row + 1) * row_bytes]);
    }
    bytes
}

/// Largest absolute disagreement between two equal-length outputs.
fn max_deviation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "outputs must be the same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Root-mean-square magnitude of an output, so a deviation can be read
/// relative to the scale of the numbers it sits in.
fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

/// Cosine similarity: the measure under which a wrong answer still looks right.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

// ---------------------------------------------------------------------------
// Half 1 — GGUF is silent
// ---------------------------------------------------------------------------

/// One weight, two activation contracts, both of which run.
///
/// `fused_dot_q6k` reads the activation as exact f32 and accumulates in f32.
/// `fused_dot_q6k_q8k` first quantizes the activation to Q8_K 8-bit codes and
/// accumulates on integers. Neither returns a `Result`: the signature itself is
/// the evidence that nothing in the GGUF path has a way to object. The caller
/// gets a number either way.
///
/// The assertions gate three things a reader should not have to take on trust:
/// the two answers are NOT the same, the difference is material against the
/// output's own scale, and it is nonetheless invisible to the similarity check
/// people actually use to sanity-check a quantized model.
#[test]
fn gguf_runs_one_weight_under_two_activation_contracts_and_reports_neither() {
    let weight_bytes = q6k_weight_bytes();
    let row_bytes = weight_bytes.len() / N;
    let activation = values(K, 977);

    // Contract A: exact f32 activations, f32 accumulator.
    let exact_f32: Vec<f32> = (0..N)
        .map(|j| {
            fused_dot_q6k(
                &activation,
                &weight_bytes[j * row_bytes..(j + 1) * row_bytes],
                K,
            )
        })
        .collect();

    // Contract B: activations quantized to 8-bit codes, integer accumulator.
    let mut act_q8k = vec![0u8; (K / 256) * Q8K_BLOCK_BYTES];
    quantize_f32_to_q8k(&activation, &mut act_q8k);
    let dynamic_int8: Vec<f32> = (0..N)
        .map(|j| {
            fused_dot_q6k_q8k(
                &act_q8k,
                &weight_bytes[j * row_bytes..(j + 1) * row_bytes],
                K,
            )
        })
        .collect();

    // Both ran. Neither could have refused.
    assert!(
        exact_f32.iter().all(|v| v.is_finite()),
        "f32-activation path produced a non-finite value: {exact_f32:?}"
    );
    assert!(
        dynamic_int8.iter().all(|v| v.is_finite()),
        "int8-activation path produced a non-finite value: {dynamic_int8:?}"
    );

    let deviation = max_deviation(&exact_f32, &dynamic_int8);
    let scale = rms(&exact_f32);
    let relative = deviation / scale;
    let similarity = cosine(&exact_f32, &dynamic_int8);

    println!(
        "GGUF, one Q6_K weight, two activation contracts:\n  \
         max absolute deviation {deviation:e}\n  \
         output RMS             {scale:e}\n  \
         relative deviation     {relative:e}\n  \
         cosine similarity      {similarity:.9}"
    );

    // The premise. If these two paths ever became numerically identical there
    // would be nothing to protect against, and this file's claim would be
    // false — so it fails here rather than passing quietly.
    assert!(
        deviation > 0.0,
        "the two activation contracts produced identical results, so this file's premise no longer holds"
    );

    // A plausible wrong answer, not obvious garbage: the answer differs by a
    // visible fraction of its own magnitude while cosine similarity stays high
    // enough that the usual quantization sanity check waves it through.
    assert!(
        relative > 1e-4,
        "deviation {relative:e} relative to output scale is below the noise a reader would dismiss"
    );
    assert!(
        similarity > 0.99,
        "cosine similarity {similarity} is low enough to be caught by inspection, which is not the case being demonstrated"
    );

    // And the point of the whole half: nothing in the GGUF representation
    // records which of the two the weight was prepared for.
    let (_client, device) = cpu_setup();
    let weight =
        QuantTensor::<CpuRuntime>::from_bytes(&weight_bytes, QuantFormat::Q6K, &[N, K], &device)
            .expect("Q6_K weight");
    assert!(
        weight.activation_contract().is_none(),
        "a GGUF weight cannot declare an activation contract: ggml_type has no field for one"
    );
}

/// The shipping dispatcher picks one of those two contracts, and picks it on
/// shape.
///
/// `quant_matmul` returns `Ok` and hands back the activation-quantizing answer
/// because `K` is a multiple of 256. No caller asked for that, no caller was
/// told, and no error path exists to tell them.
#[test]
fn gguf_dispatch_silently_selects_the_quantizing_contract() {
    let weight_bytes = q6k_weight_bytes();
    let row_bytes = weight_bytes.len() / N;
    let activation = values(K, 977);

    let exact_f32: Vec<f32> = (0..N)
        .map(|j| {
            fused_dot_q6k(
                &activation,
                &weight_bytes[j * row_bytes..(j + 1) * row_bytes],
                K,
            )
        })
        .collect();
    let mut act_q8k = vec![0u8; (K / 256) * Q8K_BLOCK_BYTES];
    quantize_f32_to_q8k(&activation, &mut act_q8k);
    let dynamic_int8: Vec<f32> = (0..N)
        .map(|j| {
            fused_dot_q6k_q8k(
                &act_q8k,
                &weight_bytes[j * row_bytes..(j + 1) * row_bytes],
                K,
            )
        })
        .collect();

    let (client, device) = cpu_setup();
    let act = Tensor::<CpuRuntime>::from_slice(&activation, &[1, K], &device).expect("activation");
    let weight =
        QuantTensor::<CpuRuntime>::from_bytes(&weight_bytes, QuantFormat::Q6K, &[N, K], &device)
            .expect("Q6_K weight");

    // It runs. There is no second return value saying which contract was used.
    let dispatched = client
        .quant_matmul(&act, &weight)
        .expect("a GGUF weight has nothing to check, so dispatch always succeeds")
        .to_vec::<f32>();

    let gap = max_deviation(&exact_f32, &dynamic_int8);
    let to_int8 = max_deviation(&dispatched, &dynamic_int8);
    let to_f32 = max_deviation(&dispatched, &exact_f32);
    println!(
        "GGUF dispatch on shape [{N}, {K}]: distance to int8-activation answer {to_int8:e}, to f32-activation answer {to_f32:e}"
    );

    assert!(
        to_int8 <= gap * 1e-3,
        "dispatch was expected to take the activation-quantizing path for K={K}; distance to it was {to_int8:e} against a contract gap of {gap:e}"
    );
    assert!(
        to_f32 > to_int8,
        "dispatch matched the f32-activation answer, so the premise of this test no longer holds"
    );
}

// ---------------------------------------------------------------------------
// Half 2 — TCF refuses
// ---------------------------------------------------------------------------

/// A `ContractRecord` as a TCF file stores it. Only the dispatch-bearing fields
/// vary between the cases below.
fn contract_record(
    input_representation: InputRepresentation,
    dot_accumulator: DotAccumulator,
    quant_group: u16,
    quant_range: (i16, i16),
    digest: u8,
) -> ContractRecord {
    ContractRecord {
        contract_id: 1,
        input_representation,
        quant_group,
        quant_axis: QuantAxis::Last,
        rounding_mode: RoundingMode::RnEven,
        qmin: quant_range.0,
        qmax: quant_range.1,
        scale_compute_dtype: ScaleComputeDtype::F32,
        dot_accumulator,
        output_dtype: OutputDtype::F32,
        math_mode: MathMode::ReassociationAllowed,
        kernel_semantics_id: 5,
        calibration_id: 0,
        flags: ContractFlags::NONE,
        contract_digest: [digest; 16],
    }
}

/// The contract of a weight prepared for exact f32 activations.
fn declares_exact_f32(tensor: &str) -> ActivationContract {
    ActivationContract::from_record(
        tensor,
        ExecutionRole::Matmul,
        &contract_record(
            InputRepresentation::F32,
            DotAccumulator::F32,
            0,
            (0, 0),
            0x11,
        ),
    )
}

/// The contract of a weight prepared for 8-bit dynamically quantized
/// activations — the dp4a and integer-MMA kernel family.
fn declares_dynamic_int8(tensor: &str) -> ActivationContract {
    ActivationContract::from_record(
        tensor,
        ExecutionRole::Matmul,
        &contract_record(
            InputRepresentation::A8S32Dynamic,
            DotAccumulator::I32ThenF32Scale,
            32,
            (-127, 127),
            0x22,
        ),
    )
}

/// A TCF-encoded weight, packed by `tcf-core`'s own writer, with no contract
/// attached yet.
fn tcf_weight(device: &CpuDevice) -> QuantTensor<CpuRuntime> {
    let source = values(N * K, 1);
    let dims: Vec<u64> = [N as u64, K as u64].to_vec();
    let tiles = quantize(&source, &dims, 2, NativeEncoding::Q8S32T64.layout()).expect("quantizes");
    let payload = pack(&tiles, NativeEncoding::Q8S32T64.layout()).expect("packs");
    QuantTensor::<CpuRuntime>::from_bytes(
        &payload,
        TcfEncoding::new(NativeEncoding::Q8S32T64),
        &[N, K],
        device,
    )
    .expect("TCF weight")
}

/// The refusal, taken through the real dispatch rather than through the check
/// in isolation.
///
/// The CPU backend's only TCF matmul kernel reads exact f32 activations. A
/// weight declaring the 8-bit dynamic contract is arithmetic that kernel does
/// not compute, so `quant_matmul` returns an error instead of a number. The
/// error names the tensor, its encoding, what the file declared, and what the
/// selected kernel would have computed — everything needed to act on it.
#[test]
fn tcf_refuses_a_weight_whose_contract_the_selected_kernel_does_not_satisfy() {
    let (client, device) = cpu_setup();
    let activation = values(K, 977);
    let act = Tensor::<CpuRuntime>::from_slice(&activation, &[1, K], &device).expect("activation");
    let weight =
        tcf_weight(&device).with_activation_contract(declares_dynamic_int8("blk.0.attn_q.weight"));

    let err = client
        .quant_matmul(&act, &weight)
        .expect_err("an f32-activation kernel must not run a weight declaring 8-bit activations");

    match &err {
        Error::ActivationContractMismatch(detail) => {
            assert_eq!(detail.tensor, "blk.0.attn_q.weight");
            assert_eq!(detail.encoding, "Q8S32_T64");
            // What the file declared.
            assert_eq!(
                detail.declared.input_representation,
                InputRepresentation::A8S32Dynamic
            );
            assert_eq!(
                detail.declared.dot_accumulator,
                DotAccumulator::I32ThenF32Scale
            );
            assert_eq!(detail.declared.quant_group, 32);
            assert_eq!(detail.declared.role, ExecutionRole::Matmul);
            // What the kernel that would have run actually computes.
            assert_eq!(detail.kernel.input_representation, InputRepresentation::F32);
            assert_eq!(detail.kernel.dot_accumulator, DotAccumulator::F32);
            assert!(
                detail.kernel.quant_group.is_none(),
                "an f32-activation kernel quantizes nothing"
            );
        }
        other => panic!("expected E_ACTIVATION_CONTRACT_MISMATCH, got {other:?}"),
    }

    let text = err.to_string();
    println!("TCF refusal: {text}");
    assert!(text.contains("E_ACTIVATION_CONTRACT_MISMATCH"), "{text}");
    assert!(text.contains("blk.0.attn_q.weight"), "{text}");
    assert!(text.contains("A8S32_DYNAMIC"), "{text}");
}

/// The mirror, which is what makes the refusal above meaningful.
///
/// Same weight bytes, same kernel, same activation — only the declared contract
/// changes, and now the weight runs. The check therefore keys on the contract
/// and nothing else, and it costs the matching case nothing: the numbers come
/// back identical to the same weight carrying no contract at all.
#[test]
fn tcf_runs_the_same_weight_when_the_declared_contract_matches_the_kernel() {
    let (client, device) = cpu_setup();
    let activation = values(K, 977);
    let act = Tensor::<CpuRuntime>::from_slice(&activation, &[1, K], &device).expect("activation");

    let matching =
        tcf_weight(&device).with_activation_contract(declares_exact_f32("blk.0.attn_q.weight"));
    let with_contract = client
        .quant_matmul(&act, &matching)
        .expect("the CPU kernel computes exactly what this contract declares")
        .to_vec::<f32>();

    let uncontracted = client
        .quant_matmul(&act, &tcf_weight(&device))
        .expect("no contract, no check")
        .to_vec::<f32>();

    assert!(
        with_contract.iter().all(|v| v.is_finite()),
        "matching contract produced a non-finite value: {with_contract:?}"
    );
    for (index, (a, b)) in with_contract.iter().zip(uncontracted.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "the check changed the arithmetic at index {index}: {a} vs {b}"
        );
    }
}

/// The case the container exists for, stated in the direction a producer hits
/// it: a weight rounded against exact f32 activations, offered to the integer
/// kernel family.
///
/// That family is the dp4a and integer-MMA path — GPU-side on this codebase, so
/// the check is exercised here against its declared `KernelContract` rather
/// than by launching it. The check is the same code the CPU, wgpu and CUDA
/// dispatch sites all call, and TCF Section 9 defines no float fallback, so the
/// answer is a refusal and not a reroute onto a kernel that happens to fit.
#[test]
fn tcf_refuses_an_exact_f32_weight_offered_to_an_int8_activation_kernel() {
    let (_client, device) = cpu_setup();
    let weight =
        tcf_weight(&device).with_activation_contract(declares_exact_f32("blk.0.ffn_down.weight"));
    let kernel = KernelContract::dynamic_int8_activation("tcf_mmq_feat_major");

    let err = weight
        .check_activation_contract(&kernel)
        .expect_err("quantizing the activation is not the arithmetic this weight was prepared for");

    match &err {
        Error::ActivationContractMismatch(detail) => {
            assert_eq!(detail.tensor, "blk.0.ffn_down.weight");
            assert_eq!(
                detail.declared.input_representation,
                InputRepresentation::F32
            );
            assert_eq!(detail.declared.dot_accumulator, DotAccumulator::F32);
            assert_eq!(
                detail.kernel.input_representation,
                InputRepresentation::A8S32Dynamic
            );
            assert_eq!(detail.kernel.quant_group, Some(32));
            assert_eq!(detail.kernel.kernel, "tcf_mmq_feat_major");
        }
        other => panic!("expected E_ACTIVATION_CONTRACT_MISMATCH, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The difference, in one assertion
// ---------------------------------------------------------------------------

/// Ask both weights the same question: which activation contract were you
/// prepared for?
///
/// The TCF weight answers. The GGUF weight has no answer to give, and its
/// `None` means "the format cannot say" — never "any kernel will do". That
/// distinction is why the GGUF weight below still dispatches unchecked: adding
/// the contract check took nothing away from a format that never carried one.
#[test]
fn only_the_tcf_weight_can_say_which_contract_it_was_prepared_for() {
    let (_client, device) = cpu_setup();

    let gguf = QuantTensor::<CpuRuntime>::from_bytes(
        &q6k_weight_bytes(),
        QuantFormat::Q6K,
        &[N, K],
        &device,
    )
    .expect("Q6_K weight");
    assert!(gguf.activation_contract().is_none());

    // And so it passes every kernel, including one whose arithmetic differs
    // from whatever it was prepared for. Unchanged behaviour, by design.
    gguf.check_activation_contract(&KernelContract::f32_activation("cpu_quant_matmul_f32"))
        .expect("a GGUF weight has no contract to check");
    gguf.check_activation_contract(&KernelContract::dynamic_int8_activation(
        "tcf_mmq_feat_major",
    ))
    .expect("a GGUF weight has no contract to check");

    let tcf =
        tcf_weight(&device).with_activation_contract(declares_exact_f32("blk.0.attn_q.weight"));
    let declared = tcf
        .activation_contract()
        .expect("a TCF weight declares a contract for every tensor");
    assert_eq!(declared.input_representation, InputRepresentation::F32);
    assert_eq!(declared.dot_accumulator, DotAccumulator::F32);
    assert_eq!(declared.digest_hex(), "11".repeat(16));

    // The same weight, offered to the same pair of kernels, answers differently.
    tcf.check_activation_contract(&KernelContract::f32_activation("cpu tcf_matmul_f32"))
        .expect("the f32 kernel computes what this contract declares");
    let err = tcf
        .check_activation_contract(&KernelContract::dynamic_int8_activation(
            "tcf_mmq_feat_major",
        ))
        .expect_err("the int8 kernel does not");
    assert!(matches!(err, Error::ActivationContractMismatch(_)), "{err}");
}
