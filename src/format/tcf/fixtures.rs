//! Test fixtures: known-good TCF files, and the values they must decode to.
//!
//! The expected values and the expected packed bytes are computed here, from
//! SPECIFICATION.md Section 13.0 and Section 14.1, and never from what the
//! reader returns. CONFORMANCE.md Section 0.1 forbids a checker sharing
//! packing code with the producer it checks: nothing in this file calls
//! `tcf_core::pack` or `tcf_core::unpack`.

use std::io::Write;
use tcf_core::{
    Code64, ContractFlags, ContractRecord, DotAccumulator, Encoding, ExecutionRole, FallbackReason,
    GroupParams, HEADER_BYTES, HEADER_DIGEST_RANGE, Header, InputRepresentation, LayoutId,
    LogicalTile, MathMode, ModuleRecord, ModuleRole, NativeEncoding, OutputDtype, PolicyFlags,
    ProofFormat, QuantAxis, RawEncoding, Record, ResidencyClass, Role, RoundingMode,
    ScaleComputeDtype, StateDtype, StateFlags, StringRef, TcfWriter, TensorFlags, TensorRecord,
    hash_128,
};
use tempfile::NamedTempFile;

/// binary16 `1.0`.
pub const F16_ONE: u16 = 0x3c00;
/// binary16 `0.5`.
pub const F16_HALF: u16 = 0x3800;

/// `Q4S32_T64`, one tile, shape `[1, 64]`.
pub const T_Q4: usize = 0;
/// Raw `F32`, shape `[4]`.
pub const T_RAW_F32: usize = 1;
/// Raw `F16`, shape `[4]`.
pub const T_RAW_F16: usize = 2;
/// Raw `F16` in a module preferring `Q4S32_T64`, so it carries a
/// `fallback_reason`. Section 8.6.
pub const T_FALLBACK: usize = 3;

/// The literal values the raw F32 tensor stores.
pub const RAW_F32_VALUES: [f32; 4] = [1.0, -2.5, 0.0, 3.25];

/// The binary16 bit patterns the raw F16 tensor stores: `1.0`, `-2.0`,
/// `0.5`, `0.0`.
pub const RAW_F16_BITS: [u16; 4] = [0x3c00, 0xc000, 0x3800, 0x0000];

/// The 64 signed codes the `Q4S32_T64` tensor stores.
///
/// Every value stays inside `-7..=7`: `-8` is the reserved most-negative
/// code a conforming payload never contains (Section 13.2).
pub fn q4_codes() -> [i8; 64] {
    let mut codes = [0i8; 64];
    for (i, slot) in codes.iter_mut().enumerate() {
        *slot = (i % 15) as i8 - 7;
    }
    codes
}

/// Section 13.0 applied by hand: `x_hat_i = f32(d) * f32(q_i)`, with group 0
/// scaled by `1.0` and group 1 by `0.5`.
pub fn expected_q4_values() -> Vec<f32> {
    q4_codes()
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let scale = if i < 32 { 1.0f32 } else { 0.5f32 };
            scale * f32::from(*q)
        })
        .collect()
}

/// Section 14.1 applied by hand: a 32-byte code plane pairing adjacent
/// elements, then the scale plane as two little-endian binary16 values.
pub fn expected_q4_payload() -> Vec<u8> {
    let codes = q4_codes();
    let mut out = Vec::with_capacity(36);
    for k in 0..32 {
        let low = (codes[2 * k] as u8) & 0x0f;
        let high = (codes[2 * k + 1] as u8) & 0x0f;
        out.push(low | (high << 4));
    }
    out.extend_from_slice(&F16_ONE.to_le_bytes());
    out.extend_from_slice(&F16_HALF.to_le_bytes());
    out
}

fn module(module_id: u32, name: StringRef, preferred: Option<Encoding>) -> ModuleRecord {
    ModuleRecord {
        module_id,
        parent_id: tcf_core::ROOT_PARENT_ID,
        name,
        module_role: ModuleRole::Ffn,
        fallback_encoding: None,
        preferred_encoding: [preferred, None, None, None],
        activation_contract_id: 1,
        policy_flags: PolicyFlags::NONE,
        min_quant_k: 64,
        default_residency: ResidencyClass::Warm,
        state_dtype: StateDtype::F32,
        state_flags: StateFlags::NONE,
        policy_digest: [0u8; 16],
    }
}

fn contract() -> ContractRecord {
    ContractRecord {
        contract_id: 1,
        input_representation: InputRepresentation::A8S32Dynamic,
        quant_group: 32,
        quant_axis: QuantAxis::Last,
        rounding_mode: RoundingMode::RnEven,
        qmin: -127,
        qmax: 127,
        scale_compute_dtype: ScaleComputeDtype::F32,
        dot_accumulator: DotAccumulator::I32ThenF32Scale,
        output_dtype: OutputDtype::F32,
        math_mode: MathMode::ReassociationAllowed,
        kernel_semantics_id: 5,
        calibration_id: 0,
        flags: ContractFlags::NONE,
        contract_digest: [0u8; 16],
    }
}

fn tensor(tensor_id: u32, name: StringRef, encoding: Encoding, dims: [u64; 8]) -> TensorRecord {
    TensorRecord {
        tensor_id,
        module_id: 0,
        name,
        role: Role::LinearWeight,
        encoding,
        fallback_reason: FallbackReason::None,
        residency_class: ResidencyClass::Hot,
        flags: TensorFlags::NONE,
        rank: 2,
        calibration_id: 0,
        dims,
        activation_contract_id: 1,
        layout_id: LayoutId::RowMajorDense,
        data_offset: 0,
        logical_payload_bytes: 0,
        physical_span_bytes: 0,
        resident_bytes: 0,
        transfer_bytes: 0,
        sensitivity_delta: 0.0,
        sensitivity_ci95: 0.0,
        accesses_per_generation: 0.0,
        bytes_read_per_generation: 0.0,
        sensitivity_samples: 0,
        sensitivity_seed_count: 0,
        access_profile_samples: 0,
        execution_role: ExecutionRole::Matmul,
        workload_profile_id: 0,
        semantic_digest: [0u8; 16],
        payload_digest: [0u8; 16],
        proof_rel_off: 0,
        proof_count: 0,
        proof_format: ProofFormat::None,
    }
}

/// A rank-1 raw tensor, in `module_id`, carrying `fallback_reason`.
fn raw_tensor(
    tensor_id: u32,
    name: StringRef,
    raw: RawEncoding,
    len: u64,
    module_id: u32,
    fallback_reason: FallbackReason,
) -> TensorRecord {
    let mut record = tensor(
        tensor_id,
        name,
        Encoding::Raw(raw),
        [len, 0, 0, 0, 0, 0, 0, 0],
    );
    record.rank = 1;
    record.module_id = module_id;
    record.role = Role::Bias;
    record.execution_role = ExecutionRole::Elementwise;
    record.fallback_reason = fallback_reason;
    record
}

/// One known-good TCF file: a `Q4S32_T64` weight, a raw F32 bias, a raw F16
/// vector, and a raw F16 tensor whose module prefers `Q4S32_T64`.
pub fn good_file() -> Vec<u8> {
    let mut w = TcfWriter::new();
    let plain = w.intern("model.layers.0.ffn").expect("interns");
    let picky = w.intern("model.layers.0.attn").expect("interns");
    w.add_module(module(0, plain, None)).expect("adds");
    w.add_module(module(
        1,
        picky,
        Some(Encoding::Native(NativeEncoding::Q4S32T64)),
    ))
    .expect("adds");
    w.add_contract(contract()).expect("adds");

    let weight = w.intern("layer.w").expect("interns");
    let tile = LogicalTile {
        group0: GroupParams {
            scale: F16_ONE,
            min: None,
        },
        group1: Some(GroupParams {
            scale: F16_HALF,
            min: None,
        }),
        code: Code64::Signed(q4_codes()),
    };
    w.add_quantized_tensor(
        tensor(
            0,
            weight,
            Encoding::Native(NativeEncoding::Q4S32T64),
            [1, 64, 0, 0, 0, 0, 0, 0],
        ),
        vec![tile],
    )
    .expect("adds");

    let bias = w.intern("layer.bias").expect("interns");
    let mut f32_bytes = Vec::new();
    for v in RAW_F32_VALUES {
        f32_bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.add_raw_tensor(
        raw_tensor(1, bias, RawEncoding::F32, 4, 0, FallbackReason::None),
        f32_bytes,
    )
    .expect("adds");

    let scale = w.intern("layer.scale").expect("interns");
    let mut f16_bytes = Vec::new();
    for bits in RAW_F16_BITS {
        f16_bytes.extend_from_slice(&bits.to_le_bytes());
    }
    w.add_raw_tensor(
        raw_tensor(2, scale, RawEncoding::F16, 4, 0, FallbackReason::None),
        f16_bytes.clone(),
    )
    .expect("adds");

    let pinned = w.intern("layer.pinned").expect("interns");
    w.add_raw_tensor(
        raw_tensor(
            3,
            pinned,
            RawEncoding::F16,
            4,
            1,
            FallbackReason::UserPinnedPrecision,
        ),
        f16_bytes,
    )
    .expect("adds");

    w.finish().expect("writes a valid file")
}

/// Write `bytes` to a temporary `.tcf` file the caller keeps alive.
pub fn write_temp(bytes: &[u8]) -> NamedTempFile {
    let mut file = tempfile::Builder::new()
        .suffix(".tcf")
        .tempfile()
        .expect("creates a temp file");
    file.write_all(bytes).expect("writes");
    file.flush().expect("flushes");
    file
}

fn header_of(bytes: &[u8]) -> Header {
    Header::decode(&bytes[..HEADER_BYTES as usize]).expect("header decodes")
}

fn tensor_offset(bytes: &[u8], index: usize) -> usize {
    header_of(bytes).tensor_off as usize + index * TensorRecord::SIZE
}

/// The decoded record of tensor `index`.
pub fn tensor_at(bytes: &[u8], index: usize) -> TensorRecord {
    let off = tensor_offset(bytes, index);
    TensorRecord::decode(&bytes[off..off + TensorRecord::SIZE]).expect("record decodes")
}

/// Recompute `header_digest` over `[0,192)` with `[144,160)` zero. Section 5.3.
fn reseal_header(bytes: &mut [u8]) {
    let mut image = [0u8; HEADER_BYTES as usize];
    image.copy_from_slice(&bytes[..HEADER_BYTES as usize]);
    image[HEADER_DIGEST_RANGE].fill(0);
    let digest = *hash_128(&image).as_bytes();
    bytes[HEADER_DIGEST_RANGE].copy_from_slice(&digest);
}

/// Recompute `directory_digest` over `[192, data_off)`, then `header_digest`.
/// Section 5.3 fixes that order.
pub fn reseal_directory(bytes: &mut [u8]) {
    let data_off = header_of(bytes).data_off as usize;
    let digest = *hash_128(&bytes[HEADER_BYTES as usize..data_off]).as_bytes();
    bytes[160..176].copy_from_slice(&digest);
    reseal_header(bytes);
}

/// Overwrite tensor `index`'s `encoding` field with `raw`, then reseal the
/// directory so the encoding is the only thing left for a reader to reject.
pub fn set_encoding(bytes: &mut [u8], index: usize, raw: u16) {
    let off = tensor_offset(bytes, index) + 22;
    bytes[off..off + 2].copy_from_slice(&raw.to_le_bytes());
    reseal_directory(bytes);
}

/// Flip one bit of tensor `index`'s first payload byte, leaving every digest
/// as written. Section 15.1's `payload_digest` is what must catch this.
pub fn corrupt_payload(bytes: &mut [u8], index: usize) {
    let start = tensor_at(bytes, index).data_offset as usize;
    bytes[start] ^= 0x01;
}
