//! Directory-level checks `TcfFile::open` runs after decoding: recomputed
//! digests and the per-tensor cross-record rules. Section 7 through
//! Section 15.

use std::collections::{HashMap, HashSet};

use crate::tcf::consts::SECTION_ALIGN;
use crate::tcf::digest::{Digest128, contract_digest, policy_digest, relation_digest};
use crate::tcf::error::TcfError;
use crate::tcf::flags::TensorFlags;
use crate::tcf::proof::PROOF_BYTES;
use crate::tcf::record::{ContractRecord, Header, ModuleRecord, RelationRecord, TensorRecord};

use super::sections::{SectionSpan, bounds, record_slice, string_bytes};

/// Section 7, Section 9: recompute every `policy_digest` and reject a
/// mismatch with `E_POLICY_DIGEST_MISMATCH`.
pub(super) fn check_policy_digests(
    directory: &[u8],
    sections: &[SectionSpan],
    header: &Header,
    modules: &[ModuleRecord],
) -> Result<(), TcfError> {
    for (i, module) in modules.iter().enumerate() {
        let raw = record_slice::<ModuleRecord>(directory, sections, 0, i)?;
        let name = string_bytes(directory, header, module.name)?;
        if policy_digest(raw, name)? != Digest128::from_bytes(module.policy_digest) {
            return Err(TcfError::PolicyDigestMismatch {
                module_id: module.module_id,
            });
        }
    }
    Ok(())
}

/// Section 9: recompute every `contract_digest` and reject a mismatch with
/// `E_CONTRACT_DIGEST_MISMATCH`.
///
/// This digest is an integrity identity, so a stored value nobody
/// recomputes is free to hide a corrupted record.
pub(super) fn check_contract_record_digests(
    directory: &[u8],
    sections: &[SectionSpan],
    contracts: &[ContractRecord],
) -> Result<(), TcfError> {
    for (i, contract) in contracts.iter().enumerate() {
        let raw = record_slice::<ContractRecord>(directory, sections, 2, i)?;
        if contract_digest(raw)? != Digest128::from_bytes(contract.contract_digest) {
            return Err(TcfError::ContractDigestMismatch {
                contract_id: contract.contract_id,
            });
        }
    }
    Ok(())
}

/// Section 11, Section 9: recompute every `relation_digest` and reject a
/// mismatch with `E_RELATION_DIGEST_MISMATCH`.
pub(super) fn check_relation_digests(
    directory: &[u8],
    sections: &[SectionSpan],
    relations: &[RelationRecord],
) -> Result<(), TcfError> {
    for (i, relation) in relations.iter().enumerate() {
        let raw = record_slice::<RelationRecord>(directory, sections, 4, i)?;
        if relation_digest(raw)? != Digest128::from_bytes(relation.relation_digest) {
            return Err(TcfError::RelationDigestMismatch {
                output_tensor_id: relation.output_tensor_id,
            });
        }
    }
    Ok(())
}

/// Section 17: a `contract_id` that resolves to a differing digest.
pub(super) fn check_contract_digests(contracts: &[ContractRecord]) -> Result<(), TcfError> {
    let mut seen: HashMap<u32, [u8; 16]> = HashMap::with_capacity(contracts.len());
    for contract in contracts {
        if let Some(previous) = seen.insert(contract.contract_id, contract.contract_digest)
            && previous != contract.contract_digest
        {
            return Err(TcfError::ContractDigestMismatch {
                contract_id: contract.contract_id,
            });
        }
    }
    Ok(())
}

/// The per-tensor directory invariants: Section 3
/// (`activation_contract_id` resolves), Section 7 (`module_id` resolves),
/// Section 8 (`data_offset`), Section 8.0.1 (payload length), Section 8.1
/// (resident bytes), Section 8.6 (`fallback_reason`), Section 10.5.1
/// (`workload_profile_id`), Section 12 and Section 14 (block shape).
///
/// Every check reads directory fields only.
///
/// `modules_by_id`, `workload_ids` and `contract_ids` are built once by the
/// caller so this runs in O(1) lookups per tensor rather than a linear scan
/// of an array for every tensor.
pub(super) fn check_tensor(
    t: &TensorRecord,
    header: &Header,
    modules_by_id: &HashMap<u32, &ModuleRecord>,
    workload_ids: &HashSet<u32>,
    contract_ids: &HashSet<u32>,
) -> Result<(), TcfError> {
    check_payload_length(t)?;

    // Section 7: every `module_id` MUST resolve to a `ModuleRecord` in this
    // file. Without this, a tensor's preferred encoding and policy flags
    // silently resolve to nothing, and `fallback_reason` cannot be checked
    // at all.
    let Some(&module) = modules_by_id.get(&t.module_id) else {
        return Err(bounds("TensorRecord.module_id"));
    };

    // Section 3: every tensor carries an `activation_contract_id` resolving
    // to a `ContractRecord`. Section 9 defines no "no contract" value, so
    // `0` is an ordinary id and resolves like any other. A weight encoding
    // does not define the operation on its own: a tensor whose contract is
    // absent leaves the activation representation and the accumulation
    // unstated, which is the assumption the format exists to prevent. The
    // failure is an unresolvable reference, so it takes Section 7's code
    // for a dangling `module_id` rather than a dispatch error.
    if !contract_ids.contains(&t.activation_contract_id) {
        return Err(bounds("TensorRecord.activation_contract_id"));
    }

    // Section 8.6: mandatory whenever `encoding` differs from the module's
    // highest-ranked `preferred_encoding`. A module that ranks none
    // expresses no preference to differ from, so it mandates no reason.
    if let Some(preferred) = module.top_preferred_encoding()
        && preferred != t.encoding
        && t.fallback_reason == crate::tcf::enums::FallbackReason::None
    {
        return Err(TcfError::MissingFallbackReason {
            tensor_id: t.tensor_id,
        });
    }

    // Section 10.5.1: the measurement without its provenance is a number
    // with no authority.
    if t.flags.contains(TensorFlags::ACCESS_PROFILE_VALID)
        && !workload_ids.contains(&t.workload_profile_id)
    {
        return Err(TcfError::MissingWorkloadProfile {
            tensor_id: t.tensor_id,
        });
    }

    // Section 8.1.
    if t.resident_bytes != t.physical_span_bytes || t.transfer_bytes != t.physical_span_bytes {
        return Err(TcfError::ResidentBytesViolation {
            tensor_id: t.tensor_id,
        });
    }

    // Section 8, Section 14.4.
    if !t.data_offset.is_multiple_of(SECTION_ALIGN) {
        return Err(TcfError::MisalignedSection {
            section: "TensorRecord.data_offset",
        });
    }
    if t.data_offset < header.data_off || t.logical_payload_bytes > t.physical_span_bytes {
        return Err(bounds("tensor payload"));
    }
    let end = t
        .data_offset
        .checked_add(t.physical_span_bytes)
        .ok_or(bounds("tensor payload"))?;
    if end > header.file_len {
        return Err(bounds("tensor payload"));
    }

    // Section 15.3: a quantized tensor's 128 proof bytes lie in the proof
    // section.
    if t.proof_count > 0 {
        let proof_bytes =
            u64::try_from(PROOF_BYTES).map_err(|_| TcfError::TileArithmeticOverflow)?;
        let proof_end = t
            .proof_rel_off
            .checked_add(proof_bytes)
            .ok_or(bounds("proof section"))?;
        if proof_end > header.proof_len {
            return Err(bounds("proof section"));
        }
    }
    Ok(())
}

/// Section 8.0.1: `logical_payload_bytes` is determined by the shape and
/// the encoding, so a stored value that disagrees is rejected with
/// `E_INVALID_QUANT_SHAPE`.
///
/// A block-encoded tensor must also be at least rank 2 and a whole number
/// of blocks wide (Section 12); a raw tensor's length is
/// `product(dims) * width`, which is the only constraint the file places on
/// one — without it a truncated or padded raw payload passes silently end
/// to end.
pub(super) fn check_payload_length(t: &TensorRecord) -> Result<(), TcfError> {
    let expected = t.determined_payload_bytes()?;
    if t.logical_payload_bytes != expected {
        return Err(TcfError::InvalidQuantShape {
            tensor_id: t.tensor_id,
        });
    }
    Ok(())
}
