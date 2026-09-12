//! The [`TcfWriter`] type, its record `add_*` and [`TcfWriter::intern`]
//! API, and the derived record digests pass 1 starts with. Section 6,
//! Section 7, Section 9, Section 11.

use std::collections::HashMap;

use crate::tcf::consts::{CONTRACT_RECORD_BYTES, MODULE_RECORD_BYTES, RELATION_RECORD_BYTES};
use crate::tcf::digest::{contract_digest, policy_digest, relation_digest};
use crate::tcf::error::TcfError;
use crate::tcf::record::field::StringRef;
use crate::tcf::record::{
    CalibrationRecord, ContractRecord, ModuleRecord, Record, RelationRecord, TensorRecord,
    WorkloadProfileRecord,
};

use super::layout::header_count;
use super::payload::Payload;
use super::tensors::writer_owned;

/// A layout arithmetic step that would overflow `u64` or leave `usize`.
/// Section 4 requires the same rejection on the reading side.
pub(crate) const LAYOUT_BOUNDS: TcfError = TcfError::SectionBounds {
    section: "file layout",
};

/// The TCF file writer. Section 4.1.
///
/// A caller adds records and tensors in the order they should appear in the
/// file, then calls [`TcfWriter::finish`] once. See the module documentation
/// for the three-pass order and the division of ownership between caller and
/// writer.
pub struct TcfWriter {
    pub(super) modules: Vec<ModuleRecord>,
    pub(super) tensors: Vec<TensorRecord>,
    pub(super) contracts: Vec<ContractRecord>,
    pub(super) calibrations: Vec<CalibrationRecord>,
    pub(super) relations: Vec<RelationRecord>,
    pub(super) workloads: Vec<WorkloadProfileRecord>,
    /// One entry per tensor, in the same order as `tensors`. `None` is a
    /// tensor registered without a payload, which only
    /// [`TcfWriter::finish_streaming`] can complete.
    pub(super) payloads: Vec<Option<Payload>>,
    /// The string table's bytes, in intern order. Section 6.
    pub(super) strings: Vec<u8>,
    /// Name to its already-interned reference, so a repeated name costs no
    /// second copy. Section 6: names are provenance, so sharing one is free.
    interned: HashMap<String, StringRef>,
}

impl Default for TcfWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TcfWriter {
    /// An empty writer. Section 4.1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            tensors: Vec::new(),
            contracts: Vec::new(),
            calibrations: Vec::new(),
            relations: Vec::new(),
            workloads: Vec::new(),
            payloads: Vec::new(),
            strings: Vec::new(),
            interned: HashMap::new(),
        }
    }

    /// Append a `ModuleRecord`, returning its index in the module array.
    /// Section 7.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_module(&mut self, m: ModuleRecord) -> Result<u32, TcfError> {
        let index = header_count(self.modules.len(), "ModuleRecord[]")?;
        self.modules.push(m);
        Ok(index)
    }

    /// Append a `ContractRecord`, returning its index in the contract array.
    /// Section 9.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_contract(&mut self, c: ContractRecord) -> Result<u32, TcfError> {
        let index = header_count(self.contracts.len(), "ContractRecord[]")?;
        self.contracts.push(c);
        Ok(index)
    }

    /// Append a `CalibrationRecord`, returning its index in the calibration
    /// array. Section 10.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_calibration(&mut self, c: CalibrationRecord) -> Result<u32, TcfError> {
        let index = header_count(self.calibrations.len(), "CalibrationRecord[]")?;
        self.calibrations.push(c);
        Ok(index)
    }

    /// Append a `WorkloadProfileRecord`, returning its index in the workload
    /// array. Section 10.5. Its presence sets `required_features` bit 5
    /// (Section 5.2).
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_workload_profile(&mut self, w: WorkloadProfileRecord) -> Result<u32, TcfError> {
        let index = header_count(self.workloads.len(), "WorkloadProfileRecord[]")?;
        self.workloads.push(w);
        Ok(index)
    }

    /// Append a `RelationRecord`. Section 11. Its presence sets
    /// `required_features` bit 4 (Section 5.2).
    ///
    /// A relation has no index a later call needs, so this returns nothing.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_relation(&mut self, r: RelationRecord) -> Result<(), TcfError> {
        header_count(self.relations.len(), "RelationRecord[]")?;
        self.relations.push(r);
        Ok(())
    }

    /// Intern a name into the string table, returning its [`StringRef`].
    /// Section 6.
    ///
    /// The same name interned twice yields the same reference and stores one
    /// copy. The empty name is `(0, 0)` and stores nothing, which is the
    /// value a record carrying no name holds.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the table or the name exceeds the
    /// offsets the record fields can hold.
    pub fn intern(&mut self, name: &str) -> Result<StringRef, TcfError> {
        if name.is_empty() {
            return Ok(StringRef::new(0, 0));
        }
        if let Some(existing) = self.interned.get(name) {
            return Ok(*existing);
        }
        let off = u64::try_from(self.strings.len()).map_err(|_| LAYOUT_BOUNDS)?;
        let len = u32::try_from(name.len()).map_err(|_| LAYOUT_BOUNDS)?;
        self.strings.extend_from_slice(name.as_bytes());
        let reference = StringRef::new(off, len);
        self.interned.insert(name.to_owned(), reference);
        Ok(reference)
    }

    /// Compute `policy_digest`, `contract_digest`, and `relation_digest`
    /// on every record that carries one. Section 7, Section 9, Section 11.
    ///
    /// Each is BLAKE3-128 over a range of the record's own encoded bytes,
    /// so each record is encoded into a scratch buffer with its digest
    /// field still zero, hashed, and the result written back. Section 9
    /// makes all three mandatory for a producer, and all three verifiable
    /// by a reader recomputing exactly this.
    ///
    /// Runs before `plan`, because the digests cover only fields the caller
    /// supplied — no digest here depends on a layout offset.
    pub(super) fn fill_record_digests(&mut self) -> Result<(), TcfError> {
        let strings = &self.strings;
        for module in &mut self.modules {
            if module.policy_digest != [0u8; 16] {
                return Err(writer_owned("ModuleRecord.policy_digest"));
            }
            let mut image = [0u8; MODULE_RECORD_BYTES];
            module.encode(&mut image)?;
            let name = name_bytes(strings, module.name)?;
            module.policy_digest = *policy_digest(&image, name)?.as_bytes();
        }
        for contract in &mut self.contracts {
            if contract.contract_digest != [0u8; 16] {
                return Err(writer_owned("ContractRecord.contract_digest"));
            }
            let mut image = [0u8; CONTRACT_RECORD_BYTES];
            contract.encode(&mut image)?;
            contract.contract_digest = *contract_digest(&image)?.as_bytes();
        }
        for relation in &mut self.relations {
            if relation.relation_digest != [0u8; 16] {
                return Err(writer_owned("RelationRecord.relation_digest"));
            }
            let mut image = [0u8; RELATION_RECORD_BYTES];
            relation.encode(&mut image)?;
            relation.relation_digest = *relation_digest(&image)?.as_bytes();
        }
        Ok(())
    }
}

/// The string-table bytes a [`StringRef`] names, for the name
/// `policy_digest` concatenates. Section 6, Section 7.
///
/// The empty name is `(0, 0)` and yields no bytes, which is what a record
/// carrying no name contributes to its digest.
fn name_bytes(strings: &[u8], r: StringRef) -> Result<&[u8], TcfError> {
    let off = usize::try_from(r.off()).map_err(|_| LAYOUT_BOUNDS)?;
    let len = usize::try_from(r.len()).map_err(|_| LAYOUT_BOUNDS)?;
    let end = off.checked_add(len).ok_or(LAYOUT_BOUNDS)?;
    strings.get(off..end).ok_or(TcfError::SectionBounds {
        section: "String table",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_the_same_name_stores_one_copy() {
        let mut w = TcfWriter::new();
        let a = w.intern("shared").expect("interns");
        let b = w.intern("shared").expect("interns");
        assert_eq!(a, b);
        assert_eq!(w.strings.len(), "shared".len());
        assert_eq!(w.intern("").expect("interns"), StringRef::new(0, 0));
    }
}
