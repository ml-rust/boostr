//! In-memory importance matrix: entries, validation on insert, and the
//! check against a model's weights.

use std::collections::BTreeMap;

use crate::error::{Error, Result};

/// First 8 bytes of every importance file.
pub const IMATRIX_MAGIC: [u8; 8] = *b"BSTRIMTX";
/// Format version this module writes, and the only one it reads.
pub const IMATRIX_VERSION: u32 = 1;
/// Bytes before the first entry.
pub const IMATRIX_HEADER_LEN: usize = 32;
/// Longest tensor name accepted, in bytes. A checkpoint key is far shorter;
/// the cap exists so a corrupt `name_len` fails immediately instead of
/// reserving an absurd allocation.
pub const MAX_NAME_LEN: usize = 1024;

pub(super) fn invalid(reason: impl Into<String>) -> Error {
    Error::InvalidArgument {
        arg: "imatrix",
        reason: reason.into(),
    }
}

/// One weight's column importance.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportanceEntry {
    /// Checkpoint tensor name, exactly as the quantizer sees it.
    pub name: String,
    /// Activation rows summed into `sums`. The divisor for a mean.
    pub rows: u64,
    /// `sum(x_j^2)` per input column `j`. Length is the weight's `in_features`.
    pub sums: Vec<f32>,
}

impl ImportanceEntry {
    /// Columns this entry describes — the weight's `in_features`.
    pub fn in_features(&self) -> usize {
        self.sums.len()
    }

    /// Per-column mean squared activation, `sums / rows`.
    ///
    /// `None` when `rows` is zero: a mean over nothing has no value, and a
    /// zero vector would read as "measured, and unimportant".
    pub fn mean_square(&self) -> Option<Vec<f32>> {
        if self.rows == 0 {
            return None;
        }
        let divisor = self.rows as f32;
        Some(self.sums.iter().map(|v| v / divisor).collect())
    }
}

/// What a file holds: one entry per MEASURED tensor, plus the run's token count.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportanceMatrix {
    pub(super) entries: BTreeMap<String, ImportanceEntry>,
    pub(super) token_count: u64,
}

impl ImportanceMatrix {
    /// Empty matrix for a run that will accumulate over `token_count` tokens.
    pub fn new(token_count: u64) -> Self {
        Self {
            entries: BTreeMap::new(),
            token_count,
        }
    }

    /// Tokens the whole run accumulated over.
    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Add one measured tensor. A duplicate name is an ERROR: silently
    /// keeping one of two disagreeing vectors picks a quantization at random.
    pub fn insert(&mut self, entry: ImportanceEntry) -> Result<()> {
        if entry.name.is_empty() {
            return Err(invalid("tensor name is empty"));
        }
        if entry.name.len() > MAX_NAME_LEN {
            return Err(invalid(format!(
                "tensor name is {} bytes, over the {MAX_NAME_LEN}-byte cap",
                entry.name.len()
            )));
        }
        if entry.sums.is_empty() {
            return Err(invalid(format!(
                "'{}': zero-length column vector",
                entry.name
            )));
        }
        check_values(&entry.name, &entry.sums)?;
        if self.entries.contains_key(&entry.name) {
            return Err(invalid(format!("'{}': recorded twice", entry.name)));
        }
        self.entries.insert(entry.name.clone(), entry);
        Ok(())
    }

    /// One measured tensor by name, or `None` when the run never exercised it.
    pub fn get(&self, name: &str) -> Option<&ImportanceEntry> {
        self.entries.get(name)
    }

    /// Every entry, in name order — the order the file stores them in.
    pub fn entries(&self) -> impl Iterator<Item = &ImportanceEntry> {
        self.entries.values()
    }

    /// Measured tensors in this matrix.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the run measured nothing at all.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compare against the weights of the model being quantized.
    ///
    /// `expected` is `(tensor name, in_features)` per quantizable weight.
    /// A name present in BOTH whose `in_features` disagree is an ERROR — that
    /// is the file-does-not-match-this-model case, and importance read at the
    /// wrong width is worse than no importance at all. Everything else is
    /// reported for the caller to act on: an absent name means the run never
    /// exercised that weight, and an unknown name means the file describes a
    /// weight this model does not have.
    pub fn check_against(&self, expected: &[(String, usize)]) -> Result<ImportanceCheck> {
        let mut check = ImportanceCheck::default();
        for (name, in_features) in expected {
            match self.entries.get(name) {
                Some(entry) if entry.in_features() != *in_features => {
                    return Err(invalid(format!(
                        "'{name}': file holds {} column(s), this model's weight has {in_features}",
                        entry.in_features()
                    )));
                }
                Some(_) => check.present.push(name.clone()),
                None => check.absent.push(name.clone()),
            }
        }
        let known: std::collections::HashSet<&str> =
            expected.iter().map(|(name, _)| name.as_str()).collect();
        for name in self.entries.keys() {
            if !known.contains(name.as_str()) {
                check.unknown.push(name.clone());
            }
        }
        Ok(check)
    }
}

/// What [`ImportanceMatrix::check_against`] found. A shape disagreement never
/// reaches here — it is an error, not a finding.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportanceCheck {
    /// Model weights the file measured, at the matching width.
    pub present: Vec<String>,
    /// Model weights the run never exercised. NOT zero importance.
    pub absent: Vec<String>,
    /// File entries naming a weight this model does not have.
    pub unknown: Vec<String>,
}

/// A sum of squares is finite and non-negative. Anything else means the bytes
/// were interpreted as something they are not.
fn check_values(name: &str, values: &[f32]) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 {
            return Err(invalid(format!(
                "'{name}' column {index}: {value} is not a finite non-negative sum of squares"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) fn entry(name: &str, rows: u64, sums: &[f32]) -> ImportanceEntry {
        ImportanceEntry {
            name: name.to_string(),
            rows,
            sums: sums.to_vec(),
        }
    }

    pub(in super::super) fn sample() -> ImportanceMatrix {
        let mut matrix = ImportanceMatrix::new(4096);
        matrix
            .insert(entry(
                "model.layers.0.mlp.down_proj.weight",
                512,
                &[1.0, 2.0],
            ))
            .unwrap();
        matrix
            .insert(entry(
                "model.layers.0.self_attn.q_proj.weight",
                512,
                &[3.0, 4.0, 5.0],
            ))
            .unwrap();
        matrix
    }

    #[test]
    fn entries_are_written_in_name_order() {
        let matrix = sample();
        let ordered: Vec<&str> = matrix.entries().map(|e| e.name.as_str()).collect();
        let mut expected = ordered.clone();
        expected.sort_unstable();
        assert_eq!(ordered, expected);
    }

    #[test]
    fn duplicate_name_is_refused() {
        let mut matrix = sample();
        let err = matrix
            .insert(entry("model.layers.0.mlp.down_proj.weight", 1, &[9.0]))
            .unwrap_err();
        assert!(err.to_string().contains("recorded twice"), "{err}");
    }

    #[test]
    fn negative_and_non_finite_values_are_refused() {
        let mut matrix = ImportanceMatrix::new(1);
        assert!(matrix.insert(entry("a", 1, &[-1.0])).is_err());
        assert!(matrix.insert(entry("b", 1, &[f32::NAN])).is_err());
        assert!(matrix.insert(entry("c", 1, &[f32::INFINITY])).is_err());
    }

    #[test]
    fn a_width_disagreement_is_an_error_not_a_finding() {
        let expected = vec![("model.layers.0.mlp.down_proj.weight".to_string(), 3)];
        let err = sample().check_against(&expected).unwrap_err();
        assert!(err.to_string().contains("column"), "{err}");
    }

    #[test]
    fn check_separates_present_absent_and_unknown() {
        let expected = vec![
            ("model.layers.0.mlp.down_proj.weight".to_string(), 2),
            ("model.layers.0.mlp.up_proj.weight".to_string(), 2),
        ];
        let check = sample().check_against(&expected).unwrap();
        assert_eq!(check.present, vec!["model.layers.0.mlp.down_proj.weight"]);
        assert_eq!(check.absent, vec!["model.layers.0.mlp.up_proj.weight"]);
        assert_eq!(
            check.unknown,
            vec!["model.layers.0.self_attn.q_proj.weight"]
        );
    }

    #[test]
    fn an_unmeasured_tensor_has_no_entry_at_all() {
        let matrix = sample();
        assert!(matrix.get("model.layers.0.mlp.up_proj.weight").is_none());
    }

    #[test]
    fn mean_square_divides_by_rows_and_refuses_an_empty_count() {
        assert_eq!(
            entry("a", 4, &[8.0, 4.0]).mean_square(),
            Some(vec![2.0, 1.0])
        );
        assert_eq!(entry("a", 0, &[8.0]).mean_square(), None);
    }
}
