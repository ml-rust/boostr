//! Group the keys of a state dictionary by parameter.

use super::keys::{STEP_KEY, fail};
use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::{BTreeMap, HashMap};

/// The state tensors of one parameter, found by field.
pub(crate) struct ParamEntry<'a, R: Runtime> {
    pub(crate) id: TensorId,
    pub(crate) name: &'a str,
    fields: HashMap<&'static str, &'a Tensor<R>>,
}

impl<'a, R: Runtime> ParamEntry<'a, R> {
    /// The key `<name>.<field>`.
    pub(crate) fn key(&self, field: &str) -> String {
        format!("{}.{field}", self.name)
    }

    /// The tensor of `field`, if present.
    pub(crate) fn get(&self, field: &str) -> Option<&'a Tensor<R>> {
        self.fields.get(field).copied()
    }

    /// The tensor of `field`, or an error that names the missing key.
    pub(crate) fn require(&self, field: &'static str) -> Result<&'a Tensor<R>> {
        self.get(field).ok_or_else(|| {
            fail(format!(
                "optimizer state key '{}' is missing: the other keys of '{}' are present",
                self.key(field),
                self.name
            ))
        })
    }
}

/// Split `key` into the parameter name and the field it ends with.
fn split_key<'k>(key: &'k str, fields: &[&'static str]) -> Option<(&'k str, &'static str)> {
    fields.iter().find_map(|&field| {
        key.strip_suffix(field)
            .and_then(|prefix| prefix.strip_suffix('.'))
            .map(|name| (name, field))
    })
}

/// Group every per-parameter key of `state` by parameter.
///
/// `ids` maps parameter names to ids. The `__step` key is skipped.
///
/// # Errors
/// Returns an error that names the key when a key has an unknown field, a
/// parameter name that is not in `ids`, or a parameter lacks a `required`
/// field. Two names that map to one id are also an error.
pub(crate) fn group_entries<'a, R: Runtime>(
    ids: &'a HashMap<String, TensorId>,
    state: &'a HashMap<String, Tensor<R>>,
    required: &[&'static str],
    optional: &[&'static str],
) -> Result<Vec<ParamEntry<'a, R>>> {
    let fields: Vec<&'static str> = required.iter().chain(optional).copied().collect();
    let mut keyed: Vec<(&'a String, &'a Tensor<R>)> = state
        .iter()
        .filter(|(k, _)| k.as_str() != STEP_KEY)
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(b.0));

    let mut by_name: BTreeMap<&'a str, ParamEntry<'a, R>> = BTreeMap::new();
    for (key, tensor) in keyed {
        let (name, field) = split_key(key, &fields).ok_or_else(|| {
            fail(format!(
                "optimizer state key '{key}': unknown field, expected a name ending in one of {fields:?}"
            ))
        })?;
        let (known, id) = ids.get_key_value(name).ok_or_else(|| {
            fail(format!(
                "optimizer state key '{key}': parameter '{name}' is not in `ids`"
            ))
        })?;
        by_name
            .entry(known.as_str())
            .or_insert_with(|| ParamEntry {
                id: *id,
                name: known.as_str(),
                fields: HashMap::new(),
            })
            .fields
            .insert(field, tensor);
    }

    let mut seen: HashMap<TensorId, &str> = HashMap::new();
    for entry in by_name.values() {
        for &field in required {
            entry.require(field)?;
        }
        if let Some(previous) = seen.insert(entry.id, entry.name) {
            return Err(fail(format!(
                "optimizer state for '{previous}' and '{}' maps to one parameter id",
                entry.name
            )));
        }
    }
    Ok(by_name.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn tensor(device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], device).unwrap()
    }

    #[test]
    fn test_groups_dotted_names() {
        let device = CpuDevice::new();
        let mut ids = HashMap::new();
        ids.insert("layer.0.w".to_string(), TensorId::new());
        let mut state = HashMap::new();
        state.insert("layer.0.w.exp_avg".to_string(), tensor(&device));
        state.insert("layer.0.w.exp_avg_sq".to_string(), tensor(&device));
        let entries = group_entries(&ids, &state, &["exp_avg", "exp_avg_sq"], &["master"]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "layer.0.w");
    }

    #[test]
    fn test_unknown_name_and_field_are_errors() {
        let device = CpuDevice::new();
        let mut ids = HashMap::new();
        ids.insert("w".to_string(), TensorId::new());
        let mut state = HashMap::new();
        state.insert("other.exp_avg".to_string(), tensor(&device));
        let err = group_entries(&ids, &state, &["exp_avg"], &[])
            .err()
            .unwrap();
        assert!(err.to_string().contains("other.exp_avg"));

        let mut state = HashMap::new();
        state.insert("w.bogus".to_string(), tensor(&device));
        let err = group_entries(&ids, &state, &["exp_avg"], &[])
            .err()
            .unwrap();
        assert!(err.to_string().contains("w.bogus"));
    }

    #[test]
    fn test_partial_entry_is_an_error() {
        let device = CpuDevice::new();
        let mut ids = HashMap::new();
        ids.insert("w".to_string(), TensorId::new());
        let mut state = HashMap::new();
        state.insert("w.exp_avg".to_string(), tensor(&device));
        let err = group_entries(&ids, &state, &["exp_avg", "exp_avg_sq"], &[])
            .err()
            .unwrap();
        assert!(err.to_string().contains("w.exp_avg_sq"));
    }
}
