//! Helpers that write optimizer state into a state dictionary.

use super::keys::fail;
use crate::error::Result;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// Look up the name of the parameter `id`.
pub(crate) fn param_name(names: &HashMap<TensorId, String>, id: TensorId) -> Result<&str> {
    names.get(&id).map(String::as_str).ok_or_else(|| {
        fail(format!(
            "optimizer state for parameter id {id} has no entry in `names`"
        ))
    })
}

/// Write a deep copy of `t` under `<name>.<field>`.
///
/// The copy owns its storage, so a later step cannot modify the exported tensor.
pub(crate) fn put<R: Runtime<DType = DType>>(
    out: &mut HashMap<String, Tensor<R>>,
    name: &str,
    field: &str,
    t: &Tensor<R>,
) -> Result<()> {
    let key = format!("{name}.{field}");
    if out.contains_key(&key) {
        return Err(fail(format!(
            "optimizer state key '{key}' is written twice: two parameters share the name '{name}'"
        )));
    }
    out.insert(key, t.clone_deep()?);
    Ok(())
}
