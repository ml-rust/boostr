//! Interception of `torch` rebuild/dtype/storage callables and root flattening.
//!
//! These functions translate the pickle object graph into [`PtTensorMeta`]
//! without executing any Python — `REDUCE`/`BINPERSID` are intercepted for the
//! specific `torch` globals `torch.save` emits, and everything else becomes an
//! opaque placeholder or a parse error.

use super::pickle::PValue;
use super::types::{PtContents, PtTensorMeta};
use crate::error::{Error, Result};
use numr::dtype::DType;
use std::collections::HashMap;

pub(super) fn apply_reduce(func: PValue, args: Vec<PValue>) -> Result<PValue> {
    let (module, name) = match &func {
        PValue::Global(m, n) => (m.as_str(), n.as_str()),
        _ => {
            return Err(Error::ModelError {
                reason: format!("REDUCE called on non-Global: {func:?}"),
            });
        }
    };
    match (module, name) {
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
            // Args: (storage_ref, storage_offset, size, stride, requires_grad, backward_hooks)
            if args.len() < 3 {
                return Err(Error::ModelError {
                    reason: format!("_rebuild_tensor_v2 expects ≥3 args, got {}", args.len()),
                });
            }
            let storage_ref = args[0].clone();
            let storage_offset = as_i64(&args[1])? as usize;
            let size = as_usize_tuple(&args[2])?;

            let (dtype, storage_id, numel) = match storage_ref {
                PValue::PersistentRef {
                    dtype,
                    storage_id,
                    numel,
                } => (dtype, storage_id, numel),
                other => {
                    return Err(Error::ModelError {
                        reason: format!("storage ref is not a persistent id: {other:?}"),
                    });
                }
            };

            Ok(PValue::Tensor(PtTensorMeta {
                dtype,
                shape: size,
                storage_id,
                storage_offset,
                storage_numel: numel,
                storage_elem_size: dtype.size_in_bytes(),
            }))
        }
        ("collections", "OrderedDict") => Ok(PValue::Dict(Vec::new())),
        _ => Ok(PValue::Opaque(module.to_string(), name.to_string())),
    }
}

pub(super) fn resolve_persistent_id(pid: PValue) -> Result<PValue> {
    // Tuple: ("storage", dtype_global, storage_id, device, numel)
    let items = match pid {
        PValue::Tuple(t) => t,
        other => {
            return Err(Error::ModelError {
                reason: format!("persistent_load(pid) expects tuple, got {other:?}"),
            });
        }
    };
    if items.len() < 5 {
        return Err(Error::ModelError {
            reason: format!("storage persistent_id expects 5 items, got {}", items.len()),
        });
    }
    let kind = match &items[0] {
        PValue::Str(s) => s.clone(),
        _ => {
            return Err(Error::ModelError {
                reason: "persistent_id kind must be a string".into(),
            });
        }
    };
    if kind != "storage" {
        return Err(Error::ModelError {
            reason: format!("unsupported persistent_id kind: {kind}"),
        });
    }
    let dtype_global = &items[1];
    let storage_id = match &items[2] {
        PValue::Str(s) => s.clone(),
        _ => {
            return Err(Error::ModelError {
                reason: "storage_id must be a string".into(),
            });
        }
    };
    let numel = as_i64(&items[4])? as usize;
    let dtype = dtype_from_global(dtype_global)?;
    Ok(PValue::PersistentRef {
        dtype,
        storage_id,
        numel,
    })
}

pub(super) fn dtype_from_global(v: &PValue) -> Result<DType> {
    let (module, name) = match v {
        PValue::Global(m, n) => (m.as_str(), n.as_str()),
        _ => {
            return Err(Error::ModelError {
                reason: format!("expected dtype global, got {v:?}"),
            });
        }
    };
    if module != "torch" {
        return Err(Error::ModelError {
            reason: format!("dtype module must be torch, got {module}"),
        });
    }
    match name {
        "FloatStorage" | "float32" => Ok(DType::F32),
        "DoubleStorage" | "float64" => Ok(DType::F64),
        "HalfStorage" | "float16" => Ok(DType::F16),
        "BFloat16Storage" | "bfloat16" => Ok(DType::BF16),
        "LongStorage" | "int64" => Ok(DType::I64),
        "IntStorage" | "int32" => Ok(DType::I32),
        "BoolStorage" | "bool" => Ok(DType::Bool),
        other => Err(Error::ModelError {
            reason: format!("unsupported torch dtype: torch.{other}"),
        }),
    }
}

pub(super) fn as_i64(v: &PValue) -> Result<i64> {
    match v {
        PValue::Int(i) => Ok(*i),
        PValue::Bool(b) => Ok(*b as i64),
        other => Err(Error::ModelError {
            reason: format!("expected int, got {other:?}"),
        }),
    }
}

pub(super) fn as_usize_tuple(v: &PValue) -> Result<Vec<usize>> {
    match v {
        PValue::Tuple(items) | PValue::List(items) => items
            .iter()
            .map(|it| as_i64(it).map(|i| i as usize))
            .collect(),
        other => Err(Error::ModelError {
            reason: format!("expected shape tuple, got {other:?}"),
        }),
    }
}

pub(super) fn finalize_root(root: PValue) -> Result<PtContents> {
    let mut raw = HashMap::new();
    match root {
        PValue::Tensor(meta) => {
            raw.insert(String::new(), meta);
        }
        PValue::Dict(_) => {
            // Recursively flatten nested dicts. Upstream Kokoro's
            // `kokoro-v1_0.pth` is `{top: {module: {sub: tensor, …}}, …}`.
            flatten_into("", &root, &mut raw);
            if raw.is_empty() {
                return Err(Error::ModelError {
                    reason: "dict contained no recognizable tensors".into(),
                });
            }
        }
        other => {
            return Err(Error::ModelError {
                reason: format!("unexpected pickle root: {other:?}"),
            });
        }
    }
    // Strip `module.` segments inserted by `torch.nn.DataParallel`. They can
    // appear anywhere in the path (top, second-level, rarely deeper) and
    // aren't semantically meaningful. Flattening is done first so the
    // transformation is a plain string operation on final keys.
    let tensors: HashMap<String, PtTensorMeta> = raw
        .into_iter()
        .map(|(k, v)| (strip_module_segments(&k), v))
        .collect();
    Ok(PtContents { tensors })
}

fn strip_module_segments(key: &str) -> String {
    key.split('.')
        .filter(|seg| *seg != "module")
        .collect::<Vec<_>>()
        .join(".")
}

/// Walk the pickle dict tree, appending `.`-separated key segments and
/// inserting tensor leaves into `out`. `module.` stripping happens in
/// [`finalize_root`] as a post-pass.
fn flatten_into(prefix: &str, value: &PValue, out: &mut HashMap<String, PtTensorMeta>) {
    match value {
        PValue::Tensor(meta) => {
            out.insert(prefix.to_string(), meta.clone());
        }
        PValue::Dict(entries) => {
            for (k, v) in entries {
                let key = match k {
                    PValue::Str(s) => s.as_str(),
                    _ => continue,
                };
                let next_prefix = match prefix {
                    "" => key.to_string(),
                    p => format!("{p}.{key}"),
                };
                flatten_into(&next_prefix, v, out);
            }
        }
        _ => {
            // List/tuple/opaque values are ignored — upstream kokoro
            // state_dicts only nest dicts and leaf tensors.
        }
    }
}
