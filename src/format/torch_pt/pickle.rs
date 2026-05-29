//! Minimal pickle virtual machine covering the opcodes `torch.save` emits.

use super::reader::{
    from_utf8, memo_get, pop, read_exact, read_i32, read_line, read_signed_long, read_u16,
    read_u32, read_u64,
};
use super::reduce::{apply_reduce, finalize_root, resolve_persistent_id};
use super::types::{PtContents, PtTensorMeta};
use crate::error::{Error, Result};
use numr::dtype::DType;
use std::collections::HashMap;
use std::io::{Cursor, Read};

#[derive(Debug, Clone)]
pub(super) enum PValue {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Tuple(Vec<PValue>),
    List(Vec<PValue>),
    Dict(Vec<(PValue, PValue)>),
    /// `(module, qualname)` reference to a Python callable; used to intercept
    /// `torch._utils._rebuild_tensor_v2`, dtype objects, etc.
    Global(String, String),
    /// `persistent_load(pid)` — for tensors this is `("storage", dtype_global,
    /// storage_id, device, numel)` where storage_id is the filename in
    /// `data/*`.
    PersistentRef {
        dtype: DType,
        storage_id: String,
        numel: usize,
    },
    /// A `REDUCE`d tensor. Fully resolved; holds the metadata we care about.
    Tensor(PtTensorMeta),
    /// Placeholder for things we don't interpret (e.g. OrderedDict, Size,
    /// DeviceObj).
    #[allow(dead_code)]
    Opaque(String, String),
}

pub(super) fn parse_pickle(bytes: &[u8]) -> Result<PtContents> {
    let mut cur = Cursor::new(bytes);
    let mut stack: Vec<PValue> = Vec::new();
    let mut memo: HashMap<u32, PValue> = HashMap::new();
    let mut marks: Vec<usize> = Vec::new();

    loop {
        let mut op = [0u8; 1];
        cur.read_exact(&mut op).map_err(|e| Error::ModelError {
            reason: format!("truncated pickle: {e}"),
        })?;
        match op[0] {
            0x80 => {
                // PROTO
                let mut v = [0u8; 1];
                read_exact(&mut cur, &mut v)?;
            }
            0x95 => {
                // FRAME
                let mut v = [0u8; 8];
                read_exact(&mut cur, &mut v)?;
            }
            b'}' => stack.push(PValue::Dict(Vec::new())),
            b']' => stack.push(PValue::List(Vec::new())),
            b')' => stack.push(PValue::Tuple(Vec::new())),
            b'(' => marks.push(stack.len()),
            0x8c => {
                // SHORT_BINUNICODE
                let mut len = [0u8; 1];
                read_exact(&mut cur, &mut len)?;
                let mut buf = vec![0u8; len[0] as usize];
                read_exact(&mut cur, &mut buf)?;
                stack.push(PValue::Str(from_utf8(buf)?));
            }
            0x8d | b'X' => {
                // BINUNICODE8 (8-byte len) / BINUNICODE (4-byte len)
                let len = if op[0] == 0x8d {
                    read_u64(&mut cur)? as usize
                } else {
                    read_u32(&mut cur)? as usize
                };
                let mut buf = vec![0u8; len];
                read_exact(&mut cur, &mut buf)?;
                stack.push(PValue::Str(from_utf8(buf)?));
            }
            b'K' => {
                let mut v = [0u8; 1];
                read_exact(&mut cur, &mut v)?;
                stack.push(PValue::Int(v[0] as i64));
            }
            b'M' => {
                let v = read_u16(&mut cur)?;
                stack.push(PValue::Int(v as i64));
            }
            b'J' => {
                let v = read_i32(&mut cur)?;
                stack.push(PValue::Int(v as i64));
            }
            0x8a => {
                // LONG1: one-byte length, little-endian signed
                let mut len = [0u8; 1];
                read_exact(&mut cur, &mut len)?;
                stack.push(PValue::Int(read_signed_long(&mut cur, len[0] as usize)?));
            }
            0x8b => {
                // LONG4: four-byte length
                let len = read_u32(&mut cur)? as usize;
                stack.push(PValue::Int(read_signed_long(&mut cur, len)?));
            }
            0x85 => {
                // TUPLE1
                let a = pop(&mut stack)?;
                stack.push(PValue::Tuple(vec![a]));
            }
            0x86 => {
                // TUPLE2
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(PValue::Tuple(vec![a, b]));
            }
            0x87 => {
                // TUPLE3
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(PValue::Tuple(vec![a, b, c]));
            }
            b't' => {
                let mark = marks.pop().ok_or_else(|| Error::ModelError {
                    reason: "TUPLE opcode without MARK".into(),
                })?;
                let items = stack.split_off(mark);
                stack.push(PValue::Tuple(items));
            }
            b'l' => {
                let mark = marks.pop().ok_or_else(|| Error::ModelError {
                    reason: "LIST opcode without MARK".into(),
                })?;
                let items = stack.split_off(mark);
                stack.push(PValue::List(items));
            }
            b'u' => {
                // SETITEMS: pop items from mark, set into dict below
                let mark = marks.pop().ok_or_else(|| Error::ModelError {
                    reason: "SETITEMS without MARK".into(),
                })?;
                let items = stack.split_off(mark);
                let dict = stack.last_mut().ok_or_else(|| Error::ModelError {
                    reason: "SETITEMS without dict on stack".into(),
                })?;
                let mut iter = items.into_iter();
                if let PValue::Dict(entries) = dict {
                    while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
                        entries.push((k, v));
                    }
                } else {
                    return Err(Error::ModelError {
                        reason: "SETITEMS target is not a dict".into(),
                    });
                }
            }
            b's' => {
                // SETITEM: single key-value
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                let dict = stack.last_mut().ok_or_else(|| Error::ModelError {
                    reason: "SETITEM without dict on stack".into(),
                })?;
                if let PValue::Dict(entries) = dict {
                    entries.push((k, v));
                } else {
                    return Err(Error::ModelError {
                        reason: "SETITEM target is not a dict".into(),
                    });
                }
            }
            b'e' => {
                // APPENDS: list extend from mark
                let mark = marks.pop().ok_or_else(|| Error::ModelError {
                    reason: "APPENDS without MARK".into(),
                })?;
                let items = stack.split_off(mark);
                if let Some(PValue::List(l)) = stack.last_mut() {
                    l.extend(items);
                } else {
                    return Err(Error::ModelError {
                        reason: "APPENDS target is not a list".into(),
                    });
                }
            }
            b'c' => {
                // GLOBAL: two newline-terminated strings (module, qualname)
                let module = read_line(&mut cur)?;
                let name = read_line(&mut cur)?;
                stack.push(PValue::Global(module, name));
            }
            0x93 => {
                // STACK_GLOBAL
                let name = match pop(&mut stack)? {
                    PValue::Str(s) => s,
                    _ => {
                        return Err(Error::ModelError {
                            reason: "STACK_GLOBAL: name is not a string".into(),
                        });
                    }
                };
                let module = match pop(&mut stack)? {
                    PValue::Str(s) => s,
                    _ => {
                        return Err(Error::ModelError {
                            reason: "STACK_GLOBAL: module is not a string".into(),
                        });
                    }
                };
                stack.push(PValue::Global(module, name));
            }
            b'R' => {
                // REDUCE: args = pop; func = pop; call func(*args)
                let args = match pop(&mut stack)? {
                    PValue::Tuple(a) => a,
                    other => vec![other],
                };
                let func = pop(&mut stack)?;
                let result = apply_reduce(func, args)?;
                stack.push(result);
            }
            b'b' => {
                // BUILD: state = pop; obj = top; obj.__setstate__(state).
                // For tensors / dtype, the state is generally irrelevant or
                // sets the requires_grad / backward_hooks — no-op for us.
                let _state = pop(&mut stack)?;
                // leave obj on stack
            }
            b'Q' => {
                // BINPERSID: pop(pid) and push persistent_load(pid).
                let pid = pop(&mut stack)?;
                stack.push(resolve_persistent_id(pid)?);
            }
            0x94 => {
                // MEMOIZE
                let top = stack.last().ok_or_else(|| Error::ModelError {
                    reason: "MEMOIZE with empty stack".into(),
                })?;
                memo.insert(memo.len() as u32, top.clone());
            }
            b'q' => {
                // BINPUT: 1-byte memo index
                let mut idx = [0u8; 1];
                read_exact(&mut cur, &mut idx)?;
                if let Some(top) = stack.last() {
                    memo.insert(idx[0] as u32, top.clone());
                }
            }
            b'r' => {
                // LONG_BINPUT: 4-byte memo index
                let idx = read_u32(&mut cur)?;
                if let Some(top) = stack.last() {
                    memo.insert(idx, top.clone());
                }
            }
            b'h' => {
                let mut idx = [0u8; 1];
                read_exact(&mut cur, &mut idx)?;
                let v = memo_get(&memo, idx[0] as u32)?;
                stack.push(v);
            }
            b'j' => {
                let idx = read_u32(&mut cur)?;
                let v = memo_get(&memo, idx)?;
                stack.push(v);
            }
            0x89 => stack.push(PValue::Bool(false)),
            0x88 => stack.push(PValue::Bool(true)),
            b'N' => stack.push(PValue::None),
            b'.' => break, // STOP
            other => {
                return Err(Error::ModelError {
                    reason: format!("unsupported pickle opcode {:#04x}", other),
                });
            }
        }
    }

    let root = stack.pop().ok_or_else(|| Error::ModelError {
        reason: "pickle ended with empty stack".into(),
    })?;
    finalize_root(root)
}

#[cfg(test)]
mod tests {
    use super::super::api::load_tensor_pt;
    use super::super::reduce::{as_usize_tuple, dtype_from_global};
    use super::*;

    #[test]
    fn rejects_non_zip() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let tmp = std::env::temp_dir().join("boostr_torch_pt_non_zip.pt");
        std::fs::write(&tmp, b"not a zip").unwrap();
        let device = CpuDevice::new();
        let res = load_tensor_pt::<CpuRuntime>(&tmp, None, &device);
        assert!(res.is_err());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn rejects_missing_file() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let res = load_tensor_pt::<CpuRuntime>("/nonexistent-kokoro-voice-xyz.pt", None, &device);
        assert!(res.is_err());
    }

    #[test]
    fn dtype_from_global_maps_common_torch_types() {
        let f32 = PValue::Global("torch".into(), "FloatStorage".into());
        assert_eq!(dtype_from_global(&f32).unwrap(), DType::F32);
        let f64 = PValue::Global("torch".into(), "DoubleStorage".into());
        assert_eq!(dtype_from_global(&f64).unwrap(), DType::F64);
        let bad = PValue::Global("torch".into(), "NewFangledStorage".into());
        assert!(dtype_from_global(&bad).is_err());
    }

    #[test]
    fn as_usize_tuple_rejects_non_tuple() {
        assert!(as_usize_tuple(&PValue::Int(3)).is_err());
    }

    #[test]
    fn pickle_stack_underflow_is_an_error() {
        let mut stack: Vec<PValue> = Vec::new();
        assert!(pop(&mut stack).is_err());
    }
}
