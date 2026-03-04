//! Shared helper utilities for LLaMA building blocks.

use numr::autograd::Var;
use numr::runtime::Runtime;

/// Make a Var contiguous (copies data if non-contiguous layout).
pub fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    Var::new(v.tensor().contiguous(), v.requires_grad())
}

/// Repeat KV heads for GQA: [B, H_kv, S, D] -> [B, H_kv * repeat, S, D]
pub fn repeat_kv<R: Runtime>(x: &Var<R>, repeat: usize) -> numr::error::Result<Var<R>> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let [b, h_kv, s, d] = [shape[0], shape[1], shape[2], shape[3]];

    // Contiguous required: reshape needs contiguous layout, and inputs
    // (e.g. V after permute) may be strided.
    let expanded = x.tensor().contiguous().reshape(&[b, h_kv, 1, s, d])?;
    let expanded = expanded.broadcast_to(&[b, h_kv, repeat, s, d])?;
    let result = expanded.contiguous().reshape(&[b, h_kv * repeat, s, d])?;
    Ok(Var::new(result, x.requires_grad()))
}
