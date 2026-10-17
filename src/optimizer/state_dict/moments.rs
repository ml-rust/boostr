//! Export and import of the `m`, `v` and master tensors that AdamW and Lamb share.

use super::check::{check_matches, check_state_dtype};
use super::export::put;
use super::group::ParamEntry;
use super::keys::{EXP_AVG, EXP_AVG_SQ, MASTER};
use crate::error::Result;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

/// Checked, deep-copied moment state of one parameter.
pub(crate) struct Moments<R: Runtime> {
    pub(crate) m: Tensor<R>,
    pub(crate) v: Tensor<R>,
    pub(crate) master: Option<Tensor<R>>,
}

/// Write `exp_avg`, `exp_avg_sq` and, when present, `master` for `name`.
pub(crate) fn put_moments<R: Runtime<DType = DType>>(
    out: &mut HashMap<String, Tensor<R>>,
    name: &str,
    m: &Tensor<R>,
    v: &Tensor<R>,
    master: Option<&Tensor<R>>,
) -> Result<()> {
    put(out, name, EXP_AVG, m)?;
    put(out, name, EXP_AVG_SQ, v)?;
    if let Some(master) = master {
        put(out, name, MASTER, master)?;
    }
    Ok(())
}

/// Check the moment tensors of `entry` and copy them.
///
/// `exp_avg_sq` and `master` must match `exp_avg` in shape and dtype.
pub(crate) fn take_moments<R: Runtime<DType = DType>>(
    entry: &ParamEntry<'_, R>,
) -> Result<Moments<R>> {
    let m = entry.require(EXP_AVG)?;
    let v = entry.require(EXP_AVG_SQ)?;
    let m_key = entry.key(EXP_AVG);
    check_state_dtype(&m_key, m)?;
    check_matches(&entry.key(EXP_AVG_SQ), v, &m_key, m)?;
    let master = entry.get(MASTER);
    if let Some(master) = master {
        check_matches(&entry.key(MASTER), master, &m_key, m)?;
    }
    Ok(Moments {
        m: m.clone_deep()?,
        v: v.clone_deep()?,
        master: master.map(Tensor::clone_deep).transpose()?,
    })
}
