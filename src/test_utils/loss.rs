//! Scalar losses that give every output element its own upstream gradient.

use numr::autograd::{Var, var_mul, var_sum};
use numr::runtime::cpu::{CpuClient, CpuRuntime};

/// `sum(y * w)` over a rank-3 `y`.
pub(crate) fn weighted_loss(
    client: &CpuClient,
    y: &Var<CpuRuntime>,
    w: &Var<CpuRuntime>,
) -> Var<CpuRuntime> {
    let prod = var_mul(y, w, client).unwrap();
    var_sum(&prod, &[0, 1, 2], false, client).unwrap()
}
