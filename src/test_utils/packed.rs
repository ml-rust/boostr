//! Shared checks for the packed-document tests of the sequence layers.
//!
//! A layer is a closure from `(client, x, segments)` to its output. Each check
//! runs it on packed rows and on the documents alone, then compares outputs
//! and gradients.

use super::{Lcg, cpu_setup, ids_for, max_abs_diff_f64, to_f64};
use crate::nn::segment_masks::SegmentMasks;
use numr::autograd::{Var, backward, var_mul, var_sum};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::{Tensor, TensorId};

pub(crate) type Fwd<'a> = &'a dyn Fn(
    &CpuClient,
    &Var<CpuRuntime>,
    Option<&SegmentMasks<CpuRuntime>>,
) -> crate::error::Result<Var<CpuRuntime>>;

pub(crate) type Params<'a> = &'a [(TensorId, &'a Var<CpuRuntime>)];

/// An F64 CPU tensor with values in `[-scale, scale)`.
pub(crate) fn t64(rng: &mut Lcg, shape: &[usize], scale: f64) -> Tensor<CpuRuntime> {
    let (_, device) = cpu_setup();
    let n: usize = shape.iter().product();
    Tensor::<CpuRuntime>::from_slice(&rng.vec(n, scale), shape, &device).unwrap()
}

struct Out {
    y: Vec<f64>,
    gx: Vec<f64>,
    gp: Vec<Vec<f64>>,
}

/// Run `fwd` on `x` (`batch` rows of `seq` tokens, width `width`). The loss is
/// the sum of squared outputs.
fn run(
    fwd: Fwd<'_>,
    params: Params<'_>,
    x: &[f64],
    batch: usize,
    width: usize,
    ids: Option<&[i32]>,
    d_conv: usize,
) -> Out {
    let (client, device) = cpu_setup();
    let seq = x.len() / (batch * width);
    let masks = ids.map(|ids| {
        let t = Tensor::<CpuRuntime>::from_slice(ids, &[batch, seq], &device).unwrap();
        SegmentMasks::from_ids(&client, &t, d_conv, DType::F64).unwrap()
    });
    let xt = Tensor::<CpuRuntime>::from_slice(x, &[batch, seq, width], &device).unwrap();
    let xv = Var::new(xt, true);
    let y = fwd(&client, &xv, masks.as_ref()).unwrap();
    let sq = var_mul(&y, &y, &client).unwrap();
    let loss = var_sum(&sq, &[0, 1, 2], false, &client).unwrap();
    let g = backward(&loss, &client).unwrap();
    let grad = |id: TensorId, len: usize| g.get(id).map_or(vec![0.0; len], to_f64);
    Out {
        y: to_f64(y.tensor()),
        gx: grad(xv.id(), x.len()),
        gp: params
            .iter()
            .map(|(id, p)| grad(*id, p.tensor().numel()))
            .collect(),
    }
}

/// Default absolute bound of the comparison checks.
const TOL: f64 = 1e-12;

/// Packed documents match separate runs in output, input gradient, and every
/// parameter gradient (summed over documents), within 1e-12.
pub(crate) fn check_packed_vs_separate(
    fwd: Fwd<'_>,
    params: Params<'_>,
    width: usize,
    d_conv: usize,
    lens: &[usize],
) {
    check_packed_vs_separate_tol(fwd, params, width, d_conv, lens, TOL);
}

/// [`check_packed_vs_separate`] with an explicit absolute bound.
pub(crate) fn check_packed_vs_separate_tol(
    fwd: Fwd<'_>,
    params: Params<'_>,
    width: usize,
    d_conv: usize,
    lens: &[usize],
    tol: f64,
) {
    let seq: usize = lens.iter().sum();
    let x = Lcg(11).vec(seq * width, 1.0);
    let packed = run(fwd, params, &x, 1, width, Some(&ids_for(lens)), d_conv);
    let (mut y, mut gx) = (Vec::new(), Vec::new());
    let mut gp: Vec<Vec<f64>> = packed.gp.iter().map(|g| vec![0.0; g.len()]).collect();
    let mut at = 0;
    for &n in lens {
        let doc = &x[at * width..(at + n) * width];
        let o = run(fwd, params, doc, 1, width, None, d_conv);
        y.extend(o.y);
        gx.extend(o.gx);
        for (sum, g) in gp.iter_mut().zip(&o.gp) {
            sum.iter_mut().zip(g).for_each(|(a, v)| *a += v);
        }
        at += n;
    }
    let tag = format!("lens={lens:?}");
    assert!(max_abs_diff_f64(&packed.y, &y) < tol, "y {tag}");
    assert!(max_abs_diff_f64(&packed.gx, &gx) < tol, "gx {tag}");
    for (i, (p, s)) in packed.gp.iter().zip(&gp).enumerate() {
        assert!(max_abs_diff_f64(p, s) < tol, "param {i} grad {tag}");
    }
}

/// Changing the tokens of the first document leaves every later document
/// bit-identical.
pub(crate) fn check_no_leak(
    fwd: Fwd<'_>,
    params: Params<'_>,
    width: usize,
    d_conv: usize,
    first: usize,
    rest: usize,
) {
    check_no_leak_tol(fwd, params, width, d_conv, first, rest, 0.0);
}

/// [`check_no_leak`] with an absolute bound on the later documents' change.
pub(crate) fn check_no_leak_tol(
    fwd: Fwd<'_>,
    params: Params<'_>,
    width: usize,
    d_conv: usize,
    first: usize,
    rest: usize,
    tol: f64,
) {
    let seq = first + rest;
    let ids = ids_for(&[first, rest]);
    let x = Lcg(21).vec(seq * width, 1.0);
    let mut x2 = x.clone();
    let noise = Lcg(22).vec(first * width, 3.0);
    x2[..first * width]
        .iter_mut()
        .zip(&noise)
        .for_each(|(a, n)| *a += n);
    let a = run(fwd, params, &x, 1, width, Some(&ids), d_conv);
    let b = run(fwd, params, &x2, 1, width, Some(&ids), d_conv);
    assert_ne!(
        a.y[..first * width],
        b.y[..first * width],
        "perturbation had no effect"
    );
    let leak = max_abs_diff_f64(&a.y[first * width..], &b.y[first * width..]);
    assert!(leak <= tol, "later documents changed by {leak}");
}

/// `Some` of one-document ids equals `None`, for two rows, within 1e-12.
pub(crate) fn check_single_document_equals_none(
    fwd: Fwd<'_>,
    params: Params<'_>,
    width: usize,
    d_conv: usize,
) {
    check_single_document_equals_none_tol(fwd, params, width, d_conv, TOL);
}

/// [`check_single_document_equals_none`] with an explicit absolute bound.
pub(crate) fn check_single_document_equals_none_tol(
    fwd: Fwd<'_>,
    params: Params<'_>,
    width: usize,
    d_conv: usize,
    tol: f64,
) {
    let (batch, seq) = (2, 6);
    let x = Lcg(31).vec(batch * seq * width, 1.0);
    let ids = vec![5i32; batch * seq];
    let none = run(fwd, params, &x, batch, width, None, d_conv);
    let some = run(fwd, params, &x, batch, width, Some(&ids), d_conv);
    assert!(max_abs_diff_f64(&none.y, &some.y) < tol);
    assert!(max_abs_diff_f64(&none.gx, &some.gx) < tol);
    for (p, s) in none.gp.iter().zip(&some.gp) {
        assert!(max_abs_diff_f64(p, s) < tol);
    }
}
