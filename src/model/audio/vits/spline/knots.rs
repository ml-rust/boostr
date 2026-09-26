//! Search-knot construction: unnormalized bin logits mapped to knot positions
//! and bin sizes on one axis (widths or heights) of the spline.

use crate::error::Result;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::RqSplineClient;

/// Knot positions `[..., num_bins + 1]` and bin sizes `[..., num_bins]` on one axis.
pub(super) struct Knots<R: Runtime> {
    pub(super) cum: Tensor<R>,
    pub(super) bins: Tensor<R>,
}

/// Maps unnormalized bin logits to knots on `[-bound, bound]`.
///
/// The first and last knots are exactly `-bound` and `bound`.
pub(super) fn knots<R, C>(
    client: &C,
    unnormalized: &Tensor<R>,
    min_bin: f64,
    bound: f64,
) -> Result<Knots<R>>
where
    R: Runtime<DType = DType>,
    C: RqSplineClient<R>,
{
    let num_bins = unnormalized.shape().last().copied().unwrap_or(0);
    let probs = client.softmax(unnormalized, -1)?;
    // Separate multiply and add, rounded twice, as HF computes them.
    let scaled = client.mul_scalar(&probs, 1.0 - min_bin * num_bins as f64)?;
    let bins = client.add_scalar(&scaled, min_bin)?;
    let cum = client.cumsum(&bins, -1)?;
    let interior = cum.narrow(-1, 0, num_bins - 1)?;
    let interior = client.mul_scalar(&interior, 2.0 * bound)?;
    let interior = client.add_scalar(&interior, -bound)?;
    let cum = client.pad(&interior, &[1, 0], -bound)?;
    let cum = client.pad(&cum, &[0, 1], bound)?;
    let bins = client.sub(&cum.narrow(-1, 1, num_bins)?, &cum.narrow(-1, 0, num_bins)?)?;
    Ok(Knots { cum, bins })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn endpoints_sit_on_the_bound_and_bins_span_it() {
        let (client, device) = cpu_setup();
        let logits =
            Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.4, -0.2, 0.9], &[1, 4], &device).unwrap();
        let result = knots(&client, &logits, 1e-3, 5.0).unwrap();

        let cum = result.cum.to_vec::<f32>();
        assert_eq!(cum.len(), 5);
        assert!((cum[0] - (-5.0)).abs() < 1e-5, "first knot: {cum:?}");
        assert!((cum[4] - 5.0).abs() < 1e-5, "last knot: {cum:?}");

        let bins = result.bins.to_vec::<f32>();
        let total: f32 = bins.iter().sum();
        assert!((total - 10.0).abs() < 1e-4, "bin widths sum: {bins:?}");
    }
}
