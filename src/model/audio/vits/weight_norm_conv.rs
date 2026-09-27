//! Shared weight-norm-aware conv loaders for VITS.
//!
//! Every VITS conv/conv-transpose that a real checkpoint may store
//! `torch.nn.utils.parametrizations.weight_norm`-reparameterized (the
//! WaveNet stack, the HiFi-GAN upsampler and residual blocks) reads through
//! [`load_weight_normed_conv1d`] or [`load_weight_normed_conv_transpose1d`].
//! Detection is per tensor: modern `parametrizations.weight.{original0,
//! original1}` pair, else legacy `weight_g`/`weight_v`, else a plain
//! `weight` — so a source that folds weight-norm ahead of time (as
//! compressr does when embedding a checkpoint) still loads correctly.

use crate::error::Result;
use crate::nn::{Conv1d, VarBuilder, fuse_weight_norm};
use numr::dtype::DType;
use numr::ops::{BinaryOps, PaddingMode, ReduceOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Reads the current prefix's weight tensor, fusing whichever weight-norm
/// encoding (if any) is present. `dim` is the output-channel axis: `0` for
/// a plain Conv1d weight `[C_out, C_in/groups, K]`, `1` for a
/// ConvTranspose1d weight `[C_in, C_out/groups, K]`.
fn load_weight<R, C>(client: &C, conv_vb: &mut VarBuilder<'_, R>, dim: usize) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    if conv_vb.contains("parametrizations.weight.original0") {
        let g = conv_vb.take_tensor("parametrizations.weight.original0")?;
        let v = conv_vb.take_tensor("parametrizations.weight.original1")?;
        fuse_weight_norm(client, &v, &g, dim)
    } else if conv_vb.contains("weight_g") {
        let g = conv_vb.take_tensor("weight_g")?;
        let v = conv_vb.take_tensor("weight_v")?;
        fuse_weight_norm(client, &v, &g, dim)
    } else {
        conv_vb.take_tensor("weight")
    }
}

/// Reads a weight-norm-reparameterized Conv1d at `{vb}.{prefix}`. See the
/// module doc for the detection order.
#[allow(clippy::too_many_arguments)]
pub fn load_weight_normed_conv1d<R, C>(
    client: &C,
    vb: &mut VarBuilder<'_, R>,
    prefix: &str,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
) -> Result<Conv1d<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    let mut conv_vb = vb.pp(prefix);
    let weight = load_weight(client, &mut conv_vb, 0)?;
    let bias = conv_vb.take_tensor_optional("bias")?;
    Ok(Conv1d::new(
        weight, bias, stride, padding, dilation, groups, false,
    ))
}

/// Reads a weight-norm-reparameterized ConvTranspose1d weight/bias pair at
/// `{vb}.{prefix}`. Unlike Conv1d, a ConvTranspose1d weight is laid out
/// `[C_in, C_out/groups, K]`, so the output-channel (weight-norm) axis is
/// `1`, not `0` — see [`fuse_weight_norm`]'s `dim` doc. Returns the raw
/// `(weight, bias)` pair rather than a module: this crate has no
/// `nn::ConvTranspose1d` wrapper, so [`super::hifigan`] holds the
/// transposed-conv call site directly.
pub fn load_weight_normed_conv_transpose1d<R, C>(
    client: &C,
    vb: &mut VarBuilder<'_, R>,
    prefix: &str,
) -> Result<(Tensor<R>, Option<Tensor<R>>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    let mut conv_vb = vb.pp(prefix);
    let weight = load_weight(client, &mut conv_vb, 1)?;
    let bias = conv_vb.take_tensor_optional("bias")?;
    Ok((weight, bias))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::varmap::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn load_weight_normed_conv1d_falls_back_to_plain_weight() {
        let (client, device) = cpu_setup();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "conv.weight".into(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 1, 3], &device).unwrap(),
        );
        map.insert(
            "conv.bias".into(),
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.5], &[2], &device).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &device);
        let conv = load_weight_normed_conv1d(&client, &mut vb, "conv", 1, PaddingMode::Valid, 1, 1)
            .unwrap();
        let w: Vec<f32> = conv.weight().tensor().to_vec();
        assert_eq!(w, vec![1.0f32; 6]);
    }

    #[test]
    fn load_weight_normed_conv1d_fuses_legacy_weight_g_v() {
        let (client, device) = cpu_setup();
        let mut map = VarMap::<CpuRuntime>::new();
        // v: unit-norm direction per output channel, g: requested magnitude.
        map.insert(
            "conv.weight_v".into(),
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0],
                &[2, 1, 3],
                &device,
            )
            .unwrap(),
        );
        map.insert(
            "conv.weight_g".into(),
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2, 1, 1], &device).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &device);
        let conv = load_weight_normed_conv1d(&client, &mut vb, "conv", 1, PaddingMode::Valid, 1, 1)
            .unwrap();
        let w: Vec<f32> = conv.weight().tensor().to_vec();
        assert!((w[0] - 3.0).abs() < 1e-5);
        assert!((w[4] - 4.0).abs() < 1e-5);
        assert!(conv.bias().is_none());
    }

    #[test]
    fn load_weight_normed_conv_transpose1d_fuses_dim1() {
        let (client, device) = cpu_setup();
        let mut map = VarMap::<CpuRuntime>::new();
        // ConvTranspose1d layout [C_in=1, C_out=2, K=3]; norm axis is dim 1.
        map.insert(
            "up.weight_v".into(),
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0],
                &[1, 2, 3],
                &device,
            )
            .unwrap(),
        );
        map.insert(
            "up.weight_g".into(),
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[1, 2, 1], &device).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &device);
        let (weight, bias) = load_weight_normed_conv_transpose1d(&client, &mut vb, "up").unwrap();
        let w: Vec<f32> = weight.contiguous().unwrap().to_vec();
        assert!((w[0] - 3.0).abs() < 1e-5);
        assert!((w[4] - 4.0).abs() < 1e-5);
        assert!(bias.is_none());
    }
}
