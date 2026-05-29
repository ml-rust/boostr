//! Tier-2 helpers: load one Kokoro primitive from a checkpoint prefix.

use crate::error::{Error, Result};
use crate::model::audio::kokoro::{
    AdaINResBlock1, AdaLayerNorm, AdainResBlk1d, AlbertConfig, AlbertEmbeddings, AlbertLayer,
    KokoroAdaIn1d, PoolParams,
};
use crate::nn::{Conv1d, Embedding};
use numr::dtype::DType;
use numr::ops::{BinaryOps, PaddingMode, ReduceOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};

use super::super::loader::{load_linear_tensors, load_weight_norm_pair, load_weight_normed_conv1d};

/// Load a Kokoro `AdaIN1d` from `{prefix}.fc.*`, plus optional `{prefix}.norm.*`.
///
/// Upstream Kokoro instantiates `InstanceNorm1d(affine=False)` inside each
/// `AdaIN1d`, so the checkpoint does NOT contain `norm.weight` /
/// `norm.bias`. This loader synthesizes identity-affine tensors (ones +
/// zeros) when those keys are missing, preserving the AdaIN math.
/// Checkpoints that do carry learnable norm affine (custom StyleTTS2
/// variants) still load unchanged.
pub fn load_kokoro_adain<R: Runtime<DType = DType>>(
    st: &mut super::super::weight_source::KokoroWeightSource,
    prefix: &str,
    eps: f32,
    device: &R::Device,
) -> Result<KokoroAdaIn1d<R>> {
    let fc_w = st.load_tensor::<R>(&format!("{prefix}.fc.weight"), device)?;
    let fc_b = st.load_tensor::<R>(&format!("{prefix}.fc.bias"), device)?;
    // `fc_w` is `[2*channels, style_dim]` — derive `channels` from it.
    let fc_shape = fc_w.shape();
    let channels = fc_shape[0] / 2;

    let norm_w = if st.has_tensor(&format!("{prefix}.norm.weight")) {
        st.load_tensor::<R>(&format!("{prefix}.norm.weight"), device)?
    } else {
        ones_1d::<R>(channels, device)
    };
    let norm_b = if st.has_tensor(&format!("{prefix}.norm.bias")) {
        st.load_tensor::<R>(&format!("{prefix}.norm.bias"), device)?
    } else {
        zeros_1d::<R>(channels, device)
    };
    KokoroAdaIn1d::new(fc_w, fc_b, norm_w, norm_b, eps)
}

fn ones_1d<R: Runtime<DType = DType>>(n: usize, device: &R::Device) -> numr::tensor::Tensor<R> {
    let data: Vec<f32> = vec![1.0; n];
    numr::tensor::Tensor::<R>::from_slice(&data, &[n], device)
}

fn zeros_1d<R: Runtime<DType = DType>>(n: usize, device: &R::Device) -> numr::tensor::Tensor<R> {
    let data: Vec<f32> = vec![0.0; n];
    numr::tensor::Tensor::<R>::from_slice(&data, &[n], device)
}

/// Load an `AdaLayerNorm` from `{prefix}.fc.*` (no learnable `norm` in
/// upstream — LayerNorm here is functional).
pub fn load_ada_layer_norm<R: Runtime<DType = DType>>(
    st: &mut super::super::weight_source::KokoroWeightSource,
    prefix: &str,
    eps: f32,
    device: &R::Device,
) -> Result<AdaLayerNorm<R>> {
    let fc_w = st.load_tensor::<R>(&format!("{prefix}.fc.weight"), device)?;
    let fc_b = st.load_tensor::<R>(&format!("{prefix}.fc.bias"), device)?;
    AdaLayerNorm::new(fc_w, fc_b, eps)
}

/// Options controlling how one `AdainResBlk1d` is read from a checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct AdainResBlk1dLoadOpts {
    /// Load `{prefix}.conv1x1` as the learned shortcut. Required when the
    /// block changes channel count; omitted otherwise.
    pub learned_sc: bool,
    /// Load `{prefix}.pool` as a transposed-conv upsampler using this
    /// stride. `None` skips the pool (no upsampling on this block).
    pub upsample_stride: Option<usize>,
    /// ε for the wrapped `InstanceNorm1d` inside each `AdaIN1d`.
    pub norm_eps: f32,
    /// Slope for the leaky-ReLU activations between stages.
    pub leaky_slope: f64,
}

impl Default for AdainResBlk1dLoadOpts {
    fn default() -> Self {
        Self {
            learned_sc: false,
            upsample_stride: None,
            norm_eps: 1e-5,
            leaky_slope: 0.2,
        }
    }
}

/// Load a single-path `AdainResBlk1d`.
pub fn load_adain_resblk1d<R, C>(
    client: &C,
    st: &mut super::super::weight_source::KokoroWeightSource,
    prefix: &str,
    device: &R::Device,
    opts: AdainResBlk1dLoadOpts,
) -> Result<AdainResBlk1d<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    let eps = opts.norm_eps;
    let leaky_slope = opts.leaky_slope;
    let learned_sc = opts.learned_sc;
    let upsample_stride = opts.upsample_stride;
    let adain1 = load_kokoro_adain::<R>(st, &format!("{prefix}.norm1"), eps, device)?;
    let adain2 = load_kokoro_adain::<R>(st, &format!("{prefix}.norm2"), eps, device)?;
    let conv1 = load_weight_normed_conv1d::<R, C>(
        client,
        st,
        &format!("{prefix}.conv1"),
        1,
        PaddingMode::Same,
        1,
        1,
        device,
    )?;
    let conv2 = load_weight_normed_conv1d::<R, C>(
        client,
        st,
        &format!("{prefix}.conv2"),
        1,
        PaddingMode::Same,
        1,
        1,
        device,
    )?;
    let conv1x1 = if learned_sc {
        Some(load_weight_normed_conv1d::<R, C>(
            client,
            st,
            &format!("{prefix}.conv1x1"),
            1,
            PaddingMode::Valid,
            1,
            1,
            device,
        )?)
    } else {
        None
    };
    let pool = match upsample_stride {
        Some(stride) => {
            let (g, v) = load_weight_norm_pair::<R>(st, &format!("{prefix}.pool"), device)?;
            let w = crate::nn::fuse_weight_norm(client, &v, &g, 0)?;
            let bias = st
                .load_tensor::<R>(&format!("{prefix}.pool.bias"), device)
                .ok();
            // Upstream `AdainResBlk1d.pool` is depthwise:
            // `ConvTranspose1d(C_in, C_in, groups=C_in, kernel=3, stride=2,
            //   padding=1, output_padding=1)`.
            // Detect groups from the weight shape `[C_in, C_out/groups, K]`:
            // when C_out/groups == 1, groups == C_in.
            let w_shape = w.shape();
            let c_in = w_shape[0];
            let out_per_group = w_shape[1];
            let groups = if out_per_group == 1 { c_in } else { 1 };
            Some(PoolParams {
                weight: w,
                bias,
                stride,
                padding: PaddingMode::Custom(1, 1, 0, 0),
                output_padding: 1,
                dilation: 1,
                groups,
            })
        }
        None => None,
    };
    Ok(AdainResBlk1d::new(
        adain1,
        adain2,
        conv1,
        conv2,
        conv1x1,
        pool,
        leaky_slope,
    ))
}

/// Options controlling how one `AdaINResBlock1` is read from a checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct AdainResBlock1LoadOpts {
    /// Per-conv dilation factors applied to `convs1.{0,1,2}`. `convs2` always
    /// uses dilation=1 upstream.
    pub dilations: [usize; 3],
    /// Kernel size for every conv in the block (matches upstream's
    /// `resblock_kernel_sizes[i]`).
    pub kernel: usize,
    /// ε for each `AdaIN1d`'s `InstanceNorm1d`.
    pub norm_eps: f32,
    /// ε floor inside the Snake activation's `1/(α+ε)` term.
    pub snake_eps: f64,
}

impl Default for AdainResBlock1LoadOpts {
    fn default() -> Self {
        Self {
            dilations: [1, 3, 5],
            kernel: 3,
            norm_eps: 1e-5,
            snake_eps: 1e-9,
        }
    }
}

/// Load a 3-tier `AdaINResBlock1` from `{prefix}.{convs1,convs2,adain1,adain2,alpha1,alpha2}.{0,1,2}`.
pub fn load_adain_resblock1<R, C>(
    client: &C,
    st: &mut super::super::weight_source::KokoroWeightSource,
    prefix: &str,
    device: &R::Device,
    opts: AdainResBlock1LoadOpts,
) -> Result<AdaINResBlock1<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    let dilations = &opts.dilations;
    let kernel = opts.kernel;
    let eps = opts.norm_eps;
    let snake_eps = opts.snake_eps;
    let conv = |client: &C,
                st: &mut super::super::weight_source::KokoroWeightSource,
                sub: &str,
                i: usize,
                dilation: usize|
     -> Result<Conv1d<R>> {
        load_weight_normed_conv1d::<R, C>(
            client,
            st,
            &format!("{prefix}.{sub}.{i}"),
            1,
            PaddingMode::Custom(
                (kernel - 1) * dilation / 2,
                (kernel - 1) * dilation / 2,
                0,
                0,
            ),
            dilation,
            1,
            device,
        )
    };

    let convs1 = [
        conv(client, st, "convs1", 0, dilations[0])?,
        conv(client, st, "convs1", 1, dilations[1])?,
        conv(client, st, "convs1", 2, dilations[2])?,
    ];
    let convs2 = [
        conv(client, st, "convs2", 0, 1)?,
        conv(client, st, "convs2", 1, 1)?,
        conv(client, st, "convs2", 2, 1)?,
    ];
    let adain1 = [
        load_kokoro_adain::<R>(st, &format!("{prefix}.adain1.0"), eps, device)?,
        load_kokoro_adain::<R>(st, &format!("{prefix}.adain1.1"), eps, device)?,
        load_kokoro_adain::<R>(st, &format!("{prefix}.adain1.2"), eps, device)?,
    ];
    let adain2 = [
        load_kokoro_adain::<R>(st, &format!("{prefix}.adain2.0"), eps, device)?,
        load_kokoro_adain::<R>(st, &format!("{prefix}.adain2.1"), eps, device)?,
        load_kokoro_adain::<R>(st, &format!("{prefix}.adain2.2"), eps, device)?,
    ];
    let alpha1 = [
        st.load_tensor::<R>(&format!("{prefix}.alpha1.0"), device)?,
        st.load_tensor::<R>(&format!("{prefix}.alpha1.1"), device)?,
        st.load_tensor::<R>(&format!("{prefix}.alpha1.2"), device)?,
    ];
    let alpha2 = [
        st.load_tensor::<R>(&format!("{prefix}.alpha2.0"), device)?,
        st.load_tensor::<R>(&format!("{prefix}.alpha2.1"), device)?,
        st.load_tensor::<R>(&format!("{prefix}.alpha2.2"), device)?,
    ];
    AdaINResBlock1::new(convs1, convs2, adain1, adain2, alpha1, alpha2, snake_eps)
}

/// Load one `AlbertLayer` (the single shared layer reused 12 times).
pub fn load_albert_layer<R: Runtime<DType = DType>>(
    st: &mut super::super::weight_source::KokoroWeightSource,
    prefix: &str,
    device: &R::Device,
) -> Result<AlbertLayer<R>> {
    let attn = format!("{prefix}.attention");
    let (q_w, q_b) = load_linear_tensors::<R>(st, &format!("{attn}.query"), device)?;
    let (k_w, k_b) = load_linear_tensors::<R>(st, &format!("{attn}.key"), device)?;
    let (v_w, v_b) = load_linear_tensors::<R>(st, &format!("{attn}.value"), device)?;
    let (dense_w, dense_b) = load_linear_tensors::<R>(st, &format!("{attn}.dense"), device)?;
    let attn_ln_w = st.load_tensor::<R>(&format!("{attn}.LayerNorm.weight"), device)?;
    let attn_ln_b = st.load_tensor::<R>(&format!("{attn}.LayerNorm.bias"), device)?;
    let (ffn_w, ffn_b) = load_linear_tensors::<R>(st, &format!("{prefix}.ffn"), device)?;
    let (ffn_out_w, ffn_out_b) =
        load_linear_tensors::<R>(st, &format!("{prefix}.ffn_output"), device)?;
    let full_ln_w =
        st.load_tensor::<R>(&format!("{prefix}.full_layer_layer_norm.weight"), device)?;
    let full_ln_b = st.load_tensor::<R>(&format!("{prefix}.full_layer_layer_norm.bias"), device)?;

    Ok(AlbertLayer {
        q_weight: q_w,
        q_bias: q_b.ok_or_else(missing_bias("query"))?,
        k_weight: k_w,
        k_bias: k_b.ok_or_else(missing_bias("key"))?,
        v_weight: v_w,
        v_bias: v_b.ok_or_else(missing_bias("value"))?,
        attn_dense_weight: dense_w,
        attn_dense_bias: dense_b.ok_or_else(missing_bias("dense"))?,
        attn_ln_weight: attn_ln_w,
        attn_ln_bias: attn_ln_b,
        ffn_weight: ffn_w,
        ffn_bias: ffn_b.ok_or_else(missing_bias("ffn"))?,
        ffn_output_weight: ffn_out_w,
        ffn_output_bias: ffn_out_b.ok_or_else(missing_bias("ffn_output"))?,
        full_ln_weight: full_ln_w,
        full_ln_bias: full_ln_b,
    })
}

pub(super) fn missing_bias(what: &'static str) -> impl Fn() -> Error {
    move || Error::ModelError {
        reason: format!("{what}.bias is required in ALBERT checkpoint"),
    }
}

/// Load ALBERT embeddings.
pub fn load_albert_embeddings<R: Runtime<DType = DType>>(
    st: &mut super::super::weight_source::KokoroWeightSource,
    prefix: &str,
    config: &AlbertConfig,
    device: &R::Device,
) -> Result<AlbertEmbeddings<R>> {
    let word = st.load_tensor::<R>(&format!("{prefix}.word_embeddings.weight"), device)?;
    let position = st.load_tensor::<R>(&format!("{prefix}.position_embeddings.weight"), device)?;
    let tok_type =
        st.load_tensor::<R>(&format!("{prefix}.token_type_embeddings.weight"), device)?;
    let ln_w = st.load_tensor::<R>(&format!("{prefix}.LayerNorm.weight"), device)?;
    let ln_b = st.load_tensor::<R>(&format!("{prefix}.LayerNorm.bias"), device)?;
    Ok(AlbertEmbeddings::new(
        Embedding::new(word, false),
        Embedding::new(position, false),
        Embedding::new(tok_type, false),
        ln_w,
        ln_b,
        config.layer_norm_eps,
        config.max_position_embeddings,
    ))
}
