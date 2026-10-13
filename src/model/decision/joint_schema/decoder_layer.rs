//! [`DecoderLayer`]: one pre-norm transformer decoder layer over the field
//! rows, torch `nn.TransformerDecoderLayer` with `norm_first=True`, GELU,
//! no masks and no dropout.

use super::blocks::{Attention, FeedForward, Norm};
use crate::error::Result;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One decoder layer: field self-attention, cross-attention to the raw
/// memory, and a feed-forward block, each a pre-norm residual step.
pub struct DecoderLayer<R: Runtime> {
    pub(super) norm1: Norm<R>,
    pub(super) self_attn: Attention<R>,
    pub(super) norm2: Norm<R>,
    pub(super) cross_attn: Attention<R>,
    pub(super) norm3: Norm<R>,
    pub(super) feedforward: FeedForward<R>,
}

impl<R: Runtime<DType = DType>> DecoderLayer<R> {
    /// The field rows `fields` `[Q, W]` after the layer, reading `memory`
    /// `[L, W]` as given (not normed):
    ///
    /// - `x = fields + SelfAttention(norm1(fields))`: every field attends to
    ///   every field
    /// - `x = x + CrossAttention(norm2(x), memory)`
    /// - `x + feedforward(norm3(x))`
    pub(super) fn forward<C: ModelClient<R>>(
        &self,
        client: &C,
        fields: &Tensor<R>,
        memory: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let normed = self.norm1.forward(client, fields)?;
        let x = client.add(fields, &self.self_attn.forward(client, &normed, &normed)?)?;
        let normed = self.norm2.forward(client, &x)?;
        let x = client.add(&x, &self.cross_attn.forward(client, &normed, memory)?)?;
        let ff = self
            .feedforward
            .forward(client, &self.norm3.forward(client, &x)?)?;
        Ok(client.add(&x, &ff)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::blocks::tests::{HostWeights, host_ff, host_mha, host_norm, values};
    use super::super::config::JointSchemaConfig;
    use super::super::weights::tests::{tiny_config, tiny_head, tiny_host};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// Decoder layer `i` on the host: `x` `[q, W]` reads `memory` `[len,
    /// W]` as given.
    pub(crate) fn host_decoder(
        m: &HostWeights,
        c: &JointSchemaConfig,
        i: usize,
        x: &mut [f64],
        memory: &[f64],
    ) {
        let w = c.width;
        let (q, len) = (x.len() / w, memory.len() / w);
        let p = format!("layers.{i}");
        let n1 = host_norm(m, x, q, &format!("{p}.norm1"));
        let sa = host_mha(m, &n1, q, &n1, q, &format!("{p}.self_attn"), c.heads);
        x.iter_mut().zip(&sa).for_each(|(a, b)| *a += b);
        let n2 = host_norm(m, x, q, &format!("{p}.norm2"));
        let ca = host_mha(
            m,
            &n2,
            q,
            memory,
            len,
            &format!("{p}.multihead_attn"),
            c.heads,
        );
        x.iter_mut().zip(&ca).for_each(|(a, b)| *a += b);
        let n3 = host_norm(m, x, q, &format!("{p}.norm3"));
        let ff = host_ff(
            m,
            &n3,
            q,
            &format!("{p}.linear1"),
            &format!("{p}.linear2"),
            w,
        );
        x.iter_mut().zip(&ff).for_each(|(a, b)| *a += b);
    }

    /// Both decoder layers of the tiny head over 3 fields and 7 memory
    /// rows against the host.
    #[test]
    fn decoder_layers_match_host() {
        let (client, device) = cpu_setup();
        let head = tiny_head(&client, &device, 16);
        let c = tiny_config(16);
        let m = tiny_host(&c);
        let (q, len, w) = (3usize, 7usize, c.width);
        let fields = values(q * w, 61);
        let memory = values(len * w, 67);
        let mut x = Tensor::<CpuRuntime>::from_slice(&fields, &[q, w], &device).unwrap();
        let mem = Tensor::<CpuRuntime>::from_slice(&memory, &[len, w], &device).unwrap();
        let mut want: Vec<f64> = fields.iter().map(|&v| f64::from(v)).collect();
        let memory: Vec<f64> = memory.iter().map(|&v| f64::from(v)).collect();
        for (i, layer) in head.layers.iter().enumerate() {
            x = layer.forward(&client, &x, &mem).unwrap();
            host_decoder(&m, &c, i, &mut want, &memory);
            for (j, (g, h)) in x.to_vec::<f32>().iter().zip(&want).enumerate() {
                assert!(
                    (f64::from(*g) - h).abs() < 1e-4,
                    "layer {i} element {j}: {g} vs {h}"
                );
            }
        }
    }
}
