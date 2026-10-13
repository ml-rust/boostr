//! [`RoutingLayer`]: one evidence-routing layer. The option rows read the
//! normed memory through cross-attention, then a feed-forward block, each
//! a pre-norm residual step. Options never attend to each other here.

use super::blocks::{Attention, FeedForward, Norm};
use crate::error::Result;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One evidence-routing layer.
pub struct RoutingLayer<R: Runtime> {
    pub(super) query_norm: Norm<R>,
    pub(super) memory_norm: Norm<R>,
    pub(super) attention: Attention<R>,
    pub(super) feedforward_norm: Norm<R>,
    pub(super) feedforward: FeedForward<R>,
}

impl<R: Runtime<DType = DType>> RoutingLayer<R> {
    /// The option rows `options` `[N, W]` after the layer, reading `memory`
    /// `[L, W]`:
    ///
    /// - `options + Attention(query_norm(options), memory_norm(memory))`
    /// - then `x + feedforward(feedforward_norm(x))`
    pub(super) fn forward<C: ModelClient<R>>(
        &self,
        client: &C,
        options: &Tensor<R>,
        memory: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let queries = self.query_norm.forward(client, options)?;
        let memory = self.memory_norm.forward(client, memory)?;
        let routed = self.attention.forward(client, &queries, &memory)?;
        let x = client.add(options, &routed)?;
        let ff = self
            .feedforward
            .forward(client, &self.feedforward_norm.forward(client, &x)?)?;
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

    /// Evidence-routing layer `i` on the host: `r` `[n, W]` reads
    /// `memory` `[len, W]`.
    pub(crate) fn host_routing(
        m: &HostWeights,
        c: &JointSchemaConfig,
        i: usize,
        r: &mut [f64],
        memory: &[f64],
    ) {
        let w = c.width;
        let (n, len) = (r.len() / w, memory.len() / w);
        let p = format!("evidence_layers.{i}");
        let queries = host_norm(m, r, n, &format!("{p}.query_norm"));
        let normed = host_norm(m, memory, len, &format!("{p}.memory_norm"));
        let routed = host_mha(
            m,
            &queries,
            n,
            &normed,
            len,
            &format!("{p}.attention"),
            c.heads,
        );
        r.iter_mut().zip(&routed).for_each(|(x, y)| *x += y);
        let ff_in = host_norm(m, r, n, &format!("{p}.feedforward_norm"));
        let ff = host_ff(
            m,
            &ff_in,
            n,
            &format!("{p}.feedforward.0"),
            &format!("{p}.feedforward.3"),
            w,
        );
        r.iter_mut().zip(&ff).for_each(|(x, y)| *x += y);
    }

    /// Both routing layers of the tiny head over 4 options and 6 memory
    /// rows against the host.
    #[test]
    fn routing_layers_match_host() {
        let (client, device) = cpu_setup();
        let head = tiny_head(&client, &device, 16);
        let c = tiny_config(16);
        let m = tiny_host(&c);
        let (n, len, w) = (4usize, 6usize, c.width);
        let options = values(n * w, 51);
        let memory = values(len * w, 53);
        let mut x = Tensor::<CpuRuntime>::from_slice(&options, &[n, w], &device).unwrap();
        let mem = Tensor::<CpuRuntime>::from_slice(&memory, &[len, w], &device).unwrap();
        let mut want: Vec<f64> = options.iter().map(|&v| f64::from(v)).collect();
        let memory: Vec<f64> = memory.iter().map(|&v| f64::from(v)).collect();
        for (i, layer) in head.evidence_layers.iter().enumerate() {
            x = layer.forward(&client, &x, &mem).unwrap();
            host_routing(&m, &c, i, &mut want, &memory);
            for (j, (g, h)) in x.to_vec::<f32>().iter().zip(&want).enumerate() {
                assert!(
                    (f64::from(*g) - h).abs() < 1e-4,
                    "layer {i} element {j}: {g} vs {h}"
                );
            }
        }
    }
}
