//! LSTM and BiLSTM modules (inference-only).
//!
//! Pure composition from primitive numr ops (matmul + sigmoid + tanh + mul +
//! add + narrow + stack). No fused kernel — the per-timestep gate computation
//! is memory-bound on small hidden sizes (Kokoro uses `hidden=128..512`) and a
//! fused cell would save at most one pass, not worth a dedicated kernel until
//! benchmarks demand it.
//!
//! Weight layout matches PyTorch `torch.nn.LSTM`:
//!
//! * `weight_ih` — `[4*hidden, input]`, gate order `[i, f, g, o]`
//! * `weight_hh` — `[4*hidden, hidden]`, same gate order
//! * `bias_ih`, `bias_hh` — `[4*hidden]` each (PyTorch carries both; their sum
//!   is the effective bias, but we keep them separate to load checkpoints
//!   unchanged)
//!
//! Gates: `i = σ(·)`, `f = σ(·)`, `g = tanh(·)`, `o = σ(·)`.
//! Cell:  `c' = f*c + i*g`; `h' = o * tanh(c')`.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, MatmulOps, TensorOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// A single-direction LSTM layer.
pub struct Lstm<R: Runtime> {
    weight_ih: Tensor<R>,
    weight_hh: Tensor<R>,
    bias_ih: Tensor<R>,
    bias_hh: Tensor<R>,
    input_size: usize,
    hidden_size: usize,
}

impl<R: Runtime> Lstm<R> {
    /// Construct from raw weight tensors with PyTorch layouts.
    pub fn new(
        weight_ih: Tensor<R>,
        weight_hh: Tensor<R>,
        bias_ih: Tensor<R>,
        bias_hh: Tensor<R>,
    ) -> Result<Self> {
        let ih_shape = weight_ih.shape();
        if ih_shape.len() != 2 || ih_shape[0] % 4 != 0 {
            return Err(Error::InvalidArgument {
                arg: "weight_ih",
                reason: format!("expected [4*hidden, input], got {ih_shape:?}"),
            });
        }
        let hidden_size = ih_shape[0] / 4;
        let input_size = ih_shape[1];
        let hh_shape = weight_hh.shape();
        if hh_shape != [4 * hidden_size, hidden_size] {
            return Err(Error::InvalidArgument {
                arg: "weight_hh",
                reason: format!(
                    "expected [{}, {}], got {hh_shape:?}",
                    4 * hidden_size,
                    hidden_size
                ),
            });
        }
        if bias_ih.shape() != [4 * hidden_size] {
            return Err(Error::InvalidArgument {
                arg: "bias_ih",
                reason: format!("expected [{}], got {:?}", 4 * hidden_size, bias_ih.shape()),
            });
        }
        if bias_hh.shape() != [4 * hidden_size] {
            return Err(Error::InvalidArgument {
                arg: "bias_hh",
                reason: format!("expected [{}], got {:?}", 4 * hidden_size, bias_hh.shape()),
            });
        }
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
        })
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Forward pass returning the sequence of hidden states and the final (h, c).
    ///
    /// * `x` — `[B, T, input_size]`.
    /// * `reverse` — process the sequence from t = T-1 down to 0. The returned
    ///   outputs are still ordered by the ORIGINAL t index (output[t] = hidden
    ///   computed while the cell was positioned at timestep t), matching how
    ///   PyTorch BiLSTM concatenates the reverse direction's outputs.
    ///
    /// Returns `(outputs [B, T, hidden], h_final [B, hidden], c_final [B, hidden])`.
    #[allow(clippy::type_complexity)]
    pub fn forward<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        reverse: bool,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + TensorOps<R>
            + UtilityOps<R>,
    {
        let x_shape = x.shape();
        if x_shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, T, input], got {x_shape:?}"),
            });
        }
        let (b, t, i) = (x_shape[0], x_shape[1], x_shape[2]);
        if i != self.input_size {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("input size {} expected, got {i}", self.input_size),
            });
        }

        let h_sz = self.hidden_size;
        let dtype = x.dtype();

        // Precompute x @ W_ih^T + b_ih for all timesteps in one matmul.
        //   x_flat: [B*T, I]
        //   W_ih^T: [I, 4H]
        //   result: [B*T, 4H] reshape to [B, T, 4H]
        let w_ih_t = self.weight_ih.transpose(0, 1).map_err(Error::Numr)?;
        let x_flat = x.reshape(&[b * t, i]).map_err(Error::Numr)?;
        let xh_flat = client
            .matmul_bias(&x_flat, &w_ih_t, &self.bias_ih)
            .map_err(Error::Numr)?;
        let xh_all = xh_flat.reshape(&[b, t, 4 * h_sz]).map_err(Error::Numr)?;

        let w_hh_t = self.weight_hh.transpose(0, 1).map_err(Error::Numr)?;

        let mut h = client.fill(&[b, h_sz], 0.0, dtype).map_err(Error::Numr)?;
        let mut c = client.fill(&[b, h_sz], 0.0, dtype).map_err(Error::Numr)?;

        let mut outputs: Vec<Tensor<R>> = Vec::with_capacity(t);
        outputs.resize_with(t, || {
            // Placeholder; replaced below. `fill` on a zero-sized tensor would be
            // cheaper but we always overwrite each slot anyway, so just clone h.
            h.clone()
        });

        for step in 0..t {
            let idx = if reverse { t - 1 - step } else { step };

            // x-part of gates at this timestep: [B, 4H].
            let xh_t = xh_all
                .narrow(1, idx, 1)
                .map_err(Error::Numr)?
                .contiguous()
                .map_err(Error::Numr)?
                .reshape(&[b, 4 * h_sz])
                .map_err(Error::Numr)?;

            // h-part: h @ W_hh^T + b_hh. [B, H] @ [H, 4H] -> [B, 4H].
            let hh = client
                .matmul_bias(&h, &w_hh_t, &self.bias_hh)
                .map_err(Error::Numr)?;

            // Sum the two gate contributions.
            let gates = client.add(&xh_t, &hh).map_err(Error::Numr)?;

            // Split into 4 gate pre-activations along the last dim.
            let i_pre = gates.narrow(1, 0, h_sz).map_err(Error::Numr)?;
            let f_pre = gates.narrow(1, h_sz, h_sz).map_err(Error::Numr)?;
            let g_pre = gates.narrow(1, 2 * h_sz, h_sz).map_err(Error::Numr)?;
            let o_pre = gates.narrow(1, 3 * h_sz, h_sz).map_err(Error::Numr)?;

            let i_gate = client.sigmoid(&i_pre).map_err(Error::Numr)?;
            let f_gate = client.sigmoid(&f_pre).map_err(Error::Numr)?;
            let g_gate = client.tanh(&g_pre).map_err(Error::Numr)?;
            let o_gate = client.sigmoid(&o_pre).map_err(Error::Numr)?;

            // c' = f*c + i*g
            let fc = client.mul(&f_gate, &c).map_err(Error::Numr)?;
            let ig = client.mul(&i_gate, &g_gate).map_err(Error::Numr)?;
            c = client.add(&fc, &ig).map_err(Error::Numr)?;

            // h' = o * tanh(c')
            let c_tanh = client.tanh(&c).map_err(Error::Numr)?;
            h = client.mul(&o_gate, &c_tanh).map_err(Error::Numr)?;

            outputs[idx] = h.clone();
        }

        // Stack outputs along the time axis. Each element is [B, H]; stacking on
        // dim=1 gives [B, T, H].
        let output_refs: Vec<&Tensor<R>> = outputs.iter().collect();
        let stacked = client.stack(&output_refs, 1).map_err(Error::Numr)?;
        Ok((stacked, h, c))
    }
}

/// Bidirectional LSTM — runs a forward pass and a reverse pass, concatenates
/// their outputs along the channel axis. Final output shape `[B, T, 2*hidden]`.
pub struct BiLstm<R: Runtime> {
    forward: Lstm<R>,
    backward: Lstm<R>,
}

impl<R: Runtime> BiLstm<R> {
    pub fn new(forward: Lstm<R>, backward: Lstm<R>) -> Result<Self> {
        if forward.hidden_size != backward.hidden_size || forward.input_size != backward.input_size
        {
            return Err(Error::InvalidArgument {
                arg: "backward",
                reason: "forward and backward LSTMs must share hidden/input sizes".into(),
            });
        }
        Ok(Self { forward, backward })
    }

    pub fn hidden_size(&self) -> usize {
        self.forward.hidden_size
    }

    /// Forward pass returning concatenated outputs `[B, T, 2*hidden]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + TensorOps<R>
            + UtilityOps<R>,
    {
        let (fwd, _, _) = self.forward.forward(client, x, false)?;
        let (bwd, _, _) = self.backward.forward(client, x, true)?;
        client.cat(&[&fwd, &bwd], 2).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }

    #[test]
    fn zero_input_zero_weights_yields_zero_output() {
        let (client, device) = cpu_setup();
        let h = 3;
        let i = 2;
        let lstm = Lstm::new(
            zeros(&[4 * h, i], &device),
            zeros(&[4 * h, h], &device),
            zeros(&[4 * h], &device),
            zeros(&[4 * h], &device),
        )
        .unwrap();

        let x = zeros(&[1, 4, i], &device);
        let (out, _hn, _cn) = lstm.forward(&client, &x, false).unwrap();
        assert_eq!(out.shape(), &[1, 4, h]);
        // With all-zero weights/biases/inputs, gates collapse to sigmoid(0)=0.5,
        // tanh(0)=0. Then c' = 0.5*c + 0.5*0 = 0.5*c → stays 0 forever. h' = 0.
        for v in out.to_vec::<f32>() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn output_shape_matches_pytorch_semantics() {
        let (client, device) = cpu_setup();
        let h = 4;
        let i = 3;
        let t = 5;
        let b = 2;
        let lstm = Lstm::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.01f32; 4 * h * i], &[4 * h, i], &device),
            Tensor::<CpuRuntime>::from_slice(&vec![0.01f32; 4 * h * h], &[4 * h, h], &device),
            zeros(&[4 * h], &device),
            zeros(&[4 * h], &device),
        )
        .unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(&vec![0.5f32; b * t * i], &[b, t, i], &device);
        let (out, hn, cn) = lstm.forward(&client, &x, false).unwrap();
        assert_eq!(out.shape(), &[b, t, h]);
        assert_eq!(hn.shape(), &[b, h]);
        assert_eq!(cn.shape(), &[b, h]);
        for v in out.to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn reverse_direction_produces_different_output() {
        let (client, device) = cpu_setup();
        let h = 2;
        let i = 2;
        // Non-zero weights so direction actually matters.
        let lstm = Lstm::new(
            Tensor::<CpuRuntime>::from_slice(
                &(0..(4 * h * i))
                    .map(|k| 0.05 * k as f32)
                    .collect::<Vec<_>>(),
                &[4 * h, i],
                &device,
            ),
            Tensor::<CpuRuntime>::from_slice(
                &(0..(4 * h * h))
                    .map(|k| 0.05 * k as f32)
                    .collect::<Vec<_>>(),
                &[4 * h, h],
                &device,
            ),
            zeros(&[4 * h], &device),
            zeros(&[4 * h], &device),
        )
        .unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0],
            &[1, 3, 2],
            &device,
        );
        let (fwd, _, _) = lstm.forward(&client, &x, false).unwrap();
        let (bwd, _, _) = lstm.forward(&client, &x, true).unwrap();
        let a: Vec<f32> = fwd.to_vec();
        let b: Vec<f32> = bwd.to_vec();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "forward and reverse should differ, max diff {max_diff}"
        );
    }

    #[test]
    fn bilstm_output_doubles_channel_dim() {
        let (client, device) = cpu_setup();
        let h = 3;
        let i = 2;
        let lstm_f = Lstm::new(
            zeros(&[4 * h, i], &device),
            zeros(&[4 * h, h], &device),
            zeros(&[4 * h], &device),
            zeros(&[4 * h], &device),
        )
        .unwrap();
        let lstm_b = Lstm::new(
            zeros(&[4 * h, i], &device),
            zeros(&[4 * h, h], &device),
            zeros(&[4 * h], &device),
            zeros(&[4 * h], &device),
        )
        .unwrap();
        let bi = BiLstm::new(lstm_f, lstm_b).unwrap();
        let x = zeros(&[1, 4, i], &device);
        let out = bi.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2 * h]);
    }

    #[test]
    fn rejects_bad_weight_shapes() {
        let (_client, device) = cpu_setup();
        // weight_ih first dim not divisible by 4
        let bad = Lstm::new(
            zeros(&[5, 2], &device),
            zeros(&[8, 2], &device),
            zeros(&[8], &device),
            zeros(&[8], &device),
        );
        assert!(bad.is_err());
    }

    #[test]
    fn bilstm_rejects_mismatched_inner_sizes() {
        let (_client, device) = cpu_setup();
        let a = Lstm::new(
            zeros(&[8, 2], &device),
            zeros(&[8, 2], &device),
            zeros(&[8], &device),
            zeros(&[8], &device),
        )
        .unwrap();
        // Different hidden size.
        let b = Lstm::new(
            zeros(&[12, 2], &device),
            zeros(&[12, 3], &device),
            zeros(&[12], &device),
            zeros(&[12], &device),
        )
        .unwrap();
        assert!(BiLstm::new(a, b).is_err());
    }
}
