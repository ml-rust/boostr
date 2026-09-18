//! Mamba2 conv helper: the causal conv over `xBC` with the cached window,
//! one call for prefill and decode.

use super::layer::Mamba2;
use crate::error::Result;
use crate::nn::causal_conv1d;
use numr::dtype::DType;
use numr::ops::{ConvOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime> Mamba2<R> {
    /// Causal conv over `xbc_ncl` `[batch, conv_channels, seq_len]` with the
    /// cached window `[batch, conv_channels, d_conv - 1]` as left context,
    /// via [`causal_conv1d`]. The window then advances past the new tokens.
    ///
    /// A zero window (fresh state) equals `d_conv - 1` zeros of left padding.
    /// An initialized window makes a continuation prefill see the previous
    /// tokens, matching how the SSM half consumes `state.h()`.
    pub(in crate::model::mamba::mamba2) fn cached_conv<C>(
        &self,
        client: &C,
        xbc_ncl: &Tensor<R>,
        state: &mut crate::inference::SsmState<R>,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + ConvOps<R> + ShapeOps<R>,
    {
        let (out, window) = causal_conv1d(
            client,
            xbc_ncl,
            self.conv1d.weight().tensor(),
            self.conv1d.bias().map(|b| b.tensor()),
            state.conv_state(),
        )?;
        state.update_conv_state(window);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::SsmState;
    use crate::model::mamba::mamba2::config::Mamba2Config;
    use crate::model::mamba::mamba2::layer::{Mamba2, Mamba2Weights};
    use crate::nn::{Conv1d, Linear};
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::dtype::DType;
    use numr::ops::{PaddingMode, ShapeOps};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lo + (hi - lo) * ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }

        fn tensor(
            &mut self,
            device: &CpuDevice,
            shape: &[usize],
            scale: f32,
        ) -> Tensor<CpuRuntime> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| self.uniform(-scale, scale)).collect();
            Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
        }
    }

    /// Tiny layer with varied weights so a wrong conv window changes the output.
    fn varied_mamba2(device: &CpuDevice) -> (Mamba2<CpuRuntime>, Mamba2Config) {
        let config = Mamba2Config::new(8)
            .with_nheads(2)
            .with_d_state(4)
            .with_expand(2)
            .with_dt_softplus(true)
            .with_use_dt_bias(true)
            .with_use_d(true);
        let mut rng = Lcg(0x5eed_c0de);
        let d_inner = config.d_inner();
        let conv_channels = config.conv_channels();
        let weights = Mamba2Weights {
            in_proj: Linear::new(
                rng.tensor(device, &[config.proj_dim(), 8], 0.3),
                None,
                false,
            ),
            conv1d: Conv1d::new(
                rng.tensor(device, &[conv_channels, 1, config.d_conv], 0.5),
                Some(rng.tensor(device, &[conv_channels], 0.1)),
                1,
                PaddingMode::Custom(config.d_conv - 1, 0, 0, 0),
                1,
                conv_channels,
                false,
            ),
            out_proj: Linear::new(rng.tensor(device, &[8, d_inner], 0.3), None, false),
            a_log: rng.tensor(device, &[config.nheads], 0.5),
            dt_bias: Some(rng.tensor(device, &[config.nheads], 0.5)),
            d_param: Some(rng.tensor(device, &[config.nheads], 0.5)),
            norm: None,
        };
        (Mamba2::new(config.clone(), weights, false), config)
    }

    fn max_abs_diff(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> f32 {
        let a = a.to_vec::<f32>();
        let b = b.to_vec::<f32>();
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn prefill(
        client: &CpuClient,
        mamba: &Mamba2<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        state: &mut SsmState<CpuRuntime>,
    ) -> Tensor<CpuRuntime> {
        mamba.forward_inference(client, x, state).unwrap()
    }

    /// Prefill of 8 tokens equals prefill of 5 then a continuation prefill
    /// of 3 with the state carried. Fails if the conv ignores the window.
    #[test]
    fn continuation_prefill_uses_the_conv_window() {
        let (client, device) = cpu_setup();
        let (mamba, config) = varied_mamba2(&device);
        let x = Lcg(17).tensor(&device, &[1, 8, 8], 1.0);

        let mut full_state = SsmState::<CpuRuntime>::new(1, &config, DType::F32, &device).unwrap();
        let full = prefill(&client, &mamba, &x, &mut full_state);

        let mut state = SsmState::<CpuRuntime>::new(1, &config, DType::F32, &device).unwrap();
        let head = x.narrow(1, 0, 5).unwrap().contiguous().unwrap();
        let tail = x.narrow(1, 5, 3).unwrap().contiguous().unwrap();
        let a = prefill(&client, &mamba, &head, &mut state);
        let b = prefill(&client, &mamba, &tail, &mut state);
        let split = client.cat(&[&a, &b], 1).unwrap();

        let y_diff = max_abs_diff(&full, &split);
        assert!(y_diff < 1e-5, "output diff {y_diff}");
        let conv_diff = max_abs_diff(full_state.conv_state(), state.conv_state());
        assert!(conv_diff < 1e-5, "conv window diff {conv_diff}");
        let h_diff = max_abs_diff(full_state.h(), state.h());
        assert!(h_diff < 1e-5, "ssm state diff {h_diff}");
    }

    /// A fresh-state inference prefill equals the training forward, whose
    /// conv is the `Conv1d` module with `d_conv - 1` zeros of left padding.
    #[test]
    fn fresh_prefill_matches_training_forward() {
        let (client, device) = cpu_setup();
        let (mamba, config) = varied_mamba2(&device);
        let x = Lcg(23).tensor(&device, &[1, 6, 8], 1.0);

        let mut state = SsmState::<CpuRuntime>::new(1, &config, DType::F32, &device).unwrap();
        let inference = prefill(&client, &mamba, &x, &mut state);
        let training = mamba
            .forward(&client, &Var::new(x.clone(), false))
            .unwrap()
            .tensor()
            .contiguous()
            .unwrap();

        let diff = max_abs_diff(&inference, &training);
        assert!(diff < 1e-4, "inference vs training diff {diff}");
        assert_eq!(
            state.conv_state().shape(),
            &[1, config.conv_channels(), config.d_conv - 1]
        );
    }
}
