//! Load-or-initialize for [`TrainableLstm`], as `torch.nn.LSTM` initializes.
//!
//! Every weight and bias absent from the map draws from
//! `U(-1/sqrt(H), 1/sqrt(H))`. A seeded `VarBuilder` derives each draw from
//! the full parameter name, so the result does not depend on build order.

use super::config::RnnConfig;
use super::lstm::TrainableLstm;
use super::params::build_cells;
use crate::error::Result;
use crate::nn::var_builder::VarBuilder;
use crate::nn::varmap::Init;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> TrainableLstm<R> {
    /// Take each weight from `vb`, or initialize it in `dtype` when absent.
    ///
    /// A present weight with the wrong shape is an error. Taken weights
    /// leave the map. Seed the builder with `VarBuilder::with_seed` for a
    /// reproducible draw.
    pub fn init<C>(
        vb: &mut VarBuilder<'_, R>,
        config: RnnConfig,
        dtype: DType,
        client: &C,
        trainable: bool,
    ) -> Result<Self>
    where
        C: RuntimeClient<R>
            + RandomOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + TensorOps<R>,
    {
        config.validate()?;
        let init = Init::Uniform((1.0 / (config.hidden_size as f64).sqrt()) as f32);
        let var = |name: String, shape: &[usize]| -> Result<Var<R>> {
            let tensor = vb.take_or_init_tensor(&name, shape, dtype, init, client)?;
            Ok(Var::new(tensor, trainable))
        };
        let cells = build_cells(&config, var)?;
        Self::from_parts(config, cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::varmap::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn config() -> RnnConfig {
        RnnConfig {
            num_layers: 2,
            bidirectional: true,
            ..RnnConfig::new(5, 16)
        }
    }

    fn values(lstm: &TrainableLstm<CpuRuntime>) -> Vec<(String, Vec<f32>)> {
        lstm.named_parameters()
            .into_iter()
            .map(|(n, v)| (n, v.tensor().to_vec::<f32>()))
            .collect()
    }

    #[test]
    fn draws_within_pytorch_bound() {
        let (client, device) = cpu_setup();
        let mut map = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut map, &device).with_seed(7);
        let lstm = TrainableLstm::init(&mut vb, config(), DType::F32, &client, true).unwrap();
        let bound = 1.0 / 4.0;
        for (name, data) in values(&lstm) {
            assert!(
                data.iter().all(|v| v.abs() <= bound + 1e-6),
                "{name} exceeds {bound}"
            );
            assert!(data.iter().any(|v| *v != 0.0), "{name} is all zero");
        }
        assert_eq!(lstm.trainable_parameters().len(), 16);
    }

    #[test]
    fn same_seed_same_weights_and_names_differ() {
        let (client, device) = cpu_setup();
        let build = |seed| {
            let mut map = VarMap::<CpuRuntime>::new();
            let mut vb = VarBuilder::new(&mut map, &device).with_seed(seed);
            values(&TrainableLstm::init(&mut vb, config(), DType::F32, &client, false).unwrap())
        };
        let a = build(11);
        assert_eq!(a, build(11));
        assert_ne!(a, build(12));
        assert_ne!(a[0].1[..8], a[4].1[..8], "forward and reverse must differ");
    }

    #[test]
    fn present_weight_is_taken_and_checked() {
        let (client, device) = cpu_setup();
        let cfg = RnnConfig::new(2, 1);
        let mut map = VarMap::<CpuRuntime>::new();
        let w = Tensor::<CpuRuntime>::from_slice(&[0.5f32; 8], &[4, 2], &device).unwrap();
        map.insert("weight_ih_l0".into(), w);
        let mut vb = VarBuilder::new(&mut map, &device);
        let lstm = TrainableLstm::init(&mut vb, cfg.clone(), DType::F32, &client, true).unwrap();
        assert_eq!(values(&lstm)[0].1, vec![0.5f32; 8]);

        let mut bad = VarMap::<CpuRuntime>::new();
        let w = Tensor::<CpuRuntime>::from_slice(&[0.5f32; 6], &[3, 2], &device).unwrap();
        bad.insert("weight_ih_l0".into(), w);
        let mut vb = VarBuilder::new(&mut bad, &device);
        assert!(TrainableLstm::init(&mut vb, cfg, DType::F32, &client, true).is_err());
    }
}
