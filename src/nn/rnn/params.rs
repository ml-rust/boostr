//! Recurrent parameters under `torch.nn.LSTM` and `torch.nn.GRU` state-dict names.
//!
//! Layer `k` holds `weight_ih_l{k}`, `weight_hh_l{k}`, `bias_ih_l{k}` and
//! `bias_hh_l{k}`. The reverse direction adds the suffix `_reverse`. Names
//! are flat, so [`RnnStack::named_parameters`] maps 1:1 to PyTorch.
//! The order is layer, then direction, then `weight_ih`, `weight_hh`,
//! `bias_ih`, `bias_hh`, as PyTorch `named_parameters` lists them.

use super::cell::CellKind;
use super::config::RnnConfig;
use super::stack::RnnStack;
use crate::error::{Error, Result};
use crate::nn::module::Module;
use crate::nn::var_builder::VarBuilder;
use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::TensorId;

/// Weights of one direction of one layer, in PyTorch layout.
///
/// With `G` gates, `weight_ih` is `[G*H, I]`, `weight_hh` is `[G*H, H]` and
/// each bias is `[G*H]`. LSTM has `G = 4`, rows `[i, f, g, o]`. GRU has
/// `G = 3`, rows `[r, z, n]`.
pub struct RnnCellParams<R: Runtime> {
    pub(super) weight_ih: Var<R>,
    pub(super) weight_hh: Var<R>,
    pub(super) bias_ih: Option<Var<R>>,
    pub(super) bias_hh: Option<Var<R>>,
}

/// LSTM cell weights: `[4H, I]`, `[4H, H]`, rows `[i, f, g, o]`.
pub type LstmCellParams<R> = RnnCellParams<R>;

/// GRU cell weights: `[3H, I]`, `[3H, H]`, rows `[r, z, n]`.
pub type GruCellParams<R> = RnnCellParams<R>;

impl<R: Runtime> RnnCellParams<R> {
    /// Build from `Var`s. Pass `Var::with_id` values to keep optimizer ids.
    ///
    /// `bias` is `(bias_ih, bias_hh)`. Shapes are checked by [`RnnStack::from_parts`].
    pub fn new(weight_ih: Var<R>, weight_hh: Var<R>, bias: Option<(Var<R>, Var<R>)>) -> Self {
        let (bias_ih, bias_hh) = match bias {
            Some((ih, hh)) => (Some(ih), Some(hh)),
            None => (None, None),
        };
        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    pub fn weight_ih(&self) -> &Var<R> {
        &self.weight_ih
    }

    pub fn weight_hh(&self) -> &Var<R> {
        &self.weight_hh
    }

    pub fn bias_ih(&self) -> Option<&Var<R>> {
        self.bias_ih.as_ref()
    }

    pub fn bias_hh(&self) -> Option<&Var<R>> {
        self.bias_hh.as_ref()
    }

    /// Return an error unless every shape matches `config` at `layer`.
    ///
    /// `gate` is the row count of each weight and bias.
    pub(super) fn check(
        &self,
        config: &RnnConfig,
        gate: usize,
        layer: usize,
        reverse: bool,
    ) -> Result<()> {
        let suffix = param_suffix(layer, reverse);
        let shape_is = |name: &str, var: &Var<R>, want: &[usize]| -> Result<()> {
            if var.shape() == want {
                Ok(())
            } else {
                Err(Error::InvalidArgument {
                    arg: "params",
                    reason: format!("{name}_{suffix}: expected {want:?}, got {:?}", var.shape()),
                })
            }
        };
        shape_is(
            "weight_ih",
            &self.weight_ih,
            &[gate, config.layer_input_size(layer)],
        )?;
        shape_is("weight_hh", &self.weight_hh, &[gate, config.hidden_size])?;
        match (&self.bias_ih, &self.bias_hh, config.bias) {
            (Some(ih), Some(hh), true) => {
                shape_is("bias_ih", ih, &[gate])?;
                shape_is("bias_hh", hh, &[gate])
            }
            (None, None, false) => Ok(()),
            _ => Err(Error::InvalidArgument {
                arg: "params",
                reason: format!(
                    "layer {suffix}: config bias = {} disagrees with the given biases",
                    config.bias
                ),
            }),
        }
    }

    /// Named parameters of this cell, each name ending in `_{suffix}`.
    fn named(&self, suffix: &str) -> Vec<(String, &Var<R>)> {
        let mut params = vec![
            (format!("weight_ih_{suffix}"), &self.weight_ih),
            (format!("weight_hh_{suffix}"), &self.weight_hh),
        ];
        if let (Some(ih), Some(hh)) = (&self.bias_ih, &self.bias_hh) {
            params.push((format!("bias_ih_{suffix}"), ih));
            params.push((format!("bias_hh_{suffix}"), hh));
        }
        params
    }
}

/// Name suffix of layer `layer`: `l{layer}`, or `l{layer}_reverse`.
pub fn param_suffix(layer: usize, reverse: bool) -> String {
    if reverse {
        format!("l{layer}_reverse")
    } else {
        format!("l{layer}")
    }
}

/// Build one cell per layer and direction, taking each weight from `var(name, shape)`.
///
/// `gate` is the row count of each weight and bias.
pub(super) fn build_cells<R: Runtime>(
    config: &RnnConfig,
    gate: usize,
    mut var: impl FnMut(String, &[usize]) -> Result<Var<R>>,
) -> Result<Vec<RnnCellParams<R>>> {
    let mut cells = Vec::with_capacity(config.num_layers * config.num_directions());
    for layer in 0..config.num_layers {
        for dir in 0..config.num_directions() {
            let suffix = param_suffix(layer, dir == 1);
            let weight_ih = var(
                format!("weight_ih_{suffix}"),
                &[gate, config.layer_input_size(layer)],
            )?;
            let weight_hh = var(format!("weight_hh_{suffix}"), &[gate, config.hidden_size])?;
            let bias = if config.bias {
                Some((
                    var(format!("bias_ih_{suffix}"), &[gate])?,
                    var(format!("bias_hh_{suffix}"), &[gate])?,
                ))
            } else {
                None
            };
            cells.push(RnnCellParams::new(weight_ih, weight_hh, bias));
        }
    }
    Ok(cells)
}

impl<R: Runtime, K: CellKind> RnnStack<R, K> {
    /// Load every weight from `vb` under PyTorch names, shape-checked.
    ///
    /// The tensors are cloned out of the map. A missing name or a shape
    /// mismatch is an error that names the full key.
    pub fn load(vb: &VarBuilder<'_, R>, config: RnnConfig, trainable: bool) -> Result<Self> {
        config.validate()?;
        let var = |name: String, shape: &[usize]| -> Result<Var<R>> {
            let tensor = vb.get_with_shape(&name, shape)?;
            Ok(Var::new(tensor.clone(), trainable))
        };
        let cells = build_cells(&config, Self::gate_rows(&config), var)?;
        Self::from_parts(config, cells)
    }
}

impl<R: Runtime, K> RnnStack<R, K> {
    /// All parameters with their stable autograd ids, in PyTorch order.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.named_parameters()
            .into_iter()
            .map(|(_, var)| (var.id(), var))
            .collect()
    }

    /// Parameters that require grad.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Parameters under PyTorch names, in PyTorch order.
    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let dirs = self.config.num_directions();
        self.cells
            .iter()
            .enumerate()
            .flat_map(|(slot, cell)| cell.named(&param_suffix(slot / dirs, slot % dirs == 1)))
            .collect()
    }
}

impl<R: Runtime, K> Module<R> for RnnStack<R, K> {
    fn parameters(&self) -> Vec<&Var<R>> {
        RnnStack::named_parameters(self)
            .into_iter()
            .map(|(_, var)| var)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        RnnStack::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        RnnStack::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        RnnStack::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::rnn::lstm::TrainableLstm;
    use crate::nn::varmap::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::dtype::DType;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use std::collections::BTreeSet;

    fn zeros(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, DType::F32, device).unwrap()
    }

    /// Fill `map` with zero tensors under PyTorch names for `cfg`.
    fn fill(map: &mut VarMap<CpuRuntime>, cfg: &RnnConfig, device: &CpuDevice) {
        let gate = cfg.lstm_gate_size();
        for layer in 0..cfg.num_layers {
            for dir in 0..cfg.num_directions() {
                let s = param_suffix(layer, dir == 1);
                let inp = cfg.layer_input_size(layer);
                map.insert(format!("weight_ih_{s}"), zeros(&[gate, inp], device));
                map.insert(
                    format!("weight_hh_{s}"),
                    zeros(&[gate, cfg.hidden_size], device),
                );
                if cfg.bias {
                    map.insert(format!("bias_ih_{s}"), zeros(&[gate], device));
                    map.insert(format!("bias_hh_{s}"), zeros(&[gate], device));
                }
            }
        }
    }

    fn two_layer_bi() -> RnnConfig {
        RnnConfig {
            num_layers: 2,
            bidirectional: true,
            ..RnnConfig::new(3, 4)
        }
    }

    /// The key set and order equal `torch.nn.LSTM(3, 4, 2, bidirectional=True)`.
    #[test]
    fn names_match_pytorch_two_layer_bidirectional() {
        let (_, device) = cpu_setup();
        let cfg = two_layer_bi();
        let mut map = VarMap::new();
        fill(&mut map, &cfg, &device);
        let vb = VarBuilder::new(&mut map, &device);
        let lstm = TrainableLstm::load(&vb, cfg, true).unwrap();
        let names: Vec<String> = lstm
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let want = [
            "weight_ih_l0",
            "weight_hh_l0",
            "bias_ih_l0",
            "bias_hh_l0",
            "weight_ih_l0_reverse",
            "weight_hh_l0_reverse",
            "bias_ih_l0_reverse",
            "bias_hh_l0_reverse",
            "weight_ih_l1",
            "weight_hh_l1",
            "bias_ih_l1",
            "bias_hh_l1",
            "weight_ih_l1_reverse",
            "weight_hh_l1_reverse",
            "bias_ih_l1_reverse",
            "bias_hh_l1_reverse",
        ];
        assert_eq!(names, want);
        let set: BTreeSet<&str> = names.iter().map(String::as_str).collect();
        assert_eq!(set.len(), want.len());

        let ids: Vec<TensorId> = lstm.parameters().into_iter().map(|(id, _)| id).collect();
        let named_ids: Vec<TensorId> = lstm
            .named_parameters()
            .into_iter()
            .map(|(_, v)| v.id())
            .collect();
        assert_eq!(ids, named_ids);
        assert_eq!(lstm.trainable_parameters().len(), 16);
        assert_eq!(Module::parameters(&lstm).len(), 16);
        assert_eq!(
            lstm.named_parameters()[8].1.shape(),
            &[16, 8],
            "layer 1 reads both directions"
        );
    }

    #[test]
    fn no_bias_omits_bias_names_and_frozen_trains_nothing() {
        let (_, device) = cpu_setup();
        let cfg = RnnConfig {
            bias: false,
            ..RnnConfig::new(3, 4)
        };
        let mut map = VarMap::new();
        fill(&mut map, &cfg, &device);
        let vb = VarBuilder::new(&mut map, &device);
        let lstm = TrainableLstm::load(&vb, cfg, false).unwrap();
        let names: Vec<String> = lstm
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, ["weight_ih_l0", "weight_hh_l0"]);
        assert!(lstm.trainable_parameters().is_empty());
    }

    #[test]
    fn load_reports_missing_and_misshaped_keys() {
        let (_, device) = cpu_setup();
        let cfg = two_layer_bi();
        let mut map = VarMap::new();
        fill(&mut map, &cfg, &device);
        map.insert("weight_hh_l1_reverse".into(), zeros(&[16, 3], &device));
        let vb = VarBuilder::new(&mut map, &device);
        let err = TrainableLstm::load(&vb, cfg.clone(), true)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("weight_hh_l1_reverse"), "{err}");

        let mut empty = VarMap::<CpuRuntime>::new();
        let vb = VarBuilder::new(&mut empty, &device);
        assert!(TrainableLstm::load(&vb, cfg, true).is_err());
    }

    #[test]
    fn from_parts_rejects_wrong_count_and_bias_mismatch() {
        let (_, device) = cpu_setup();
        let cfg = RnnConfig::new(3, 2);
        let cell = |bias: bool| {
            LstmCellParams::new(
                Var::new(zeros(&[8, 3], &device), true),
                Var::new(zeros(&[8, 2], &device), true),
                bias.then(|| {
                    (
                        Var::new(zeros(&[8], &device), true),
                        Var::new(zeros(&[8], &device), true),
                    )
                }),
            )
        };
        assert!(TrainableLstm::from_parts(cfg.clone(), vec![cell(true)]).is_ok());
        assert!(TrainableLstm::from_parts(cfg.clone(), vec![cell(false)]).is_err());
        assert!(TrainableLstm::<CpuRuntime>::from_parts(cfg.clone(), vec![]).is_err());
        assert!(TrainableLstm::from_parts(cfg, vec![cell(true), cell(true)]).is_err());
    }
}
