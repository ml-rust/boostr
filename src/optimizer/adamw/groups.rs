//! Parameter groups: a per-group learning-rate scale and weight decay.
//!
//! `AdamWConfig::lr` stays the base rate, so `set_lr` and every scheduler keep
//! their meaning. A group's effective rate is `config.lr * lr_scale`.

use super::types::AdamW;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::TensorId;
use std::collections::HashSet;

/// A set of parameters that share a learning-rate scale and a weight decay.
///
/// A parameter in no group uses `lr_scale = 1.0` and `AdamWConfig::weight_decay`.
/// A parameter can belong to one group only.
///
/// The usual rule keeps biases and norm weights out of weight decay:
///
/// ```
/// use boostr::CpuRuntime;
/// use boostr::optimizer::{AdamW, AdamWConfig, ParamGroup};
/// use numr::tensor::TensorId;
///
/// fn build(named: &[(&str, TensorId)]) -> boostr::error::Result<AdamW<CpuRuntime>> {
///     let no_decay = ParamGroup::no_decay_from_names(named.iter().copied(), |name| {
///         name.ends_with(".bias") || name.contains("norm")
///     });
///     AdamW::with_param_groups(AdamWConfig::default(), vec![no_decay])
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ParamGroup {
    /// Parameters in this group.
    pub ids: HashSet<TensorId>,
    /// Multiplier on the base learning rate. Must be finite and not negative.
    pub lr_scale: f64,
    /// Weight decay for this group. `None` uses `AdamWConfig::weight_decay`.
    pub weight_decay: Option<f64>,
}

impl ParamGroup {
    /// Create a group with `lr_scale = 1.0` and the config weight decay.
    pub fn new(ids: impl IntoIterator<Item = TensorId>) -> Self {
        Self {
            ids: ids.into_iter().collect(),
            lr_scale: 1.0,
            weight_decay: None,
        }
    }

    /// Create a group with `weight_decay = Some(0.0)` and `lr_scale = 1.0`.
    ///
    /// The group holds every id whose name satisfies `predicate`.
    pub fn no_decay_from_names<'a>(
        named: impl IntoIterator<Item = (&'a str, TensorId)>,
        predicate: impl Fn(&str) -> bool,
    ) -> Self {
        Self {
            ids: named
                .into_iter()
                .filter(|(name, _)| predicate(name))
                .map(|(_, id)| id)
                .collect(),
            lr_scale: 1.0,
            weight_decay: Some(0.0),
        }
    }
}

fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

fn check_scale(lr_scale: f64) -> Result<()> {
    if lr_scale.is_finite() && lr_scale >= 0.0 {
        Ok(())
    } else {
        Err(invalid(
            "lr_scale",
            format!("must be finite and not negative, got {lr_scale}"),
        ))
    }
}

fn check_decay(weight_decay: Option<f64>) -> Result<()> {
    match weight_decay {
        Some(wd) if !(wd.is_finite() && wd >= 0.0) => Err(invalid(
            "weight_decay",
            format!("must be finite and not negative, got {wd}"),
        )),
        _ => Ok(()),
    }
}

/// Learning rate, weight decay and corrected step size for one step.
#[derive(Debug, Clone, Copy)]
pub(super) struct Rates {
    pub(super) lr: f64,
    pub(super) wd: f64,
    pub(super) step_size: f64,
}

/// The `Rates` of every bucket for one step.
///
/// Bucket 0 holds the parameters in no group. Bucket `i + 1` is group `i`.
pub(super) struct StepRates {
    base: Rates,
    groups: Vec<Rates>,
}

impl StepRates {
    pub(super) fn of(&self, bucket: usize) -> Rates {
        match bucket.checked_sub(1) {
            None => self.base,
            Some(i) => self.groups.get(i).copied().unwrap_or(self.base),
        }
    }

    pub(super) fn buckets(&self) -> usize {
        self.groups.len() + 1
    }
}

impl<R: Runtime<DType = DType>> AdamW<R> {
    /// Create an optimizer with `groups`.
    ///
    /// # Errors
    /// Returns an error that names the id when an id is in two groups.
    /// Returns an error when a `lr_scale` is not finite or is negative,
    /// or when a `weight_decay` is not finite or is negative.
    pub fn with_param_groups(config: super::AdamWConfig, groups: Vec<ParamGroup>) -> Result<Self> {
        let mut opt = Self::new(config);
        for group in groups {
            opt.add_param_group(group)?;
        }
        Ok(opt)
    }

    /// Add a group. The optimizer is unchanged after an error.
    ///
    /// # Errors
    /// Same conditions as [`Self::with_param_groups`], including an id that
    /// is already in an earlier group.
    pub fn add_param_group(&mut self, group: ParamGroup) -> Result<()> {
        check_scale(group.lr_scale)?;
        check_decay(group.weight_decay)?;
        if let Some(id) = group.ids.iter().find(|id| self.group_of.contains_key(id)) {
            return Err(invalid(
                "groups",
                format!("parameter {id:?} is in two groups"),
            ));
        }
        let index = self.groups.len();
        for &id in &group.ids {
            self.group_of.insert(id, index);
        }
        self.groups.push(group);
        Ok(())
    }

    /// The groups, in the order they were added.
    pub fn param_groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Set the learning-rate scale of group `idx`.
    ///
    /// # Errors
    /// Returns an error when `idx` is out of range or `lr_scale` is not finite
    /// or is negative.
    pub fn set_group_lr_scale(&mut self, idx: usize, lr_scale: f64) -> Result<()> {
        check_scale(lr_scale)?;
        self.group_mut(idx)?.lr_scale = lr_scale;
        Ok(())
    }

    /// Set the weight decay of group `idx`. `None` uses the config weight decay.
    ///
    /// # Errors
    /// Returns an error when `idx` is out of range or `weight_decay` is not
    /// finite or is negative.
    pub fn set_group_weight_decay(&mut self, idx: usize, weight_decay: Option<f64>) -> Result<()> {
        check_decay(weight_decay)?;
        self.group_mut(idx)?.weight_decay = weight_decay;
        Ok(())
    }

    fn group_mut(&mut self, idx: usize) -> Result<&mut ParamGroup> {
        let len = self.groups.len();
        self.groups
            .get_mut(idx)
            .ok_or_else(|| invalid("idx", format!("group {idx} is out of range, {len} groups")))
    }

    /// Bucket of `id`: 0 for no group, `i + 1` for group `i`.
    pub(super) fn bucket(&self, id: TensorId) -> usize {
        self.group_of.get(&id).map_or(0, |&g| g + 1)
    }

    /// Rates of every bucket. Each step size is `lr * sqrt(bc2) / bc1`.
    pub(super) fn step_rates(&self, bc1: f64, bc2: f64) -> StepRates {
        let base = self.config.lr;
        let rate = |lr: f64, wd: f64| Rates {
            lr,
            wd,
            step_size: lr * bc2.sqrt() / bc1,
        };
        let groups = self
            .groups
            .iter()
            .map(|g| {
                let wd = g.weight_decay.unwrap_or(self.config.weight_decay);
                rate(base * g.lr_scale, wd)
            })
            .collect();
        StepRates {
            base: rate(base, self.config.weight_decay),
            groups,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::f32_reference;
    use super::super::types::AdamWConfig;
    use super::*;
    use crate::optimizer::Optimizer;
    use crate::optimizer::state_dict::test_support::{
        Fixture, assert_same_params, assert_same_state, deep, run,
    };
    use crate::test_utils::cpu_setup;
    use numr::autograd::GradStore;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use std::collections::HashMap;

    type Params = HashMap<TensorId, Tensor<CpuRuntime>>;

    fn tensor(device: &CpuDevice, values: &[f32]) -> Tensor<CpuRuntime> {
        Tensor::from_slice(values, &[values.len()], device).unwrap()
    }

    fn config(lr: f64, weight_decay: f64) -> AdamWConfig {
        AdamWConfig {
            lr,
            weight_decay,
            ..Default::default()
        }
    }

    fn scaled(ids: &[TensorId], lr_scale: f64, weight_decay: Option<f64>) -> ParamGroup {
        ParamGroup {
            ids: ids.iter().copied().collect(),
            lr_scale,
            weight_decay,
        }
    }

    /// Step `steps` times with fixed gradients `(id, values)`.
    fn step_n(
        opt: &mut AdamW<CpuRuntime>,
        client: &CpuClient,
        device: &CpuDevice,
        params: &mut Params,
        grads: &[(TensorId, &[f32])],
        steps: usize,
    ) {
        let mut store = GradStore::new();
        for (id, values) in grads {
            store.insert(*id, tensor(device, values));
        }
        for _ in 0..steps {
            opt.step(client, params, &store).unwrap();
        }
    }

    fn bits(params: &Params, id: TensorId) -> Vec<u32> {
        params[&id]
            .to_vec::<f32>()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn test_no_groups_matches_f32_reference_bits() {
        let (client, device) = cpu_setup();
        let cfg = config(2e-5, 0.01);
        let (w0, g) = (0.02f32, 0.001f32);
        let expected = f32_reference(w0, g, &cfg, 4).to_bits();

        let plain_id = tensor(&device, &[w0]);
        let id = plain_id.id();
        let mut params = HashMap::from([(id, plain_id)]);
        let mut plain = AdamW::<CpuRuntime>::new(cfg.clone());
        step_n(&mut plain, &client, &device, &mut params, &[(id, &[g])], 4);
        assert_eq!(bits(&params, id), vec![expected]);

        let empty = AdamW::<CpuRuntime>::with_param_groups(cfg.clone(), vec![]).unwrap();
        assert!(empty.param_groups().is_empty());

        // A neutral group runs the same arithmetic as no group.
        let t = tensor(&device, &[w0]);
        let gid = t.id();
        let mut params = HashMap::from([(gid, t)]);
        let mut grouped =
            AdamW::<CpuRuntime>::with_param_groups(cfg, vec![scaled(&[gid], 1.0, None)]).unwrap();
        step_n(
            &mut grouped,
            &client,
            &device,
            &mut params,
            &[(gid, &[g])],
            4,
        );
        assert_eq!(bits(&params, gid), vec![expected]);
    }

    #[test]
    fn test_zero_decay_group_is_untouched_by_decay() {
        let (client, device) = cpu_setup();
        let a = tensor(&device, &[5.0, 3.0]);
        let b = tensor(&device, &[5.0, 3.0]);
        let (ia, ib) = (a.id(), b.id());
        let mut params = HashMap::from([(ia, a), (ib, b)]);
        let mut opt = AdamW::<CpuRuntime>::with_param_groups(
            config(0.1, 0.1),
            vec![scaled(&[ia], 1.0, Some(0.0))],
        )
        .unwrap();
        let zero: &[f32] = &[0.0, 0.0];
        step_n(
            &mut opt,
            &client,
            &device,
            &mut params,
            &[(ia, zero), (ib, zero)],
            3,
        );

        assert_eq!(params[&ia].to_vec::<f32>(), vec![5.0, 3.0]);
        let decayed = params[&ib].to_vec::<f32>();
        assert!(decayed[0] < 5.0 && decayed[1] < 3.0, "got {decayed:?}");
    }

    #[test]
    fn test_zero_lr_scale_freezes_params_but_not_moments() {
        let (client, device) = cpu_setup();
        let a = tensor(&device, &[1.0, -2.0]);
        let ia = a.id();
        let mut params = HashMap::from([(ia, a)]);
        let mut opt = AdamW::<CpuRuntime>::with_param_groups(
            config(0.1, 0.1),
            vec![scaled(&[ia], 0.0, None)],
        )
        .unwrap();
        step_n(
            &mut opt,
            &client,
            &device,
            &mut params,
            &[(ia, &[0.5, -0.5])],
            2,
        );

        assert_eq!(params[&ia].to_vec::<f32>(), vec![1.0, -2.0]);
        let names = HashMap::from([(ia, "a".to_string())]);
        let state = opt.state_dict(&names).unwrap();
        assert!(state["a.exp_avg"].to_vec::<f32>().iter().all(|m| *m != 0.0));
        assert!(
            state["a.exp_avg_sq"]
                .to_vec::<f32>()
                .iter()
                .all(|v| *v != 0.0)
        );
    }

    #[test]
    fn test_lr_scale_two_equals_separate_double_lr_optimizer() {
        let (client, device) = cpu_setup();
        let values = [0.5f32, -0.25, 0.75];
        let grad: &[f32] = &[0.1, -0.2, 0.3];
        let (a, b) = (tensor(&device, &values), tensor(&device, &values));
        let (ia, ib) = (a.id(), b.id());
        let mut params =
            HashMap::from([(ia, a.clone_deep().unwrap()), (ib, b.clone_deep().unwrap())]);
        let mut opt = AdamW::<CpuRuntime>::with_param_groups(
            config(0.01, 0.1),
            vec![scaled(&[ia], 2.0, None)],
        )
        .unwrap();
        step_n(
            &mut opt,
            &client,
            &device,
            &mut params,
            &[(ia, grad), (ib, grad)],
            3,
        );

        let mut fast = HashMap::from([(ia, a)]);
        let mut fast_opt = AdamW::<CpuRuntime>::new(config(0.02, 0.1));
        step_n(&mut fast_opt, &client, &device, &mut fast, &[(ia, grad)], 3);
        let mut slow = HashMap::from([(ib, b)]);
        let mut slow_opt = AdamW::<CpuRuntime>::new(config(0.01, 0.1));
        step_n(&mut slow_opt, &client, &device, &mut slow, &[(ib, grad)], 3);

        assert_eq!(bits(&params, ia), bits(&fast, ia));
        assert_eq!(bits(&params, ib), bits(&slow, ib));
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_widened_path_uses_group_lr() {
        use half::bf16;
        let (client, device) = cpu_setup();
        let bf = |v: f32| bf16::from_f32(v);
        let make =
            |v: f32| Tensor::<CpuRuntime>::from_slice(&[bf(v), bf(-v)], &[2], &device).unwrap();
        let grad = Tensor::<CpuRuntime>::from_slice(&[bf(0.01), bf(0.02)], &[2], &device).unwrap();

        let w = make(0.5);
        let wid = w.id();
        let f32_param = tensor(&device, &[0.5, -0.5]);
        let fid = f32_param.id();
        let mut params = HashMap::from([(wid, w.clone_deep().unwrap()), (fid, f32_param)]);
        let mut opt = AdamW::<CpuRuntime>::with_param_groups(
            config(1e-3, 0.0),
            vec![scaled(&[wid], 2.0, None)],
        )
        .unwrap();
        let mut store = GradStore::new();
        store.insert(wid, grad.clone());
        store.insert(fid, tensor(&device, &[0.01, 0.02]));
        for _ in 0..3 {
            opt.step(&client, &mut params, &store).unwrap();
        }

        let mut reference = HashMap::from([(wid, w)]);
        let mut ref_opt = AdamW::<CpuRuntime>::new(config(2e-3, 0.0));
        let mut ref_store = GradStore::new();
        ref_store.insert(wid, grad);
        for _ in 0..3 {
            ref_opt.step(&client, &mut reference, &ref_store).unwrap();
        }
        assert_eq!(
            params[&wid].to_bytes().unwrap(),
            reference[&wid].to_bytes().unwrap()
        );
    }

    #[test]
    fn test_id_in_two_groups_is_rejected() {
        let (_client, device) = cpu_setup();
        let id = tensor(&device, &[1.0]).id();
        let err = AdamW::<CpuRuntime>::with_param_groups(
            AdamWConfig::default(),
            vec![scaled(&[id], 1.0, None), scaled(&[id], 0.5, None)],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(err.contains(&format!("{id:?}")), "got {err}");

        let mut opt = AdamW::<CpuRuntime>::with_param_groups(
            AdamWConfig::default(),
            vec![scaled(&[id], 1.0, None)],
        )
        .unwrap();
        assert!(opt.add_param_group(scaled(&[id], 1.0, None)).is_err());
        assert_eq!(opt.param_groups().len(), 1);
    }

    #[test]
    fn test_invalid_scale_and_decay_are_rejected() {
        let mut opt = AdamW::<CpuRuntime>::new(AdamWConfig::default());
        for bad in [f64::NAN, f64::INFINITY, -0.5] {
            assert!(opt.add_param_group(scaled(&[], bad, None)).is_err());
        }
        assert!(opt.add_param_group(scaled(&[], 1.0, Some(-1.0))).is_err());
        assert!(opt.param_groups().is_empty());

        opt.add_param_group(scaled(&[], 1.0, None)).unwrap();
        assert!(opt.set_group_lr_scale(0, f64::NAN).is_err());
        assert!(opt.set_group_lr_scale(1, 1.0).is_err());
        assert!(opt.set_group_weight_decay(1, None).is_err());
        opt.set_group_lr_scale(0, 0.25).unwrap();
        opt.set_group_weight_decay(0, Some(0.0)).unwrap();
        assert_eq!(opt.param_groups()[0].lr_scale, 0.25);
        assert_eq!(opt.param_groups()[0].weight_decay, Some(0.0));
    }

    #[test]
    fn test_set_lr_scales_effective_rate() {
        let (client, device) = cpu_setup();
        let values = [0.5f32, -0.25];
        let grad: &[f32] = &[0.1, -0.2];
        let a = tensor(&device, &values);
        let ia = a.id();
        let mut params = HashMap::from([(ia, a.clone_deep().unwrap())]);
        let mut opt = AdamW::<CpuRuntime>::with_param_groups(
            config(0.01, 0.1),
            vec![scaled(&[ia], 0.5, None)],
        )
        .unwrap();
        opt.set_lr(0.02);
        assert_eq!(Optimizer::lr(&opt), 0.02);
        step_n(&mut opt, &client, &device, &mut params, &[(ia, grad)], 2);

        let mut reference = HashMap::from([(ia, a)]);
        let mut ref_opt = AdamW::<CpuRuntime>::new(config(0.01, 0.1));
        step_n(
            &mut ref_opt,
            &client,
            &device,
            &mut reference,
            &[(ia, grad)],
            2,
        );
        assert_eq!(bits(&params, ia), bits(&reference, ia));
    }

    #[test]
    fn test_no_decay_from_names_selects_by_predicate() {
        let (_client, device) = cpu_setup();
        let (w, b, n) = (
            tensor(&device, &[1.0]).id(),
            tensor(&device, &[1.0]).id(),
            tensor(&device, &[1.0]).id(),
        );
        let named = [
            ("layer.weight", w),
            ("layer.bias", b),
            ("final_norm.weight", n),
        ];
        let group = ParamGroup::no_decay_from_names(named.iter().copied(), |name| {
            name.ends_with(".bias") || name.contains("norm")
        });
        assert_eq!(group.ids, HashSet::from([b, n]));
        assert_eq!(group.weight_decay, Some(0.0));
        assert_eq!(group.lr_scale, 1.0);
    }

    #[test]
    fn test_state_dict_round_trip_with_groups_resumes_identically() {
        let (client, device) = cpu_setup();
        let fx = Fixture::new(&client, &device, false);
        let w = fx.ids["layer.w"];
        let make = || {
            AdamW::<CpuRuntime>::with_param_groups(
                config(0.01, 0.1),
                vec![scaled(&[w], 0.5, Some(0.0))],
            )
            .unwrap()
        };
        let mut params = deep(&fx.params);
        let mut opt = make();
        run(&mut opt, &client, &fx, &mut params, 0..3);
        let state = opt.state_dict(&fx.names).unwrap();

        let mut resumed_params = deep(&params);
        run(&mut opt, &client, &fx, &mut params, 3..5);

        let mut resumed = make();
        resumed.load_state_dict(&fx.ids, &state).unwrap();
        run(&mut resumed, &client, &fx, &mut resumed_params, 3..5);

        assert_same_params(&params, &resumed_params);
        assert_same_state(
            &opt.state_dict(&fx.names).unwrap(),
            &resumed.state_dict(&fx.names).unwrap(),
        );
    }
}
