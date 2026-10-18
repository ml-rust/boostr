//! Shared fixtures for the optimizer state dictionary tests.

use crate::optimizer::Optimizer;
use crate::test_utils::cpu_setup;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;
use std::ops::Range;

pub(crate) type Params = HashMap<TensorId, Tensor<CpuRuntime>>;
pub(crate) type State = HashMap<String, Tensor<CpuRuntime>>;

/// Two named parameters: `layer.w` with shape `[2, 3]` and `bias` with shape `[3]`.
pub(crate) struct Fixture {
    pub(crate) params: Params,
    pub(crate) names: HashMap<TensorId, String>,
    pub(crate) ids: HashMap<String, TensorId>,
}

fn widen_or_cast(client: &CpuClient, t: Tensor<CpuRuntime>, narrow: bool) -> Tensor<CpuRuntime> {
    if narrow {
        client.cast(&t, DType::BF16).unwrap()
    } else {
        t
    }
}

impl Fixture {
    /// Build the parameters. `narrow` makes `layer.w` a BF16 parameter.
    pub(crate) fn new(client: &CpuClient, device: &CpuDevice, narrow: bool) -> Self {
        let w = Tensor::<CpuRuntime>::from_slice(
            &[0.5f32, -0.25, 0.75, 1.0, -1.5, 0.125],
            &[2, 3],
            device,
        )
        .unwrap();
        let w = widen_or_cast(client, w, narrow);
        let b = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3], &[3], device).unwrap();
        let mut params = HashMap::new();
        let mut names = HashMap::new();
        let mut ids = HashMap::new();
        for (name, t) in [("layer.w", w), ("bias", b)] {
            let id = t.id();
            names.insert(id, name.to_string());
            ids.insert(name.to_string(), id);
            params.insert(id, t);
        }
        Self { params, names, ids }
    }

    /// Deterministic gradients for step number `step`, at each parameter's dtype.
    pub(crate) fn grads(&self, client: &CpuClient, step: usize) -> GradStore<CpuRuntime> {
        let mut grads = GradStore::new();
        for (id, param) in &self.params {
            let values: Vec<f32> = (0..param.numel())
                .map(|i| ((step * 7 + i) as f32).sin() * 0.1)
                .collect();
            let g =
                Tensor::<CpuRuntime>::from_slice(&values, param.shape(), param.device()).unwrap();
            let g = if param.dtype() == DType::F32 {
                g
            } else {
                client.cast(&g, param.dtype()).unwrap()
            };
            grads.insert(*id, g);
        }
        grads
    }
}

/// Copy every tensor of `params` into new storage.
pub(crate) fn deep(params: &Params) -> Params {
    params
        .iter()
        .map(|(id, t)| (*id, t.clone_deep().unwrap()))
        .collect()
}

/// Run the steps numbered by `steps`.
pub(crate) fn run<O: Optimizer<CpuRuntime>>(
    opt: &mut O,
    client: &CpuClient,
    fx: &Fixture,
    params: &mut Params,
    steps: Range<usize>,
) {
    for step in steps {
        opt.step(client, params, &fx.grads(client, step)).unwrap();
    }
}

/// Assert that two parameter maps hold the same bytes.
pub(crate) fn assert_same_params(a: &Params, b: &Params) {
    assert_eq!(a.len(), b.len());
    for (id, t) in a {
        let other = &b[id];
        assert_eq!(t.dtype(), other.dtype());
        assert_eq!(t.to_bytes().unwrap(), other.to_bytes().unwrap());
    }
}

/// Assert that two state dictionaries hold the same keys and bytes.
pub(crate) fn assert_same_state(a: &State, b: &State) {
    let mut ka: Vec<_> = a.keys().collect();
    let mut kb: Vec<_> = b.keys().collect();
    ka.sort();
    kb.sort();
    assert_eq!(ka, kb);
    for (key, t) in a {
        assert_eq!(t.dtype(), b[key].dtype(), "{key}");
        assert_eq!(t.to_bytes().unwrap(), b[key].to_bytes().unwrap(), "{key}");
    }
}

/// Step three times, export, load into a fresh optimizer, then step once on both.
///
/// The original steps before the load, so the test also checks that the export
/// is a snapshot. Returns the exported state.
pub(crate) fn assert_round_trip<O: Optimizer<CpuRuntime>>(
    make: impl Fn() -> O,
    narrow: bool,
) -> State {
    let (client, device) = cpu_setup();
    let fx = Fixture::new(&client, &device, narrow);
    let mut params = deep(&fx.params);
    let mut opt = make();
    run(&mut opt, &client, &fx, &mut params, 0..3);
    let state = opt.state_dict(&fx.names).unwrap();

    let mut params_fresh = deep(&params);
    run(&mut opt, &client, &fx, &mut params, 3..4);

    let mut fresh = make();
    fresh.load_state_dict(&fx.ids, &state).unwrap();
    run(&mut fresh, &client, &fx, &mut params_fresh, 3..4);

    assert_same_params(&params, &params_fresh);
    assert_same_state(
        &opt.state_dict(&fx.names).unwrap(),
        &fresh.state_dict(&fx.names).unwrap(),
    );
    state
}

/// Load `bad` into an optimizer that holds `good`, and check the error and the rollback.
pub(crate) fn assert_load_fails<O: Optimizer<CpuRuntime>>(
    make: &impl Fn() -> O,
    fx: &Fixture,
    good: &State,
    bad: &State,
    needle: &str,
) {
    let mut opt = make();
    opt.load_state_dict(&fx.ids, good).unwrap();
    let err = opt.load_state_dict(&fx.ids, bad).err().unwrap().to_string();
    assert!(err.contains(needle), "error '{err}' lacks '{needle}'");
    assert_same_state(&opt.state_dict(&fx.names).unwrap(), good);
}

/// Strict-load cases shared by AdamW and Lamb.
pub(crate) fn assert_moment_strictness<O: Optimizer<CpuRuntime>>(make: impl Fn() -> O) {
    let (client, device) = cpu_setup();
    let fx = Fixture::new(&client, &device, false);
    let mut params = deep(&fx.params);
    let mut opt = make();
    run(&mut opt, &client, &fx, &mut params, 0..2);
    let good = opt.state_dict(&fx.names).unwrap();
    let zeros =
        |shape: &[usize], dtype: DType| Tensor::<CpuRuntime>::zeros(shape, dtype, &device).unwrap();

    let mut unknown_name = good.clone();
    unknown_name.insert("ghost.exp_avg".into(), zeros(&[3], DType::F32));
    assert_load_fails(&make, &fx, &good, &unknown_name, "ghost.exp_avg");

    let mut unknown_field = good.clone();
    unknown_field.insert("bias.bogus".into(), zeros(&[3], DType::F32));
    assert_load_fails(&make, &fx, &good, &unknown_field, "bias.bogus");

    let mut bad_shape = good.clone();
    bad_shape.insert("bias.exp_avg_sq".into(), zeros(&[2], DType::F32));
    assert_load_fails(&make, &fx, &good, &bad_shape, "bias.exp_avg_sq");

    let mut bad_dtype = good.clone();
    bad_dtype.insert("bias.exp_avg_sq".into(), zeros(&[3], DType::F64));
    assert_load_fails(&make, &fx, &good, &bad_dtype, "bias.exp_avg_sq");

    let mut bad_master = good.clone();
    bad_master.insert("bias.master".into(), zeros(&[2], DType::F32));
    assert_load_fails(&make, &fx, &good, &bad_master, "bias.master");

    let mut partial = good.clone();
    partial.remove("bias.exp_avg_sq");
    assert_load_fails(&make, &fx, &good, &partial, "bias.exp_avg_sq");

    let mut no_step = good.clone();
    no_step.remove("__step");
    assert_load_fails(&make, &fx, &good, &no_step, "__step");

    let mut bad_step = good.clone();
    bad_step.insert("__step".into(), zeros(&[1], DType::F32));
    assert_load_fails(&make, &fx, &good, &bad_step, "__step");

    let err = opt.state_dict(&HashMap::new()).err().unwrap().to_string();
    assert!(err.contains("names"), "{err}");
}

/// Assert the step key holds `expected` as a 1-element F64 tensor.
pub(crate) fn assert_step_is(state: &State, expected: f64) {
    let step = &state["__step"];
    assert_eq!(step.dtype(), DType::F64);
    assert_eq!(step.shape(), &[1]);
    assert_eq!(step.to_vec::<f64>(), vec![expected]);
}

/// Run the steps numbered by `steps`, with one `step_partial` per parameter.
///
/// Parameters go in name order, one gradient per call, as a bucketed caller does.
pub(crate) fn run_partial<O: Optimizer<CpuRuntime>>(
    opt: &mut O,
    client: &CpuClient,
    fx: &Fixture,
    params: &mut Params,
    steps: Range<usize>,
) {
    let mut order: Vec<(&String, &TensorId)> = fx.ids.iter().collect();
    order.sort_by(|a, b| a.0.cmp(b.0));
    for step in steps {
        let grads = fx.grads(client, step);
        opt.begin_step();
        for (_, id) in &order {
            let mut subset = GradStore::new();
            subset.insert(**id, grads.get(**id).unwrap().clone());
            opt.step_partial(client, params, &subset).unwrap();
        }
    }
}

/// Check that one `step_partial` per parameter gives the bits of one `step`.
///
/// Three bucketed steps must match three whole steps in parameters and state.
/// A state exported after two bucketed steps must resume to the same bits.
pub(crate) fn assert_partial_matches_step<O: Optimizer<CpuRuntime>>(
    make: impl Fn(&Fixture) -> O,
    narrow: bool,
) {
    // The BF16 case needs the CPU cast kernel, which exists only with `f16`.
    if narrow && !cfg!(feature = "f16") {
        return;
    }
    let (client, device) = cpu_setup();
    let fx = Fixture::new(&client, &device, narrow);

    let mut whole = make(&fx);
    let mut whole_params = deep(&fx.params);
    run(&mut whole, &client, &fx, &mut whole_params, 0..3);
    let whole_state = whole.state_dict(&fx.names).unwrap();

    let mut split = make(&fx);
    let mut split_params = deep(&fx.params);
    run_partial(&mut split, &client, &fx, &mut split_params, 0..2);
    let state = split.state_dict(&fx.names).unwrap();
    let mut resumed_params = deep(&split_params);
    run_partial(&mut split, &client, &fx, &mut split_params, 2..3);

    assert_same_params(&whole_params, &split_params);
    assert_same_state(&whole_state, &split.state_dict(&fx.names).unwrap());

    let mut resumed = make(&fx);
    resumed.load_state_dict(&fx.ids, &state).unwrap();
    run_partial(&mut resumed, &client, &fx, &mut resumed_params, 2..3);

    assert_same_params(&whole_params, &resumed_params);
    assert_same_state(&whole_state, &resumed.state_dict(&fx.names).unwrap());
}

/// Check that `step_partial` refuses to run outside a begun step.
///
/// The cases are a new optimizer, one after `reset`, and one after `load_state_dict`.
/// A refused call must leave the parameters unchanged.
pub(crate) fn assert_partial_needs_begin<O: Optimizer<CpuRuntime>>(make: impl Fn() -> O) {
    let (client, device) = cpu_setup();
    let fx = Fixture::new(&client, &device, false);
    let grads = fx.grads(&client, 0);
    let mut params = deep(&fx.params);
    let refused = |opt: &mut O, params: &mut Params| {
        let err = opt
            .step_partial(&client, params, &grads)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("begin_step"), "{err}");
    };

    let mut opt = make();
    refused(&mut opt, &mut params);
    assert_same_params(&params, &fx.params);

    opt.step(&client, &mut params, &grads).unwrap();
    let state = opt.state_dict(&fx.names).unwrap();
    opt.reset();
    refused(&mut opt, &mut params);

    opt.load_state_dict(&fx.ids, &state).unwrap();
    refused(&mut opt, &mut params);
}
