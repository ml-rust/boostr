//! A Hugging Face Qwen3.5 decision backbone built for the real-model
//! decision tests, with an optional PEFT adapter, and the numeric helpers
//! their comparisons share.

use std::path::Path;

use boostr::format::hf_qwen35::is_dropped;
use boostr::format::{PeftAdapter, SafeTensorsLoader, restructure_hf_checkpoint};
use boostr::model::{LoadedModel, ModelClient, Qwen35Model, UniversalConfig, load_config_auto};
use boostr::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;

/// The config of the checkpoint in `dir`.
pub fn config(dir: &Path) -> UniversalConfig {
    let path = dir.join("config.json");
    load_config_auto(&path).unwrap_or_else(|e| panic!("load_config_auto {}: {e}", path.display()))
}

/// The model of `dir` on `device` through the Hugging Face load path:
/// tensors loaded to the device, the layout edits made in the file
/// dtypes, every float tensor cast to `dtype` on the device, then
/// `adapter` attached unmerged when given.
pub fn build_model<R>(
    dir: &Path,
    config: &UniversalConfig,
    device: &R::Device,
    dtype: DType,
    adapter: Option<&PeftAdapter<R>>,
) -> Box<Qwen35Model<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    let client = R::default_client(device);
    let mut loader = SafeTensorsLoader::open(dir)
        .unwrap_or_else(|e| panic!("SafeTensorsLoader::open {}: {e}", dir.display()));
    let mut map = VarMap::<R>::new();
    for name in loader.tensor_names() {
        if is_dropped(&name) {
            continue;
        }
        let t = loader
            .load_tensor::<R>(&name, device)
            .unwrap_or_else(|e| panic!("load {name}: {e}"));
        map.insert(name, t);
    }
    restructure_hf_checkpoint(&mut map, config, device)
        .unwrap_or_else(|e| panic!("restructure_hf_checkpoint: {e}"));
    let names: Vec<String> = map.names().map(str::to_string).collect();
    for name in names {
        let t = map
            .take_tensor(&name)
            .unwrap_or_else(|e| panic!("take {name}: {e}"));
        let t = if matches!(t.dtype(), DType::F32 | DType::F16 | DType::BF16) && t.dtype() != dtype
        {
            client
                .cast(&t, dtype)
                .unwrap_or_else(|e| panic!("cast {name} to {dtype:?}: {e}"))
        } else {
            t
        };
        map.insert(name, t);
    }
    let mut vb = VarBuilder::new(&mut map, device);
    let loaded = LoadedModel::<R>::load_with_adapter(config, &mut vb, adapter)
        .unwrap_or_else(|e| panic!("LoadedModel::load_with_adapter: {e}"));
    let model = match loaded {
        LoadedModel::Qwen35(model) => model,
        _ => panic!("LoadedModel::load_with_adapter returned a model other than Qwen35"),
    };
    assert_eq!(
        map.len(),
        0,
        "loader left {} tensors behind: {:?}",
        map.len(),
        map.names().take(8).collect::<Vec<_>>()
    );
    if let Some(adapter) = adapter {
        assert_eq!(
            model.adapted_modules().len(),
            adapter.layers.len(),
            "every adapter module attaches"
        );
    }
    model
}

/// The first index of the largest value, and the gap to the runner-up
/// (`inf` for one value).
pub fn top<T: Copy + Into<f64>>(values: &[T]) -> (usize, f64) {
    let v: Vec<f64> = values.iter().map(|&x| x.into()).collect();
    let best = (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    let second = (0..v.len())
        .filter(|&i| i != best)
        .map(|i| v[i])
        .fold(f64::NEG_INFINITY, f64::max);
    (best, v[best] - second)
}

/// Every entry of `got` within `tol` of `want`; returns the largest
/// difference.
pub fn close_f64(got: &[f64], want: &[f64], tol: f64, what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(d <= tol, "{what} entry {i}: {g} vs reference {w}");
        worst = worst.max(d);
    }
    worst
}
