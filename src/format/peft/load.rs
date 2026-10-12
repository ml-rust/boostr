//! [`PeftAdapter`]: a PEFT LoRA adapter read from `adapter_config.json` and
//! `adapter_model.safetensors` into per-module A/B pairs on a device.

use super::config::{PeftConfig, prefix_reason};
use super::names::{LoraPart, parse_adapter_key};
use super::paths::resolve_paths;
use crate::error::{Error, Result};
use crate::format::safetensors::{SafeTensors, TensorInfo};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::BTreeMap;
use std::path::Path;

/// One adapted module's LoRA factors.
///
/// The module output gains `scaling * (x @ a^T) @ b^T`.
pub struct LoraPair<R: Runtime> {
    /// `lora_A`: `[r, in_features]`.
    pub a: Tensor<R>,
    /// `lora_B`: `[out_features, r]`.
    pub b: Tensor<R>,
    /// `alpha / r`, or `alpha / sqrt(r)` with rsLoRA, per the module's patterns.
    pub scaling: f32,
}

impl<R: Runtime> Clone for LoraPair<R> {
    fn clone(&self) -> Self {
        Self {
            a: self.a.clone(),
            b: self.b.clone(),
            scaling: self.scaling,
        }
    }
}

/// A loaded PEFT LoRA adapter.
pub struct PeftAdapter<R: Runtime> {
    /// The parsed `adapter_config.json`.
    pub config: PeftConfig,
    /// Factors keyed by base-checkpoint module name, e.g. `model.layers.0.self_attn.q_proj`.
    pub layers: BTreeMap<String, LoraPair<R>>,
}

impl<R: Runtime> PeftAdapter<R> {
    /// The factors for base-checkpoint module `module`, if the adapter targets it.
    pub fn get(&self, module: &str) -> Option<&LoraPair<R>> {
        self.layers.get(module)
    }
}

impl<R: Runtime<DType = DType>> PeftAdapter<R> {
    /// Load an adapter onto `device`.
    ///
    /// `path` is a directory holding `adapter_config.json` and
    /// `adapter_model.safetensors`, or a `.safetensors` file with
    /// `adapter_config.json` beside it. Every factor keeps its file dtype
    /// (F32, F16 or BF16). The one exception is a backend that computes
    /// floats in F32 only (WebGPU): the SafeTensors decoder widens F16 and
    /// BF16 to F32 there.
    ///
    /// # Errors
    ///
    /// Errors when the config is missing or refused (see [`PeftConfig`]),
    /// the weights are `adapter_model.bin` only, a tensor key is not a
    /// `lora_A`/`lora_B` factor or has an unknown prefix, a factor has no
    /// partner, two keys name the same factor, a factor is not 2-D, a rank
    /// differs from the config's `r` or its `rank_pattern` entry, or the
    /// factor dtypes are unsupported or differ within a pair.
    pub fn load(path: &Path, device: &R::Device) -> Result<Self> {
        let (config_path, weights_path) = resolve_paths(path)?;
        let config = PeftConfig::from_file(&config_path)?;
        let mut file = SafeTensors::open(&weights_path).map_err(|e| Error::ModelError {
            reason: format!(
                "cannot open PEFT adapter weights {}: {e}",
                weights_path.display()
            ),
        })?;
        let plan = plan_pairs(&file, &config)
            .map_err(|e| prefix_reason(&weights_path.display().to_string(), e))?;

        let mut layers = BTreeMap::new();
        for (module, planned) in plan {
            let a = file.load_tensor::<R>(&planned.a_key, device)?;
            let b = file.load_tensor::<R>(&planned.b_key, device)?;
            let scaling = planned.scaling;
            layers.insert(module, LoraPair { a, b, scaling });
        }
        Ok(Self { config, layers })
    }
}

/// The tensor keys and scaling of one module's pair, checked before any load.
struct PlannedPair {
    a_key: String,
    b_key: String,
    scaling: f32,
}

#[derive(Default)]
struct Slot {
    peft_module: String,
    a: Option<String>,
    b: Option<String>,
}

fn plan_pairs(file: &SafeTensors, config: &PeftConfig) -> Result<BTreeMap<String, PlannedPair>> {
    let mut slots: BTreeMap<String, Slot> = BTreeMap::new();
    for key in file.tensor_names() {
        let parsed = parse_adapter_key(key)?;
        let slot = slots.entry(parsed.module.clone()).or_insert_with(|| Slot {
            peft_module: parsed.peft_module.clone(),
            ..Slot::default()
        });
        if slot.peft_module != parsed.peft_module {
            return Err(Error::ModelError {
                reason: format!(
                    "adapter tensor `{key}` names module `{}`, which another tensor names as \
                     `{}`; one adapter uses one prefix",
                    parsed.module, slot.peft_module
                ),
            });
        }
        let (factor, name) = match parsed.part {
            LoraPart::A => (&mut slot.a, "lora_A"),
            LoraPart::B => (&mut slot.b, "lora_B"),
        };
        if let Some(previous) = factor {
            return Err(Error::ModelError {
                reason: format!(
                    "adapter tensors `{previous}` and `{key}` are both the {name} factor of \
                     module `{}`",
                    parsed.module
                ),
            });
        }
        *factor = Some(key.to_string());
    }
    if slots.is_empty() {
        return Err(Error::ModelError {
            reason: "adapter holds no LoRA tensors".into(),
        });
    }

    let mut plan = BTreeMap::new();
    for (module, slot) in slots {
        let (a_key, b_key) = match (slot.a, slot.b) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => return Err(unpaired(&a, "lora_B", &module)),
            (None, Some(b)) => return Err(unpaired(&b, "lora_A", &module)),
            (None, None) => {
                return Err(Error::ModelError {
                    reason: format!("module `{module}` has no LoRA tensors"),
                });
            }
        };
        let a_info = file.tensor_info(&a_key)?;
        let b_info = file.tensor_info(&b_key)?;
        let (a_rank, _) = factor_shape(&a_key, a_info)?;
        let (_, b_rank) = factor_shape(&b_key, b_info)?;
        check_dtypes(&a_key, a_info, &b_key, b_info)?;

        let rank = config.rank_for(&slot.peft_module);
        if a_rank != rank || b_rank != rank {
            let source = match config.rank_pattern_key(&slot.peft_module) {
                Some(pattern) => format!("rank_pattern key `{pattern}`"),
                None => "`r`".to_string(),
            };
            return Err(Error::ModelError {
                reason: format!(
                    "module `{module}`: `{a_key}` has shape {:?} and `{b_key}` has shape {:?}, \
                     but adapter_config.json sets rank {rank} through {source}",
                    a_info.shape, b_info.shape
                ),
            });
        }
        let scaling = config.scaling_for(&slot.peft_module);
        plan.insert(
            module,
            PlannedPair {
                a_key,
                b_key,
                scaling,
            },
        );
    }
    Ok(plan)
}

fn unpaired(present: &str, missing: &str, module: &str) -> Error {
    Error::ModelError {
        reason: format!(
            "adapter tensor `{present}` has no matching {missing} factor for module `{module}`"
        ),
    }
}

fn factor_shape(key: &str, info: &TensorInfo) -> Result<(usize, usize)> {
    match info.shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        other => Err(Error::ModelError {
            reason: format!("adapter tensor `{key}` has shape {other:?}; LoRA factors are 2-D"),
        }),
    }
}

fn check_dtypes(a_key: &str, a: &TensorInfo, b_key: &str, b: &TensorInfo) -> Result<()> {
    for (key, info) in [(a_key, a), (b_key, b)] {
        if !matches!(info.dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::ModelError {
                reason: format!(
                    "adapter tensor `{key}` has dtype {:?}; LoRA factors load as F32, F16 or BF16",
                    info.dtype
                ),
            });
        }
    }
    if a.dtype != b.dtype {
        return Err(Error::ModelError {
            reason: format!(
                "adapter tensors `{a_key}` ({:?}) and `{b_key}` ({:?}) form one pair but differ \
                 in dtype",
                a.dtype, b.dtype
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::paths::{ADAPTER_BIN_FILE, ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE};
    use super::*;
    use crate::format::safetensors::save_safetensors;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use std::collections::HashMap;

    const PREFIX: &str = "base_model.model.model.layers";

    fn tensor(rows: usize, cols: usize, start: f32) -> Tensor<CpuRuntime> {
        let data: Vec<f32> = (0..rows * cols).map(|i| start + i as f32).collect();
        Tensor::<CpuRuntime>::from_slice(&data, &[rows, cols], &CpuDevice::new()).unwrap()
    }

    fn write_adapter(dir: &Path, config: &str, tensors: Vec<(String, Tensor<CpuRuntime>)>) {
        std::fs::write(dir.join(ADAPTER_CONFIG_FILE), config).unwrap();
        let map: HashMap<String, Tensor<CpuRuntime>> = tensors.into_iter().collect();
        save_safetensors(dir.join(ADAPTER_WEIGHTS_FILE), &map, None).unwrap();
    }

    fn load_err(path: &Path) -> String {
        match PeftAdapter::<CpuRuntime>::load(path, &CpuDevice::new()) {
            Ok(_) => panic!("{} loaded but must error", path.display()),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn round_trips_through_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let config = r#"{"peft_type": "LORA", "r": 4, "lora_alpha": 8,
                         "rank_pattern": {"v_proj": 2}, "target_modules": ["q_proj", "v_proj"]}"#;
        write_adapter(
            dir.path(),
            config,
            vec![
                (
                    format!("{PREFIX}.0.self_attn.q_proj.lora_A.weight"),
                    tensor(4, 6, 0.0),
                ),
                (
                    format!("{PREFIX}.0.self_attn.q_proj.lora_B.weight"),
                    tensor(5, 4, 100.0),
                ),
                (
                    format!("{PREFIX}.1.self_attn.v_proj.lora_A.default.weight"),
                    tensor(2, 6, 7.0),
                ),
                (
                    format!("{PREFIX}.1.self_attn.v_proj.lora_B.default.weight"),
                    tensor(3, 2, 9.0),
                ),
            ],
        );

        let adapter = PeftAdapter::<CpuRuntime>::load(dir.path(), &CpuDevice::new()).unwrap();
        assert_eq!(adapter.config.r, 4);
        assert_eq!(adapter.layers.len(), 2);

        let q = adapter.get("model.layers.0.self_attn.q_proj").unwrap();
        assert_eq!(q.a.shape(), &[4, 6]);
        assert_eq!(q.b.shape(), &[5, 4]);
        assert_eq!(q.a.dtype(), DType::F32);
        assert_eq!(q.a.to_vec::<f32>(), tensor(4, 6, 0.0).to_vec::<f32>());
        assert_eq!(q.b.to_vec::<f32>(), tensor(5, 4, 100.0).to_vec::<f32>());
        assert_eq!(q.scaling, 2.0);

        let v = adapter.get("model.layers.1.self_attn.v_proj").unwrap();
        assert_eq!(v.a.shape(), &[2, 6]);
        assert_eq!(v.b.to_vec::<f32>(), tensor(3, 2, 9.0).to_vec::<f32>());
        assert_eq!(v.scaling, 4.0);
    }

    #[test]
    fn loads_language_model_prefix_from_a_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let lm = "base_model.model.model.language_model.layers.2.mlp.down_proj";
        write_adapter(
            dir.path(),
            r#"{"r": 2, "lora_alpha": 4, "use_rslora": true}"#,
            vec![
                (format!("{lm}.lora_A.weight"), tensor(2, 3, 0.0)),
                (format!("{lm}.lora_B.weight"), tensor(4, 2, 0.0)),
            ],
        );
        let file = dir.path().join(ADAPTER_WEIGHTS_FILE);
        let adapter = PeftAdapter::<CpuRuntime>::load(&file, &CpuDevice::new()).unwrap();
        let pair = adapter.get("model.layers.2.mlp.down_proj").unwrap();
        assert!(
            (pair.scaling - 2.0 * 2.0f32.sqrt()).abs() < 1e-6,
            "{}",
            pair.scaling
        );
    }

    #[test]
    fn unpaired_factor_errors() {
        let dir = tempfile::tempdir().unwrap();
        let a_key = format!("{PREFIX}.0.self_attn.q_proj.lora_A.weight");
        write_adapter(
            dir.path(),
            r#"{"r": 4, "lora_alpha": 8}"#,
            vec![
                (a_key.clone(), tensor(4, 6, 0.0)),
                (
                    format!("{PREFIX}.0.self_attn.k_proj.lora_A.weight"),
                    tensor(4, 6, 0.0),
                ),
                (
                    format!("{PREFIX}.0.self_attn.k_proj.lora_B.weight"),
                    tensor(5, 4, 0.0),
                ),
            ],
        );
        let text = load_err(dir.path());
        assert!(text.contains(&a_key), "{text}");
        assert!(text.contains("no matching lora_B"), "{text}");
    }

    #[test]
    fn rank_mismatch_errors() {
        let dir = tempfile::tempdir().unwrap();
        write_adapter(
            dir.path(),
            r#"{"r": 8, "lora_alpha": 8}"#,
            vec![
                (
                    format!("{PREFIX}.0.mlp.up_proj.lora_A.weight"),
                    tensor(4, 6, 0.0),
                ),
                (
                    format!("{PREFIX}.0.mlp.up_proj.lora_B.weight"),
                    tensor(5, 4, 0.0),
                ),
            ],
        );
        let text = load_err(dir.path());
        assert!(text.contains("sets rank 8 through `r`"), "{text}");

        let patterned = tempfile::tempdir().unwrap();
        write_adapter(
            patterned.path(),
            r#"{"r": 4, "lora_alpha": 8, "rank_pattern": {"up_proj": 2}}"#,
            vec![
                (
                    format!("{PREFIX}.0.mlp.up_proj.lora_A.weight"),
                    tensor(4, 6, 0.0),
                ),
                (
                    format!("{PREFIX}.0.mlp.up_proj.lora_B.weight"),
                    tensor(5, 4, 0.0),
                ),
            ],
        );
        let text = load_err(patterned.path());
        assert!(
            text.contains("rank 2 through rank_pattern key `up_proj`"),
            "{text}"
        );
    }

    #[test]
    fn non_lora_tensor_errors() {
        let dir = tempfile::tempdir().unwrap();
        let key = "base_model.model.model.embed_tokens.lora_embedding_A";
        write_adapter(
            dir.path(),
            r#"{"r": 4, "lora_alpha": 8}"#,
            vec![(key.to_string(), tensor(4, 6, 0.0))],
        );
        let text = load_err(dir.path());
        assert!(
            text.contains(key) && text.contains("not a LoRA factor"),
            "{text}"
        );
    }

    #[test]
    fn duplicate_factor_errors() {
        let dir = tempfile::tempdir().unwrap();
        write_adapter(
            dir.path(),
            r#"{"r": 4, "lora_alpha": 8}"#,
            vec![
                (
                    format!("{PREFIX}.0.mlp.up_proj.lora_A.weight"),
                    tensor(4, 6, 0.0),
                ),
                (
                    format!("{PREFIX}.0.mlp.up_proj.lora_A.other.weight"),
                    tensor(4, 6, 0.0),
                ),
                (
                    format!("{PREFIX}.0.mlp.up_proj.lora_B.weight"),
                    tensor(5, 4, 0.0),
                ),
            ],
        );
        let text = load_err(dir.path());
        assert!(text.contains("are both the lora_A factor"), "{text}");
    }

    #[test]
    fn bin_only_and_missing_config_error() {
        let bin_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            bin_dir.path().join(ADAPTER_CONFIG_FILE),
            r#"{"r": 4, "lora_alpha": 8}"#,
        )
        .unwrap();
        std::fs::write(bin_dir.path().join(ADAPTER_BIN_FILE), b"pickle").unwrap();
        let text = load_err(bin_dir.path());
        assert!(text.contains(".bin format, which is unsupported"), "{text}");
        let text = load_err(&bin_dir.path().join(ADAPTER_BIN_FILE));
        assert!(text.contains(".bin format, which is unsupported"), "{text}");

        let no_config = tempfile::tempdir().unwrap();
        let map: HashMap<String, Tensor<CpuRuntime>> = HashMap::from([(
            format!("{PREFIX}.0.mlp.up_proj.lora_A.weight"),
            tensor(4, 6, 0.0),
        )]);
        save_safetensors(no_config.path().join(ADAPTER_WEIGHTS_FILE), &map, None).unwrap();
        let text = load_err(no_config.path());
        assert!(text.contains("has no adapter_config.json"), "{text}");
    }

    #[test]
    fn refused_config_errors_before_weights_load() {
        let dir = tempfile::tempdir().unwrap();
        write_adapter(
            dir.path(),
            r#"{"r": 4, "lora_alpha": 8, "use_dora": true}"#,
            vec![
                (
                    format!("{PREFIX}.0.mlp.up_proj.lora_A.weight"),
                    tensor(4, 6, 0.0),
                ),
                (
                    format!("{PREFIX}.0.mlp.up_proj.lora_B.weight"),
                    tensor(5, 4, 0.0),
                ),
            ],
        );
        let text = load_err(dir.path());
        assert!(text.contains("use_dora = true is unsupported"), "{text}");
    }
}
