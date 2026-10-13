//! Building a [`JointSchemaHead`] from named tensors
//! ([`JointSchemaHead::from_tensors`]) or from a checkpoint directory's
//! `joint_head_config.json` and `joint_head.safetensors`
//! ([`JointSchemaHead::from_dir`]), with every key and shape checked.

use super::blocks::FeedForward;
use super::config::{CONFIG_FILE, FIELD_TYPES, HeadDtype, JointSchemaConfig, WEIGHTS_FILE};
use super::decoder_layer::DecoderLayer;
use super::head::JointSchemaHead;
use super::routing::RoutingLayer;
use super::weights::{Weights, model_error};
use crate::error::{Error, Result};
use crate::format::SafeTensors;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// The three scalar parameters of the head, as stored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointScalars {
    /// `prior_logit_scale`: the prior scale is `exp(min(it, ln 100))`.
    pub prior_logit_scale: f64,
    /// `joint_logit_scale`: the cosine scale is `exp(min(it, ln 100))`.
    pub joint_logit_scale: f64,
    /// `residual_gate`: the joint term is weighted by `sigmoid(it)`.
    pub residual_gate: f64,
}

/// The keys of the scalar parameters.
const SCALAR_KEYS: [&str; 3] = ["prior_logit_scale", "joint_logit_scale", "residual_gate"];

/// `v` rounded to `dtype`, as a 0-dim tensor of that dtype holds it.
pub(super) fn round_to(dtype: DType, v: f64) -> f64 {
    match dtype {
        DType::BF16 => half::bf16::from_f64(v).to_f64(),
        DType::F16 => half::f16::from_f64(v).to_f64(),
        DType::F32 => f64::from(v as f32),
        _ => v,
    }
}

impl<R: Runtime<DType = DType>> JointSchemaHead<R> {
    /// A head from its tensors under the torch `JointSchemaHead` state-dict
    /// names (the scalars aside), each cast to `dtype`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `config` fails its check or `dtype`
    /// is not F32, F16 or BF16. [`Error::ModelError`] when a key is
    /// missing, a tensor's shape differs from the config's (naming the key
    /// and both shapes), a tensor is not a float dtype, a key is left over
    /// (naming every such key), or a scalar is not finite. Device errors of
    /// the casts propagate.
    pub fn from_tensors<C: ModelClient<R>>(
        client: &C,
        config: JointSchemaConfig,
        tensors: HashMap<String, Tensor<R>>,
        scalars: JointScalars,
        dtype: DType,
    ) -> Result<Self> {
        config.validate()?;
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("a joint schema head computes in F32, F16 or BF16, got {dtype:?}"),
            });
        }
        let values = [
            scalars.prior_logit_scale,
            scalars.joint_logit_scale,
            scalars.residual_gate,
        ];
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            return Err(model_error(format!(
                "joint schema head: {} is {}, expected a finite number",
                SCALAR_KEYS[i], values[i]
            )));
        }
        let c = config;
        let (h, w) = (c.hidden_size, c.width);
        let mut wt = Weights::new(client, tensors, dtype);
        let hidden_norm = wt.norm("hidden_norm", h)?;
        let memory_projection = wt.affine("memory_projection", h, w, false)?;
        let question_projection = wt.affine("question_projection", h, w, false)?;
        let option_question_projection = wt.affine("option_question_projection", h, w, false)?;
        let global_projection = wt.affine("global_projection", h, w, false)?;
        let option_context_projection = wt.affine("option_context_projection", h, w, false)?;
        let option_lexical_projection = wt.affine("option_lexical_projection", h, w, false)?;
        let type_embedding = wt.take("type_embedding.weight", &[FIELD_TYPES, w])?;
        let mut evidence_layers = Vec::with_capacity(c.routing_layers);
        for i in 0..c.routing_layers {
            let p = format!("evidence_layers.{i}");
            evidence_layers.push(RoutingLayer {
                query_norm: wt.norm(&format!("{p}.query_norm"), w)?,
                memory_norm: wt.norm(&format!("{p}.memory_norm"), w)?,
                attention: wt.attention(&format!("{p}.attention"), &c)?,
                feedforward_norm: wt.norm(&format!("{p}.feedforward_norm"), w)?,
                feedforward: wt.feedforward(
                    &format!("{p}.feedforward.0"),
                    &format!("{p}.feedforward.3"),
                    &c,
                )?,
            });
        }
        let option_summary_norm = wt.norm("option_summary_norm", w)?;
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let p = format!("layers.{i}");
            layers.push(DecoderLayer {
                norm1: wt.norm(&format!("{p}.norm1"), w)?,
                self_attn: wt.attention(&format!("{p}.self_attn"), &c)?,
                norm2: wt.norm(&format!("{p}.norm2"), w)?,
                cross_attn: wt.attention(&format!("{p}.multihead_attn"), &c)?,
                norm3: wt.norm(&format!("{p}.norm3"), w)?,
                feedforward: wt.feedforward(
                    &format!("{p}.linear1"),
                    &format!("{p}.linear2"),
                    &c,
                )?,
            });
        }
        let field_norm = wt.norm("field_norm", w)?;
        let option_norm = wt.norm("option_norm", w)?;
        let residual_scorer = FeedForward {
            inner: wt.affine("residual_scorer.0", 4 * w, w, true)?,
            outer: wt.affine("residual_scorer.3", w, 1, true)?,
        };
        wt.finish()?;

        let cap = 100f64.ln();
        let scale = |v: f64| round_to(dtype, round_to(dtype, v.min(cap)).exp());
        let gate = round_to(dtype, 1.0 / (1.0 + (-scalars.residual_gate).exp()));
        Ok(Self {
            config: c,
            dtype,
            hidden_norm,
            memory_projection,
            question_projection,
            option_question_projection,
            global_projection,
            option_context_projection,
            option_lexical_projection,
            type_embedding,
            evidence_layers,
            option_summary_norm,
            layers,
            field_norm,
            option_norm,
            residual_scorer,
            prior_scale: scale(scalars.prior_logit_scale),
            joint_scale: scale(scalars.joint_logit_scale),
            gate,
        })
    }

    /// The head of a checkpoint directory: `joint_head_config.json` and
    /// `joint_head.safetensors`, on `client`'s device. [`HeadDtype::F32`]
    /// casts every weight to F32 once; [`HeadDtype::Checkpoint`] keeps the
    /// stored dtype.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] naming the file when the config fails
    /// [`JointSchemaConfig::from_file`], when the weights do not open, when
    /// a scalar is missing or is not a 0-dim float, when
    /// [`HeadDtype::Checkpoint`] meets tensors of more than one dtype or a
    /// dtype the backend does not compute; errors of
    /// [`Self::from_tensors`] otherwise.
    pub fn from_dir<C: ModelClient<R>>(
        client: &C,
        dir: impl AsRef<Path>,
        head_dtype: HeadDtype,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let config = JointSchemaConfig::from_file(dir.join(CONFIG_FILE))?;
        let path = dir.join(WEIGHTS_FILE);
        let file_error = |reason: String| model_error(format!("{}: {reason}", path.display()));
        let mut st = SafeTensors::open(&path).map_err(|e| file_error(e.to_string()))?;
        let names: Vec<String> = st.tensor_names().map(str::to_string).collect();

        let mut scalars = [0.0f64; 3];
        for (slot, key) in scalars.iter_mut().zip(SCALAR_KEYS) {
            let info = st
                .tensor_info(key)
                .map_err(|_| file_error(format!("no scalar {key}")))?
                .clone();
            if !info.shape.is_empty() {
                return Err(file_error(format!(
                    "{key} is {:?}, expected a 0-dim scalar",
                    info.shape
                )));
            }
            let bytes = st.read_tensor_bytes(key)?;
            *slot = decode_scalar(&bytes, info.dtype).ok_or_else(|| {
                file_error(format!("{key} is {:?}, expected a float", info.dtype))
            })?;
        }

        let dtype = match head_dtype {
            HeadDtype::F32 => DType::F32,
            HeadDtype::Checkpoint => {
                let mut stored: Vec<DType> = Vec::new();
                for name in &names {
                    let d = st.tensor_info(name)?.dtype;
                    if !stored.contains(&d) {
                        stored.push(d);
                    }
                }
                let [only] = stored.as_slice() else {
                    return Err(file_error(format!(
                        "HeadDtype::Checkpoint needs one stored dtype, the file holds {stored:?}"
                    )));
                };
                if !R::computes_float_dtype(*only) {
                    return Err(file_error(format!(
                        "the file stores {only:?}, which this backend does not compute; \
                         build with the f16 feature or load with HeadDtype::F32"
                    )));
                }
                *only
            }
        };

        let device = client.device();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            if SCALAR_KEYS.contains(&name.as_str()) {
                continue;
            }
            let t = st
                .load_tensor::<R>(&name, device)
                .map_err(|e| file_error(format!("{name}: {e}")))?;
            tensors.insert(name, t);
        }
        let scalars = JointScalars {
            prior_logit_scale: scalars[0],
            joint_logit_scale: scalars[1],
            residual_gate: scalars[2],
        };
        Self::from_tensors(client, config, tensors, scalars, dtype)
            .map_err(|e| file_error(e.to_string()))
    }
}

/// One little-endian float of `dtype` from `bytes`.
fn decode_scalar(bytes: &[u8], dtype: DType) -> Option<f64> {
    match dtype {
        DType::BF16 => Some(half::bf16::from_le_bytes(bytes.try_into().ok()?).to_f64()),
        DType::F16 => Some(half::f16::from_le_bytes(bytes.try_into().ok()?).to_f64()),
        DType::F32 => Some(f64::from(f32::from_le_bytes(bytes.try_into().ok()?))),
        DType::F64 => Some(f64::from_le_bytes(bytes.try_into().ok()?)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::weights::tests::{SCALARS, tiny_config, tiny_head, tiny_tensors};
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn scalars_round_as_the_dtype_holds_them() {
        assert_eq!(round_to(DType::BF16, 9.942_499_8), 9.9375);
        assert_eq!(round_to(DType::BF16, 0.119_202_92), 0.119_140_625);
        assert_eq!(round_to(DType::F32, 0.1), f64::from(0.1f32));
        assert_eq!(decode_scalar(&[0x13, 0x40], DType::BF16), Some(2.296875));
        assert_eq!(decode_scalar(&[0, 0, 0, 0xc0], DType::F32), Some(-2.0));
        assert_eq!(decode_scalar(&[0, 0], DType::I32), None);
        assert_eq!(decode_scalar(&[0], DType::BF16), None);
    }

    #[test]
    fn scales_are_capped_and_gated() {
        let (client, device) = cpu_setup();
        let head = tiny_head(&client, &device, 16);
        assert!((head.prior_scale() - f64::from(0.75f64.exp() as f32)).abs() < 1e-12);
        assert!((head.joint_scale() - f64::from(100f32)).abs() < 1e-4);
        let gate = 1.0 / (1.0 + 0.5f64.exp());
        assert!((head.gate() - gate).abs() < 1e-7);
        assert_eq!(head.dtype(), DType::F32);
    }

    fn from_tensors_err(edit: impl FnOnce(&mut HashMap<String, Tensor<CpuRuntime>>)) -> String {
        let (client, device) = cpu_setup();
        let c = tiny_config(16);
        let mut tensors = tiny_tensors(&device, &c);
        edit(&mut tensors);
        match JointSchemaHead::from_tensors(&client, c, tensors, SCALARS, DType::F32) {
            Ok(_) => panic!("the edited tensors must be refused"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn missing_extra_and_misshapen_tensors_are_refused() {
        let e = from_tensors_err(|t| {
            t.remove("layers.1.norm2.bias");
        });
        assert!(e.contains("no tensor layers.1.norm2.bias"), "{e}");
        let e = from_tensors_err(|t| {
            let (_, device) = cpu_setup();
            let extra = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();
            t.insert("layers.4.norm1.bias".into(), extra);
        });
        assert!(
            e.contains("unexpected tensors [\"layers.4.norm1.bias\"]"),
            "{e}"
        );
        let e = from_tensors_err(|t| {
            let (_, device) = cpu_setup();
            let wrong = Tensor::<CpuRuntime>::zeros(&[8, 15], DType::F32, &device).unwrap();
            t.insert("memory_projection.weight".into(), wrong);
        });
        assert!(
            e.contains("memory_projection.weight is [8, 15], expected [8, 16]"),
            "{e}"
        );
        let e = from_tensors_err(|t| {
            let (_, device) = cpu_setup();
            let ints = Tensor::<CpuRuntime>::zeros(&[3, 8], DType::I64, &device).unwrap();
            t.insert("type_embedding.weight".into(), ints);
        });
        assert!(e.contains("type_embedding.weight is I64"), "{e}");
    }

    #[test]
    fn bad_dtype_and_scalars_are_refused() {
        let (client, device) = cpu_setup();
        let c = tiny_config(16);
        let Err(e) = JointSchemaHead::from_tensors(
            &client,
            c,
            tiny_tensors(&device, &c),
            SCALARS,
            DType::F64,
        ) else {
            panic!("an F64 head must be refused");
        };
        assert!(e.to_string().contains("F32, F16 or BF16, got F64"), "{e}");
        let nan = JointScalars {
            residual_gate: f64::NAN,
            ..SCALARS
        };
        let Err(e) =
            JointSchemaHead::from_tensors(&client, c, tiny_tensors(&device, &c), nan, DType::F32)
        else {
            panic!("a NaN gate must be refused");
        };
        assert!(e.to_string().contains("residual_gate is NaN"), "{e}");
    }
}
