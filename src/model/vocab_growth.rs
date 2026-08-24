//! Policy for reconciling a checkpoint's weight row count with `ModelConfig`.
//!
//! The math (mean-init row growth) lives in `crate::nn::vocab_resize`, which
//! must stay architecture-agnostic. This module owns the decision of when
//! growth is allowed, keyed off `ModelConfig::grow_vocab`.

use crate::error::{Error, Result};
use crate::model::config::ModelConfig;
use crate::nn::vocab_resize::resize_rows_mean_init;
use numr::dtype::DType;
use numr::ops::{ReduceOps, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Reconcile `weight`'s row count with `config.vocab_size`.
///
/// - Row count already matches: returned unchanged.
/// - Row count is smaller and `config.grow_vocab` is set: grown via
///   mean-init (see `resize_rows_mean_init`).
/// - Row count is smaller and `grow_vocab` is not set: hard error naming
///   `what`, both row counts, and the flag.
/// - Row count is *larger*: returned unchanged. Checkpoints routinely pad the
///   embedding table above the declared `vocab_size` (a multiple of 64 or 128
///   for kernel alignment), so this is a normal load, not a mismatch. The
///   surplus rows are unreachable token ids; keep them masked at sampling time
///   via `SpeechVocab::sampling_forbidden_ids`.
///
/// The hard error on the *smaller* case is the point of this function. Growing
/// without being asked would hide a wrong-config/wrong-checkpoint pairing: the
/// worst bug class here, since the model would train and look fine while the
/// rows it was told to use do not exist in the checkpoint.
pub fn fit_vocab_rows<R: Runtime<DType = DType>>(
    weight: Tensor<R>,
    config: &ModelConfig,
    device: &R::Device,
    what: &str,
) -> Result<Tensor<R>>
where
    R::Client: ReduceOps<R> + ShapeOps<R>,
{
    let shape = weight.shape();
    let rows = shape.first().copied().unwrap_or(0);
    let target = config.vocab_size;

    // `>=` and not `==`: a checkpoint whose table is padded above the declared
    // vocab_size is a normal, widely used layout, and rejecting it would break
    // loads that worked before this check existed.
    if rows >= target {
        return Ok(weight);
    }

    if config.grow_vocab {
        let client = R::default_client(device);
        return resize_rows_mean_init(&client, &weight, target);
    }

    Err(Error::ModelError {
        reason: format!(
            "{what}: checkpoint has {rows} rows but config.vocab_size is {target}; \
             the checkpoint cannot supply the missing {} rows. Set grow_vocab: true \
             to mean-init them, or correct the config to match the checkpoint",
            target - rows
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn base_config(vocab_size: usize, grow_vocab: bool) -> ModelConfig {
        ModelConfig {
            model_type: "llama".into(),
            vocab_size,
            hidden_size: 3,
            num_layers: 1,
            max_seq_len: 8,
            intermediate_size: None,
            rms_norm_eps: 1e-5,
            attention: None,
            ssm: None,
            moe: None,
            hybrid_layers: None,
            tie_word_embeddings: false,
            grow_vocab,
            vision: None,
            audio: None,
        }
    }

    fn weight(device: &numr::runtime::cpu::CpuDevice) -> Tensor<CpuRuntime> {
        #[rustfmt::skip]
        let w = Tensor::<CpuRuntime>::try_from_slice(
            &[1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0],
            &[2, 3],
            device,
        ).unwrap();
        w
    }

    #[test]
    fn passes_through_when_rows_already_match() {
        let (_client, device) = cpu_setup();
        let config = base_config(2, false);
        let out = fit_vocab_rows(weight(&device), &config, &device, "w").unwrap();
        assert_eq!(out.shape(), &[2, 3]);
    }

    #[test]
    fn grows_when_flag_is_set() {
        let (_client, device) = cpu_setup();
        let config = base_config(4, true);
        let out = fit_vocab_rows(weight(&device), &config, &device, "w").unwrap();
        assert_eq!(out.shape(), &[4, 3]);
    }

    #[test]
    fn errors_when_flag_is_unset() {
        let (_client, device) = cpu_setup();
        let config = base_config(4, false);
        assert!(fit_vocab_rows(weight(&device), &config, &device, "w").is_err());
    }

    /// A table padded above `vocab_size` is a normal checkpoint layout, not a
    /// mismatch — it must load untouched whether or not growth is enabled.
    #[test]
    fn padded_checkpoint_passes_through_unchanged() {
        let (_client, device) = cpu_setup();
        for grow in [false, true] {
            let config = base_config(1, grow);
            let out = fit_vocab_rows(weight(&device), &config, &device, "w").unwrap();
            assert_eq!(out.shape(), &[2, 3], "grow_vocab={grow}");
        }
    }
}
