//! CUDA kernel loading for boostr attention operations

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::{Error, Result};

/// Directory containing compiled PTX files (set by build.rs)
const KERNEL_DIR: &str = env!("CUDA_KERNEL_DIR");

/// Load PTX from compiled file.
fn load_ptx(name: &str) -> Ptx {
    let path = format!("{}/{}.ptx", KERNEL_DIR, name);
    Ptx::from_file(path)
}

/// Module names
pub const DECODE_ATTENTION_MODULE: &str = "decode_attention";
pub const KV_INSERT_MODULE: &str = "kv_insert";
pub const PAGED_DECODE_ATTENTION_MODULE: &str = "paged_decode_attention";
pub const FLASH_V2_MODULE: &str = "flash_v2";
pub const FLASH_V2_BWD_MODULE: &str = "flash_v2_bwd";
pub const PAGED_ATTENTION_MODULE: &str = "paged_attention";
pub const PAGED_ATTENTION_BWD_MODULE: &str = "paged_attention_bwd";
pub const FLASH_V3_MODULE: &str = "flash_v3";
pub const FLASH_V3_BWD_MODULE: &str = "flash_v3_bwd";
pub const KV_CACHE_UPDATE_MODULE: &str = "kv_cache_update";
pub const VARLEN_ATTENTION_MODULE: &str = "varlen_attention";
pub const VARLEN_ATTENTION_BWD_MODULE: &str = "varlen_attention_bwd";
pub const MQA_GQA_MODULE: &str = "mqa_gqa";
pub const MQA_GQA_BWD_MODULE: &str = "mqa_gqa_bwd";
pub const ALIBI_MODULE: &str = "alibi";
pub const ALIBI_BWD_MODULE: &str = "alibi_bwd";
pub const KV_CACHE_INT4_MODULE: &str = "kv_cache_int4";
pub const KV_CACHE_FP8_MODULE: &str = "kv_cache_fp8";
pub const KV_CACHE_FP8_BWD_MODULE: &str = "kv_cache_fp8_bwd";
pub const KV_CACHE_QUANT_MODULE: &str = "kv_cache_quant";
pub const RESHAPE_AND_CACHE_MODULE: &str = "reshape_and_cache";
pub const FUSED_ADAMW_MODULE: &str = "fused_adamw";
pub const FUSED_SGD_MODULE: &str = "fused_sgd";
pub const FUSED_ADAGRAD_MODULE: &str = "fused_adagrad";
pub const FUSED_LAMB_MODULE: &str = "fused_lamb";
pub const FUSED_MULTI_TENSOR_MODULE: &str = "fused_multi_tensor";
pub const ROPE_MODULE: &str = "rope";
pub const ROPE_INTERLEAVED_MODULE: &str = "rope_interleaved";
pub const ROPE_YARN_MODULE: &str = "rope_yarn";
pub const SDPA_MODULE: &str = "sdpa";
pub const FUSED_QKV_MODULE: &str = "fused_qkv";
pub const MOE_ROUTING_MODULE: &str = "moe_routing";
pub const MOE_PERMUTE_MODULE: &str = "moe_permute";
pub const MOE_GROUPED_GEMM_MODULE: &str = "moe_grouped_gemm";
pub const SSD_STATE_PASSING_MODULE: &str = "ssd_state_passing";
pub const FUSED_GRAD_UNSCALE_CLIP_MODULE: &str = "fused_grad_unscale_clip";
pub const SPECULATIVE_VERIFY_MODULE: &str = "speculative_verify";
pub const SAMPLING_PENALTIES_MODULE: &str = "sampling_penalties";
pub const SAMPLING_MODULE: &str = "sampling";
pub const LOGITS_TO_TOKEN_MODULE: &str = "logits_to_token";
pub const CALIBRATION_MODULE: &str = "calibration";

/// Cache for loaded CUDA modules, keyed by (device_index, module_name)
#[allow(clippy::type_complexity)]
static MODULE_CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), Arc<CudaModule>>>> =
    OnceLock::new();

/// Get or load a CUDA module from PTX.
pub fn get_or_load_module(
    context: &Arc<CudaContext>,
    device_index: usize,
    module_name: &'static str,
) -> Result<Arc<CudaModule>> {
    let cache = MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().map_err(|e| Error::KernelError {
        reason: format!("kernel cache mutex poisoned: {e}"),
    })?;

    let key = (device_index, module_name);
    if let Some(module) = guard.get(&key) {
        return Ok(module.clone());
    }

    let ptx = load_ptx(module_name);
    let module = context.load_module(ptx).map_err(|e| Error::KernelError {
        reason: format!(
            "Failed to load CUDA module '{}': {:?}. \
             Ensure CUDA kernels were compiled correctly by build.rs.",
            module_name, e
        ),
    })?;

    guard.insert(key, module.clone());
    Ok(module)
}

/// Pre-load a list of CUDA modules to avoid JIT compilation latency on first use.
pub fn preload_modules(
    context: &Arc<CudaContext>,
    device_index: usize,
    module_names: &[&'static str],
) -> Result<()> {
    for name in module_names {
        get_or_load_module(context, device_index, name)?;
    }
    Ok(())
}

/// Get a kernel function from a loaded module.
pub fn get_kernel_function(module: &Arc<CudaModule>, kernel_name: &str) -> Result<CudaFunction> {
    module
        .load_function(kernel_name)
        .map_err(|e| Error::KernelError {
            reason: format!(
                "Failed to get kernel '{}': {:?}. \
                 Check that the kernel name matches the CUDA source.",
                kernel_name, e
            ),
        })
}
