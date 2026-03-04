/// Module names for quantization CUDA kernel PTX files.

pub const DEQUANT_MODULE: &str = "dequant";
pub const DEQUANT_GENERIC_MODULE: &str = "dequant_generic";
pub const QUANT_MATMUL_GENERIC_MODULE: &str = "quant_matmul_generic";
pub const QUANT_MATMUL_MODULE: &str = "quant_matmul";
pub const QUANT_GEMV_MODULE: &str = "quant_gemv";
pub const INT4_GEMM_MODULE: &str = "int4_gemm";
pub const INT4_GEMM_GPTQ_MODULE: &str = "int4_gemm_gptq";
pub const NF4_QUANT_MODULE: &str = "nf4_quant";
pub const MARLIN_GEMM_MODULE: &str = "marlin_gemm";
pub const FUSED_INT4_SWIGLU_MODULE: &str = "fused_int4_swiglu";
pub const FUSED_INT4_QKV_MODULE: &str = "fused_int4_qkv";
pub const QUANT_ACT_MODULE: &str = "quant_act";
