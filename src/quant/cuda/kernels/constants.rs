//! Module names for quantization CUDA kernel PTX files.

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

// Per-format GEMV kernels
pub const GEMV_Q5_K_MODULE: &str = "gemv_q5_k";
pub const GEMV_Q3_K_MODULE: &str = "gemv_q3_k";
pub const GEMV_Q2_K_MODULE: &str = "gemv_q2_k";
pub const GEMV_Q5_0_MODULE: &str = "gemv_q5_0";

// Per-format IQ GEMV kernels
pub const GEMV_IQ4_NL_MODULE: &str = "gemv_iq4_nl";
pub const GEMV_IQ4_XS_MODULE: &str = "gemv_iq4_xs";
pub const GEMV_IQ3_S_MODULE: &str = "gemv_iq3_s";
pub const GEMV_IQ2_XS_MODULE: &str = "gemv_iq2_xs";

// Per-format GEMM kernels
pub const GEMM_Q5_K_MODULE: &str = "gemm_q5_k";
pub const GEMM_Q3_K_MODULE: &str = "gemm_q3_k";
pub const GEMM_Q2_K_MODULE: &str = "gemm_q2_k";
pub const GEMM_Q5_0_MODULE: &str = "gemm_q5_0";

// Per-format IQ GEMM kernels
pub const GEMM_IQ4_NL_MODULE: &str = "gemm_iq4_nl";
pub const GEMM_IQ4_XS_MODULE: &str = "gemm_iq4_xs";
pub const GEMM_IQ3_S_MODULE: &str = "gemm_iq3_s";
pub const GEMM_IQ2_XS_MODULE: &str = "gemm_iq2_xs";

// Low-priority GEMV kernels
pub const GEMV_Q4_1_MODULE: &str = "gemv_q4_1";
pub const GEMV_Q5_1_MODULE: &str = "gemv_q5_1";
pub const GEMV_Q8_1_MODULE: &str = "gemv_q8_1";
pub const GEMV_Q8_K_MODULE: &str = "gemv_q8_k";
pub const GEMV_IQ1_S_MODULE: &str = "gemv_iq1_s";
pub const GEMV_IQ1_M_MODULE: &str = "gemv_iq1_m";
pub const GEMV_IQ2_XXS_MODULE: &str = "gemv_iq2_xxs";
pub const GEMV_IQ2_S_MODULE: &str = "gemv_iq2_s";
pub const GEMV_IQ3_XXS_MODULE: &str = "gemv_iq3_xxs";
pub const GEMV_TQ1_0_MODULE: &str = "gemv_tq1_0";
pub const GEMV_TQ2_0_MODULE: &str = "gemv_tq2_0";

// Low-priority GEMM kernels
pub const GEMM_Q4_1_MODULE: &str = "gemm_q4_1";
pub const GEMM_Q5_1_MODULE: &str = "gemm_q5_1";
pub const GEMM_Q8_1_MODULE: &str = "gemm_q8_1";
pub const GEMM_Q8_K_MODULE: &str = "gemm_q8_k";
pub const GEMM_IQ1_S_MODULE: &str = "gemm_iq1_s";
pub const GEMM_IQ1_M_MODULE: &str = "gemm_iq1_m";
pub const GEMM_IQ2_XXS_MODULE: &str = "gemm_iq2_xxs";
pub const GEMM_IQ2_S_MODULE: &str = "gemm_iq2_s";
pub const GEMM_IQ3_XXS_MODULE: &str = "gemm_iq3_xxs";
pub const GEMM_TQ1_0_MODULE: &str = "gemm_tq1_0";
pub const GEMM_TQ2_0_MODULE: &str = "gemm_tq2_0";
