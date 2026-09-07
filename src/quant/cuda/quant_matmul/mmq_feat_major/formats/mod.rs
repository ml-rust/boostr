mod descriptor;
mod iquant;
mod kquant;
mod legacy;
mod tcf;

pub(in crate::quant::cuda::quant_matmul) use iquant::{
    IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_XS,
};
pub(in crate::quant::cuda::quant_matmul) use kquant::{Q2_K, Q3_K, Q4_K, Q5_K, Q6_K};
pub(in crate::quant::cuda::quant_matmul) use legacy::{IQ4_NL, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};
// The TCF descriptors are reached through `feat_major_format`, which is the
// one place that maps a native encoding to its kernel.
pub(in crate::quant::cuda::quant_matmul) use tcf::feat_major_format;

pub(in crate::quant::cuda::quant_matmul) use descriptor::FeatMajorFormat;
