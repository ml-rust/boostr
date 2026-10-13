//! The kernel family is parameterized over the weight format; everything that
//! differs per format is a field of `FeatMajorFormat`. Q8_0, Q4_0, Q4_1,
//! Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS,
//! IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, PQ2_0, Q2_0, Q1_0, PTQ1_0 and the TCF native
//! encodings `Q8S32T64` and `Q4AS32DT64` are the formats compiled today.

mod descriptor;
mod iquant;
mod kquant;
mod legacy;
mod lowbit;

pub(in crate::quant::cuda::quant_matmul) use descriptor::FeatMajorFormat;
pub(in crate::quant::cuda::quant_matmul) use iquant::{
    IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_XS,
};
pub(in crate::quant::cuda::quant_matmul) use kquant::{Q2_K, Q3_K, Q4_K, Q5_K, Q6_K};
pub(in crate::quant::cuda::quant_matmul) use legacy::{IQ4_NL, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};
pub(in crate::quant::cuda::quant_matmul) use lowbit::{PQ2_0, PTQ1_0, Q1_0, Q2_0};
