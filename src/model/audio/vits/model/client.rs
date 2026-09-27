//! [`VitsModelClient`]: the client bound [`super::VitsModel`]'s methods
//! compose across every submodule they call (text encoder, both
//! duration-predictor variants, length regulation, prior sampling, flow,
//! decoder, `embed_speaker`).

use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, CumulativeOps, IndexingOps,
    MatmulOps, NormalizationOps, RandomOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};

/// A concrete backend client (e.g. `CpuClient`) implements every one of
/// these; this trait exists only to keep [`super::VitsModel`]'s method
/// signatures short.
pub trait VitsModelClient<R: Runtime>:
    RuntimeClient<R>
    + ActivationOps<R>
    + BinaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + ConvOps<R>
    + CumulativeOps<R>
    + IndexingOps<R>
    + MatmulOps<R>
    + NormalizationOps<R>
    + RandomOps<R>
    + ReduceOps<R>
    + ScalarOps<R>
    + ShapeOps<R>
    + TensorOps<R>
    + TypeConversionOps<R>
    + UnaryOps<R>
    + UtilityOps<R>
{
}

impl<R, C> VitsModelClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ConvOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MatmulOps<R>
        + NormalizationOps<R>
        + RandomOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + UtilityOps<R>,
{
}
