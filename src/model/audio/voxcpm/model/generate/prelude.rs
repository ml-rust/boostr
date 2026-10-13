//! The imports every file of the per-patch loop shares, plus the shape checks
//! and stop decision from `validate`. Each file pulls them in with
//! `use super::prelude::*`.

pub(super) use crate::error::{Error, Result};
pub(super) use crate::model::audio::voxcpm::fsq::{AuxProjections, ScalarQuantization};
pub(super) use crate::model::audio::voxcpm::local_dit::{CfmOptions, LocalDit};
pub(super) use crate::model::audio::voxcpm::local_encoder::LocalEncoder;
pub(super) use crate::model::audio::voxcpm::minicpm4::MiniCpm4Model;
pub(super) use crate::model::audio::voxcpm::model::config::VoxCpm2Config;
pub(super) use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
pub(super) use crate::model::audio::voxcpm::model::prefill::PrefillState;
pub(super) use crate::model::traits::ModelClient;
pub(super) use crate::quant::traits::DequantOps;
pub(super) use numr::autograd::{Var, var_mul_scalar};
pub(super) use numr::dtype::DType;
pub(super) use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, HostCopyOps, IndexingOps, RandomOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
pub(super) use numr::runtime::Runtime;
pub(super) use numr::tensor::Tensor;

pub(super) use super::validate::{check_patch, check_row, stop_predicted};
