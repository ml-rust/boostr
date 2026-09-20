//! Load the tower from an mmproj GGUF.

use super::config::Qwen3VlVisionConfig;
use super::encoder::Qwen3VlVision;
use crate::error::Result;
use crate::format::gguf::Gguf;
use crate::nn::{VarBuilder, VarMap};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::path::Path;

/// Read `clip.*` metadata and every tensor of an mmproj GGUF, then build
/// the tower on `device`. Quantized matrices stay quantized; every other
/// tensor loads as F32.
pub fn load_qwen3vl_vision_gguf<R, P>(path: P, device: &R::Device) -> Result<Qwen3VlVision<R>>
where
    R: Runtime<DType = DType>,
    R::Client: DequantOps<R>,
    P: AsRef<Path>,
{
    let mut gguf = Gguf::open(path)?;
    load_qwen3vl_vision_from_reader(&mut gguf, device)
}

/// Same as [`load_qwen3vl_vision_gguf`] on an open reader.
pub fn load_qwen3vl_vision_from_reader<R>(
    gguf: &mut Gguf,
    device: &R::Device,
) -> Result<Qwen3VlVision<R>>
where
    R: Runtime<DType = DType>,
    R::Client: DequantOps<R>,
{
    let cfg = Qwen3VlVisionConfig::from_gguf(gguf.metadata())?;
    let names: Vec<String> = gguf.tensor_names().map(str::to_string).collect();
    let mut map = VarMap::<R>::new();
    for name in &names {
        let quantized = gguf.tensor_info(name)?.ggml_type.is_quantized();
        if quantized {
            let qt = gguf.load_tensor_quantized::<R>(name, device)?;
            map.insert_quant(name.clone(), qt);
        } else {
            let t = gguf.load_tensor_f32::<R>(name, device)?;
            map.insert(name.clone(), t);
        }
    }
    let mut vb = VarBuilder::new(&mut map, device);
    Qwen3VlVision::from_varbuilder(&mut vb, &cfg)
}
