//! The [`WeightSource`] trait: reads a named tensor out of a checkpoint,
//! either dense or in its most compact stored form.

use crate::error::Result;
use crate::format::gguf::Gguf;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::nn::Weight;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A checkpoint a VoxCPM2 sub-loader can read a named tensor out of.
///
/// The file format is an I/O detail: every sub-loader below knows the
/// *layout* (which key holds which weight, and what shape it must have),
/// which is identical whether the bytes arrive from safetensors or from a
/// GGUF file. Keeping the read behind this trait is what stops that layout
/// knowledge from being duplicated once per format.
///
/// Public because the sub-model `from_source` constructors take it as a
/// bound, and those are the API a caller uses to load several sub-models
/// out of ONE open checkpoint instead of reopening it per sub-model.
pub trait WeightSource<R: Runtime<DType = DType>> {
    /// Load the tensor stored under exactly `name` as a DENSE tensor.
    ///
    /// "Dense" is the contract, not an implementation detail: a GGUF source
    /// DEQUANTIZES here. Every caller that needs real elements — norms,
    /// biases, conv kernels, the embedding table, anything that is not a
    /// matmul weight — goes through this and keeps working unchanged
    /// whatever the file format stores.
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>>;

    /// Load the tensor stored under exactly `name` in the MOST COMPACT form
    /// the source holds it in.
    ///
    /// This is the other half of the split: `load_named` means "give me a
    /// dense tensor", this means "give me whatever you have, quantized if
    /// that is how it is stored". A GGUF source returns
    /// [`Weight::Quantized`] for a block-quantized tensor, which is what
    /// keeps a 1.2 GB Q4_K file from expanding to a full-size F32 weight set
    /// in memory; the caller then multiplies it with `quant_matmul` instead
    /// of `matmul`.
    ///
    /// The default is correct for any source that only ever stores dense
    /// tensors (safetensors), so only GGUF overrides it.
    fn load_named_weight(&mut self, name: &str, device: &R::Device) -> Result<Weight<R>> {
        Ok(Weight::Standard(self.load_named(name, device)?))
    }
}

impl<R: Runtime<DType = DType>> WeightSource<R> for SafeTensorsLoader {
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        self.load_tensor::<R>(name, device)
    }
}

impl<R: Runtime<DType = DType>> WeightSource<R> for Gguf {
    /// DEQUANTIZES, deliberately. `load_tensor_f32` expands a K-quant block
    /// tensor to a dense F32 tensor, which is exactly what the dense callers
    /// (norms, biases, conv kernels, the embedding table) need and the only
    /// thing they can consume.
    ///
    /// This is therefore NOT the path that shrinks resident memory — a
    /// weight read through here costs the same as a full-precision
    /// checkpoint's would. `load_named_weight` below is the path that keeps
    /// block-quantized weights packed.
    ///
    /// The tensor names are the checkpoint's ORIGINAL HuggingFace names:
    /// `compressr convert --format gguf` writes them verbatim, so
    /// `gguf_to_hf_name` is deliberately not applied.
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        self.load_tensor_f32::<R>(name, device)
    }

    /// Keeps a block-quantized tensor PACKED: `load_tensor_quantized` copies
    /// the GGML blocks through verbatim into a `QuantTensor`, so a Q4_K
    /// weight stays at roughly a quarter of its F32 size and is multiplied
    /// on the fly by `quant_matmul`.
    ///
    /// The GGML type decides, not the tensor name: `to_quant_format` returns
    /// `None` for F32/F16/BF16/integer tensors, which fall back to the dense
    /// read. A GGUF mixes both — llama.cpp writes norms and often the
    /// embedding table unquantized in an otherwise Q4_K file.
    fn load_named_weight(&mut self, name: &str, device: &R::Device) -> Result<Weight<R>> {
        let quantized = self
            .tensor_info(name)?
            .ggml_type
            .to_quant_format()
            .is_some();
        if quantized {
            Ok(Weight::Quantized(
                self.load_tensor_quantized::<R>(name, device)?,
            ))
        } else {
            Ok(Weight::Standard(self.load_tensor_f32::<R>(name, device)?))
        }
    }
}
