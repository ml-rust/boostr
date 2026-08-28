use super::tensor_loader::checked_tensor;
use super::*;
use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::nn::{MaybeLoraLinear, MaybeQuantLinear, Weight};
use crate::quant::{QuantFormat, QuantTensor};
use crate::test_utils::cpu_setup;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::io::Write;
use tempfile::NamedTempFile;

/// One `weight` tensor of shape `[2, 3]`, values `1.0 ..= 6.0`.
fn one_tensor_file() -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("temp file");
    let header = serde_json::json!({
        "weight": { "dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24] }
    })
    .to_string();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("header len");
    file.write_all(header.as_bytes()).expect("header");
    for f in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        file.write_all(&f.to_le_bytes()).expect("data");
    }
    file.flush().expect("flush");
    file
}

/// `SafeTensorsLoader` reaches the same bytes through the trait as it
/// does through its own `load_tensor`, and the shape gate still fires.
#[test]
fn safetensors_source_loads_named_tensor() {
    let (_, device) = cpu_setup();
    let file = one_tensor_file();
    let mut loader = SafeTensorsLoader::open(file.path()).expect("open");

    let t: Tensor<CpuRuntime> = loader.load_named("weight", &device).expect("load_named");
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let ok = checked_tensor::<CpuRuntime, _>(&mut loader, &device, "", "weight", &[2, 3]);
    assert!(ok.is_ok());
    let wrong_shape = checked_tensor::<CpuRuntime, _>(&mut loader, &device, "", "weight", &[6]);
    assert!(wrong_shape.is_err());
}

/// A source whose every weight is block-quantized, standing in for a
/// GGUF. Written by hand rather than through a real GGUF file because
/// the gates under test are `TensorLoader::linear`'s, not the reader's.
struct QuantSource {
    shape: Vec<usize>,
}

impl WeightSource<CpuRuntime> for QuantSource {
    fn load_named(&mut self, name: &str, _device: &CpuDevice) -> Result<Tensor<CpuRuntime>> {
        Err(Error::ModelError {
            reason: format!("{name}: dense read not expected in this test"),
        })
    }

    fn load_named_weight(&mut self, _name: &str, device: &CpuDevice) -> Result<Weight<CpuRuntime>> {
        let numel: usize = self.shape.iter().product();
        // Q4_0: 32 elements per 18-byte block. Values are irrelevant —
        // nothing here multiplies.
        let bytes = vec![0u8; numel / 32 * 18];
        Ok(Weight::Quantized(QuantTensor::<CpuRuntime>::from_bytes(
            &bytes,
            QuantFormat::Q4_0,
            &self.shape,
            device,
        )?))
    }
}

/// The shape gate is not something a packed weight gets to skip:
/// `Weight::shape` is the logical element shape for both variants.
#[test]
fn quantized_weight_is_shape_checked() {
    let (_, device) = cpu_setup();
    let mut source = QuantSource { shape: vec![2, 32] };
    let mut tl = TensorLoader::<CpuRuntime, _> {
        loader: &mut source,
        device: &device,
        prefix: "base_lm.layers.0.self_attn".to_string(),
        dtype: None,
    };

    let ok = tl.linear("q_proj", 2, 32, false);
    assert!(ok.is_ok(), "matching shape rejected: {:?}", ok.err());

    // `MaybeLoraLinear` is not `Debug`, so `expect_err` cannot be used.
    let Err(err) = tl.linear("q_proj", 4, 32, false) else {
        panic!("wrong out_features accepted");
    };
    let msg = err.to_string();
    assert!(msg.contains("expected shape [4, 32]"), "got {msg}");
    assert!(msg.contains("[2, 32]"), "got {msg}");
}

/// BF16/F16 plus a quantized weight cannot run: `quant_matmul` requires
/// F32 activations. The load errors and names the dtype rather than
/// dequantizing behind the caller's back or dropping the request.
#[test]
fn quantized_weight_rejects_a_narrow_dtype() {
    let (_, device) = cpu_setup();
    for want in [DType::BF16, DType::F16] {
        let mut source = QuantSource { shape: vec![2, 32] };
        let mut tl = TensorLoader::<CpuRuntime, _> {
            loader: &mut source,
            device: &device,
            prefix: "base_lm.layers.0.self_attn".to_string(),
            dtype: Some(want),
        };

        let Err(err) = tl.linear("q_proj", 2, 32, false) else {
            panic!("a narrow dtype was accepted alongside a quantized weight");
        };
        let msg = err.to_string();
        assert!(msg.contains(&format!("{want:?}")), "got {msg}");
        assert!(msg.contains("F32"), "got {msg}");
        assert!(msg.contains("q_proj.weight"), "got {msg}");
    }
}

/// F32 (and `None`) keep the weight packed — the whole point of the path.
#[test]
fn quantized_weight_survives_an_f32_request() {
    let (_, device) = cpu_setup();
    for dtype in [None, Some(DType::F32)] {
        let mut source = QuantSource { shape: vec![2, 32] };
        let mut tl = TensorLoader::<CpuRuntime, _> {
            loader: &mut source,
            device: &device,
            prefix: "base_lm".to_string(),
            dtype,
        };

        let layer = tl.linear("q_proj", 2, 32, false).expect("linear");
        assert!(
            matches!(
                layer,
                MaybeLoraLinear::Plain(MaybeQuantLinear::Quantized(_))
            ),
            "the weight was dequantized instead of staying packed"
        );
    }
}
