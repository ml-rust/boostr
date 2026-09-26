//! Activation function enum for configurable model architectures

use crate::error::{Error, Result as BoostrResult};
use numr::ops::{ActivationOps, BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Activation function selector for model configuration.
///
/// Used in model configs (YAML, etc.) to pick an activation at runtime
/// without hard-coding a specific function in the architecture.
///
/// `Gelu` is the exact/erf form (HF `"gelu"`, `"gelu_python"`, PyTorch
/// `F.gelu(x, approximate="none")`). `GeluTanh` is the tanh approximation
/// (HF `"gelu_new"`, `"gelu_pytorch_tanh"`, `"gelu_fast"`). These are NOT
/// interchangeable: loading a checkpoint trained with one and running the
/// other shifts every activated value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Relu,
    /// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    Gelu,
    /// Tanh-approximate GELU.
    GeluTanh,
    /// QuickGELU: `x * sigmoid(1.702 * x)`.
    QuickGelu,
    Silu,
    Sigmoid,
    Tanh,
}

impl Activation {
    /// Parse a HuggingFace `config.json` / GGUF `hidden_act` string.
    ///
    /// The ONE place HF activation names are parsed. Returns an error naming
    /// the unrecognized string and the accepted names.
    pub fn from_hf_name(name: &str) -> BoostrResult<Self> {
        match name {
            "relu" => Ok(Activation::Relu),
            "gelu" | "gelu_python" => Ok(Activation::Gelu),
            "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" => Ok(Activation::GeluTanh),
            "quick_gelu" => Ok(Activation::QuickGelu),
            "silu" | "swish" => Ok(Activation::Silu),
            "sigmoid" => Ok(Activation::Sigmoid),
            "tanh" => Ok(Activation::Tanh),
            other => Err(Error::InvalidArgument {
                arg: "hidden_act",
                reason: format!(
                    "unknown activation name '{other}', expected one of: \
                     relu, gelu, gelu_python, gelu_new, gelu_pytorch_tanh, \
                     gelu_fast, quick_gelu, silu, swish, sigmoid, tanh"
                ),
            }),
        }
    }

    /// Apply this activation to a tensor.
    pub fn forward<R, C>(&self, client: &C, x: &Tensor<R>) -> numr::error::Result<Tensor<R>>
    where
        R: Runtime,
        C: ActivationOps<R> + UnaryOps<R> + BinaryOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        match self {
            Activation::Relu => client.relu(x),
            Activation::Gelu => client.gelu_erf(x),
            Activation::GeluTanh => client.gelu(x),
            Activation::QuickGelu => {
                // x * sigmoid(1.702 * x). numr has no dedicated op, so this is
                // composed from primitives already required by every backend.
                let scaled = client.mul_scalar(x, 1.702)?;
                let gate = client.sigmoid(&scaled)?;
                client.mul(x, &gate)
            }
            Activation::Silu => client.silu(x),
            Activation::Sigmoid => client.sigmoid(x),
            Activation::Tanh => client.tanh(x),
        }
    }
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::Relu => write!(f, "relu"),
            Activation::Gelu => write!(f, "gelu"),
            Activation::GeluTanh => write!(f, "gelu_tanh"),
            Activation::QuickGelu => write!(f, "quick_gelu"),
            Activation::Silu => write!(f, "silu"),
            Activation::Sigmoid => write!(f, "sigmoid"),
            Activation::Tanh => write!(f, "tanh"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_activation_forward() {
        let (client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 1.0], &[3], &device).unwrap();

        for act in [
            Activation::Relu,
            Activation::Gelu,
            Activation::GeluTanh,
            Activation::QuickGelu,
            Activation::Silu,
            Activation::Sigmoid,
            Activation::Tanh,
        ] {
            let out = act.forward(&client, &x).unwrap();
            assert_eq!(out.shape(), &[3]);
        }
    }

    #[test]
    fn test_relu_values() {
        let (client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[-2.0f32, 0.0, 3.0], &[3], &device).unwrap();
        let out = Activation::Relu.forward(&client, &x).unwrap();
        let data: Vec<f32> = out.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_quick_gelu_values() {
        let (client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[-2.0f32, 0.0, 1.0, 2.0], &[4], &device).unwrap();
        let out = Activation::QuickGelu.forward(&client, &x).unwrap();
        let data: Vec<f32> = out.to_vec();
        for (xi, yi) in [-2.0f32, 0.0, 1.0, 2.0].iter().zip(data.iter()) {
            let expected = xi / (1.0 + (-1.702 * xi).exp());
            assert!(
                (yi - expected).abs() < 1e-5,
                "quick_gelu({xi}) = {yi}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gelu_erf_and_tanh_differ() {
        let (client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.5f32], &[1], &device).unwrap();
        let erf_out: Vec<f32> = Activation::Gelu.forward(&client, &x).unwrap().to_vec();
        let tanh_out: Vec<f32> = Activation::GeluTanh.forward(&client, &x).unwrap().to_vec();
        assert!(
            (erf_out[0] - tanh_out[0]).abs() > 1e-4,
            "exact and tanh GELU must diverge at x=1.5: erf={}, tanh={}",
            erf_out[0],
            tanh_out[0]
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let act = Activation::Gelu;
        let json = serde_json::to_string(&act).unwrap();
        assert_eq!(json, "\"gelu\"");
        let back: Activation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Activation::Gelu);
    }

    #[test]
    fn test_serde_roundtrip_gelu_tanh() {
        let act = Activation::GeluTanh;
        let json = serde_json::to_string(&act).unwrap();
        assert_eq!(json, "\"gelu_tanh\"");
        let back: Activation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Activation::GeluTanh);
    }

    #[test]
    fn test_from_hf_name_mapping() {
        assert_eq!(Activation::from_hf_name("gelu").unwrap(), Activation::Gelu);
        assert_eq!(
            Activation::from_hf_name("gelu_python").unwrap(),
            Activation::Gelu
        );
        assert_eq!(
            Activation::from_hf_name("gelu_new").unwrap(),
            Activation::GeluTanh
        );
        assert_eq!(
            Activation::from_hf_name("gelu_pytorch_tanh").unwrap(),
            Activation::GeluTanh
        );
        assert_eq!(
            Activation::from_hf_name("gelu_fast").unwrap(),
            Activation::GeluTanh
        );
        assert_eq!(
            Activation::from_hf_name("quick_gelu").unwrap(),
            Activation::QuickGelu
        );
        assert_eq!(Activation::from_hf_name("silu").unwrap(), Activation::Silu);
        assert_eq!(Activation::from_hf_name("swish").unwrap(), Activation::Silu);
        assert_eq!(Activation::from_hf_name("relu").unwrap(), Activation::Relu);
        assert_eq!(Activation::from_hf_name("tanh").unwrap(), Activation::Tanh);
        assert_eq!(
            Activation::from_hf_name("sigmoid").unwrap(),
            Activation::Sigmoid
        );
    }

    #[test]
    fn test_from_hf_name_rejects_unknown() {
        let err = Activation::from_hf_name("mystery_act").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mystery_act"));
    }
}
