//! Activation function enum for configurable model architectures

use numr::ops::{ActivationOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Activation function selector for model configuration.
///
/// Used in model configs (YAML, etc.) to pick an activation at runtime
/// without hard-coding a specific function in the architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Relu,
    Gelu,
    Silu,
    Sigmoid,
    Tanh,
}

impl Activation {
    /// Apply this activation to a tensor.
    pub fn forward<R, C>(&self, client: &C, x: &Tensor<R>) -> numr::error::Result<Tensor<R>>
    where
        R: Runtime,
        C: ActivationOps<R> + UnaryOps<R> + RuntimeClient<R>,
    {
        match self {
            Activation::Relu => client.relu(x),
            Activation::Gelu => client.gelu(x),
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
            Activation::Silu => write!(f, "silu"),
            Activation::Sigmoid => write!(f, "sigmoid"),
            Activation::Tanh => write!(f, "tanh"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn cpu_setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_activation_forward() {
        let (client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 1.0], &[3], &device);

        for act in [
            Activation::Relu,
            Activation::Gelu,
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
        let x = Tensor::<CpuRuntime>::from_slice(&[-2.0f32, 0.0, 3.0], &[3], &device);
        let out = Activation::Relu.forward(&client, &x).unwrap();
        let data: Vec<f32> = out.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_serde_roundtrip() {
        let act = Activation::Gelu;
        let json = serde_json::to_string(&act).unwrap();
        assert_eq!(json, "\"gelu\"");
        let back: Activation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Activation::Gelu);
    }
}
