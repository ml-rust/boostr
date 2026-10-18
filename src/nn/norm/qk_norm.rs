//! Query/key normalisation shared by the encoder layer and the decoder
//! attention blocks.
//!
//! Both families normalise Q and K over the LAST axis of the tensor they are
//! handed. The caller decides what that axis is: the whole `hidden` width
//! (jina-bert-v2, before the reshape into heads) or `head_dim` (Gemma, Qwen3,
//! `qwen35`, after it). [`apply_qk_norm`] is the axis-agnostic pair helper;
//! [`QkNorm`] is the per-head RMS variant the decoder blocks own.

use crate::error::{Error, Result};
use crate::nn::{LayerNorm, Module, RmsNorm};
use numr::autograd::Var;
use numr::ops::{NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// A norm over the last axis of its input.
pub trait LastAxisNorm<R: Runtime> {
    /// Normalise `x` over its last axis.
    fn forward_last_axis<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>;
}

impl<R: Runtime> LastAxisNorm<R> for RmsNorm<R> {
    fn forward_last_axis<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        self.forward(client, x)
    }
}

impl<R: Runtime> LastAxisNorm<R> for LayerNorm<R> {
    fn forward_last_axis<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        self.forward(client, x)
    }
}

/// Apply an optional norm to `q` and an optional norm to `k`.
///
/// `None` passes the tensor through unchanged. The norms run over the last
/// axis, so `q` and `k` must already be shaped with the normalised width last.
pub fn apply_qk_norm<R, C, N>(
    client: &C,
    q: Var<R>,
    k: Var<R>,
    q_norm: Option<&N>,
    k_norm: Option<&N>,
) -> Result<(Var<R>, Var<R>)>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    N: LastAxisNorm<R>,
{
    let q = match q_norm {
        Some(n) => n.forward_last_axis(client, &q)?,
        None => q,
    };
    let k = match k_norm {
        Some(n) => n.forward_last_axis(client, &k)?,
        None => k,
    };
    Ok((q, k))
}

/// Per-head RMS normalisation of Q and K: one `[head_dim]` weight each,
/// applied to tensors whose last axis is `head_dim`.
pub struct QkNorm<R: Runtime> {
    q_norm: RmsNorm<R>,
    k_norm: RmsNorm<R>,
    head_dim: usize,
}

impl<R: Runtime> QkNorm<R> {
    /// Build from `q_weight` and `k_weight`, both `[head_dim]`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when either weight is not `[head_dim]`.
    pub fn new(
        q_weight: Tensor<R>,
        k_weight: Tensor<R>,
        head_dim: usize,
        eps: f32,
    ) -> Result<Self> {
        for (name, w) in [("q_norm", &q_weight), ("k_norm", &k_weight)] {
            if w.shape() != [head_dim] {
                return Err(Error::ModelError {
                    reason: format!("{name}: expected weight [{head_dim}], got {:?}", w.shape()),
                });
            }
        }
        Ok(Self {
            q_norm: RmsNorm::new(q_weight, eps, false),
            k_norm: RmsNorm::new(k_weight, eps, false),
            head_dim,
        })
    }

    /// Build from both weights, each with its stable autograd id, and
    /// `requires_grad = trainable` on both.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_ids(
        q_weight: (Tensor<R>, TensorId),
        k_weight: (Tensor<R>, TensorId),
        head_dim: usize,
        eps: f32,
        trainable: bool,
    ) -> Result<Self> {
        let (q, q_id) = q_weight;
        let (k, k_id) = k_weight;
        let checked = Self::new(q, k, head_dim, eps)?;
        let q = checked.q_norm.weight().tensor().clone();
        let k = checked.k_norm.weight().tensor().clone();
        Ok(Self {
            q_norm: RmsNorm::with_id(q, q_id, eps, trainable),
            k_norm: RmsNorm::with_id(k, k_id, eps, trainable),
            head_dim,
        })
    }

    /// The same weights and ids with `requires_grad = trainable` on both.
    pub fn with_trainable(&self, trainable: bool) -> Self {
        let rebuild = |norm: &RmsNorm<R>| {
            let weight = norm.weight();
            RmsNorm::with_id(weight.tensor().clone(), weight.id(), norm.eps(), trainable)
        };
        Self {
            q_norm: rebuild(&self.q_norm),
            k_norm: rebuild(&self.k_norm),
            head_dim: self.head_dim,
        }
    }

    /// Normalise `q` and `k`, each `[.., head_dim]`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when the last axis of `q` or `k` is not `head_dim`.
    pub fn forward<C>(&self, client: &C, q: &Var<R>, k: &Var<R>) -> Result<(Var<R>, Var<R>)>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        for (name, t) in [("q", q), ("k", k)] {
            let shape = t.shape();
            if shape.last() != Some(&self.head_dim) {
                return Err(Error::ModelError {
                    reason: format!(
                        "qk_norm: {name} last axis must be head_dim={}, got {shape:?}",
                        self.head_dim
                    ),
                });
            }
        }
        apply_qk_norm(
            client,
            q.alias(),
            k.alias(),
            Some(&self.q_norm),
            Some(&self.k_norm),
        )
    }

    /// Normalised width.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The Q norm.
    pub fn q_norm(&self) -> &RmsNorm<R> {
        &self.q_norm
    }

    /// The K norm.
    pub fn k_norm(&self) -> &RmsNorm<R> {
        &self.k_norm
    }

    /// Both weights with their stable autograd IDs.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = self.q_norm.parameters();
        params.extend(self.k_norm.parameters());
        params
    }

    /// Cheap duplicate over the same weight storage, keeping both
    /// `TensorId`s.
    pub fn alias(&self) -> Self {
        Self {
            q_norm: self.q_norm.alias(),
            k_norm: self.k_norm.alias(),
            head_dim: self.head_dim,
        }
    }
}

impl<R: Runtime> Module<R> for QkNorm<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        vec![self.q_norm.weight(), self.k_norm.weight()]
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        vec![
            ("q_norm.weight".to_string(), self.q_norm.weight()),
            ("k_norm.weight".to_string(), self.k_norm.weight()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn per_head_norm_matches_flat_rmsnorm() {
        let (client, device) = cpu_setup();
        let qw = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
        let kw = Tensor::<CpuRuntime>::from_slice(&[0.5f32; 4], &[4], &device).unwrap();
        let norm = QkNorm::new(qw.clone(), kw, 4, 1e-6).unwrap();

        let data: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32 * 0.3 - 1.0).collect();
        let q = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &device).unwrap(),
            false,
        );
        let k = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &device).unwrap(),
            false,
        );
        let (qn, kn) = norm.forward(&client, &q, &k).unwrap();
        assert_eq!(qn.shape(), &[2, 3, 4]);

        let flat = RmsNorm::new(qw, 1e-6, false);
        let want: Vec<f32> = flat.forward(&client, &q).unwrap().tensor().to_vec();
        let got: Vec<f32> = qn.tensor().to_vec();
        assert_eq!(got, want);

        // k uses its own weight: every row is scaled by 0.5 instead.
        let kn: Vec<f32> = kn.tensor().to_vec();
        assert!(kn.iter().zip(&got).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn rejects_wrong_widths() {
        let (client, device) = cpu_setup();
        let w4 = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device).unwrap();
        let w3 = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 3], &[3], &device).unwrap();
        assert!(QkNorm::new(w4.clone(), w3, 4, 1e-6).is_err());

        let norm = QkNorm::new(w4.clone(), w4, 4, 1e-6).unwrap();
        let bad = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap(),
            false,
        );
        assert!(norm.forward(&client, &bad, &bad).is_err());
    }

    /// `with_ids` and `with_trainable` keep every id and set the flag.
    #[test]
    fn ids_and_trainable_flag_carry_over() {
        let (_, device) = cpu_setup();
        let w = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device).unwrap();
        let (q_id, k_id) = (TensorId::new(), TensorId::new());
        let norm = QkNorm::with_ids((w.clone(), q_id), (w, k_id), 4, 1e-6, true).unwrap();
        assert_eq!(
            (norm.q_norm().weight().id(), norm.k_norm().weight().id()),
            (q_id, k_id)
        );
        assert!(norm.q_norm().weight().requires_grad());

        let frozen = norm.with_trainable(false);
        assert_eq!(frozen.q_norm().weight().id(), q_id);
        assert_eq!(frozen.k_norm().weight().id(), k_id);
        assert!(!frozen.k_norm().weight().requires_grad());
        assert_eq!(frozen.q_norm().eps(), 1e-6);
    }

    #[test]
    fn none_norms_pass_through() {
        let (client, device) = cpu_setup();
        let q = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap(),
            false,
        );
        let k = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[1, 2], &device).unwrap(),
            false,
        );
        let (q2, k2) =
            apply_qk_norm::<CpuRuntime, _, RmsNorm<CpuRuntime>>(&client, q, k, None, None).unwrap();
        assert_eq!(q2.tensor().to_vec::<f32>(), vec![1.0, 2.0]);
        assert_eq!(k2.tensor().to_vec::<f32>(), vec![3.0, 4.0]);
    }
}
