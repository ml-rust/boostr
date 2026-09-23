//! Writing and clearing one sequence's lane of a batched GDN state.
//!
//! A server keeps one lane per sequence. A sequence is admitted by
//! prefilling it on its own and installing the result into a free lane, and
//! released by clearing that lane. Both work in place, so the buffers keep
//! the addresses a captured graph holds.

use crate::error::{Error, Result};
use crate::inference::gdn_state::{GdnState, LayeredGdnState};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Row `lane` of `t`, as a view into `t`'s own storage.
fn lane_view<R: Runtime<DType = DType>>(t: &Tensor<R>, lane: usize) -> Result<Tensor<R>> {
    Ok(t.narrow(0, lane, 1)?)
}

impl<R: Runtime<DType = DType>> GdnState<R> {
    /// Copy a one-sequence state into lane `lane`.
    ///
    /// The lane then holds exactly what `src` holds, so the sequence
    /// continues where its own prefill left off.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch, `src`
    /// carries more than one sequence, or the two states disagree on
    /// anything but the batch.
    pub fn install_lane<C: BinaryOps<R>>(
        &mut self,
        client: &C,
        lane: usize,
        src: &GdnState<R>,
    ) -> Result<()> {
        if lane >= self.batch() {
            return Err(Error::InferenceError {
                reason: format!("lane {lane} is outside batch {}", self.batch()),
            });
        }
        if src.batch() != 1 {
            return Err(Error::InferenceError {
                reason: format!("install_lane takes one sequence, got {}", src.batch()),
            });
        }
        for (name, stored, new) in [
            ("conv", &self.conv, src.conv()),
            ("ssm", &self.ssm, src.ssm()),
        ] {
            let lane_buf = lane_view(stored, lane)?;
            if lane_buf.shape() != new.shape() {
                return Err(Error::InferenceError {
                    reason: format!(
                        "gdn state {name}: lane is {:?}, source is {:?}",
                        lane_buf.shape(),
                        new.shape()
                    ),
                });
            }
            client.copy_into(&lane_buf, new).map_err(Error::Numr)?;
        }
        self.initialized = true;
        Ok(())
    }

    /// Zero lane `lane` and leave every other lane as it is.
    ///
    /// Releases a finished sequence's lane for the next one. The buffers
    /// keep their addresses, so a captured graph stays valid.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch; a fill
    /// error propagates.
    pub fn reset_lane<C: BinaryOps<R>>(&mut self, client: &C, lane: usize) -> Result<()> {
        if lane >= self.batch() {
            return Err(Error::InferenceError {
                reason: format!("lane {lane} is outside batch {}", self.batch()),
            });
        }
        let device = self.ssm.device().clone();
        let dtype = self.ssm.dtype();
        for stored in [&self.conv, &self.ssm] {
            let lane_buf = lane_view(stored, lane)?;
            let zeros = Tensor::<R>::zeros(lane_buf.shape(), dtype, &device)?;
            client.copy_into(&lane_buf, &zeros).map_err(Error::Numr)?;
        }
        Ok(())
    }
}

impl<R: Runtime<DType = DType>> LayeredGdnState<R> {
    /// Install a one-sequence state into lane `lane` of every layer.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when the two states hold a different
    /// number of layers, or a layer refuses the write.
    pub fn install_lane<C: BinaryOps<R>>(
        &mut self,
        client: &C,
        lane: usize,
        src: &LayeredGdnState<R>,
    ) -> Result<()> {
        if src.num_layers() != self.num_layers() {
            return Err(Error::InferenceError {
                reason: format!(
                    "gdn state holds {} layers, source holds {}",
                    self.num_layers(),
                    src.num_layers()
                ),
            });
        }
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let from = src.layer(idx).ok_or_else(|| Error::InferenceError {
                reason: format!("source gdn state missing layer {idx}"),
            })?;
            layer.install_lane(client, lane, from)?;
        }
        Ok(())
    }

    /// Zero lane `lane` of every layer, leaving the other lanes as they are.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when a layer refuses the clear.
    pub fn reset_lane<C: BinaryOps<R>>(&mut self, client: &C, lane: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.reset_lane(client, lane)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::GdnConfig;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn cfg() -> GdnConfig {
        GdnConfig {
            hidden_size: 8,
            conv_kernel: 4,
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            inner_size: 16,
            rms_eps: 1e-6,
            chunk_size: 64,
            v_grouped: false,
        }
    }

    /// An installed lane holds the source state, the other lanes keep
    /// theirs, and the buffers keep their addresses.
    #[test]
    fn install_writes_one_lane_in_place() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let mut batched = GdnState::<CpuRuntime>::zeros(&cfg(), 3, DType::F32, &device).unwrap();
        let (conv_ptr, ssm_ptr) = (batched.conv().ptr(), batched.ssm().ptr());

        let mut src = GdnState::<CpuRuntime>::zeros(&cfg(), 1, DType::F32, &device).unwrap();
        let conv = Tensor::<CpuRuntime>::ones(&[1, 32, 3], DType::F32, &device).unwrap();
        let ssm = Tensor::<CpuRuntime>::ones(&[1, 4, 4, 4], DType::F32, &device).unwrap();
        src.update(&client, &conv, &ssm).unwrap();

        batched.install_lane(&client, 1, &src).unwrap();
        assert!(batched.is_initialized());
        assert_eq!(batched.conv().ptr(), conv_ptr);
        assert_eq!(batched.ssm().ptr(), ssm_ptr);

        let rows = batched.ssm().to_vec::<f32>();
        let row = 4 * 4 * 4;
        assert!(rows[..row].iter().all(|x| *x == 0.0));
        assert!(rows[row..2 * row].iter().all(|x| *x == 1.0));
        assert!(rows[2 * row..].iter().all(|x| *x == 0.0));
    }

    /// Clearing a lane leaves every other lane untouched.
    #[test]
    fn reset_clears_one_lane_only() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let mut batched = GdnState::<CpuRuntime>::zeros(&cfg(), 2, DType::F32, &device).unwrap();
        let conv = Tensor::<CpuRuntime>::ones(&[2, 32, 3], DType::F32, &device).unwrap();
        let ssm = Tensor::<CpuRuntime>::ones(&[2, 4, 4, 4], DType::F32, &device).unwrap();
        batched.update(&client, &conv, &ssm).unwrap();

        batched.reset_lane(&client, 0).unwrap();
        let rows = batched.ssm().to_vec::<f32>();
        let row = 4 * 4 * 4;
        assert!(rows[..row].iter().all(|x| *x == 0.0));
        assert!(rows[row..].iter().all(|x| *x == 1.0));
    }

    /// A lane outside the batch and a source with more than one sequence
    /// are both refused.
    #[test]
    fn refuses_bad_lane_or_source() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let mut batched = GdnState::<CpuRuntime>::zeros(&cfg(), 2, DType::F32, &device).unwrap();
        let src = GdnState::<CpuRuntime>::zeros(&cfg(), 1, DType::F32, &device).unwrap();
        let two = GdnState::<CpuRuntime>::zeros(&cfg(), 2, DType::F32, &device).unwrap();

        assert!(batched.install_lane(&client, 2, &src).is_err());
        assert!(batched.install_lane(&client, 0, &two).is_err());
        assert!(batched.reset_lane(&client, 5).is_err());
    }

    /// Every layer's lane is written, and the layer counts must agree.
    #[test]
    fn layered_install_covers_every_layer() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let mut batched =
            LayeredGdnState::<CpuRuntime>::zeros(2, &cfg(), 2, DType::F32, &device).unwrap();
        let mut src =
            LayeredGdnState::<CpuRuntime>::zeros(2, &cfg(), 1, DType::F32, &device).unwrap();
        let conv = Tensor::<CpuRuntime>::ones(&[1, 32, 3], DType::F32, &device).unwrap();
        let ssm = Tensor::<CpuRuntime>::ones(&[1, 4, 4, 4], DType::F32, &device).unwrap();
        for idx in 0..2 {
            src.layer_mut(idx)
                .unwrap()
                .update(&client, &conv, &ssm)
                .unwrap();
        }

        batched.install_lane(&client, 1, &src).unwrap();
        for idx in 0..2 {
            let rows = batched.layer(idx).unwrap().ssm().to_vec::<f32>();
            let row = 4 * 4 * 4;
            assert!(rows[..row].iter().all(|x| *x == 0.0));
            assert!(rows[row..].iter().all(|x| *x == 1.0));
        }

        let three =
            LayeredGdnState::<CpuRuntime>::zeros(3, &cfg(), 1, DType::F32, &device).unwrap();
        assert!(batched.install_lane(&client, 0, &three).is_err());
    }
}
