//! One tensor-core launch of the feature-major family, on either schedule.

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaFunction, CudaModule, LaunchConfig};
use numr::dtype::DType;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use super::super::super::kernels;
use super::formats::FeatMajorFormat;
use super::tiling::{Role, Tiling};

/// Opts a function in to more than the static shared-memory limit. Required
/// before the first launch of every variant; the limit is per function.
fn opt_in_shared(func: &CudaFunction, bytes: u32, name: &str) -> Result<()> {
    func.set_attribute(
        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        bytes as i32,
    )
    .map_err(|e| Error::QuantError {
        reason: format!("CUDA {name} shared-memory opt-in failed: {e:?}"),
    })
}

/// One feature-major launch, on either schedule. `tiling::tile_parallel_tune`
/// builds one to time both schedules, so the probe issues exactly the
/// launches the dispatch does.
pub(super) struct Launch<'a> {
    pub format: &'a FeatMajorFormat,
    pub client: &'a CudaClient,
    pub module: &'a std::sync::Arc<CudaModule>,
    pub output_ptr: u64,
    pub q8_ptr: u64,
    pub weight_ptr: u64,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub ntok: u32,
    pub tiling: Tiling,
    pub smem: u32,
    /// (token tiles, feature tiles).
    pub grid: (u32, u32),
    pub splits: u32,
}

impl Launch<'_> {
    fn function(&self, role: Role, smem: u32) -> Result<(CudaFunction, String)> {
        let name = self.tiling.kernel_name(self.format, role);
        let func = kernels::get_kernel_function(self.module, &name)?;
        if smem > 0 {
            opt_in_shared(&func, smem, &name)?;
        }
        Ok((func, name))
    }

    /// One block per output tile. One range takes the plain kernel; more
    /// take the multi-range kernel, which runs them back to back.
    pub fn tile_parallel(&self) -> Result<()> {
        let role = if self.splits > 1 {
            Role::Fused
        } else {
            Role::TileParallel
        };
        let (func, name) = self.function(role, self.smem)?;
        let cfg = LaunchConfig {
            grid_dim: (self.grid.0, self.grid.1, 1),
            block_dim: (self.tiling.threads(), 1, 1),
            shared_mem_bytes: self.smem,
        };
        unsafe {
            let mut builder = self.client.stream().launch_builder(&func);
            builder.arg(&self.q8_ptr);
            builder.arg(&self.weight_ptr);
            builder.arg(&self.output_ptr);
            builder.arg(&self.m);
            builder.arg(&self.k);
            builder.arg(&self.n);
            builder.arg(&self.ntok);
            if self.splits > 1 {
                builder.arg(&self.splits);
            }
            builder.launch(cfg).map_err(|e| Error::QuantError {
                reason: format!("CUDA {name} launch failed: {e:?}"),
            })?;
        }
        Ok(())
    }

    /// One block per (output tile, split range), then the fixup pass that
    /// adds ranges 1.. onto the range-0 store, in order. The fixup is a
    /// separate launch on the same stream: it reads what the first wrote.
    pub fn split_k(&self, device: &CudaDevice) -> Result<()> {
        let threads = self.tiling.threads();
        let (sk_func, sk_name) = self.function(Role::SplitK, self.smem)?;

        // `workspace[tile][s - 1]`: one dense tile per split past the first.
        let tiles = self.grid.0 as usize * self.grid.1 as usize;
        let ws_len = tiles
            * (self.splits as usize - 1)
            * self.tiling.mmq_x as usize
            * self.tiling.feat_tile as usize;
        let ws = Tensor::<CudaRuntime>::empty(&[ws_len], DType::F32, device)?;
        let ws_ptr = ws.ptr();

        let cfg_sk = LaunchConfig {
            grid_dim: (self.grid.0, self.grid.1, self.splits),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: self.smem,
        };
        unsafe {
            let mut builder = self.client.stream().launch_builder(&sk_func);
            builder.arg(&self.q8_ptr);
            builder.arg(&self.weight_ptr);
            builder.arg(&self.output_ptr);
            builder.arg(&ws_ptr);
            builder.arg(&self.m);
            builder.arg(&self.k);
            builder.arg(&self.n);
            builder.arg(&self.ntok);
            builder.launch(cfg_sk).map_err(|e| Error::QuantError {
                reason: format!("CUDA {sk_name} launch failed: {e:?}"),
            })?;
        }

        let (fx_func, fx_name) = self.function(Role::Fixup, 0)?;
        let cfg_fx = LaunchConfig {
            grid_dim: (self.grid.0, self.grid.1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = self.client.stream().launch_builder(&fx_func);
            builder.arg(&self.output_ptr);
            builder.arg(&ws_ptr);
            builder.arg(&self.m);
            builder.arg(&self.n);
            builder.arg(&self.splits);
            builder.launch(cfg_fx).map_err(|e| Error::QuantError {
                reason: format!("CUDA {fx_name} launch failed: {e:?}"),
            })?;
        }
        Ok(())
    }
}
