//! WebGPU implementation of MoEOps
//!
//! Uses custom WGSL shaders for routing, permute, unpermute, and grouped GEMM.
//! F32 only (WebGPU limitation).
//! Permute/unpermute: argsort + offset computation stays in impl_generic,
//! only routing and grouped GEMM use custom shaders.

use crate::error::{Error, Result};
use crate::ops::impl_generic::architecture::moe::{
    moe_permute_tokens_impl, moe_unpermute_tokens_impl,
};
use crate::ops::traits::architecture::moe::{MoEActivation, MoEOps};
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const MOE_ROUTING_SHADER: &str = include_str!("../shaders/architecture/moe_routing.wgsl");
const MOE_GROUPED_GEMM_SHADER: &str = include_str!("../shaders/architecture/moe_grouped_gemm.wgsl");

/// Select grouped GEMM entry point by activation (WebGPU is F32-only).
fn grouped_gemm_entry_point(activation: MoEActivation) -> &'static str {
    match activation {
        MoEActivation::None => "moe_grouped_gemm_f32",
        MoEActivation::SiLU => "moe_grouped_gemm_silu_f32",
        MoEActivation::GeLU => "moe_grouped_gemm_gelu_f32",
    }
}

fn validate_f32(t: &numr::tensor::Tensor<WgpuRuntime>, op: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("{}: WebGPU MoE requires F32, got {:?}", op, t.dtype()),
        });
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MoERoutingParams {
    num_tokens: u32,
    num_experts: u32,
    k: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MoEGemmParams {
    in_dim: u32,
    out_dim: u32,
    num_experts: u32,
    _pad: u32,
}

impl MoEOps<WgpuRuntime> for WgpuClient {
    fn moe_top_k_routing(
        &self,
        logits: &Tensor<WgpuRuntime>,
        k: usize,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        validate_f32(logits, "moe_top_k_routing")?;

        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "expected 2D [num_tokens, num_experts], got {}D",
                    shape.len()
                ),
            });
        }

        let num_tokens = shape[0];
        let num_experts = shape[1];

        if k == 0 || k > num_experts {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: format!("k={} must be in [1, num_experts={}]", k, num_experts),
            });
        }

        // Allocate outputs — I32 indices on WebGPU (no I64)
        let out_indices =
            Tensor::<WgpuRuntime>::empty(&[num_tokens, k], DType::I32, logits.device());
        let out_weights =
            Tensor::<WgpuRuntime>::empty(&[num_tokens, k], DType::F32, logits.device());

        let logits_buf = get_buffer(logits.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "logits buffer not found".into(),
        })?;
        let indices_buf =
            get_buffer(out_indices.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "indices buffer not found".into(),
            })?;
        let weights_buf =
            get_buffer(out_weights.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "weights buffer not found".into(),
            })?;

        let params = MoERoutingParams {
            num_tokens: num_tokens as u32,
            num_experts: num_experts as u32,
            k: k as u32,
            _pad: 0,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("moe_routing_params"),
            size: std::mem::size_of::<MoERoutingParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("moe_routing_f32", MOE_ROUTING_SHADER);
        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 3,
            num_uniform_buffers: 1,
            num_readonly_storage: 1,
        });
        let pipeline =
            cache.get_or_create_pipeline("moe_routing_f32", "moe_routing_f32", &module, &layout);

        let bind_group = cache.create_bind_group(
            &layout,
            &[&logits_buf, &indices_buf, &weights_buf, &params_buf],
        );

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("moe_routing"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("moe_routing"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(num_tokens as u32, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok((out_indices, out_weights))
    }

    fn moe_permute_tokens(
        &self,
        tokens: &Tensor<WgpuRuntime>,
        indices: &Tensor<WgpuRuntime>,
        num_experts: usize,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        // impl_generic composes numr's WebGPU-native argsort + index_select
        moe_permute_tokens_impl(self, tokens, indices, num_experts)
    }

    fn moe_unpermute_tokens(
        &self,
        expert_output: &Tensor<WgpuRuntime>,
        sort_indices: &Tensor<WgpuRuntime>,
        weights: &Tensor<WgpuRuntime>,
        num_tokens: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        moe_unpermute_tokens_impl(self, expert_output, sort_indices, weights, num_tokens)
    }

    fn moe_grouped_gemm(
        &self,
        permuted_tokens: &Tensor<WgpuRuntime>,
        expert_weights: &Tensor<WgpuRuntime>,
        expert_offsets: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        launch_grouped_gemm_wgpu(
            self,
            permuted_tokens,
            expert_weights,
            expert_offsets,
            grouped_gemm_entry_point(MoEActivation::None),
        )
    }

    fn moe_grouped_gemm_fused(
        &self,
        permuted_tokens: &Tensor<WgpuRuntime>,
        expert_weights: &Tensor<WgpuRuntime>,
        expert_offsets: &Tensor<WgpuRuntime>,
        activation: MoEActivation,
    ) -> Result<Tensor<WgpuRuntime>> {
        launch_grouped_gemm_wgpu(
            self,
            permuted_tokens,
            expert_weights,
            expert_offsets,
            grouped_gemm_entry_point(activation),
        )
    }
}

fn launch_grouped_gemm_wgpu(
    client: &WgpuClient,
    permuted_tokens: &Tensor<WgpuRuntime>,
    expert_weights: &Tensor<WgpuRuntime>,
    expert_offsets: &Tensor<WgpuRuntime>,
    entry_point: &'static str,
) -> Result<Tensor<WgpuRuntime>> {
    validate_f32(permuted_tokens, "moe_grouped_gemm")?;
    validate_f32(expert_weights, "moe_grouped_gemm")?;

    let pt_shape = permuted_tokens.shape();
    let ew_shape = expert_weights.shape();
    if pt_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "permuted_tokens",
            reason: format!("expected 2D, got {}D", pt_shape.len()),
        });
    }
    if ew_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "expert_weights",
            reason: format!("expected 3D, got {}D", ew_shape.len()),
        });
    }
    if pt_shape[1] != ew_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "expert_weights",
            reason: format!(
                "in_dim mismatch: tokens {}, weights {}",
                pt_shape[1], ew_shape[1]
            ),
        });
    }

    let total_tokens = pt_shape[0];
    let in_dim = pt_shape[1];
    let num_experts = ew_shape[0];
    let out_dim = ew_shape[2];
    let device = permuted_tokens.device();

    let output = Tensor::<WgpuRuntime>::empty(&[total_tokens, out_dim], DType::F32, device);

    if total_tokens == 0 {
        return Ok(output);
    }

    let tokens_buf =
        get_buffer(permuted_tokens.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "tokens buffer not found".into(),
        })?;
    let weights_buf =
        get_buffer(expert_weights.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "weights buffer not found".into(),
        })?;
    let offsets_buf =
        get_buffer(expert_offsets.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "offsets buffer not found".into(),
        })?;
    let output_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "output buffer not found".into(),
    })?;

    let params = MoEGemmParams {
        in_dim: in_dim as u32,
        out_dim: out_dim as u32,
        num_experts: num_experts as u32,
        _pad: 0,
    };

    let params_buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("moe_gemm_params"),
        size: std::mem::size_of::<MoEGemmParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module("moe_grouped_gemm", MOE_GROUPED_GEMM_SHADER);
    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 3,
    });
    let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            &tokens_buf,
            &weights_buf,
            &offsets_buf,
            &output_buf,
            &params_buf,
        ],
    );

    // grid_y uses total_tokens as a conservative upper bound — no CPU readback needed.
    // The shader reads offsets from device memory and guards: `if (row < count)`.
    const TILE: u32 = 16;
    let grid_x = (out_dim as u32 + TILE - 1) / TILE;
    let grid_y = (total_tokens as u32 + TILE - 1) / TILE;
    let grid_z = num_experts as u32;

    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("moe_grouped_gemm"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("moe_grouped_gemm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(grid_x, grid_y, grid_z);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));

    Ok(output)
}
