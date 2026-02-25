//! WebGPU implementation of KvCacheOps
//!
//! Fused KV cache update and reshape_and_cache operations.
//! F32 only (WebGPU limitation).

use crate::error::{Error, Result};
use crate::ops::traits::KvCacheOps;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const KV_CACHE_SHADER_SOURCE: &str = include_str!("../shaders/cache/kv_cache.wgsl");
const RESHAPE_CACHE_SHADER_SOURCE: &str = include_str!("../shaders/cache/reshape_and_cache.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KvCacheParams {
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    new_len: u32,
    max_seq_len: u32,
    position: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PagedCacheParams {
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    block_size: u32,
}

fn validate_f32(t: &Tensor<WgpuRuntime>, op: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("{}: WebGPU requires F32, got {:?}", op, t.dtype()),
        });
    }
    Ok(())
}

impl KvCacheOps<WgpuRuntime> for WgpuClient {
    fn kv_cache_update(
        &self,
        k_cache: &Tensor<WgpuRuntime>,
        v_cache: &Tensor<WgpuRuntime>,
        new_k: &Tensor<WgpuRuntime>,
        new_v: &Tensor<WgpuRuntime>,
        position: usize,
    ) -> Result<()> {
        validate_f32(k_cache, "kv_cache_update")?;
        validate_f32(v_cache, "kv_cache_update")?;
        validate_f32(new_k, "kv_cache_update")?;
        validate_f32(new_v, "kv_cache_update")?;

        let k_shape = k_cache.shape();
        let new_k_shape = new_k.shape();

        let batch_size = k_shape[0];
        let num_heads = k_shape[1];
        let max_seq_len = k_shape[2];
        let head_dim = k_shape[3];
        let new_len = new_k_shape[2];

        let k_buf = get_buffer(k_cache.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "k_cache buffer not found".into(),
        })?;
        let v_buf = get_buffer(v_cache.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "v_cache buffer not found".into(),
        })?;
        let new_k_buf = get_buffer(new_k.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "new_k buffer not found".into(),
        })?;
        let new_v_buf = get_buffer(new_v.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "new_v buffer not found".into(),
        })?;

        let params = KvCacheParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            new_len: new_len as u32,
            max_seq_len: max_seq_len as u32,
            position: position as u32,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_cache_params"),
            size: std::mem::size_of::<KvCacheParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("kv_cache_update_f32", KV_CACHE_SHADER_SOURCE);

        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 4,
            num_uniform_buffers: 1,
            num_readonly_storage: 2,
        });
        let pipeline = cache.get_or_create_pipeline(
            "kv_cache_update_f32",
            "kv_cache_update_f32",
            &module,
            &layout,
        );

        // Bind order must match shader: read-only first (new_k, new_v), then rw (k_cache, v_cache)
        let bind_group = cache.create_bind_group(
            &layout,
            &[&new_k_buf, &new_v_buf, &k_buf, &v_buf, &params_buf],
        );

        let total_elems = (batch_size * num_heads * new_len * head_dim) as u32;
        let workgroups = total_elems.div_ceil(256);

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("kv_cache_update"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kv_cache_update"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    fn reshape_and_cache(
        &self,
        key: &Tensor<WgpuRuntime>,
        value: &Tensor<WgpuRuntime>,
        key_cache: &Tensor<WgpuRuntime>,
        value_cache: &Tensor<WgpuRuntime>,
        slot_mapping: &Tensor<WgpuRuntime>,
        block_size: usize,
    ) -> Result<()> {
        validate_f32(key, "reshape_and_cache")?;
        validate_f32(value, "reshape_and_cache")?;
        validate_f32(key_cache, "reshape_and_cache")?;
        validate_f32(value_cache, "reshape_and_cache")?;

        let key_shape = key.shape();
        let num_tokens = key_shape[0];
        let num_heads = key_shape[1];
        let head_dim = key_shape[2];

        let key_buf = get_buffer(key.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "key buffer not found".into(),
        })?;
        let value_buf = get_buffer(value.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "value buffer not found".into(),
        })?;
        let key_cache_buf =
            get_buffer(key_cache.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "key_cache buffer not found".into(),
            })?;
        let value_cache_buf =
            get_buffer(value_cache.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "value_cache buffer not found".into(),
            })?;
        let slot_buf =
            get_buffer(slot_mapping.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "slot_mapping buffer not found".into(),
            })?;

        let params = PagedCacheParams {
            num_tokens: num_tokens as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("paged_cache_params"),
            size: std::mem::size_of::<PagedCacheParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module =
            cache.get_or_create_module("reshape_and_cache_f32", RESHAPE_CACHE_SHADER_SOURCE);

        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 5,
            num_uniform_buffers: 1,
            num_readonly_storage: 3,
        });
        let pipeline = cache.get_or_create_pipeline(
            "reshape_and_cache_f32",
            "reshape_and_cache_f32",
            &module,
            &layout,
        );

        // Bind order: read-only first (key, value, slot_mapping), then rw (key_cache, value_cache)
        let bind_group = cache.create_bind_group(
            &layout,
            &[
                &key_buf,
                &value_buf,
                &slot_buf,
                &key_cache_buf,
                &value_cache_buf,
                &params_buf,
            ],
        );

        let total_elems = (num_tokens * num_heads * head_dim) as u32;
        let workgroups = total_elems.div_ceil(256);

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("reshape_and_cache"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reshape_and_cache"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}
