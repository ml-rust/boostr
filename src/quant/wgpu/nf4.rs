//! WebGPU NF4 dispatch

use crate::error::{Error, Result};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

use super::shaders::nf4 as shader_gen;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Nf4DequantParams {
    num_bytes: u32,
    blocksize: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Nf4GemmParams {
    m: u32,
    k: u32,
    n: u32,
    blocksize: u32,
}

pub fn dispatch_nf4_dequant(
    client: &WgpuClient,
    nf4_data: &Tensor<WgpuRuntime>,
    absmax: &Tensor<WgpuRuntime>,
    output: &Tensor<WgpuRuntime>,
    num_bytes: u32,
    blocksize: u32,
) -> Result<()> {
    let shader_source = shader_gen::generate_nf4_dequant_shader();
    let entry_point = "nf4_dequant";

    let data_buf = get_buffer(nf4_data.storage().ptr()).ok_or_else(|| Error::QuantError {
        reason: "nf4_data buffer not found".into(),
    })?;
    let abs_buf = get_buffer(absmax.storage().ptr()).ok_or_else(|| Error::QuantError {
        reason: "absmax buffer not found".into(),
    })?;
    let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::QuantError {
        reason: "output buffer not found".into(),
    })?;

    let params = Nf4DequantParams {
        num_bytes,
        blocksize,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("nf4_dequant_params"),
        size: std::mem::size_of::<Nf4DequantParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(entry_point, &shader_source);
    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);
    let bind_group =
        cache.create_bind_group(&layout, &[&data_buf, &abs_buf, &out_buf, &params_buf]);

    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nf4_dequant"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nf4_dequant"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(num_bytes.div_ceil(256), 1, 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
    Ok(())
}

pub fn dispatch_nf4_gemm(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    nf4_weight: &Tensor<WgpuRuntime>,
    absmax: &Tensor<WgpuRuntime>,
    output: &Tensor<WgpuRuntime>,
    m: u32,
    k: u32,
    n: u32,
    blocksize: u32,
) -> Result<()> {
    let shader_source = shader_gen::generate_nf4_gemm_shader();
    let entry_point = "nf4_gemm";

    let inp_buf = get_buffer(input.storage().ptr()).ok_or_else(|| Error::QuantError {
        reason: "input buffer not found".into(),
    })?;
    let wt_buf = get_buffer(nf4_weight.storage().ptr()).ok_or_else(|| Error::QuantError {
        reason: "nf4_weight buffer not found".into(),
    })?;
    let abs_buf = get_buffer(absmax.storage().ptr()).ok_or_else(|| Error::QuantError {
        reason: "absmax buffer not found".into(),
    })?;
    let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::QuantError {
        reason: "output buffer not found".into(),
    })?;

    let params = Nf4GemmParams { m, k, n, blocksize };
    let params_buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("nf4_gemm_params"),
        size: std::mem::size_of::<Nf4GemmParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(entry_point, &shader_source);
    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(
        &layout,
        &[&inp_buf, &wt_buf, &abs_buf, &out_buf, &params_buf],
    );

    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nf4_gemm"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nf4_gemm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(n.div_ceil(16), m.div_ceil(16), 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
    Ok(())
}
