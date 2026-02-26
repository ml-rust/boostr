//! WebGPU INT4 GEMM dispatch (AWQ, GPTQ, Marlin)
//!
//! Helper functions that set up pipelines and dispatch shaders.
//! Called from the QuantMatmulOps trait impl in quant_matmul.rs.

use crate::error::{Error, Result};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

use super::shaders::int4_gemm as shader_gen;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Int4GemmParams {
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GptqParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_int4_gemm(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    qweight: &Tensor<WgpuRuntime>,
    scales: &Tensor<WgpuRuntime>,
    zeros: &Tensor<WgpuRuntime>,
    output: &Tensor<WgpuRuntime>,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
) -> Result<()> {
    let shader_source = shader_gen::generate_int4_gemm_shader();
    let entry_point = "int4_gemm";

    let bufs = [
        get_buffer(input.storage().ptr()),
        get_buffer(qweight.storage().ptr()),
        get_buffer(scales.storage().ptr()),
        get_buffer(zeros.storage().ptr()),
        get_buffer(output.storage().ptr()),
    ];
    for (i, b) in bufs.iter().enumerate() {
        if b.is_none() {
            return Err(Error::QuantError {
                reason: format!("int4_gemm buffer {} not found in WebGPU registry", i),
            });
        }
    }
    let [inp_buf, qw_buf, sc_buf, zr_buf, out_buf] = bufs.map(|b| b.unwrap());

    let params = Int4GemmParams {
        m,
        k,
        n,
        group_size,
    };
    let params_buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("int4_gemm_params"),
        size: std::mem::size_of::<Int4GemmParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(entry_point, &shader_source);
    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(
        &layout,
        &[&inp_buf, &qw_buf, &sc_buf, &zr_buf, &out_buf, &params_buf],
    );

    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("int4_gemm"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("int4_gemm"),
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

#[allow(clippy::too_many_arguments)]
pub fn dispatch_int4_gemm_gptq(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    qweight: &Tensor<WgpuRuntime>,
    qzeros: &Tensor<WgpuRuntime>,
    scales: &Tensor<WgpuRuntime>,
    g_idx: &Tensor<WgpuRuntime>,
    output: &Tensor<WgpuRuntime>,
    m: u32,
    k: u32,
    n: u32,
) -> Result<()> {
    let shader_source = shader_gen::generate_int4_gemm_gptq_shader();
    let entry_point = "int4_gemm_gptq";

    let bufs = [
        get_buffer(input.storage().ptr()),
        get_buffer(qweight.storage().ptr()),
        get_buffer(qzeros.storage().ptr()),
        get_buffer(scales.storage().ptr()),
        get_buffer(g_idx.storage().ptr()),
        get_buffer(output.storage().ptr()),
    ];
    for (i, b) in bufs.iter().enumerate() {
        if b.is_none() {
            return Err(Error::QuantError {
                reason: format!("int4_gemm_gptq buffer {} not found", i),
            });
        }
    }
    let [inp_buf, qw_buf, qz_buf, sc_buf, gi_buf, out_buf] = bufs.map(|b| b.unwrap());

    let params = GptqParams { m, k, n, _pad: 0 };
    let params_buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gptq_params"),
        size: std::mem::size_of::<GptqParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(entry_point, &shader_source);
    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(
        &layout,
        &[
            &inp_buf,
            &qw_buf,
            &qz_buf,
            &sc_buf,
            &gi_buf,
            &out_buf,
            &params_buf,
        ],
    );

    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("int4_gemm_gptq"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("int4_gemm_gptq"),
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

#[allow(clippy::too_many_arguments)]
pub fn dispatch_marlin_gemm(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    scales: &Tensor<WgpuRuntime>,
    zeros: &Tensor<WgpuRuntime>,
    output: &Tensor<WgpuRuntime>,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
) -> Result<()> {
    let shader_source = shader_gen::generate_marlin_gemm_shader();
    let entry_point = "marlin_gemm";

    let bufs = [
        get_buffer(input.storage().ptr()),
        get_buffer(weight.storage().ptr()),
        get_buffer(scales.storage().ptr()),
        get_buffer(zeros.storage().ptr()),
        get_buffer(output.storage().ptr()),
    ];
    for (i, b) in bufs.iter().enumerate() {
        if b.is_none() {
            return Err(Error::QuantError {
                reason: format!("marlin_gemm buffer {} not found", i),
            });
        }
    }
    let [inp_buf, wt_buf, sc_buf, zr_buf, out_buf] = bufs.map(|b| b.unwrap());

    let params = Int4GemmParams {
        m,
        k,
        n,
        group_size,
    };
    let params_buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("marlin_params"),
        size: std::mem::size_of::<Int4GemmParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(entry_point, &shader_source);
    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(
        &layout,
        &[&inp_buf, &wt_buf, &sc_buf, &zr_buf, &out_buf, &params_buf],
    );

    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("marlin_gemm"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("marlin_gemm"),
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
