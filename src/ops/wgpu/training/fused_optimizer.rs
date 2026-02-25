//! WebGPU implementation of FusedOptimizerOps
//!
//! F32 only (WebGPU is 32-bit by design).

use crate::error::{Error, Result};
use crate::ops::traits::FusedOptimizerOps;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const SHADER_SOURCE: &str = include_str!("../shaders/training/fused_optimizer.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OptimizerParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step_size: f32,
    momentum: f32,
    dampening: f32,
    nesterov: u32,
    has_buf: u32,
    n: u32,
    _pad: u32,
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

fn dispatch_kernel(
    client: &WgpuClient,
    entry: &'static str,
    buffers: &[&wgpu::Buffer],
    params: OptimizerParams,
) -> Result<()> {
    let params_buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("fused_opt_params"),
        size: std::mem::size_of::<OptimizerParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(entry, SHADER_SOURCE);

    // Always use 4 storage buffers to match shader binding layout.
    // For 3-buffer ops (sgd/adagrad), state1 is unused but must exist.
    let (num_storage, num_readonly) = (4u32, 1u32);

    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: num_storage,
        num_uniform_buffers: 1,
        num_readonly_storage: num_readonly,
    });
    let pipeline = cache.get_or_create_pipeline(entry, entry, &module, &layout);

    let mut all_bufs: Vec<&wgpu::Buffer> = buffers.to_vec();
    all_bufs.push(&params_buf);
    let bind_group = cache.create_bind_group(&layout, &all_bufs);

    let workgroups = (params.n + 255) / 256;

    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fused_opt"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_opt"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));

    Ok(())
}

impl FusedOptimizerOps<WgpuRuntime> for WgpuClient {
    fn fused_adamw_step(
        &self,
        param: &Tensor<WgpuRuntime>,
        grad: &Tensor<WgpuRuntime>,
        m: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        validate_f32(param, "fused_adamw")?;
        let n: usize = param.shape().iter().product();

        let new_param = param.clone();
        let new_m = m.clone();
        let new_v = v.clone();

        let p_buf = get_buffer(new_param.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "param buffer not found".into(),
        })?;
        let g_buf = get_buffer(grad.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "grad buffer not found".into(),
        })?;
        let m_buf = get_buffer(new_m.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "m buffer not found".into(),
        })?;
        let v_buf = get_buffer(new_v.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "v buffer not found".into(),
        })?;

        let params = OptimizerParams {
            lr: lr as f32,
            beta1: beta1 as f32,
            beta2: beta2 as f32,
            eps: eps as f32,
            wd: wd as f32,
            step_size: step_size as f32,
            momentum: 0.0,
            dampening: 0.0,
            nesterov: 0,
            has_buf: 0,
            n: n as u32,
            _pad: 0,
        };

        dispatch_kernel(
            self,
            "fused_adamw_f32",
            &[&g_buf, &p_buf, &m_buf, &v_buf],
            params,
        )?;

        Ok((new_param, new_m, new_v))
    }

    fn fused_sgd_step(
        &self,
        param: &Tensor<WgpuRuntime>,
        grad: &Tensor<WgpuRuntime>,
        momentum_buf: Option<&Tensor<WgpuRuntime>>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        validate_f32(param, "fused_sgd")?;
        let n: usize = param.shape().iter().product();

        let new_param = param.clone();
        let has_buf = momentum_buf.is_some();
        let new_buf = match momentum_buf {
            Some(b) => b.clone(),
            None => Tensor::<WgpuRuntime>::zeros(param.shape(), DType::F32, param.device()),
        };

        let p_buf = get_buffer(new_param.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "param buffer not found".into(),
        })?;
        let g_buf = get_buffer(grad.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "grad buffer not found".into(),
        })?;
        let b_buf = get_buffer(new_buf.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "buf buffer not found".into(),
        })?;

        let params = OptimizerParams {
            lr: lr as f32,
            beta1: 0.0,
            beta2: 0.0,
            eps: 0.0,
            wd: wd as f32,
            step_size: 0.0,
            momentum: momentum as f32,
            dampening: dampening as f32,
            nesterov: if nesterov { 1 } else { 0 },
            has_buf: if has_buf { 1 } else { 0 },
            n: n as u32,
            _pad: 0,
        };

        // Pass dummy state1 buffer to match 4-binding shader layout
        let dummy = Tensor::<WgpuRuntime>::zeros(&[1], DType::F32, param.device());
        let dummy_buf = get_buffer(dummy.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "dummy buffer not found".into(),
        })?;
        dispatch_kernel(
            self,
            "fused_sgd_f32",
            &[&g_buf, &p_buf, &b_buf, &dummy_buf],
            params,
        )?;

        Ok((new_param, new_buf))
    }

    fn fused_adagrad_step(
        &self,
        param: &Tensor<WgpuRuntime>,
        grad: &Tensor<WgpuRuntime>,
        accum: &Tensor<WgpuRuntime>,
        lr: f64,
        eps: f64,
        wd: f64,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        validate_f32(param, "fused_adagrad")?;
        let n: usize = param.shape().iter().product();

        let new_param = param.clone();
        let new_accum = accum.clone();

        let p_buf = get_buffer(new_param.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "param buffer not found".into(),
        })?;
        let g_buf = get_buffer(grad.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "grad buffer not found".into(),
        })?;
        let a_buf = get_buffer(new_accum.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "accum buffer not found".into(),
        })?;

        let params = OptimizerParams {
            lr: lr as f32,
            beta1: 0.0,
            beta2: 0.0,
            eps: eps as f32,
            wd: wd as f32,
            step_size: 0.0,
            momentum: 0.0,
            dampening: 0.0,
            nesterov: 0,
            has_buf: 0,
            n: n as u32,
            _pad: 0,
        };

        // Pass dummy state1 buffer to match 4-binding shader layout
        let dummy = Tensor::<WgpuRuntime>::zeros(&[1], DType::F32, param.device());
        let dummy_buf = get_buffer(dummy.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "dummy buffer not found".into(),
        })?;
        dispatch_kernel(
            self,
            "fused_adagrad_f32",
            &[&g_buf, &p_buf, &a_buf, &dummy_buf],
            params,
        )?;

        Ok((new_param, new_accum))
    }

    fn fused_lamb_step(
        &self,
        param: &Tensor<WgpuRuntime>,
        grad: &Tensor<WgpuRuntime>,
        m: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        bias_corr1: f64,
        bias_corr2: f64,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        validate_f32(param, "fused_lamb")?;
        let n: usize = param.shape().iter().product();

        // For LAMB, param buffer is used to store the update vector
        let update = param.clone();
        let new_m = m.clone();
        let new_v = v.clone();

        let u_buf = get_buffer(update.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "update buffer not found".into(),
        })?;
        let g_buf = get_buffer(grad.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "grad buffer not found".into(),
        })?;
        let m_buf = get_buffer(new_m.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "m buffer not found".into(),
        })?;
        let v_buf = get_buffer(new_v.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "v buffer not found".into(),
        })?;

        // Reuse step_size for bias_corr1, dampening for bias_corr2
        let params = OptimizerParams {
            lr: 0.0,
            beta1: beta1 as f32,
            beta2: beta2 as f32,
            eps: eps as f32,
            wd: wd as f32,
            step_size: bias_corr1 as f32,
            momentum: 0.0,
            dampening: bias_corr2 as f32,
            nesterov: 0,
            has_buf: 0,
            n: n as u32,
            _pad: 0,
        };

        dispatch_kernel(
            self,
            "fused_lamb_f32",
            &[&g_buf, &u_buf, &m_buf, &v_buf],
            params,
        )?;

        Ok((update, new_m, new_v))
    }
}
