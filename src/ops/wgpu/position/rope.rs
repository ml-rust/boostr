//! WebGPU implementation of RoPEOps — fused WGSL shader dispatch

use crate::error::{Error, Result};
use crate::ops::traits::RoPEOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const ROPE_SHADER: &str = include_str!("../shaders/position/rope.wgsl");
const ROPE_INTERLEAVED_SHADER: &str = include_str!("../shaders/position/rope_interleaved.wgsl");
const ROPE_YARN_SHADER: &str = include_str!("../shaders/position/rope_yarn.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RoPEParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct YaRNParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    attn_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

fn validate_rope_shapes(
    x: &Tensor<WgpuRuntime>,
    cos: &Tensor<WgpuRuntime>,
    sin: &Tensor<WgpuRuntime>,
) -> Result<(usize, usize, usize, usize)> {
    let x_shape = x.shape();
    if x_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected 4D [B, H, S, D], got {:?}", x_shape),
        });
    }

    let (b, h, s, d) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    if d % 2 != 0 {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!("head_dim must be even, got {}", d),
        });
    }

    let half_d = d / 2;
    let cos_shape = cos.shape();
    let sin_shape = sin.shape();
    if cos_shape.len() != 2
        || cos_shape[1] != half_d
        || sin_shape.len() != 2
        || sin_shape[1] != half_d
    {
        return Err(Error::InvalidArgument {
            arg: "cache",
            reason: format!(
                "expected [>=S, {}], got cos: {:?}, sin: {:?}",
                half_d, cos_shape, sin_shape
            ),
        });
    }

    Ok((b, h, s, d))
}

/// Launch a compute shader with 3 read-only storage buffers + 1 read-write output + 1 uniform.
#[allow(clippy::too_many_arguments)]
fn launch_rope_shader(
    client: &WgpuClient,
    shader_source: &'static str,
    entry_point: &'static str,
    x_buf: &wgpu::Buffer,
    cos_buf: &wgpu::Buffer,
    sin_buf: &wgpu::Buffer,
    out_buf: &wgpu::Buffer,
    params_bytes: &[u8],
    num_workgroups: u32,
) {
    let params_buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope_params"),
        size: params_bytes.len() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&params_buf, 0, params_bytes);

    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(entry_point, shader_source);

    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 3,
    });
    let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[x_buf, cos_buf, sin_buf, out_buf, &params_buf]);

    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(entry_point),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry_point),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
}

impl RoPEOps<WgpuRuntime> for WgpuClient {
    fn apply_rope(
        &self,
        x: &Var<WgpuRuntime>,
        cos_cache: &Var<WgpuRuntime>,
        sin_cache: &Var<WgpuRuntime>,
    ) -> Result<Var<WgpuRuntime>> {
        let x_t = x.tensor();
        let cos_t = cos_cache.tensor();
        let sin_t = sin_cache.tensor();

        validate_f32(x_t, "apply_rope")?;
        validate_f32(cos_t, "apply_rope")?;
        validate_f32(sin_t, "apply_rope")?;

        let (b, h, s, d) = validate_rope_shapes(x_t, cos_t, sin_t)?;

        let cos_narrowed = if cos_t.shape()[0] > s {
            cos_t.narrow(0, 0, s)?
        } else {
            cos_t.clone()
        };
        let sin_narrowed = if sin_t.shape()[0] > s {
            sin_t.narrow(0, 0, s)?
        } else {
            sin_t.clone()
        };

        let output = Tensor::<WgpuRuntime>::empty(&[b, h, s, d], DType::F32, x_t.device());

        let x_buf = get_buffer(x_t.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "x buffer not found".into(),
        })?;
        let cos_buf =
            get_buffer(cos_narrowed.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "cos buffer not found".into(),
            })?;
        let sin_buf =
            get_buffer(sin_narrowed.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "sin buffer not found".into(),
            })?;
        let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;

        let params = RoPEParams {
            batch_size: b as u32,
            num_heads: h as u32,
            seq_len: s as u32,
            head_dim: d as u32,
        };

        let total_pairs = (b * h * s * d / 2) as u32;
        let workgroups = total_pairs.div_ceil(256);

        launch_rope_shader(
            self,
            ROPE_SHADER,
            "rope_apply_f32",
            &x_buf,
            &cos_buf,
            &sin_buf,
            &out_buf,
            bytemuck::bytes_of(&params),
            workgroups,
        );

        Ok(Var::new(output, false))
    }

    fn apply_rope_interleaved(
        &self,
        x: &Var<WgpuRuntime>,
        cos_cache: &Var<WgpuRuntime>,
        sin_cache: &Var<WgpuRuntime>,
    ) -> Result<Var<WgpuRuntime>> {
        let x_t = x.tensor();
        let cos_t = cos_cache.tensor();
        let sin_t = sin_cache.tensor();

        validate_f32(x_t, "apply_rope_interleaved")?;
        validate_f32(cos_t, "apply_rope_interleaved")?;
        validate_f32(sin_t, "apply_rope_interleaved")?;

        let (b, h, s, d) = validate_rope_shapes(x_t, cos_t, sin_t)?;

        let cos_narrowed = if cos_t.shape()[0] > s {
            cos_t.narrow(0, 0, s)?
        } else {
            cos_t.clone()
        };
        let sin_narrowed = if sin_t.shape()[0] > s {
            sin_t.narrow(0, 0, s)?
        } else {
            sin_t.clone()
        };

        let output = Tensor::<WgpuRuntime>::empty(&[b, h, s, d], DType::F32, x_t.device());

        let x_buf = get_buffer(x_t.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "x buffer not found".into(),
        })?;
        let cos_buf =
            get_buffer(cos_narrowed.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "cos buffer not found".into(),
            })?;
        let sin_buf =
            get_buffer(sin_narrowed.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "sin buffer not found".into(),
            })?;
        let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;

        let params = RoPEParams {
            batch_size: b as u32,
            num_heads: h as u32,
            seq_len: s as u32,
            head_dim: d as u32,
        };

        let total_pairs = (b * h * s * d / 2) as u32;
        let workgroups = total_pairs.div_ceil(256);

        launch_rope_shader(
            self,
            ROPE_INTERLEAVED_SHADER,
            "rope_interleaved_f32",
            &x_buf,
            &cos_buf,
            &sin_buf,
            &out_buf,
            bytemuck::bytes_of(&params),
            workgroups,
        );

        Ok(Var::new(output, false))
    }

    fn apply_rope_yarn(
        &self,
        x: &Var<WgpuRuntime>,
        cos_cache: &Var<WgpuRuntime>,
        sin_cache: &Var<WgpuRuntime>,
        attn_scale: f32,
    ) -> Result<Var<WgpuRuntime>> {
        let x_t = x.tensor();
        let cos_t = cos_cache.tensor();
        let sin_t = sin_cache.tensor();

        validate_f32(x_t, "apply_rope_yarn")?;
        validate_f32(cos_t, "apply_rope_yarn")?;
        validate_f32(sin_t, "apply_rope_yarn")?;

        let (b, h, s, d) = validate_rope_shapes(x_t, cos_t, sin_t)?;

        let cos_narrowed = if cos_t.shape()[0] > s {
            cos_t.narrow(0, 0, s)?
        } else {
            cos_t.clone()
        };
        let sin_narrowed = if sin_t.shape()[0] > s {
            sin_t.narrow(0, 0, s)?
        } else {
            sin_t.clone()
        };

        let output = Tensor::<WgpuRuntime>::empty(&[b, h, s, d], DType::F32, x_t.device());

        let x_buf = get_buffer(x_t.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "x buffer not found".into(),
        })?;
        let cos_buf =
            get_buffer(cos_narrowed.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "cos buffer not found".into(),
            })?;
        let sin_buf =
            get_buffer(sin_narrowed.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "sin buffer not found".into(),
            })?;
        let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;

        let params = YaRNParams {
            batch_size: b as u32,
            num_heads: h as u32,
            seq_len: s as u32,
            head_dim: d as u32,
            attn_scale,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let total_pairs = (b * h * s * d / 2) as u32;
        let workgroups = total_pairs.div_ceil(256);

        launch_rope_shader(
            self,
            ROPE_YARN_SHADER,
            "rope_yarn_f32",
            &x_buf,
            &cos_buf,
            &sin_buf,
            &out_buf,
            bytemuck::bytes_of(&params),
            workgroups,
        );

        Ok(Var::new(output, false))
    }
}
