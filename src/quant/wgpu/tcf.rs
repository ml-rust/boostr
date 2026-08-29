//! WebGPU dispatch for TCF native quantized weights.
//!
//! Two entry points, one per shape of work: [`dispatch_dequant`] rebuilds a
//! whole tensor as f32, and [`dispatch_matmul`] multiplies against a packed
//! weight without ever materializing it. Both drive the same decoder in
//! `shaders::tcf` and take the same plane offsets from [`TcfPlanes`], so
//! neither can disagree with the other — or with the CUDA launches, which read
//! that type too — about the layout.
//!
//! # Uniform block
//!
//! One 16-field `u32` block serves both kernels. 64 bytes satisfies WGSL's
//! 16-byte uniform struct alignment, and a `u32` never needs interior padding,
//! so [`TcfShaderParams`] and the WGSL `TcfParams` agree field for field. The
//! trailing member rounds 60 bytes up to 64 and carries a second duty: it is
//! the always-zero value the decoder ORs into the asymmetric product's bits,
//! which is what stops a driver contracting `d * code + m` into a single
//! rounding the CPU never performs. Every offset is
//! narrowed to `u32` here, and a payload whose planes exceed `u32` is refused
//! rather than silently truncated — WebGPU's own maximum storage binding is far
//! below that bound anyway.
//!
//! # What the GPU path does not check
//!
//! `tcf-core` rejects a payload carrying Section 13.2's reserved code or a
//! Section 13.1 invalid scale. A shader cannot return that error per element,
//! so these kernels decode a payload that has already been accepted — which is
//! what reading a TCF file produces. On a payload `tcf-core` would reject, the
//! CPU path errors and the WebGPU path returns numbers.

use numr::runtime::wgpu::{WgpuClient, get_buffer};
use std::sync::Arc;
use wgpu::{Buffer, BufferUsages};

use crate::error::{Error, Result};
use crate::quant::TcfEncoding;
use crate::quant::tcf::TcfPlanes;

use super::shaders::tcf as shader_gen;

/// The uniform block both TCF shaders read. Mirrors WGSL `TcfParams`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TcfShaderParams {
    tiles: u32,
    code_high_off: u32,
    scale_off: u32,
    min_off: u32,
    super_off: u32,
    super_min_off: u32,
    bits: u32,
    group: u32,
    groups_per_tile: u32,
    symmetric: u32,
    scale_form: u32,
    sub_block_bytes: u32,
    m: u32,
    k: u32,
    n: u32,
    /// Always zero. The decoder ORs it into the bit pattern of the asymmetric
    /// product so a driver cannot fold the surrounding `bitcast` pair away and
    /// contract `d * code + m` into one rounding. See `tcf_settled` in
    /// `shaders::tcf`. It also rounds the block's 60 bytes up to the 64 WGSL's
    /// uniform alignment wants.
    zero_barrier: u32,
}

/// The `[m, n]` output geometry a matmul dispatch takes.
#[derive(Debug, Clone, Copy)]
pub struct MatmulShape {
    /// Activation rows.
    pub m: usize,
    /// Shared dimension, a whole number of execution tiles.
    pub k: usize,
    /// Weight rows, and output columns.
    pub n: usize,
}

impl TcfShaderParams {
    /// Narrow `planes` and an output geometry into the uniform block.
    ///
    /// # Errors
    /// [`Error::QuantError`] when a plane offset, a tile count or a dimension
    /// exceeds `u32`.
    fn new(planes: TcfPlanes, encoding: TcfEncoding, at: MatmulShape) -> Result<Self> {
        let name = encoding.name();
        let narrow = |value: u64, label: &str| -> Result<u32> {
            u32::try_from(value).map_err(|_| Error::QuantError {
                reason: format!("{name}: {label}={value} exceeds the u32 a WGSL uniform carries"),
            })
        };
        let narrow_dim = |value: usize, label: &str| -> Result<u32> {
            u32::try_from(value).map_err(|_| Error::QuantError {
                reason: format!("{name}: {label}={value} exceeds u32"),
            })
        };

        Ok(Self {
            tiles: narrow(planes.tiles, "tiles")?,
            code_high_off: narrow(planes.code_high_off, "code_high_off")?,
            scale_off: narrow(planes.scale_off, "scale_off")?,
            min_off: narrow(planes.min_off, "min_off")?,
            super_off: narrow(planes.super_off, "super_off")?,
            super_min_off: narrow(planes.super_min_off, "super_min_off")?,
            bits: planes.bits,
            group: planes.group,
            groups_per_tile: planes.groups_per_tile,
            symmetric: planes.symmetric,
            scale_form: planes.scale_form,
            sub_block_bytes: planes.sub_block_bytes,
            m: narrow_dim(at.m, "M")?,
            k: narrow_dim(at.k, "K")?,
            n: narrow_dim(at.n, "N")?,
            zero_barrier: 0,
        })
    }
}

/// The registry buffer behind a device pointer, named when it is missing.
fn buffer(ptr: u64, label: &str) -> Result<Arc<Buffer>> {
    get_buffer(ptr).ok_or_else(|| Error::QuantError {
        reason: format!("TCF {label} buffer not found in WebGPU registry"),
    })
}

/// Upload `params` into a fresh uniform buffer.
fn params_buffer(client: &WgpuClient, params: &TcfShaderParams) -> Buffer {
    let buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("tcf_params"),
        size: std::mem::size_of::<TcfShaderParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&buf, 0, bytemuck::bytes_of(params));
    buf
}

/// Build the pipeline for `entry` and run it over `workgroups`.
fn run(
    client: &WgpuClient,
    entry: &'static str,
    source: &str,
    storage_buffers: u32,
    bindings: &[&Buffer],
    workgroups: (u32, u32),
) {
    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(entry, source);
    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: storage_buffers,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(entry, entry, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, bindings);

    let mut encoder = client
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
}

/// Dequantize a whole TCF payload into `output`, `product(shape)` f32.
///
/// One workgroup per super-block of four execution tiles.
///
/// # Errors
/// Every error [`TcfPlanes::new`] raises, plus [`Error::QuantError`] when a
/// plane offset exceeds `u32` or a buffer is missing from the registry.
pub fn dispatch_dequant(
    client: &WgpuClient,
    payload_ptr: u64,
    output_ptr: u64,
    encoding: TcfEncoding,
    shape: &[usize],
) -> Result<()> {
    let planes = TcfPlanes::new(encoding, shape)?;
    let at = MatmulShape { m: 0, k: 0, n: 0 };
    let params = TcfShaderParams::new(planes, encoding, at)?;
    if params.tiles == 0 {
        return Ok(());
    }

    let payload_buf = buffer(payload_ptr, "payload")?;
    let output_buf = buffer(output_ptr, "dequant output")?;
    let params_buf = params_buffer(client, &params);

    let workgroups = params.tiles.div_ceil(shader_gen::DEQUANT_TILES_PER_GROUP);
    run(
        client,
        shader_gen::DEQUANT_ENTRY,
        &shader_gen::generate_tcf_dequant_shader(),
        2,
        &[&payload_buf, &output_buf, &params_buf],
        (workgroups, 1),
    );
    Ok(())
}

/// `activation [M, K] x weight [N, K]^T -> output [M, N]`, one invocation per
/// output element.
///
/// `K` must be a whole number of execution tiles: the shader walks a weight row
/// tile by tile, and a partial trailing tile would read a neighbouring row's
/// codes.
///
/// # Errors
/// Every error [`dispatch_dequant`] raises, plus [`Error::QuantError`] when `K`
/// is not a positive multiple of the tile width.
pub fn dispatch_matmul(
    client: &WgpuClient,
    act_ptr: u64,
    weight_ptr: u64,
    output_ptr: u64,
    encoding: TcfEncoding,
    at: MatmulShape,
) -> Result<()> {
    let tile = encoding.tile();
    if tile == 0 || at.k == 0 || !at.k.is_multiple_of(tile) {
        return Err(Error::QuantError {
            reason: format!(
                "{}: K={} is not a positive multiple of the tile width {tile}",
                encoding.name(),
                at.k
            ),
        });
    }
    let planes = TcfPlanes::new(encoding, &[at.n, at.k])?;
    let params = TcfShaderParams::new(planes, encoding, at)?;
    if params.m == 0 || params.n == 0 {
        return Ok(());
    }

    let act_buf = buffer(act_ptr, "activation")?;
    let weight_buf = buffer(weight_ptr, "weight")?;
    let output_buf = buffer(output_ptr, "matmul output")?;
    let params_buf = params_buffer(client, &params);

    let edge = shader_gen::MATMUL_TILE;
    run(
        client,
        shader_gen::MATMUL_ENTRY,
        &shader_gen::generate_tcf_matmul_shader(),
        3,
        &[&act_buf, &weight_buf, &output_buf, &params_buf],
        (params.n.div_ceil(edge), params.m.div_ceil(edge)),
    );
    Ok(())
}
