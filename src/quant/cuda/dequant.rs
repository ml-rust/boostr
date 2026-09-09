//! CUDA implementation of DequantOps

use crate::error::{Error, Result};
use crate::quant::traits::DequantOps;
use crate::quant::{QuantFormat, QuantScheme, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::kernels::{self, DEQUANT_GENERIC_MODULE, DEQUANT_MODULE};
use super::nf4 as nf4_dispatch;
use super::tcf as tcf_dispatch;

/// Threads one dequantization CUDA block runs, for every kernel in
/// `kernels/dequant.cu`.
const DEQUANT_BLOCK: u32 = 256;

/// Output elements one dequant thread writes, matching the `*_ELEMS_PER_THREAD`
/// macros in `kernels/dequant.cu`. Every kernel there uses the same
/// four-element run so one `float4` store covers a thread's whole output.
const DEQUANT_ELEMS_PER_THREAD: u32 = 4;

/// Output elements one 256-element super-block holds: every K-quant, and the
/// IQ formats that share their super-block size.
const SUPER_ELEMS_PER_BLOCK: u32 = 256;

/// Output elements one 32-element GGUF block holds: Q4_0, Q5_0, Q8_0, IQ4_NL.
const BLOCK32_ELEMS_PER_BLOCK: u32 = 32;

/// Grid mapping for the super-block kernels: one thread per four-element run
/// of a 256-element super-block.
const SUPER_MAPPING: DequantMapping = DequantMapping {
    elems_per_thread: DEQUANT_ELEMS_PER_THREAD,
    elems_per_block: SUPER_ELEMS_PER_BLOCK,
};

/// Grid mapping for the 32-element block kernels: one thread per four-element
/// run of a 32-element block.
const BLOCK32_MAPPING: DequantMapping = DequantMapping {
    elems_per_thread: DEQUANT_ELEMS_PER_THREAD,
    elems_per_block: BLOCK32_ELEMS_PER_BLOCK,
};

/// What one thread of a dequantization kernel owns: `elems_per_thread`
/// consecutive output elements of a quantized block holding `elems_per_block`
/// of them.
///
/// The grid follows the ELEMENT count, not the block count. Sizing an
/// element-mapped kernel's grid by block count launches a fraction of the
/// threads it needs, so each kernel names its mapping at the dispatch site and
/// the grid formula reads it from there.
#[derive(Debug, Clone, Copy)]
struct DequantMapping {
    elems_per_thread: u32,
    elems_per_block: u32,
}

impl DequantMapping {
    /// CUDA blocks of [`DEQUANT_BLOCK`] threads needed to cover `num_blocks`
    /// quantized blocks.
    ///
    /// # Errors
    /// [`Error::QuantError`] when the grid does not fit in `u32`.
    fn grid(self, num_blocks: usize, kernel_name: &str) -> Result<u32> {
        let elements = num_blocks as u64 * u64::from(self.elems_per_block);
        let threads = elements.div_ceil(u64::from(self.elems_per_thread));
        let grid = threads.div_ceil(u64::from(DEQUANT_BLOCK));
        u32::try_from(grid).map_err(|_| Error::QuantError {
            reason: format!("{kernel_name}: grid of {grid} blocks exceeds u32"),
        })
    }
}

impl DequantOps<CudaRuntime> for CudaClient {
    fn nf4_dequant(
        &self,
        nf4_data: &Tensor<CudaRuntime>,
        absmax: &Tensor<CudaRuntime>,
        blocksize: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let num_bytes = nf4_data.numel();
        let n = num_bytes * 2;

        let output = Tensor::<CudaRuntime>::empty(&[n], DType::F32, nf4_data.device())?;
        nf4_dispatch::launch_nf4_dequant(
            self,
            nf4_data,
            absmax,
            &output,
            num_bytes as u32,
            blocksize as u32,
        )?;
        Ok(output)
    }

    fn nf4_gemm(
        &self,
        input: &Tensor<CudaRuntime>,
        nf4_weight: &Tensor<CudaRuntime>,
        absmax: &Tensor<CudaRuntime>,
        n_out: usize,
        k: usize,
        blocksize: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        if input.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!("nf4_gemm input must be F32, got {:?}", input.dtype()),
            });
        }
        let in_shape = input.shape();
        let m: usize = in_shape.iter().product::<usize>() / k;
        let act_contig = input.contiguous()?;

        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(n_out);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, input.device())?;
        nf4_dispatch::launch_nf4_gemm(
            self,
            &act_contig,
            nf4_weight,
            absmax,
            &output,
            m as u32,
            k as u32,
            n_out as u32,
            blocksize as u32,
        )?;
        Ok(output)
    }

    fn dequantize(
        &self,
        qt: &QuantTensor<CudaRuntime>,
        target_dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(
            target_dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::F64
        ) {
            return Err(Error::QuantError {
                reason: format!("dequantize target must be float, got {:?}", target_dtype),
            });
        }

        // TCF is plane-major and its two-level encodings carry a second scale
        // level, so it has its own kernel rather than a `QuantFormat` arm.
        let format = match qt.scheme() {
            QuantScheme::Gguf(format) => format,
            QuantScheme::Tcf(encoding) => {
                return dequant_tcf(self, qt, encoding, target_dtype);
            }
        };

        let (kernel_name, mapping) = match format {
            QuantFormat::Q4_0 => ("dequant_q4_0_f32", BLOCK32_MAPPING),
            QuantFormat::Q5_0 => ("dequant_q5_0_f32", BLOCK32_MAPPING),
            QuantFormat::Q8_0 => ("dequant_q8_0_f32", BLOCK32_MAPPING),
            QuantFormat::Q2K => ("dequant_q2_k_f32", SUPER_MAPPING),
            QuantFormat::Q3K => ("dequant_q3_k_f32", SUPER_MAPPING),
            QuantFormat::Q4K => ("dequant_q4_k_f32", SUPER_MAPPING),
            QuantFormat::Q5K => ("dequant_q5_k_f32", SUPER_MAPPING),
            QuantFormat::Q6K => ("dequant_q6_k_f32", SUPER_MAPPING),
            QuantFormat::IQ4NL => ("dequant_iq4_nl_f32", BLOCK32_MAPPING),
            QuantFormat::IQ4XS => ("dequant_iq4_xs_f32", SUPER_MAPPING),
            QuantFormat::IQ3S => ("dequant_iq3_s_f32", SUPER_MAPPING),
            QuantFormat::IQ2XS => ("dequant_iq2_xs_f32", SUPER_MAPPING),
            // All other formats: use generic dequant kernel (format dispatch via switch)
            _ => {
                return dequant_via_generic_kernel(self, qt, target_dtype);
            }
        };

        let num_blocks = qt.num_blocks();
        let device_index = qt.device().id();

        // Get input pointer (raw quant bytes on GPU)
        let input_ptr = qt.storage().ptr();

        // Allocate f32 output tensor
        let f32_out = Tensor::<CudaRuntime>::empty(qt.shape(), DType::F32, qt.device())?;
        let output_ptr = f32_out.ptr();

        // Load module and launch kernel
        let module = kernels::get_or_load_module(self.context(), device_index, DEQUANT_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let grid_size = mapping.grid(num_blocks, kernel_name)?;
        let num_blocks_u32 = u32::try_from(num_blocks).map_err(|_| Error::QuantError {
            reason: format!("{kernel_name}: {num_blocks} blocks exceed u32"),
        })?;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (DEQUANT_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&num_blocks_u32);
            builder.launch(cfg).map_err(|e| Error::QuantError {
                reason: format!("CUDA dequant kernel launch failed: {:?}", e),
            })?;
        }

        if target_dtype == DType::F32 {
            Ok(f32_out)
        } else {
            self.cast(&f32_out, target_dtype).map_err(Error::Numr)
        }
    }
}

/// Dequantize a TCF native quantized payload on the GPU.
///
/// The device decoder in `kernels/tcf.cuh` reproduces `tcf_core::unpack`
/// followed by `tcf_core::dequantize`, and `tests/backend_parity/quant_tcf.rs`
/// is the gate that says so for every encoding.
fn dequant_tcf(
    client: &CudaClient,
    qt: &QuantTensor<CudaRuntime>,
    encoding: crate::quant::TcfEncoding,
    target_dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    let f32_out = Tensor::<CudaRuntime>::empty(qt.shape(), DType::F32, qt.device())?;
    tcf_dispatch::launch_dequant(
        client,
        qt.device().id(),
        qt.storage().ptr(),
        f32_out.ptr(),
        encoding,
        qt.shape(),
    )?;

    if target_dtype == DType::F32 {
        Ok(f32_out)
    } else {
        client.cast(&f32_out, target_dtype).map_err(Error::Numr)
    }
}

/// Dequantize using the generic CUDA kernel that handles all 23 formats
/// via format_id dispatch. Slower than dedicated kernels but universally correct.
fn dequant_via_generic_kernel(
    client: &CudaClient,
    qt: &QuantTensor<CudaRuntime>,
    target_dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    let num_blocks = qt.num_blocks();
    let device_index = qt.device().id();

    let input_ptr = qt.storage().ptr();
    let f32_out = Tensor::<CudaRuntime>::empty(qt.shape(), DType::F32, qt.device())?;
    let output_ptr = f32_out.ptr();

    let module =
        kernels::get_or_load_module(client.context(), device_index, DEQUANT_GENERIC_MODULE)?;
    let func = kernels::get_kernel_function(&module, "dequant_generic_f32")?;

    let threads = 256u32;
    let num_blocks_u32 = u32::try_from(num_blocks).map_err(|_| Error::QuantError {
        reason: format!("dequant_generic_f32: {num_blocks} blocks exceed u32"),
    })?;
    let grid_size = num_blocks_u32.div_ceil(threads);
    let format = qt.format()?;
    let format_id = format.format_id();

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&num_blocks_u32);
        builder.arg(&format_id);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!(
                "CUDA dequant_generic kernel launch failed for {}: {:?}",
                format, e
            ),
        })?;
    }

    if target_dtype == DType::F32 {
        Ok(f32_out)
    } else {
        client.cast(&f32_out, target_dtype).map_err(Error::Numr)
    }
}
