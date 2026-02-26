//! CUDA implementation of DequantOps

use crate::error::{Error, Result};
use crate::quant::traits::DequantOps;
use crate::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::kernels::{self, DEQUANT_MODULE};
use super::nf4 as nf4_dispatch;

impl DequantOps<CudaRuntime> for CudaClient {
    fn nf4_dequant(
        &self,
        nf4_data: &Tensor<CudaRuntime>,
        absmax: &Tensor<CudaRuntime>,
        blocksize: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let num_bytes = nf4_data.numel();
        let n = num_bytes * 2;

        let output = Tensor::<CudaRuntime>::empty(&[n], DType::F32, nf4_data.device());
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
        let act_contig = input.contiguous();

        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(n_out);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, input.device());
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

        let kernel_name = match qt.format() {
            QuantFormat::Q4_0 => "dequant_q4_0_f32",
            QuantFormat::Q8_0 => "dequant_q8_0_f32",
            QuantFormat::Q4K => "dequant_q4_k_f32",
            QuantFormat::Q6K => "dequant_q6_k_f32",
            other => {
                return Err(Error::UnsupportedQuantFormat {
                    format: format!("{} (CUDA dequant not implemented)", other),
                });
            }
        };

        let num_blocks = qt.num_blocks();
        let device_index = qt.device().id();

        // Get input pointer (raw quant bytes on GPU)
        let input_ptr = qt.storage().ptr();

        // Allocate f32 output tensor
        let f32_out = Tensor::<CudaRuntime>::empty(qt.shape(), DType::F32, qt.device());
        let output_ptr = f32_out.ptr();

        // Load module and launch kernel
        let module = kernels::get_or_load_module(self.context(), device_index, DEQUANT_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let block_size = 256u32;
        let grid_size = (num_blocks as u32).div_ceil(block_size);
        let num_blocks_u32 = num_blocks as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
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
