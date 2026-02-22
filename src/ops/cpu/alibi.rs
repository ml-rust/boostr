//! CPU implementation of ALiBi attention bias

use crate::error::Result;
use crate::ops::traits::AlibiOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl AlibiOps<CpuRuntime> for CpuClient {
    fn alibi_add_bias(
        &self,
        scores: &Tensor<CpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
    ) -> Result<()> {
        let ptr = scores.ptr() as *mut f32;

        unsafe {
            for b in 0..batch_size {
                for h in 0..num_heads {
                    let slope = 2.0f32.powf(-8.0 * h as f32 / num_heads as f32);
                    for qi in 0..seq_len_q {
                        for ki in 0..seq_len_k {
                            let idx = ((b * num_heads + h) * seq_len_q + qi) * seq_len_k + ki;
                            let bias = -slope * (qi as f32 - ki as f32).abs();
                            *ptr.add(idx) += bias;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn test_alibi_bias() {
        let (client, dev) = cpu_setup();
        let (b, h, sq, sk) = (1, 2, 3, 3);

        let data = vec![0.0f32; b * h * sq * sk];
        let scores = Tensor::<CpuRuntime>::from_slice(&data, &[b, h, sq, sk], &dev);

        client.alibi_add_bias(&scores, b, h, sq, sk).unwrap();

        let result = scores.to_vec::<f32>();
        // Head 0: slope = 2^0 = 1.0, diagonal (i==j) should be 0
        assert!((result[0] - 0.0).abs() < 1e-6); // [0,0] bias = -1*|0-0| = 0
        assert!((result[1] - (-1.0)).abs() < 1e-6); // [0,1] bias = -1*|0-1| = -1
        assert!((result[4] - 0.0).abs() < 1e-6); // [1,1] bias = -1*|1-1| = 0
    }
}
