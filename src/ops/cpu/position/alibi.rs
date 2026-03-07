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

    fn alibi_add_bias_causal(
        &self,
        scores: &Tensor<CpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        position: usize,
    ) -> Result<()> {
        let ptr = scores.ptr() as *mut f32;

        unsafe {
            for b in 0..batch_size {
                for h in 0..num_heads {
                    let slope = 2.0f32.powf(-8.0 * h as f32 / num_heads as f32);
                    for qi in 0..seq_len_q {
                        let abs_qi = qi + position;
                        for ki in 0..seq_len_k {
                            let idx = ((b * num_heads + h) * seq_len_q + qi) * seq_len_k + ki;
                            if ki > abs_qi {
                                *ptr.add(idx) = f32::NEG_INFINITY;
                            } else {
                                let distance = (abs_qi - ki) as f32;
                                *ptr.add(idx) += -slope * distance;
                            }
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

    #[test]
    fn test_alibi_bias_causal() {
        let (client, dev) = cpu_setup();
        // 1 batch, 2 heads, 1 query token, 4 KV tokens, position=2
        let (b, h, sq, sk) = (1, 2, 1, 4);
        let position = 2;

        let data = vec![0.0f32; b * h * sq * sk];
        let scores = Tensor::<CpuRuntime>::from_slice(&data, &[b, h, sq, sk], &dev);

        client
            .alibi_add_bias_causal(&scores, b, h, sq, sk, position)
            .unwrap();

        let result = scores.to_vec::<f32>();
        // Head 0: slope = 1.0, query at abs position 2
        // ki=0: bias = -1.0 * 2 = -2.0
        // ki=1: bias = -1.0 * 1 = -1.0
        // ki=2: bias = -1.0 * 0 = 0.0
        // ki=3: causal mask → -inf (3 > 2)
        assert!((result[0] - (-2.0)).abs() < 1e-6);
        assert!((result[1] - (-1.0)).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
        assert!(result[3] == f32::NEG_INFINITY);
    }
}
