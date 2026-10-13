//! [`MaybeRotatedLinear::row_logits_f32`]: the projection's outputs at
//! chosen weight rows per input row, computed in F32 without producing
//! the other outputs.

use super::layer::MaybeRotatedLinear;
use super::row_checks::{check_row_inputs, name_out_of_range, to_f32};
use crate::error::{Error, Result};
use crate::nn::linear::maybe_quant_linear::MaybeQuantLinear;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::dtype::DType;
use numr::ops::{BinaryOps, FwhtOps, IndexingOps, MatmulOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> MaybeRotatedLinear<R> {
    /// For each input row `q`, the outputs at weight rows `rows[q, ..]`:
    /// `out[q, k] = x[q] . W[rows[q, k]] + bias[rows[q, k]]`, F32 `[Q, K]`.
    ///
    /// - `input`: `[Q, in_features]`, a float dtype
    /// - `rows`: `[Q, K]` i64 output-row ids on `input`'s device,
    ///   `Q, K >= 1`; an id may repeat within and across input rows
    ///
    /// A `Rotated` projection rotates `input` first, in `input`'s dtype, as
    /// its full forward does. The input then goes to F32 and the dot
    /// products run in F32 over:
    ///
    /// - a dense weight: the gathered rows, cast to F32
    /// - a block-quantized weight: the gathered packed rows, dequantized
    ///   to F32
    /// - an AWQ/GPTQ weight: the F32 `int4_gemm` output of the `Q` input
    ///   rows, from which the chosen columns are taken. Its packing holds 8
    ///   output rows per word, so it has no row gather.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the projection carries a LoRA
    /// adapter, when a shape, dtype or device disagrees with the contract,
    /// or when an id lies outside `[0, out_features)`; the last names the
    /// first such id.
    pub fn row_logits_f32<C>(
        &self,
        client: &C,
        input: &Tensor<R>,
        rows: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + QuantMatmulOps<R> + DequantOps<R> + FwhtOps<R>,
    {
        let base_shape = self.base().shape();
        let (out_features, in_features) = (base_shape[0], base_shape[1]);
        let (q, k) = check_row_inputs(input, rows, in_features)?;
        let (base, input) = match self {
            Self::Plain(base) => (base, input.clone()),
            Self::Rotated(rotated) => (rotated.base(), rotated.rotation().forward(client, input)?),
            Self::Lora(_) => {
                return Err(Error::InvalidArgument {
                    arg: "self",
                    reason: "row_logits_f32: the projection carries a LoRA adapter, which has \
                             no row path"
                        .into(),
                });
            }
        };
        let input = to_f32(client, &input)?;
        let rows = rows.contiguous()?;
        let flat = rows.reshape(&[q * k])?;
        let out_of_range =
            |err: Error| name_out_of_range(err, &rows, out_features, "rows", "row id");

        match base {
            MaybeQuantLinear::Standard(linear) => {
                let weight = linear.weight().tensor();
                let picked = client
                    .index_select(weight, 0, &flat)
                    .map_err(|e| out_of_range(e.into()))?;
                let logits = dot_rows(client, &input, &to_f32(client, &picked)?, q, k)?;
                add_bias(
                    client,
                    logits,
                    linear.bias().map(|b| b.tensor()),
                    &flat,
                    q,
                    k,
                )
            }
            MaybeQuantLinear::Quantized(qlinear) => {
                let picked = qlinear
                    .weight()
                    .gather_rows(client, &flat)
                    .map_err(out_of_range)?;
                let picked = client.dequantize(&picked, DType::F32)?;
                let logits = dot_rows(client, &input, &picked, q, k)?;
                add_bias(client, logits, qlinear.bias(), &flat, q, k)
            }
            MaybeQuantLinear::DecomposedQuant(dqlinear) => {
                // F32 input keeps the output in F32; the bias is added here.
                let full = dqlinear.forward(client, &input)?;
                // `gather` does not bound its ids, so `index_select` over one
                // output row checks them on the device first.
                let first = full.narrow(0, 0, 1)?;
                client
                    .index_select(&first, 1, &flat)
                    .map_err(|e| out_of_range(e.into()))?;
                Ok(client.gather(&full, 1, &rows)?)
            }
        }
    }
}

/// `[Q, K]`: input row `q` `[H]` dotted with weight rows `q * K .. q * K + K`
/// of `weight_rows` `[Q * K, H]`, one batched F32 matmul.
fn dot_rows<R, C>(
    client: &C,
    input: &Tensor<R>,
    weight_rows: &Tensor<R>,
    q: usize,
    k: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: MatmulOps<R>,
{
    let h = input.shape()[1];
    let w = weight_rows.contiguous()?.reshape(&[q, k, h])?;
    let x = input.contiguous()?.reshape(&[q, h, 1])?;
    Ok(client.matmul(&w, &x)?.reshape(&[q, k])?)
}

/// `logits + bias[flat]` reshaped to `[Q, K]`, or `logits` without a bias.
/// `flat` is already bounds-checked by the weight gather.
fn add_bias<R, C>(
    client: &C,
    logits: Tensor<R>,
    bias: Option<&Tensor<R>>,
    flat: &Tensor<R>,
    q: usize,
    k: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R> + TypeConversionOps<R> + BinaryOps<R>,
{
    let Some(bias) = bias else {
        return Ok(logits);
    };
    let picked = client.index_select(bias, 0, flat)?;
    let picked = to_f32(client, &picked)?.reshape(&[q, k])?;
    Ok(client.add(&logits, &picked)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::linear::dense::Linear;
    use crate::nn::linear::quant_linear::QuantLinear;
    use crate::nn::linear::rotated_linear::RotatedLinear;
    use crate::quant::traits::QuantizeOps;
    use crate::quant::{DecomposedQuantLinear, DecomposedQuantMethod, DecomposedQuantTensor};
    use crate::quant::{QuantFormat, QuantTensor};
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const IN: usize = 32;
    const OUT: usize = 16;
    const Q: usize = 3;
    const K: usize = 4;
    /// Ids repeat within a row and across rows.
    const ROWS: [i64; Q * K] = [0, 15, 7, 7, 3, 0, 9, 15, 12, 1, 1, 5];

    fn tensor(device: &CpuDevice, shape: &[usize], seed: f32) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.3)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    fn rows(device: &CpuDevice, ids: &[i64]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(ids, &[Q, ids.len() / Q], device).unwrap()
    }

    /// `full[q, ROWS[q, k]]` from a `[Q, OUT]` output.
    fn pick(full: &Tensor<CpuRuntime>) -> Vec<f32> {
        let full = full.to_vec::<f32>();
        (0..Q * K)
            .map(|i| full[(i / K) * OUT + ROWS[i] as usize])
            .collect()
    }

    /// `x @ dequant(w)^T`, the whole weight dequantized: the reference a
    /// row gather must match.
    fn dequant_matmul(
        client: &CpuClient,
        x: &Tensor<CpuRuntime>,
        w: &QuantTensor<CpuRuntime>,
    ) -> Tensor<CpuRuntime> {
        let w = client.dequantize(w, DType::F32).unwrap();
        let w_t = w.t().unwrap().contiguous().unwrap();
        client.matmul(x, &w_t).unwrap()
    }

    fn assert_close(got: &Tensor<CpuRuntime>, want: &[f32]) {
        assert_eq!(got.shape(), &[Q, K]);
        assert_eq!(got.dtype(), DType::F32);
        for (i, (g, w)) in got.to_vec::<f32>().iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-5, "entry {i}: {g} vs {w}");
        }
    }

    fn rotation(device: &CpuDevice) -> HadamardRotation<CpuRuntime> {
        let signs: Vec<i8> = (0..IN).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        HadamardRotation::new(8, Some(&signs), DType::F32, device).unwrap()
    }

    #[test]
    fn dense_with_bias_matches_full_forward() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[OUT, IN], 0.37);
        let b = tensor(&device, &[OUT], 0.91);
        let layer: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::Standard(Linear::new(w, Some(b), false)).into();
        let x = tensor(&device, &[Q, IN], 0.53);
        let full = layer.forward(&client, &Var::new(x.clone(), false)).unwrap();
        let got = layer
            .row_logits_f32(&client, &x, &rows(&device, &ROWS))
            .unwrap();
        assert_close(&got, &pick(full.tensor()));
    }

    #[test]
    fn quantized_matches_whole_weight_dequant() {
        let (client, device) = cpu_setup();
        let quant = client
            .quantize(&tensor(&device, &[OUT, IN], 0.41), QuantFormat::Q8_0)
            .unwrap();
        let x = tensor(&device, &[Q, IN], 0.67);
        let want = pick(&dequant_matmul(&client, &x, &quant));
        let layer: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::Quantized(QuantLinear::new(quant, None)).into();
        let got = layer
            .row_logits_f32(&client, &x, &rows(&device, &ROWS))
            .unwrap();
        assert_close(&got, &want);
    }

    /// A rotated quantized head rotates the input, then reads dequantized
    /// rows.
    #[test]
    fn rotated_quantized_matches_rotate_then_dequant() {
        let (client, device) = cpu_setup();
        let quant = client
            .quantize(&tensor(&device, &[OUT, IN], 0.29), QuantFormat::Q8_0)
            .unwrap();
        let x = tensor(&device, &[Q, IN], 0.73);
        let rotated_x = rotation(&device).forward(&client, &x).unwrap();
        let want = pick(&dequant_matmul(&client, &rotated_x, &quant));
        let base = MaybeQuantLinear::Quantized(QuantLinear::new(quant, None));
        let layer: MaybeRotatedLinear<CpuRuntime> =
            RotatedLinear::new(base, rotation(&device)).unwrap().into();
        let got = layer
            .row_logits_f32(&client, &x, &rows(&device, &ROWS))
            .unwrap();
        assert_close(&got, &want);
    }

    #[test]
    fn decomposed_matches_int4_gemm_output() {
        let (client, device) = cpu_setup();
        let words: Vec<u32> = (0..IN * OUT / 8)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let qweight = Tensor::<CpuRuntime>::from_slice(&words, &[IN, OUT / 8], &device).unwrap();
        let scales: Vec<f32> = (0..OUT).map(|i| 0.01 + 0.002 * i as f32).collect();
        let scales = Tensor::<CpuRuntime>::from_slice(&scales, &[1, OUT], &device).unwrap();
        let qzeros = Tensor::<CpuRuntime>::from_slice(&[8.0f32; OUT], &[1, OUT], &device).unwrap();
        let weight = DecomposedQuantTensor::new(
            qweight,
            scales,
            qzeros,
            None,
            DecomposedQuantMethod::Awq { group_size: IN },
            vec![OUT, IN],
        );
        let bias = tensor(&device, &[OUT], 0.19);
        let dq = DecomposedQuantLinear::new(weight, Some(bias));
        let x = tensor(&device, &[Q, IN], 0.47);
        let want = pick(&dq.forward(&client, &x).unwrap());
        let layer: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::DecomposedQuant(Box::new(dq)).into();
        let got = layer
            .row_logits_f32(&client, &x, &rows(&device, &ROWS))
            .unwrap();
        assert_close(&got, &want);

        let mut past_end = ROWS;
        past_end[5] = OUT as i64;
        let err = layer
            .row_logits_f32(&client, &x, &rows(&device, &past_end))
            .unwrap_err()
            .to_string();
        assert!(err.contains("row id 16 at flat position 5"), "{err}");
    }

    #[test]
    fn out_of_range_ids_are_named() {
        let (client, device) = cpu_setup();
        let layer: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::Standard(Linear::new(tensor(&device, &[OUT, IN], 0.3), None, false))
                .into();
        let x = tensor(&device, &[Q, IN], 0.5);
        for (pos, value) in [(2usize, OUT as i64), (7, -1)] {
            let mut ids = ROWS;
            ids[pos] = value;
            let err = layer
                .row_logits_f32(&client, &x, &rows(&device, &ids))
                .unwrap_err()
                .to_string();
            let want = format!("row id {value} at flat position {pos} is outside [0, {OUT})");
            assert!(err.contains(&want), "{err}");
        }
    }

    #[test]
    fn bad_rows_and_lora_are_refused() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[OUT, IN], 0.3);
        let mut layer: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::Standard(Linear::new(w, None, false)).into();
        let x = tensor(&device, &[Q, IN], 0.5);
        let two_rows = Tensor::<CpuRuntime>::from_slice(&ROWS[..8], &[2, K], &device).unwrap();
        assert!(layer.row_logits_f32(&client, &x, &two_rows).is_err());

        let a = tensor(&device, &[2, IN], 0.7);
        let b = tensor(&device, &[OUT, 2], 0.9);
        layer.attach_adapter(a, b, 1.0).unwrap();
        let err = layer
            .row_logits_f32(&client, &x, &rows(&device, &ROWS))
            .unwrap_err()
            .to_string();
        assert!(err.contains("LoRA"), "{err}");
    }
}
