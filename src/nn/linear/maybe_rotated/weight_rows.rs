//! [`MaybeRotatedLinear::weight_rows_f32`]: chosen rows of the projection's
//! effective weight in F32, the vectors its outputs are dot products with.

use super::layer::MaybeRotatedLinear;
use super::row_checks::{name_out_of_range, to_f32};
use crate::error::{Error, Result};
use crate::nn::linear::maybe_quant_linear::MaybeQuantLinear;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use crate::quant::{DecomposedQuantLinear, DecomposedQuantMethod, DecomposedQuantTensor};
use numr::dtype::DType;
use numr::ops::{FwhtOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Identity rows run through an AWQ/GPTQ sub-weight per forward call.
const IDENTITY_CHUNK: usize = 512;

/// Output rows one packed word of an AWQ/GPTQ weight holds.
const PACK: usize = 8;

fn invalid(reason: String) -> Error {
    Error::InvalidArgument { arg: "ids", reason }
}

impl<R: Runtime<DType = DType>> MaybeRotatedLinear<R> {
    /// Rows `ids` of the effective weight, F32 `[ids.len(), in_features]`:
    /// row `n` is the vector an input row is dotted with to give output
    /// `ids[n]`, bias aside. An id may repeat.
    ///
    /// - a dense weight: the gathered rows, cast to F32
    /// - a block-quantized weight: the gathered packed rows, dequantized
    /// - an AWQ/GPTQ weight: the packed words holding the ids, taken as a
    ///   sub-weight and run over identity inputs, so the int4 GEMM itself
    ///   dequantizes them
    ///
    /// A `Rotated` projection rotates its input before the stored weight,
    /// so its effective row is the inverse rotation of the stored row.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `ids` is empty, when an id lies
    /// outside `[0, out_features)` (naming the first such id and its
    /// position), or when the projection carries a LoRA adapter. Device
    /// errors propagate.
    pub fn weight_rows_f32<C>(&self, client: &C, ids: &[usize]) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + QuantMatmulOps<R> + DequantOps<R> + FwhtOps<R>,
    {
        if ids.is_empty() {
            return Err(invalid("no ids: at least one weight row is read".into()));
        }
        let out_features = self.base().shape()[0];
        if let Some(n) = ids.iter().position(|&id| id >= out_features) {
            return Err(invalid(format!(
                "id {} at position {n} is outside [0, {out_features})",
                ids[n]
            )));
        }
        let (base, rotation) = match self {
            Self::Plain(base) => (base, None),
            Self::Rotated(rotated) => (rotated.base(), Some(rotated.rotation())),
            Self::Lora(_) => {
                return Err(invalid(
                    "the projection carries a LoRA adapter, whose rows are not read".into(),
                ));
            }
        };
        let device = client.device();
        let flat: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let index = Tensor::<R>::from_slice(&flat, &[flat.len()], device)?;
        let out_of_range = |err: Error| name_out_of_range(err, &index, out_features, "ids", "id");
        let rows = match base {
            MaybeQuantLinear::Standard(linear) => {
                let picked = client
                    .index_select(linear.weight().tensor(), 0, &index)
                    .map_err(|e| out_of_range(e.into()))?;
                to_f32(client, &picked)?
            }
            MaybeQuantLinear::Quantized(qlinear) => {
                let picked = qlinear
                    .weight()
                    .gather_rows(client, &index)
                    .map_err(out_of_range)?;
                client.dequantize(&picked, DType::F32)?
            }
            MaybeQuantLinear::DecomposedQuant(dq) => decomposed_rows(client, dq.weight(), ids)?,
        };
        match rotation {
            None => Ok(rows),
            Some(rotation) => rotation.inverse_cast(client, &rows),
        }
    }
}

/// F32 rows `ids` of an AWQ/GPTQ weight.
///
/// Both packings hold output rows in words of [`PACK`]: AWQ packs `qweight`
/// columns, GPTQ packs `qzeros` columns. The words holding `ids` and the
/// matching unpacked columns form a sub-weight of `PACK * words` output
/// rows, its forward over identity rows is that sub-weight transposed, and
/// the ids' columns of it are the rows.
fn decomposed_rows<R, C>(
    client: &C,
    weight: &DecomposedQuantTensor<R>,
    ids: &[usize],
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + QuantMatmulOps<R>,
{
    let device = client.device();
    let in_features = weight.shape()[1];
    let out_features = weight.shape()[0];
    let mut words: Vec<usize> = ids.iter().map(|&id| id / PACK).collect();
    words.sort_unstable();
    words.dedup();
    if let Some(&last) = words.last()
        && (last + 1) * PACK > out_features
    {
        return Err(invalid(format!(
            "an AWQ/GPTQ weight of {out_features} output rows packs {PACK} rows per word; \
             word {last} runs past it"
        )));
    }
    let word_index: Vec<i64> = words.iter().map(|&w| w as i64).collect();
    let column_index: Vec<i64> = words
        .iter()
        .flat_map(|&w| (w * PACK..(w + 1) * PACK).map(|c| c as i64))
        .collect();
    let word_index = Tensor::<R>::from_slice(&word_index, &[word_index.len()], device)?;
    let column_index = Tensor::<R>::from_slice(&column_index, &[column_index.len()], device)?;
    let scales = client.index_select(&weight.scales, 1, &column_index)?;
    let (qweight, qzeros) = match weight.method {
        DecomposedQuantMethod::Awq { .. } => (
            client.index_select(&weight.qweight, 1, &word_index)?,
            client.index_select(&weight.qzeros, 1, &column_index)?,
        ),
        DecomposedQuantMethod::Gptq { .. } => (
            client.index_select(&weight.qweight, 1, &column_index)?,
            client.index_select(&weight.qzeros, 1, &word_index)?,
        ),
    };
    let sub_rows = words.len() * PACK;
    let sub = DecomposedQuantLinear::new(
        DecomposedQuantTensor::new(
            qweight,
            scales,
            qzeros,
            weight.g_idx.clone(),
            weight.method,
            vec![sub_rows, in_features],
        ),
        None,
    );

    let mut parts = Vec::with_capacity(in_features.div_ceil(IDENTITY_CHUNK));
    for start in (0..in_features).step_by(IDENTITY_CHUNK) {
        let rows = IDENTITY_CHUNK.min(in_features - start);
        let mut eye = vec![0.0f32; rows * in_features];
        for r in 0..rows {
            eye[r * in_features + start + r] = 1.0;
        }
        let eye = Tensor::<R>::from_slice(&eye, &[rows, in_features], device)?;
        parts.push(sub.forward(client, &eye)?);
    }
    let refs: Vec<&Tensor<R>> = parts.iter().collect();
    let transposed = client.cat(&refs, 0)?;

    let picks: Vec<i64> = ids
        .iter()
        .map(|&id| {
            let rank = words.partition_point(|&w| w < id / PACK);
            (rank * PACK + id % PACK) as i64
        })
        .collect();
    let picks = Tensor::<R>::from_slice(&picks, &[picks.len()], device)?;
    let columns = client.index_select(&transposed, 1, &picks)?;
    Ok(columns.transpose(0, 1)?.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::linear::dense::Linear;
    use crate::nn::linear::quant_linear::QuantLinear;
    use crate::nn::linear::rotated_linear::RotatedLinear;
    use crate::quant::QuantFormat;
    use crate::quant::traits::QuantizeOps;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const IN: usize = 32;
    const OUT: usize = 32;
    /// Ids repeat and fall in three of the four packed words.
    const IDS: [usize; 6] = [0, 31, 7, 7, 12, 30];

    fn tensor(device: &CpuDevice, shape: &[usize], seed: f32) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.3)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    /// `row_logits_f32` at `IDS` for one input row against the rows dotted
    /// with it, bias removed by `bias`.
    fn check(client: &CpuClient, device: &CpuDevice, layer: &MaybeRotatedLinear<CpuRuntime>) {
        let rows = layer.weight_rows_f32(client, &IDS).unwrap();
        assert_eq!(rows.shape(), &[IDS.len(), IN]);
        assert_eq!(rows.dtype(), DType::F32);
        let x = tensor(device, &[1, IN], 0.61);
        let ids: Vec<i64> = IDS.iter().map(|&i| i as i64).collect();
        let ids = Tensor::<CpuRuntime>::from_slice(&ids, &[1, IDS.len()], device).unwrap();
        let want = layer
            .row_logits_f32(client, &x, &ids)
            .unwrap()
            .to_vec::<f32>();
        let bias = match layer.base() {
            MaybeQuantLinear::Standard(l) => l.bias().map(|b| b.tensor().to_vec::<f32>()),
            MaybeQuantLinear::Quantized(q) => q.bias().map(|b| b.to_vec::<f32>()),
            MaybeQuantLinear::DecomposedQuant(d) => d.bias().map(|b| b.to_vec::<f32>()),
        };
        let x = x.to_vec::<f32>();
        let rows = rows.to_vec::<f32>();
        for (n, &id) in IDS.iter().enumerate() {
            let dot: f32 = (0..IN).map(|i| rows[n * IN + i] * x[i]).sum();
            let b = bias.as_ref().map_or(0.0, |b| b[id]);
            assert!(
                (dot + b - want[n]).abs() < 1e-4,
                "id {id}: {} vs {}",
                dot + b,
                want[n]
            );
        }
    }

    fn rotation(device: &CpuDevice) -> HadamardRotation<CpuRuntime> {
        let signs: Vec<i8> = (0..IN).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        HadamardRotation::new(8, Some(&signs), DType::F32, device).unwrap()
    }

    #[test]
    fn dense_rows_match_row_logits() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[OUT, IN], 0.37);
        let b = tensor(&device, &[OUT], 0.91);
        let layer: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::Standard(Linear::new(w.clone(), Some(b), false)).into();
        check(&client, &device, &layer);
        let rows = layer
            .weight_rows_f32(&client, &IDS)
            .unwrap()
            .to_vec::<f32>();
        let w = w.to_vec::<f32>();
        for (n, &id) in IDS.iter().enumerate() {
            assert_eq!(&rows[n * IN..(n + 1) * IN], &w[id * IN..(id + 1) * IN]);
        }
    }

    #[test]
    fn quantized_and_rotated_rows_match_row_logits() {
        let (client, device) = cpu_setup();
        let quant = |seed| {
            client
                .quantize(&tensor(&device, &[OUT, IN], seed), QuantFormat::Q8_0)
                .unwrap()
        };
        let plain: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::Quantized(QuantLinear::new(quant(0.41), None)).into();
        check(&client, &device, &plain);
        let base = MaybeQuantLinear::Quantized(QuantLinear::new(quant(0.29), None));
        let rotated: MaybeRotatedLinear<CpuRuntime> =
            RotatedLinear::new(base, rotation(&device)).unwrap().into();
        check(&client, &device, &rotated);
        let dense =
            MaybeQuantLinear::Standard(Linear::new(tensor(&device, &[OUT, IN], 0.23), None, false));
        let rotated_dense: MaybeRotatedLinear<CpuRuntime> =
            RotatedLinear::new(dense, rotation(&device)).unwrap().into();
        check(&client, &device, &rotated_dense);
    }

    fn words(n: usize, salt: u32) -> Vec<u32> {
        (0..n as u32)
            .map(|i| (i ^ salt).wrapping_mul(2_654_435_761))
            .collect()
    }

    #[test]
    fn awq_rows_match_row_logits() {
        let (client, device) = cpu_setup();
        let qweight = Tensor::<CpuRuntime>::from_slice(
            &words(IN * OUT / PACK, 0),
            &[IN, OUT / PACK],
            &device,
        )
        .unwrap();
        let scales: Vec<f32> = (0..2 * OUT).map(|i| 0.01 + 0.002 * i as f32).collect();
        let scales = Tensor::<CpuRuntime>::from_slice(&scales, &[2, OUT], &device).unwrap();
        let zeros: Vec<f32> = (0..2 * OUT).map(|i| (i % 5) as f32 + 6.0).collect();
        let zeros = Tensor::<CpuRuntime>::from_slice(&zeros, &[2, OUT], &device).unwrap();
        let weight = DecomposedQuantTensor::new(
            qweight,
            scales,
            zeros,
            None,
            DecomposedQuantMethod::Awq { group_size: IN / 2 },
            vec![OUT, IN],
        );
        let dq = DecomposedQuantLinear::new(weight, Some(tensor(&device, &[OUT], 0.19)));
        let layer: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::DecomposedQuant(Box::new(dq)).into();
        check(&client, &device, &layer);
    }

    #[test]
    fn gptq_rows_match_row_logits() {
        let (client, device) = cpu_setup();
        let qweight = Tensor::<CpuRuntime>::from_slice(
            &words(IN * OUT / PACK, 7),
            &[IN / PACK, OUT],
            &device,
        )
        .unwrap();
        let qzeros =
            Tensor::<CpuRuntime>::from_slice(&words(2 * OUT / PACK, 3), &[2, OUT / PACK], &device)
                .unwrap();
        let scales: Vec<f32> = (0..2 * OUT).map(|i| 0.02 + 0.001 * i as f32).collect();
        let scales = Tensor::<CpuRuntime>::from_slice(&scales, &[2, OUT], &device).unwrap();
        let g_idx: Vec<i32> = (0..IN).map(|i| (i * 2 / IN) as i32).collect();
        let g_idx = Tensor::<CpuRuntime>::from_slice(&g_idx, &[IN], &device).unwrap();
        let weight = DecomposedQuantTensor::new(
            qweight,
            scales,
            qzeros,
            Some(g_idx),
            DecomposedQuantMethod::Gptq { group_size: IN / 2 },
            vec![OUT, IN],
        );
        let layer: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::DecomposedQuant(Box::new(DecomposedQuantLinear::new(weight, None)))
                .into();
        check(&client, &device, &layer);
    }

    #[test]
    fn bad_ids_and_lora_are_refused() {
        let (client, device) = cpu_setup();
        let mut layer: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::Standard(Linear::new(tensor(&device, &[OUT, IN], 0.3), None, false))
                .into();
        let err = layer.weight_rows_f32(&client, &[]).unwrap_err().to_string();
        assert!(err.contains("no ids"), "{err}");
        let err = layer
            .weight_rows_f32(&client, &[3, OUT])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(&format!("id {OUT} at position 1 is outside [0, {OUT})")),
            "{err}"
        );
        let a = tensor(&device, &[2, IN], 0.7);
        let b = tensor(&device, &[OUT, 2], 0.9);
        layer.attach_adapter(a, b, 1.0).unwrap();
        let err = layer
            .weight_rows_f32(&client, &[1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("LoRA"), "{err}");
    }
}
