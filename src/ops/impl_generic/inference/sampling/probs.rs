//! The canonical sampling filter, composed from numr ops on the device.

use crate::error::{Error, Result};
use crate::ops::traits::inference::sampling_probs::FilterRule;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, CumulativeOps, ReduceOps, ScalarOps, ShapeOps,
    SortingOps, TensorOps, TypeConversionOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Filters `logits` by `rule` and returns renormalized F32 probabilities.
///
/// The order is temperature, softmax, top-k, top-p, min-p, renormalize.
/// [`FilterRule`] defines each step. `logits` is `[.., vocab]` in any float
/// dtype. Every row filters on its own. No value leaves the device.
///
/// # Errors
///
/// Returns an error if `rule` fails [`FilterRule::check`], `logits` is
/// rank 0, not a float dtype, or has an empty vocab axis.
pub fn sampling_probs_impl<R: Runtime<DType = DType>>(
    client: &R::Client,
    logits: &Tensor<R>,
    rule: &FilterRule,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    rule.check()?;
    let shape = logits.shape().to_vec();
    let vocab = shape.last().copied().unwrap_or(0);
    if vocab == 0 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "sampling_probs: logits must be [.., vocab] with vocab above 0, got shape {shape:?}"
            ),
        });
    }
    let dtype = logits.dtype();
    if !dtype.is_float() {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "sampling_probs: logits must be a float dtype, got {dtype:?} — cast the logits first"
            ),
        });
    }
    let last = shape.len() - 1;

    let logits = if dtype == DType::F32 {
        logits.contiguous()?
    } else {
        client.cast(logits, DType::F32)?
    };
    let scaled = if rule.temperature == 1.0 {
        logits
    } else {
        client.div_scalar(&logits, f64::from(rule.temperature))?
    };
    let probs = client.softmax(&scaled, -1)?;
    let mut kept = probs.clone();

    let top_k = rule.top_k_active(vocab);
    let top_p = rule.top_p_active() && vocab > 1;
    if top_k || top_p {
        // Descending, so a mask at a threshold keeps a prefix and the order holds.
        let mut sorted = client.sort(&probs, -1, true)?;
        if top_k {
            let kth = sorted.narrow(-1, rule.top_k - 1, 1)?.contiguous()?;
            kept = keep_at_least(client, &kept, &kth)?;
            sorted = keep_at_least(client, &sorted, &kth)?;
        }
        if top_p {
            let tau = top_p_threshold(client, &sorted, rule.top_p, vocab, last)?;
            kept = keep_at_least(client, &kept, &tau)?;
        }
    }
    if rule.min_p_active() {
        let max_p = client.max(&probs, &[last], true)?;
        let threshold = client.mul_scalar(&max_p, f64::from(rule.min_p))?;
        kept = keep_at_least(client, &kept, &threshold)?;
    }

    let total = client.sum(&kept, &[last], true)?;
    Ok(client.div(&kept, &total)?)
}

/// `p` with every entry below the row's `threshold` set to 0.
///
/// `threshold` is `[.., 1]` and broadcasts across the vocab axis.
fn keep_at_least<R: Runtime<DType = DType>>(
    client: &R::Client,
    p: &Tensor<R>,
    threshold: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + CompareOps<R>,
{
    let mask = client.ge(p, threshold)?;
    Ok(client.mul(p, &mask)?)
}

/// The top-p threshold `tau` of each row, as `[.., 1]`.
///
/// `sorted` holds the row after top-k in descending order. `tau` is the
/// entry at the first index whose prefix sum reaches `top_p` times the row
/// mass. Every larger value has a smaller kept mass, so `tau` is the
/// largest value whose kept mass reaches the target.
fn top_p_threshold<R: Runtime<DType = DType>>(
    client: &R::Client,
    sorted: &Tensor<R>,
    top_p: f32,
    vocab: usize,
    last: usize,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let prefix = client.cumsum(sorted, -1)?;
    // The mass is the last prefix sum, so the target never exceeds the last prefix.
    let mass = prefix.narrow(-1, vocab - 1, 1)?.contiguous()?;
    let target = client.mul_scalar(&mass, f64::from(top_p))?;
    // 1 where the prefix through this index stays below the target.
    let below = client.lt(&prefix, &target)?;
    let mut lead = sorted.shape().to_vec();
    lead[last] = 1;
    let ones = Tensor::<R>::ones(&lead, DType::F32, sorted.device())?;
    let below_before = below.narrow(-1, 0, vocab - 1)?.contiguous()?;
    // 1 where every earlier prefix stays below the target.
    let open = client.cat(&[&ones, &below_before], -1)?;
    // 1 only at the first index whose prefix reaches the target.
    let crossing = client.sub(&open, &below)?;
    let at_crossing = client.mul(sorted, &crossing)?;
    Ok(client.max(&at_crossing, &[last], true)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn rule(temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> FilterRule {
        FilterRule {
            temperature,
            top_k,
            top_p,
            min_p,
        }
    }

    fn run(logits: &[f32], shape: &[usize], r: FilterRule) -> Vec<f32> {
        let (client, device) = cpu_setup();
        let t = Tensor::<CpuRuntime>::from_slice(logits, shape, &device).expect("logits");
        let out = sampling_probs_impl::<CpuRuntime>(&client, &t, &r).expect("sampling_probs");
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.shape(), shape);
        out.to_vec()
    }

    /// The softmax the filter sees, from the same numr op.
    fn softmax(logits: &[f32], device: &CpuDevice) -> Vec<f32> {
        let (client, _) = cpu_setup();
        let t = Tensor::<CpuRuntime>::from_slice(logits, &[logits.len()], device).expect("t");
        client.softmax(&t, -1).expect("softmax").to_vec()
    }

    fn kept(row: &[f32]) -> Vec<usize> {
        (0..row.len()).filter(|&i| row[i] > 0.0).collect()
    }

    fn assert_rows_sum_to_one(out: &[f32], vocab: usize) {
        for (r, row) in out.chunks(vocab).enumerate() {
            let s: f64 = row.iter().map(|&p| f64::from(p)).sum();
            assert!((s - 1.0).abs() <= 1e-6, "row {r} sums to {s}");
        }
    }

    #[test]
    fn top_k_keeps_ties_at_the_kth_value() {
        let logits = [0.4f32.ln(), 0.3f32.ln(), 0.3f32.ln()];
        let out = run(&logits, &[3], rule(1.0, 2, 1.0, 0.0));
        assert_eq!(
            kept(&out),
            vec![0, 1, 2],
            "kth largest is 0.3: both ties stay"
        );
        assert_rows_sum_to_one(&out, 3);
    }

    #[test]
    fn top_k_one_keeps_only_the_argmax() {
        let out = run(&[0.1, 2.0, -1.0, 1.9], &[4], rule(1.0, 1, 1.0, 0.0));
        assert_eq!(out, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn top_p_keeps_the_token_whose_prefix_lands_exactly_on_the_target() {
        let (_, device) = cpu_setup();
        let logits = [2.0f32, 1.5, 0.5, -0.5, -1.0];
        let p = softmax(&logits, &device);
        // Logits are already in descending order, so the sorted row is `p`.
        let mut prefix = Vec::with_capacity(p.len());
        let mut acc = 0.0f32;
        for &x in &p {
            acc += x;
            prefix.push(acc);
        }
        let mass = prefix[p.len() - 1];
        let mut top_p = prefix[1] / mass;
        let mut steps = 0;
        while top_p * mass != prefix[1] {
            top_p = if top_p * mass < prefix[1] {
                top_p.next_up()
            } else {
                top_p.next_down()
            };
            steps += 1;
            assert!(steps < 64, "no f32 top_p lands exactly on the prefix");
        }
        let at = run(&logits, &[5], rule(1.0, 0, top_p, 0.0));
        assert_eq!(
            kept(&at),
            vec![0, 1],
            "prefix equal to the target keeps index 1"
        );
        let mut past = top_p.next_up();
        while past * mass <= prefix[1] {
            past = past.next_up();
        }
        let above = run(&logits, &[5], rule(1.0, 0, past, 0.0));
        assert_eq!(
            kept(&above),
            vec![0, 1, 2],
            "a target past the prefix takes one more"
        );
    }

    #[test]
    fn top_p_measures_the_mass_top_k_left() {
        // p ~ [0.5, 0.3, 0.2]. After top_k = 2 the mass is 0.8, so the target
        // at top_p = 0.6 is 0.48 and the first token alone reaches it.
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        let out = run(&logits, &[3], rule(1.0, 2, 0.6, 0.0));
        assert_eq!(kept(&out), vec![0]);
        // Over the full mass the target is 0.6, which needs two tokens.
        let full = run(&logits, &[3], rule(1.0, 0, 0.6, 0.0));
        assert_eq!(kept(&full), vec![0, 1]);
    }

    #[test]
    fn min_p_drops_tokens_below_the_fraction_of_the_top() {
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.15f32.ln(), 0.05f32.ln()];
        let out = run(&logits, &[4], rule(1.0, 0, 1.0, 0.25));
        assert_eq!(kept(&out), vec![0, 1, 2], "threshold 0.125 drops only 0.05");
        assert_rows_sum_to_one(&out, 4);
    }

    #[test]
    fn every_filter_disabled_returns_the_softmax() {
        let (_, device) = cpu_setup();
        let logits = [0.3f32, -1.2, 2.2, 0.0, 0.7];
        let want = softmax(&logits, &device);
        for r in [
            rule(1.0, 0, 1.0, 0.0),
            rule(1.0, 5, 1.0, 0.0),
            rule(1.0, 9, 1.0, 0.0),
            rule(1.0, 0, 1.5, 0.0),
            rule(1.0, 0, 1.0, -0.5),
        ] {
            let out = run(&logits, &[5], r);
            for (o, w) in out.iter().zip(&want) {
                assert!((o - w).abs() <= 1e-6, "{r:?}: {o} vs {w}");
            }
        }
    }

    #[test]
    fn temperature_divides_the_logits() {
        let (_, device) = cpu_setup();
        let logits = [0.3f32, -1.2, 2.2, 0.0];
        let halved: Vec<f32> = logits.iter().map(|&l| l / 2.0).collect();
        let want = softmax(&halved, &device);
        let out = run(&logits, &[4], FilterRule::temperature_only(2.0));
        for (o, w) in out.iter().zip(&want) {
            assert!((o - w).abs() <= 1e-6, "{o} vs {w}");
        }
    }

    #[test]
    fn batched_rows_filter_on_their_own() {
        let rows: [[f32; 4]; 3] = [
            [0.1, 2.0, -1.0, 1.9],
            [5.0, 0.0, 0.0, 0.0],
            [-0.3, -0.2, -0.1, 0.0],
        ];
        let r = rule(0.8, 3, 0.9, 0.05);
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let batched = run(&flat, &[3, 4], r);
        assert_rows_sum_to_one(&batched, 4);
        for (i, row) in rows.iter().enumerate() {
            let single = run(row, &[4], r);
            let got = &batched[i * 4..i * 4 + 4];
            assert_eq!(kept(got), kept(&single), "row {i} kept set");
            for (g, s) in got.iter().zip(&single) {
                assert!((g - s).abs() <= 1e-7, "row {i}: {g} vs {s}");
            }
        }
    }

    #[test]
    fn one_dominant_token_takes_all_the_mass() {
        let mut logits = vec![0.0f32; 300];
        logits[123] = 60.0;
        let out = run(&logits, &[300], rule(1.0, 40, 0.9, 0.05));
        assert_eq!(kept(&out), vec![123]);
        assert_eq!(out[123], 1.0);
    }

    fn drop_below(row: &mut [f32], threshold: f32) {
        for x in row.iter_mut().filter(|x| **x < threshold) {
            *x = 0.0;
        }
    }

    /// The canonical rule on one softmax row, written out on the host.
    fn reference(p: &[f32], r: FilterRule) -> Vec<f32> {
        let vocab = p.len();
        let mut sorted = p.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let max_p = sorted[0];
        let mut kept = p.to_vec();
        if r.top_k_active(vocab) {
            let kth = sorted[r.top_k - 1];
            drop_below(&mut kept, kth);
            drop_below(&mut sorted, kth);
        }
        if r.top_p_active() && vocab > 1 {
            let mut prefix = Vec::with_capacity(vocab);
            let mut acc = 0.0f32;
            for &x in &sorted {
                acc += x;
                prefix.push(acc);
            }
            let target = r.top_p * prefix[vocab - 1];
            let j = prefix
                .iter()
                .position(|&c| c >= target)
                .expect("target <= mass");
            drop_below(&mut kept, sorted[j]);
        }
        if r.min_p_active() {
            drop_below(&mut kept, r.min_p * max_p);
        }
        let total: f32 = kept.iter().sum();
        kept.iter().map(|&x| x / total).collect()
    }

    /// Deterministic logits in `[-4, 4)`.
    fn random_logits(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
            })
            .collect()
    }

    #[test]
    fn combined_filters_match_the_host_rule_on_random_rows() {
        let (client, device) = cpu_setup();
        let rules = [
            rule(1.0, 5, 0.9, 0.05),
            rule(0.7, 40, 0.8, 0.0),
            rule(1.3, 0, 0.5, 0.1),
            rule(0.5, 3, 1.0, 0.3),
            rule(2.0, 1, 0.95, 0.0),
            rule(1.0, 0, 0.3, 0.0),
        ];
        for (case, &vocab) in [7usize, 50, 257].iter().enumerate() {
            let rows = 3;
            let logits = random_logits(rows * vocab, case as u64 + 11);
            let t = Tensor::<CpuRuntime>::from_slice(&logits, &[rows, vocab], &device).expect("t");
            for r in rules {
                let scaled = client
                    .div_scalar(&t, f64::from(r.temperature))
                    .expect("scale");
                let p: Vec<f32> = client.softmax(&scaled, -1).expect("softmax").to_vec();
                let out: Vec<f32> = sampling_probs_impl::<CpuRuntime>(&client, &t, &r)
                    .expect("sampling_probs")
                    .to_vec();
                assert_rows_sum_to_one(&out, vocab);
                for row in 0..rows {
                    let span = row * vocab..(row + 1) * vocab;
                    let want = reference(&p[span.clone()], r);
                    let got = &out[span];
                    assert_eq!(kept(got), kept(&want), "{r:?} vocab {vocab} row {row}");
                    for (g, w) in got.iter().zip(&want) {
                        assert!((g - w).abs() <= 1e-6, "{r:?} vocab {vocab}: {g} vs {w}");
                    }
                }
            }
        }
    }

    #[cfg(feature = "f16")]
    #[test]
    fn half_logits_filter_as_their_f32_values() {
        let (client, device) = cpu_setup();
        let logits = random_logits(2 * 33, 5);
        let r = rule(0.9, 6, 0.85, 0.02);
        let f16: Vec<half::f16> = logits.iter().map(|&x| half::f16::from_f32(x)).collect();
        let bf16: Vec<half::bf16> = logits.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let cases = [
            (
                Tensor::<CpuRuntime>::from_slice(&f16, &[2, 33], &device).expect("f16"),
                f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
            ),
            (
                Tensor::<CpuRuntime>::from_slice(&bf16, &[2, 33], &device).expect("bf16"),
                bf16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
            ),
        ];
        for (half_t, wide) in cases {
            let got: Vec<f32> = sampling_probs_impl::<CpuRuntime>(&client, &half_t, &r)
                .expect("half")
                .to_vec();
            let want = run(&wide, &[2, 33], r);
            assert_eq!(got, want, "{:?}", half_t.dtype());
            assert_rows_sum_to_one(&got, 33);
        }
    }

    #[test]
    fn rejects_non_float_logits_and_an_empty_vocab() {
        let (client, device) = cpu_setup();
        let ints = Tensor::<CpuRuntime>::from_slice(&[1i32, 2], &[2], &device).expect("ints");
        let r = FilterRule::temperature_only(1.0);
        let err = sampling_probs_impl::<CpuRuntime>(&client, &ints, &r).expect_err("i32");
        assert!(err.to_string().contains("float"), "{err}");
        let empty = Tensor::<CpuRuntime>::from_slice::<f32>(&[], &[2, 0], &device).expect("e");
        let err = sampling_probs_impl::<CpuRuntime>(&client, &empty, &r).expect_err("empty");
        assert!(err.to_string().contains("vocab"), "{err}");
    }

    #[test]
    fn rejects_temperature_zero() {
        let (client, device) = cpu_setup();
        let t = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[2], &device).expect("t");
        let r = FilterRule::temperature_only(0.0);
        match sampling_probs_impl::<CpuRuntime>(&client, &t, &r) {
            Err(Error::InvalidArgument { arg, .. }) => assert_eq!(arg, "temperature"),
            other => panic!("expected a temperature error, got {other:?}"),
        }
    }
}
