//! `speculative_verify` composed from numr ops. No value leaves the device.

use super::accept::leading_accepts;
use super::draw::inverse_cdf_draw;
use super::residual::next_distribution;
use super::validate::{VerifyShape, validate_verify_args};
use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{CompareOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UtilityOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Sampled speculative verification on any backend. See
/// [`SpeculativeOps::speculative_verify`](crate::ops::traits::inference::speculative::SpeculativeOps::speculative_verify).
///
/// # Errors
///
/// Returns `Error::InvalidArgument` naming the argument for a wrong rank, shape or dtype.
pub fn speculative_verify_impl<R: Runtime<DType = DType>>(
    client: &R::Client,
    target_probs: &Tensor<R>,
    draft_probs: Option<&Tensor<R>>,
    draft_tokens: &Tensor<R>,
    uniforms: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let shape = validate_verify_args(target_probs, draft_probs, draft_tokens, uniforms)?;
    let VerifyShape {
        batch,
        drafts,
        vocab,
    } = shape;
    let device = target_probs.device();
    if batch == 0 {
        return Ok(Tensor::<R>::empty(&[0, 2], DType::I32, device)?);
    }
    let target = target_probs.contiguous()?;
    let uniforms = uniforms.contiguous()?;
    let ids = client
        .arange(0.0, vocab as f64, 1.0, DType::F32)?
        .reshape(&[1, 1, vocab])?;
    let (accepted, dist) = if drafts == 0 {
        let none = Tensor::<R>::zeros(&[batch, 1], DType::F32, device)?;
        (none, target.clone())
    } else {
        let lead =
            leading_accepts::<R>(client, &target, draft_probs, draft_tokens, &uniforms, shape)?;
        let dist = next_distribution::<R>(client, &target, draft_probs, &lead, &ids, shape)?;
        (lead.accepted, dist)
    };
    let u_draw = uniforms
        .narrow(1, drafts, 1)?
        .contiguous()?
        .reshape(&[batch, 1, 1])?;
    let next = inverse_cdf_draw::<R>(client, &dist, &u_draw, &ids)?;
    let out = client.cat(&[&accepted, &next], 1)?;
    Ok(client.cast(&out, DType::I32)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::AdvancedRandomOps;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    type T = Tensor<CpuRuntime>;

    /// Runs the op on host data. `p` is `[B, K+1, V]`, `q` is `[B, K, V]`.
    fn run(bkv: [usize; 3], p: &[f32], q: Option<&[f32]>, d: &[i32], u: &[f32]) -> Vec<i32> {
        let (client, device) = cpu_setup();
        run_on(&client, &device, bkv, p, q, d, u)
    }

    fn run_on(
        client: &CpuClient,
        device: &CpuDevice,
        [b, k, v]: [usize; 3],
        p: &[f32],
        q: Option<&[f32]>,
        d: &[i32],
        u: &[f32],
    ) -> Vec<i32> {
        let p = T::from_slice(p, &[b, k + 1, v], device).expect("p");
        let q = q.map(|q| T::from_slice(q, &[b, k, v], device).expect("q"));
        let d = T::from_slice(d, &[b, k], device).expect("d");
        let u = T::from_slice(u, &[b, k + 1], device).expect("u");
        speculative_verify_impl::<CpuRuntime>(client, &p, q.as_ref(), &d, &u)
            .expect("verify")
            .to_vec::<i32>()
    }

    const ROWS: [f32; 12] = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.0, 0.0];

    #[test]
    fn all_accepted_draws_the_bonus_token() {
        assert_eq!(
            run([1, 2, 4], &ROWS, None, &[3, 2], &[0.39, 0.29, 0.6]),
            [2, 1]
        );
    }

    #[test]
    fn a_first_reject_draws_from_the_residual() {
        assert_eq!(
            run([1, 2, 4], &ROWS, None, &[3, 2], &[0.5, 0.0, 0.4]),
            [0, 1]
        );
    }

    #[test]
    fn batched_rows_are_independent() {
        let p: Vec<f32> = ROWS.iter().chain(ROWS.iter()).copied().collect();
        let u = [0.39, 0.29, 0.6, 0.5, 0.0, 0.4];
        assert_eq!(run([2, 2, 4], &p, None, &[3, 2, 3, 2], &u), [2, 1, 0, 1]);
    }

    #[test]
    fn a_middle_reject_draws_from_its_own_residual() {
        let p = [
            0.25, 0.25, 0.25, 0.25, 0.4, 0.3, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let q = [
            0.5, 0.5, 0.0, 0.0, 0.1, 0.1, 0.4, 0.4, 0.25, 0.25, 0.25, 0.25,
        ];
        let u = [0.3, 0.7, 0.0, 0.7];
        assert_eq!(run([1, 3, 4], &p, Some(&q), &[0, 2, 0], &u), [1, 1]);
    }

    #[test]
    fn a_general_draft_and_a_one_hot_draft_differ() {
        let p = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 0.0];
        let q = [0.2, 0.2, 0.3, 0.3];
        assert_eq!(run([1, 1, 4], &p, Some(&q), &[2], &[0.8, 0.5]), [1, 2]);
        assert_eq!(run([1, 1, 4], &p, None, &[2], &[0.8, 0.5]), [0, 1]);
    }

    #[test]
    fn a_zero_residual_falls_back_to_the_target_row() {
        let p = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let q = [0.5, 0.5, 0.0, 0.0];
        assert_eq!(run([1, 1, 4], &p, Some(&q), &[2], &[0.0, 0.7]), [0, 1]);
    }

    #[test]
    fn a_draft_with_zero_draft_mass_is_rejected() {
        let p = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 1.0];
        let q = [0.5, 0.5, 0.0, 0.0];
        assert_eq!(run([1, 1, 4], &p, Some(&q), &[2], &[0.0, 0.1]), [0, 2]);
    }

    #[test]
    fn an_out_of_range_draft_is_rejected() {
        let p = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(run([1, 1, 4], &p, None, &[9], &[0.0, 0.6]), [0, 2]);
        assert_eq!(run([1, 1, 4], &p, None, &[-3], &[0.0, 0.6]), [0, 2]);
    }

    #[test]
    fn zero_drafts_draw_from_target_row_zero() {
        let p = [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1];
        assert_eq!(run([2, 0, 4], &p, None, &[], &[0.95, 0.05]), [0, 3, 0, 0]);
    }

    #[test]
    fn a_zero_probability_token_is_never_drawn_near_one() {
        let p = [0.3, 0.7, 0.0, 0.0, 0.3, 0.7, 0.0, 0.0];
        let u_max = 0.999_999_94;
        assert_eq!(run([1, 1, 4], &p, None, &[1], &[0.0, u_max]), [1, 1]);
        assert_eq!(run([1, 1, 4], &p, None, &[1], &[0.9, u_max]), [0, 0]);
    }

    #[test]
    fn an_empty_batch_returns_an_empty_result() {
        let (client, device) = cpu_setup();
        assert!(run_on(&client, &device, [0, 2, 4], &[], None, &[], &[]).is_empty());
    }

    // Statistical check: the emitted tokens follow the target distribution.

    const V: usize = 8;
    const K: usize = 2;
    const RUNS: usize = 200_000;
    const BATCHES: usize = 4;
    const KEY: u64 = 0x6e75_6d72;
    const Z: f64 = 4.753;
    const P_W: [[f32; V]; 3] = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [8.0, 1.0, 6.0, 3.0, 4.0, 5.0, 2.0, 7.0],
        [3.0, 3.0, 1.0, 9.0, 2.0, 8.0, 4.0, 6.0],
    ];
    const Q_W: [[f32; V]; 3] = [
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 8.0, 8.0, 8.0, 8.0],
        [6.0, 1.0, 6.0, 1.0, 6.0, 1.0, 6.0, 1.0],
    ];

    #[derive(Clone, Copy)]
    enum Mode {
        General,
        OneHot,
        /// The negative control: accepts every draft with `p > 0`, never subtracts `q`.
        Biased,
    }

    fn row(w: &[f32; V]) -> [f32; V] {
        let s: f32 = w.iter().sum();
        w.map(|x| x / s)
    }

    fn host_draw(row: &[f32], u: f32) -> usize {
        let t = u * row.iter().sum::<f32>();
        let mut c = 0.0f32;
        for (i, &x) in row.iter().enumerate() {
            c += x;
            if x > 0.0 && c > t {
                return i;
            }
        }
        row.iter().rposition(|&x| x > 0.0).unwrap_or(0)
    }

    fn argmax(row: &[f32]) -> usize {
        (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
    }

    /// One verify call over `n` runs from position `start`. Returns the emitted tokens.
    fn emit(
        (client, device): (&CpuClient, &CpuDevice),
        mode: Mode,
        start: usize,
        n: usize,
        call: u64,
    ) -> Vec<Vec<usize>> {
        if n == 0 {
            return Vec::new();
        }
        let uniforms = |cols: usize, c: u64| {
            client
                .philox_uniform(&[n, cols], KEY, c << 32, DType::F32)
                .expect("philox")
                .to_vec::<f32>()
        };
        let (du, vu) = (uniforms(K, 2 * call), uniforms(K + 1, 2 * call + 1));
        let p_rows: Vec<[f32; V]> = (0..=K).map(|i| row(&P_W[(start + i) % 3])).collect();
        let q_rows: Vec<[f32; V]> = (0..K).map(|i| row(&Q_W[(start + i) % 3])).collect();
        let mut d = vec![0i32; n * K];
        for r in 0..n {
            for i in 0..K {
                d[r * K + i] = match mode {
                    Mode::OneHot => argmax(&q_rows[i]),
                    _ => host_draw(&q_rows[i], du[r * K + i]),
                } as i32;
            }
        }
        let out: Vec<i32> = match mode {
            Mode::Biased => (0..n)
                .flat_map(|r| {
                    let dr = &d[r * K..(r + 1) * K];
                    let m = dr
                        .iter()
                        .zip(&p_rows)
                        .take_while(|pair| pair.1[*pair.0 as usize] > 0.0)
                        .count();
                    [m as i32, host_draw(&p_rows[m], vu[r * (K + 1) + K]) as i32]
                })
                .collect(),
            _ => {
                let p: Vec<f32> = (0..n)
                    .flat_map(|_| p_rows.iter().flatten().copied())
                    .collect();
                let q: Vec<f32> = (0..n)
                    .flat_map(|_| q_rows.iter().flatten().copied())
                    .collect();
                let q = matches!(mode, Mode::General).then_some(q.as_slice());
                run_on(client, device, [n, K, V], &p, q, &d, &vu)
            }
        };
        (0..n)
            .map(|r| {
                let m = out[2 * r] as usize;
                let mut tokens: Vec<usize> =
                    d[r * K..r * K + m].iter().map(|&t| t as usize).collect();
                tokens.push(out[2 * r + 1] as usize);
                tokens
            })
            .collect()
    }

    /// Counts of the first token `[V]` and of the first two tokens `[V * V]`.
    fn first_two(mode: Mode, tag: u64) -> (Vec<f64>, Vec<f64>) {
        let (client, device) = cpu_setup();
        let backend = (&client, &device);
        let (mut first, mut joint) = (vec![0.0; V], vec![0.0; V * V]);
        let n = RUNS / BATCHES;
        for batch in 0..BATCHES as u64 {
            let call = (tag * BATCHES as u64 + batch) * 2;
            let round1 = emit(backend, mode, 0, n, call);
            let short = round1.iter().filter(|tokens| tokens.len() < 2).count();
            let round2 = emit(backend, mode, 1, short, call + 1);
            let mut later = round2.iter();
            for tokens in &round1 {
                let t1 = match tokens.get(1) {
                    Some(&t) => t,
                    None => later.next().expect("a second round")[0],
                };
                first[tokens[0]] += 1.0;
                joint[tokens[0] * V + t1] += 1.0;
            }
        }
        (first, joint)
    }

    fn chi_square(counts: &[f64], probs: &[f64]) -> f64 {
        let n: f64 = counts.iter().sum();
        counts
            .iter()
            .zip(probs)
            .map(|(&o, &p)| (o - n * p).powi(2) / (n * p))
            .sum()
    }

    /// The Wilson-Hilferty critical value at alpha = 1e-6.
    fn critical(df: usize) -> f64 {
        let h = 2.0 / (9.0 * df as f64);
        df as f64 * (1.0 - h + Z * h.sqrt()).powi(3)
    }

    fn expected() -> (Vec<f64>, Vec<f64>) {
        let (p0, p1) = (row(&P_W[0]), row(&P_W[1]));
        let first: Vec<f64> = p0.iter().map(|&x| x as f64).collect();
        let joint = (0..V * V)
            .map(|c| first[c / V] * p1[c % V] as f64)
            .collect();
        (first, joint)
    }

    #[test]
    fn emitted_tokens_follow_the_target_distribution() {
        let (want_first, want_joint) = expected();
        for (tag, mode) in [(0, Mode::General), (1, Mode::OneHot)] {
            let (first, joint) = first_two(mode, tag);
            let s1 = chi_square(&first, &want_first);
            assert!(
                s1 < critical(V - 1),
                "first token: chi2 {s1} >= {}",
                critical(V - 1)
            );
            let s2 = chi_square(&joint, &want_joint);
            assert!(
                s2 < critical(V * V - 1),
                "first two: chi2 {s2} >= {}",
                critical(V * V - 1)
            );
        }
    }

    #[test]
    fn the_biased_negative_control_fails_the_same_check() {
        let (want_first, _) = expected();
        let (first, _) = first_two(Mode::Biased, 2);
        let s1 = chi_square(&first, &want_first);
        assert!(s1 > critical(V - 1), "biased chi2 {s1} passed");
    }
}
