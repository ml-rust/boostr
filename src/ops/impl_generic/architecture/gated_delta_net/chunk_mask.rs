//! Document masks of the chunked gated delta rule for packed rows.
//!
//! `segment_ids` is an I32 `[batch, seq]` tensor. A document starts where
//! the id differs from the id one position earlier. Position 0 starts no
//! document, so the initial state reaches the first document of each row.
//!
//! Per chunk, `r[t]` is 1 where token `t` starts a document and
//! `R = cumsum(r)` within the chunk. `R` never decreases, so `R[j] == R[i]`
//! holds exactly when no document starts in `(i, j]`:
//!
//! ```text
//! same[j][i] = R[j] == R[i]       tokens i and j share a document
//! pre[j]     = R[j] == 0          token j shares the document of the
//!                                 token before the chunk
//! last[i]    = R[i] == R[cs - 1]  token i shares the document of the
//!                                 last token of the chunk
//! pre_last   = pre[cs - 1]        the carried state survives the chunk
//! ```
//!
//! The chunk terms take these factors:
//!
//! ```text
//! decay = exp(g_cs[j] - g_cs[i]) * same[j][i]
//! e_g   = exp(g_cs) * pre                   in Q_g and in the W operand
//! e_tail = exp(g_last - g_cs) * last        in K_g
//! g_last_exp = exp(g_last) * pre_last
//! ```
//!
//! Padded tokens repeat the last id, so they start no document and keep
//! the carried state of the last document. The flags are counted in F32
//! on the device, where every count is exact, then cast to the compute
//! dtype. The ids never leave the device.

use super::common::GdnDims;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{CompareOps, CumulativeOps, ShapeOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Document masks of every chunk, in the compute dtype. `N = batch * H`.
pub(super) struct ChunkMasks<R: Runtime> {
    /// `same[j][i]`: `[N, n_chunks, cs, cs]`.
    pub same: Tensor<R>,
    /// `pre[j]`: `[N, n_chunks, cs, 1]`.
    pub pre: Tensor<R>,
    /// `last[i]`: `[N, n_chunks, cs, 1]`.
    pub last: Tensor<R>,
    /// `pre[cs - 1]`: `[N, n_chunks, 1]`.
    pub pre_last: Tensor<R>,
}

/// Error unless `ids` is an I32 `[batch, seq]` tensor.
pub(super) fn check_segment_ids<R>(ids: &Tensor<R>, batch: usize, seq: usize) -> Result<()>
where
    R: Runtime<DType = DType>,
{
    let want = [batch, seq];
    if ids.dtype() != DType::I32 || ids.shape() != want.as_slice() {
        return Err(Error::InvalidArgument {
            arg: "segment_ids",
            reason: format!(
                "expected I32 {want:?} to match the batch and seq of q, got {:?} {:?}",
                ids.dtype(),
                ids.shape()
            ),
        });
    }
    Ok(())
}

/// Build the document masks of `n_chunks` chunks of `cs` tokens.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `ids` is not an I32 `[batch, seq]` tensor.
pub(super) fn chunk_masks<R, C>(
    client: &C,
    ids: &Tensor<R>,
    dims: GdnDims,
    n_chunks: usize,
    cs: usize,
    dtype: DType,
) -> Result<ChunkMasks<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + CompareOps<R> + CumulativeOps<R> + ShapeOps<R> + TypeConversionOps<R>,
{
    let (batch, seq) = (dims.batch, dims.seq);
    check_segment_ids(ids, batch, seq)?;
    let padded = n_chunks * cs;
    let device = ids.device();
    let ids = ids.contiguous()?;

    // r[t] = ids[t] != ids[t - 1]. It is 0 at t = 0 and on padded tokens.
    let mut parts = vec![Tensor::<R>::zeros(&[batch, 1], DType::F32, device)?];
    if seq > 1 {
        let later = ids.narrow(1, 1, seq - 1)?;
        let earlier = ids.narrow(1, 0, seq - 1)?;
        parts.push(client.cast(&client.ne(&later, &earlier)?, DType::F32)?);
    }
    if padded > seq {
        parts.push(Tensor::<R>::zeros(
            &[batch, padded - seq],
            DType::F32,
            device,
        )?);
    }
    let refs: Vec<&Tensor<R>> = parts.iter().collect();
    let starts = client.cat(&refs, 1)?.reshape(&[batch, n_chunks, cs])?;
    let count = client.cumsum(&starts, 2)?; // R: [batch, n_chunks, cs]

    // The comparisons broadcast their operands, so no expanded copy is built.
    let same = client.eq(&count.unsqueeze(3)?, &count.unsqueeze(2)?)?;

    let zero = Tensor::<R>::zeros(&[1, 1, 1], DType::F32, device)?;
    let pre = client.eq(&count, &zero)?;
    let last = client.eq(&count, &count.narrow(2, cs - 1, 1)?)?;
    let pre_last = pre.narrow(2, cs - 1, 1)?;

    let heads = dims.heads;
    Ok(ChunkMasks {
        same: per_head(client, &same, heads, dtype)?,
        pre: per_head(client, &pre.unsqueeze(3)?, heads, dtype)?,
        last: per_head(client, &last.unsqueeze(3)?, heads, dtype)?,
        pre_last: per_head(client, &pre_last, heads, dtype)?,
    })
}

/// F32 `[batch, ..]` flags as `[batch * H, ..]` flags in `dtype`.
///
/// Head `h` of row `b` sits at `b * H + h`, the order of the chunked inputs.
fn per_head<R, C>(client: &C, t: &Tensor<R>, heads: usize, dtype: DType) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R>,
{
    let t = if dtype == DType::F32 {
        t.clone()
    } else {
        client.cast(t, dtype)?
    };
    let mut shape = t.shape().to_vec();
    let mut wide = shape.clone();
    wide.insert(1, heads);
    shape[0] *= heads;
    Ok(t.unsqueeze(1)?
        .broadcast_to(&wide)?
        .contiguous()?
        .reshape(&shape)?)
}

#[cfg(test)]
mod tests {
    use super::super::chunk::gdn_chunk_prefill_impl;
    use super::super::chunk_bwd::gdn_chunk_backward_impl;
    use super::super::recurrent_bwd::gdn_recurrent_backward;
    use super::*;
    use crate::ops::traits::architecture::gated_delta_net::GdnGrads;
    use crate::test_utils::{Lcg, cpu_setup, ids_for, max_abs_diff_f64, to_f64};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    type T = Tensor<CpuRuntime>;

    const HEADS: usize = 2;
    /// `S_k != S_v`, so a swapped transpose fails on shape or value.
    const S_K: usize = 4;
    const S_V: usize = 5;
    /// F64 bound, relative to `max(1, largest reference entry)`.
    const TOL: f64 = 1e-9;
    const NAMES: [&str; 6] = ["o", "dq", "dk", "dv", "dg", "dbeta"];

    /// Seq 13. Row 0 starts documents at 3, 8 and 9: a one-token document,
    /// and a document over several chunks of 3. Chunk 3 resets on the
    /// boundary at 3 and 9 and mid-chunk at 8. Chunk 4 resets on the
    /// boundary at 8 and mid-chunk at 3 and 9. Row 1 starts documents at
    /// 1 and 7, mid-chunk for both sizes. Neither size divides 13.
    const SHORT: [&[usize]; 2] = [&[3, 5, 1, 4], &[1, 6, 6]];
    /// Seq 150 over chunks of 64: resets mid-chunk at 70 and 71, and on the
    /// boundary at 128 and 64. Documents span several chunks.
    const LONG: [&[usize]; 2] = [&[70, 1, 57, 22], &[64, 86]];

    fn ids_tensor(device: &CpuDevice, rows: &[&[usize]]) -> T {
        let seq: usize = rows[0].iter().sum();
        assert!(rows.iter().all(|r| r.iter().sum::<usize>() == seq));
        let data: Vec<i32> = rows.iter().flat_map(|lens| ids_for(lens)).collect();
        T::from_slice(&data, &[rows.len(), seq], device).unwrap()
    }

    fn unit_rows(rng: &mut Lcg, rows: usize, dim: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let row: Vec<f64> = (0..dim).map(|_| rng.uniform_f64(-1.0, 1.0)).collect();
            let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            out.extend(row.iter().map(|x| x / norm));
        }
        out
    }

    /// F64 `q, k, v, g, beta, state, d_o, d_state_final`.
    fn problem(device: &CpuDevice, seed: u64, batch: usize, seq: usize) -> [T; 8] {
        let mut rng = Lcg(seed);
        let rows = batch * seq * HEADS;
        let n_state = batch * HEADS * S_K * S_V;
        let q = unit_rows(&mut rng, rows, S_K);
        let k = unit_rows(&mut rng, rows, S_K);
        let mut uni = |n: usize, lo: f64, hi: f64| -> Vec<f64> {
            (0..n).map(|_| rng.uniform_f64(lo, hi)).collect()
        };
        let data = [
            q,
            k,
            uni(rows * S_V, -1.0, 1.0),
            uni(rows, -1.0, -0.05),
            uni(rows, 0.1, 0.9),
            uni(n_state, -0.5, 0.5),
            uni(rows * S_V, -1.0, 1.0),
            uni(n_state, -1.0, 1.0),
        ];
        let tok = |d: usize| vec![batch, seq, HEADS, d];
        let state = vec![batch, HEADS, S_K, S_V];
        let shapes = [
            tok(S_K),
            tok(S_K),
            tok(S_V),
            vec![batch, seq, HEADS],
            vec![batch, seq, HEADS],
            state.clone(),
            tok(S_V),
            state,
        ];
        std::array::from_fn(|i| T::from_slice(&data[i], &shapes[i], device).unwrap())
    }

    struct Run {
        o: T,
        state: T,
        grads: GdnGrads<CpuRuntime>,
    }

    fn packed(client: &CpuClient, t: &[T; 8], ids: Option<&T>, cs: usize) -> Run {
        let [q, k, v, g, beta, s0, d_o, ds] = t;
        let (o, state) = gdn_chunk_prefill_impl(client, q, k, v, g, beta, s0, ids, cs).unwrap();
        let grads =
            gdn_chunk_backward_impl(client, q, k, v, g, beta, s0, ids, d_o, Some(ds), cs).unwrap();
        Run { o, state, grads }
    }

    fn token_fields(r: &Run) -> [&T; 6] {
        let g = &r.grads;
        [&r.o, &g.dq, &g.dk, &g.dv, &g.dg, &g.dbeta]
    }

    /// Tokens `start .. start + len` of row `b`, as a batch of one.
    fn part(t: &T, b: usize, start: usize, len: usize) -> T {
        let row = t.narrow(0, b, 1).unwrap();
        row.narrow(1, start, len).unwrap().contiguous().unwrap()
    }

    fn row(t: &T, b: usize) -> T {
        t.narrow(0, b, 1).unwrap().contiguous().unwrap()
    }

    /// One document on its own. It starts from the row's initial state when
    /// `first`, else from zero, and takes `d_state_final` when `last`.
    /// Returns its output, its final state and its oracle gradients.
    #[allow(clippy::too_many_arguments, reason = "one document of one row")]
    fn separate(
        client: &CpuClient,
        t: &[T; 8],
        b: usize,
        start: usize,
        len: usize,
        first: bool,
        last: bool,
        cs: usize,
    ) -> (T, T, GdnGrads<CpuRuntime>) {
        let s = |i: usize| part(&t[i], b, start, len);
        let (q, k, v, g, beta, d_o) = (s(0), s(1), s(2), s(3), s(4), s(6));
        let state = if first {
            row(&t[5], b)
        } else {
            T::zeros(&[1, HEADS, S_K, S_V], DType::F64, t[5].device()).unwrap()
        };
        let ds = last.then(|| row(&t[7], b));
        let (o, s_end) =
            gdn_chunk_prefill_impl(client, &q, &k, &v, &g, &beta, &state, None, cs).unwrap();
        let grads =
            gdn_recurrent_backward(client, &q, &k, &v, &g, &beta, &state, &d_o, ds.as_ref())
                .unwrap();
        (o, s_end, grads)
    }

    fn assert_close(got: &T, want: &T, what: &str) {
        assert_eq!(got.shape(), want.shape(), "{what} shape");
        let (got, want) = (to_f64(got), to_f64(want));
        let scale = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
        let diff = max_abs_diff_f64(&got, &want);
        assert!(
            diff <= TOL * scale,
            "{what}: diff {diff} (bound {})",
            TOL * scale
        );
    }

    /// Packed rows against each document run alone, for the output, the
    /// final state and all six input gradients.
    fn check_packed(rows: &[&[usize]], cs: usize, seed: u64) {
        let (client, device) = cpu_setup();
        let seq: usize = rows[0].iter().sum();
        let t = problem(&device, seed, rows.len(), seq);
        let ids = ids_tensor(&device, rows);
        let run = packed(&client, &t, Some(&ids), cs);
        for (b, lens) in rows.iter().enumerate() {
            let mut start = 0;
            for (d, &len) in lens.iter().enumerate() {
                let (first, last) = (d == 0, d + 1 == lens.len());
                let (o, s_end, grads) = separate(&client, &t, b, start, len, first, last, cs);
                let want = [&o, &grads.dq, &grads.dk, &grads.dv, &grads.dg, &grads.dbeta];
                for (i, (got, want)) in token_fields(&run).iter().zip(want).enumerate() {
                    let what = format!("cs={cs} row={b} doc={d} {}", NAMES[i]);
                    assert_close(&part(got, b, start, len), want, &what);
                }
                if first {
                    let what = format!("cs={cs} row={b} dstate");
                    assert_close(&row(&run.grads.dstate, b), &grads.dstate, &what);
                }
                if last {
                    let what = format!("cs={cs} row={b} final state");
                    assert_close(&row(&run.state, b), &s_end, &what);
                }
                start += len;
            }
        }
    }

    #[test]
    fn packed_matches_separate_chunk_3() {
        check_packed(&SHORT, 3, 0x5e9_0001);
    }

    #[test]
    fn packed_matches_separate_chunk_4() {
        check_packed(&SHORT, 4, 0x5e9_0002);
    }

    /// Every document of a row in one padded chunk.
    #[test]
    fn packed_matches_separate_chunk_64_one_chunk() {
        check_packed(&SHORT, 64, 0x5e9_0003);
    }

    #[test]
    fn packed_matches_separate_chunk_64_many_chunks() {
        check_packed(&LONG, 64, 0x5e9_0004);
    }

    /// Doc A holds tokens 0..5 and doc B tokens 5..13. Chunks of 4 put the
    /// reset mid-chunk. New inputs for A, and a new initial state, leave B
    /// within 1e-12 of the old values.
    ///
    /// The bound is not bit equality. `g_cs = cumsum(g)` and the suffix sum
    /// of the gate gradient run over the whole chunk. A's gates therefore
    /// reach B's decay differences and B's `dg` through rounding. Every
    /// cross-document product itself is an exact zero.
    #[test]
    fn earlier_document_does_not_leak() {
        let (client, device) = cpu_setup();
        let (a_len, seq, cs) = (5, 13, 4);
        let b_len = seq - a_len;
        let ids = ids_tensor(&device, &[&[a_len, b_len][..]]);
        let base = problem(&device, 0x1ea_0001, 1, seq);
        let other = problem(&device, 0x1ea_0002, 1, seq);
        let mixed: [T; 8] = std::array::from_fn(|i| match i {
            5 => other[5].clone(),
            7 => base[7].clone(),
            _ => {
                let a = part(&other[i], 0, 0, a_len);
                let b = part(&base[i], 0, a_len, b_len);
                client.cat(&[&a, &b], 1).unwrap()
            }
        });
        let r0 = packed(&client, &base, Some(&ids), cs);
        let r1 = packed(&client, &mixed, Some(&ids), cs);
        let a_out = |r: &Run| to_f64(&part(&r.o, 0, 0, a_len));
        assert_ne!(a_out(&r0), a_out(&r1), "doc A must change");

        let mut pairs: Vec<(String, T, T)> = token_fields(&r0)
            .iter()
            .zip(token_fields(&r1))
            .enumerate()
            .map(|(i, (x, y))| {
                let (x, y) = (part(x, 0, a_len, b_len), part(y, 0, a_len, b_len));
                (NAMES[i].to_string(), x, y)
            })
            .collect();
        pairs.push(("final state".into(), r0.state.clone(), r1.state.clone()));
        for (name, x, y) in pairs {
            let (x, y) = (to_f64(&x), to_f64(&y));
            let scale = x.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            let diff = max_abs_diff_f64(&x, &y);
            assert!(diff <= 1e-12 * scale, "doc B {name}: diff {diff}");
        }
    }

    /// One document per row: every mask is 1, and a product with 1 is
    /// exact. The masked path equals the `None` path bit for bit.
    #[test]
    fn one_document_matches_none_exactly() {
        let (client, device) = cpu_setup();
        let seq = 13;
        let t = problem(&device, 0x5e9_0005, 2, seq);
        let ids = ids_tensor(&device, &[&[seq][..], &[seq][..]]);
        for cs in [3, 64] {
            let a = packed(&client, &t, None, cs);
            let b = packed(&client, &t, Some(&ids), cs);
            let mut fa = token_fields(&a).to_vec();
            let mut fb = token_fields(&b).to_vec();
            fa.extend([&a.state, &a.grads.dstate]);
            fb.extend([&b.state, &b.grads.dstate]);
            for (x, y) in fa.iter().zip(&fb) {
                assert_eq!(to_f64(x), to_f64(y), "cs={cs}");
            }
        }
    }

    /// Ids `[1, 1, 2, 2, 2, 3, 4]` over chunks of 3. Document starts sit at
    /// 2, 5 and 6, and tokens 7 and 8 are padding.
    #[test]
    fn masks_match_hand_values() {
        let (client, device) = cpu_setup();
        let ids = T::from_slice(&[1i32, 1, 2, 2, 2, 3, 4], &[1, 7], &device).unwrap();
        let dims = GdnDims {
            batch: 1,
            seq: 7,
            heads: HEADS,
            s_k: 1,
            s_v: 1,
        };
        let m = chunk_masks(&client, &ids, dims, 3, 3, DType::F64).unwrap();
        assert_eq!(m.same.shape(), &[HEADS, 3, 3, 3]);
        assert_eq!(m.pre.shape(), &[HEADS, 3, 3, 1]);
        assert_eq!(m.last.shape(), &[HEADS, 3, 3, 1]);
        assert_eq!(m.pre_last.shape(), &[HEADS, 3, 1]);
        let split = [1., 1., 0., 1., 1., 0., 0., 0., 1.];
        let heads = |v: Vec<f64>| -> Vec<f64> { v.repeat(HEADS) };
        assert_eq!(to_f64(&m.same), heads([split, split, [1.0; 9]].concat()));
        assert_eq!(
            to_f64(&m.pre),
            heads(vec![1., 1., 0., 1., 1., 0., 0., 0., 0.])
        );
        assert_eq!(
            to_f64(&m.last),
            heads(vec![0., 0., 1., 0., 0., 1., 1., 1., 1.])
        );
        assert_eq!(to_f64(&m.pre_last), heads(vec![0., 0., 0.]));
    }

    #[test]
    fn rejects_bad_segment_ids() {
        let (client, device) = cpu_setup();
        let seq = 7;
        let t = problem(&device, 1, 1, seq);
        let [q, k, v, g, beta, s0, ..] = &t;
        let run = |ids: &T| gdn_chunk_prefill_impl(&client, q, k, v, g, beta, s0, Some(ids), 3);
        let is_ids_error = |r: Result<(T, T)>| {
            matches!(
                r,
                Err(Error::InvalidArgument {
                    arg: "segment_ids",
                    ..
                })
            )
        };
        let f32_ids = T::from_slice(&[0.0f32; 7], &[1, 7], &device).unwrap();
        assert!(is_ids_error(run(&f32_ids)));
        let short = T::from_slice(&[0i32; 6], &[1, 6], &device).unwrap();
        assert!(is_ids_error(run(&short)));
        let flat = T::from_slice(&[0i32; 7], &[7], &device).unwrap();
        assert!(is_ids_error(run(&flat)));
        let good = T::from_slice(&[0i32; 7], &[1, 7], &device).unwrap();
        assert!(run(&good).is_ok());
    }
}
