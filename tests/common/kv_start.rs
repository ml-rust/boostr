//! Reference for the `kv_start` (per-row left padding) argument of
//! `FlashAttentionOps::flash_attention_fwd`.
//!
//! Row `b` of a left-padded batch must equal the unpadded attention over the
//! keys from `kv_start[b]` on. The reference therefore runs each row alone
//! (`B = 1`) on the SAME backend with `kv_start = None`, over K/V narrowed to
//! `[kv_start[b], S_k)`. Query positions are absolute
//! (`S_k - S_q + i`), so with `S_k' = S_k - kv_start[b]` the causal and
//! window bounds shift by exactly `kv_start[b]`, which is what narrowing K/V
//! does. Under a causal mask the query rows whose position is below the
//! start see no key at all; they are dropped from the reference call (Q is
//! narrowed too, which keeps the offset consistent) and expected to be all
//! zeros with an LSE of `-inf`.
//!
//! One constraint on non-causal windowed cases: the key offset saturates at
//! zero when `S_k' < S_q`, so a window would then measure from a different
//! position in the sliced call. Keep `S_k - kv_start[b] >= S_q` there. A
//! causal case is always consistent, since the dead rows are dropped along
//! with the padded keys.

use boostr::ops::AttnOutLayout;
use boostr::ops::traits::attention::flash::FlashAttentionOps;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Geometry of one left-padded case.
#[derive(Clone, Copy, Debug)]
pub struct KvStartGeom {
    pub batch: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub window: usize,
}

/// Deterministic values, distinct per index and per seed.
pub fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// Reads a tensor back as F32, casting first when it is stored narrower.
pub fn read_f32<R>(t: &Tensor<R>) -> Vec<f32>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    if t.dtype() == DType::F32 {
        t.to_vec::<f32>()
    } else {
        t.to_dtype(DType::F32)
            .expect("cast result back to F32")
            .to_vec::<f32>()
    }
}

/// Query rows of batch row `b` that see no key: with `causal`, the rows
/// whose absolute position `S_k - S_q + i` is below `start`; without it,
/// every row once `start` reaches `S_k`.
fn dead_rows(g: &KvStartGeom, start: usize) -> usize {
    if start >= g.seq_k {
        return g.seq_q;
    }
    if !g.causal {
        return 0;
    }
    let key_offset = g.seq_k.saturating_sub(g.seq_q);
    start.saturating_sub(key_offset).min(g.seq_q)
}

/// Expected `(output [B, H, S_q, D], lse [B, H, S_q])` for a left-padded
/// batch, computed row by row as described in the module doc. `q`, `k`, `v`
/// are the full padded tensors in the dtype under test.
pub fn sliced_reference<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    g: &KvStartGeom,
    kv_start: &[i32],
) -> (Vec<f32>, Vec<f32>)
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
    C: FlashAttentionOps<R>,
{
    let row_len = g.num_heads * g.seq_q * g.head_dim;
    let mut out = vec![0.0f32; g.batch * row_len];
    let mut lse = vec![f32::NEG_INFINITY; g.batch * g.num_heads * g.seq_q];
    for b in 0..g.batch {
        let start = usize::try_from(kv_start[b].max(0)).expect("start fits usize");
        let dead = dead_rows(g, start);
        if dead == g.seq_q {
            continue;
        }
        let live = g.seq_q - dead;
        let keys = g.seq_k - start.min(g.seq_k);
        let q_b = q
            .narrow(0, b, 1)
            .expect("narrow q batch")
            .narrow(2, dead, live)
            .expect("narrow q rows")
            .contiguous()
            .expect("q contiguous");
        let k_b = k
            .narrow(0, b, 1)
            .expect("narrow k batch")
            .narrow(2, start, keys)
            .expect("narrow k keys")
            .contiguous()
            .expect("k contiguous");
        let v_b = v
            .narrow(0, b, 1)
            .expect("narrow v batch")
            .narrow(2, start, keys)
            .expect("narrow v keys")
            .contiguous()
            .expect("v contiguous");
        let (o_b, l_b) = client
            .flash_attention_fwd(
                &q_b,
                &k_b,
                &v_b,
                g.num_heads,
                g.num_kv_heads,
                g.head_dim,
                g.causal,
                g.window,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
            .unwrap_or_else(|e| panic!("reference flash_attention_fwd failed for row {b}: {e}"));
        let o_b = read_f32(&o_b);
        let l_b = read_f32(&l_b);
        for h in 0..g.num_heads {
            for i in 0..live {
                let src = (h * live + i) * g.head_dim;
                let dst = b * row_len + (h * g.seq_q + dead + i) * g.head_dim;
                out[dst..dst + g.head_dim].copy_from_slice(&o_b[src..src + g.head_dim]);
                lse[(b * g.num_heads + h) * g.seq_q + dead + i] = l_b[h * live + i];
            }
        }
    }
    (out, lse)
}

/// Largest absolute difference, failing on any non-finite kernel value.
/// `-inf` is accepted only where the reference is `-inf` too (a dead row's
/// LSE).
pub fn max_abs_diff(actual: &[f32], expected: &[f32], label: &str) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let mut max = 0.0f32;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if e.is_infinite() {
            assert_eq!(a, e, "{label}: index {i}: kernel {a} vs reference {e}");
            continue;
        }
        assert!(
            a.is_finite(),
            "{label}: kernel produced non-finite {a} at index {i} (reference {e})"
        );
        max = max.max((a - e).abs());
    }
    max
}

/// Head-major view of a `[B, S_q, H, D]` result, so both layouts compare
/// against the same head-major reference.
pub fn token_major_to_head_major(data: &[f32], g: &KvStartGeom) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for b in 0..g.batch {
        for s in 0..g.seq_q {
            for h in 0..g.num_heads {
                let src = ((b * g.seq_q + s) * g.num_heads + h) * g.head_dim;
                let dst = ((b * g.num_heads + h) * g.seq_q + s) * g.head_dim;
                out[dst..dst + g.head_dim].copy_from_slice(&data[src..src + g.head_dim]);
            }
        }
    }
    out
}
