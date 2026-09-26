//! The spaCy CNN tagger: `tok2vec`, then the tagger's linear output layer.
//!
//! One call runs one document, as `nlp(text)` does. The input is the
//! `FeatureExtractor` array: `n_tokens` rows of `n_attrs` `u64` IDs,
//! row-major. Dropout layers are inactive at inference and do not appear.

use super::config::SpacyTaggerConfig;
use super::embed::HashEmbedStack;
use super::encoder::WindowEncoder;
use super::maxout::check_shape;
use super::weights::SpacyTaggerWeights;
use crate::error::{Error, Result};
use crate::format::safetensors::SafeTensors;
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// `spacy.Tok2Vec.v2` (`MultiHashEmbed.v2` + `MaxoutWindowEncoder.v2`) and
/// `spacy.Tagger.v2`.
pub struct SpacyTagger<R: Runtime> {
    config: SpacyTaggerConfig,
    embed: HashEmbedStack<R>,
    encoder: WindowEncoder<R>,
    /// Output layer `W` as `[width, n_labels]`.
    softmax_w_t: Tensor<R>,
    softmax_b: Tensor<R>,
    n_labels: usize,
}

impl<R: Runtime<DType = DType>> SpacyTagger<R> {
    /// Builds the network from raw tensors, checking every shape against
    /// `config`.
    pub fn new<C>(
        client: &C,
        config: SpacyTaggerConfig,
        weights: SpacyTaggerWeights<R>,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        config.validate()?;
        let n_labels = weights.softmax_w.shape().first().copied().unwrap_or(0);
        if n_labels == 0 {
            return Err(Error::ModelError {
                reason: "spaCy tagger output layer softmax.W has no labels".into(),
            });
        }
        check_shape("softmax.W", &weights.softmax_w, &[n_labels, config.width])?;
        check_shape("softmax.b", &weights.softmax_b, &[n_labels])?;
        let embed = HashEmbedStack::new(client, &config, weights.tables, weights.embed_mix)?;
        let encoder = WindowEncoder::new(&config, weights.encode)?;
        let softmax_w_t = weights.softmax_w.transpose(0, 1)?.contiguous()?;
        Ok(Self {
            config,
            embed,
            encoder,
            softmax_w_t,
            softmax_b: weights.softmax_b,
            n_labels,
        })
    }

    /// Builds the network from an in-memory SafeTensors image with the
    /// tensor names `convert_tagger.py` writes.
    pub fn from_safetensors_bytes<C>(
        client: &C,
        config: SpacyTaggerConfig,
        image: Vec<u8>,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let mut st = SafeTensors::from_bytes(image)?;
        let weights = SpacyTaggerWeights::from_safetensors(
            &mut st,
            config.n_attrs(),
            config.depth,
            client.device(),
        )?;
        Self::new(client, config, weights)
    }

    /// The geometry the network was built with.
    pub fn config(&self) -> &SpacyTaggerConfig {
        &self.config
    }

    /// Number of output classes.
    pub fn n_labels(&self) -> usize {
        self.n_labels
    }

    /// Token count of `ids`, refusing a length that is not a whole number
    /// of rows.
    fn n_tokens(&self, ids: &[u64]) -> Result<usize> {
        let n_attrs = self.config.n_attrs();
        if !ids.len().is_multiple_of(n_attrs) {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "{} attribute IDs do not form rows of {n_attrs} columns",
                    ids.len()
                ),
            });
        }
        Ok(ids.len() / n_attrs)
    }

    /// `tok2vec` output, `[n_tokens, width]`. An empty document gives a
    /// `[0, width]` tensor.
    pub fn tok2vec<C>(&self, client: &C, ids: &[u64]) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let n = self.n_tokens(ids)?;
        if n == 0 {
            return Ok(Tensor::<R>::zeros(
                &[0, self.config.width],
                DType::F32,
                client.device(),
            )?);
        }
        let embedded = self.embed.forward(client, ids, n)?;
        self.encoder.forward(client, &embedded, n)
    }

    /// Tagger scores before softmax, `[n_tokens, n_labels]`. The tagger's
    /// `Softmax_v2` skips normalization at inference, so these are what
    /// spaCy takes the argmax of.
    pub fn logits<C>(&self, client: &C, ids: &[u64]) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let n = self.n_tokens(ids)?;
        if n == 0 {
            return Ok(Tensor::<R>::zeros(
                &[0, self.n_labels],
                DType::F32,
                client.device(),
            )?);
        }
        let h = self.tok2vec(client, ids)?;
        Ok(client.matmul_bias(&h, &self.softmax_w_t, &self.softmax_b)?)
    }

    /// Label index per token: the first maximum of each row of
    /// [`Self::logits`], as spaCy's `Tagger.predict` picks it.
    pub fn predict<C>(&self, client: &C, ids: &[u64]) -> Result<Vec<usize>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        if self.n_tokens(ids)? == 0 {
            return Ok(Vec::new());
        }
        let logits = self.logits(client, ids)?;
        let best = client.argmax(&logits, 1, false)?;
        best.to_vec::<i64>()
            .into_iter()
            .map(|i| {
                usize::try_from(i).map_err(|_| Error::ModelError {
                    reason: format!("argmax returned a negative label index {i}"),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::text::spacy_tagger::hash::bucket_keys;
    use crate::model::text::spacy_tagger::weights::MaxoutWeights;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const W: usize = 3;
    const P: usize = 2;

    fn config(depth: usize, pad: usize) -> SpacyTaggerConfig {
        SpacyTaggerConfig {
            width: W,
            depth,
            window_size: 1,
            embed_pieces: P,
            encode_pieces: P,
            seeds: vec![8, 9],
            pad,
            layer_norm_eps: 1e-8,
        }
    }

    /// Deterministic values in [-1, 1).
    fn values(n: usize, salt: u64) -> Vec<f32> {
        (0..n as u64)
            .map(|i| {
                let h =
                    (i + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt.wrapping_mul(0xbf58_476d);
                ((h >> 40) % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// One Maxout layer: `nI`, `W`, `b`, LayerNorm `G`, LayerNorm `b`.
    type RawMaxout = (usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    struct Raw {
        tables: Vec<(usize, Vec<f32>)>,
        maxouts: Vec<RawMaxout>,
        softmax: (Vec<f32>, Vec<f32>),
    }

    const LABELS: usize = 4;

    fn raw(depth: usize) -> Raw {
        let tables = vec![(7, values(7 * W, 1)), (5, values(5 * W, 2))];
        let mut maxouts = Vec::new();
        for (d, n_in) in std::iter::once(2 * W)
            .chain(std::iter::repeat_n(3 * W, depth))
            .enumerate()
        {
            let s = 10 + 4 * d as u64;
            maxouts.push((
                n_in,
                values(W * P * n_in, s),
                values(W * P, s + 1),
                values(W, s + 2),
                values(W, s + 3),
            ));
        }
        Raw {
            tables,
            maxouts,
            softmax: (values(LABELS * W, 99), values(LABELS, 100)),
        }
    }

    fn build(
        c: &CpuClient,
        d: &CpuDevice,
        cfg: SpacyTaggerConfig,
        raw: &Raw,
    ) -> SpacyTagger<CpuRuntime> {
        let t = |v: &[f32], s: &[usize]| Tensor::<CpuRuntime>::from_slice(v, s, d).unwrap();
        let mo = |(n_in, w, b, g, nb): &RawMaxout| MaxoutWeights {
            w: t(w, &[W, P, *n_in]),
            b: t(b, &[W, P]),
            norm_g: t(g, &[W]),
            norm_b: t(nb, &[W]),
        };
        let weights = SpacyTaggerWeights {
            tables: raw.tables.iter().map(|(r, v)| t(v, &[*r, W])).collect(),
            embed_mix: mo(&raw.maxouts[0]),
            encode: raw.maxouts[1..].iter().map(mo).collect(),
            softmax_w: t(&raw.softmax.0, &[LABELS, W]),
            softmax_b: t(&raw.softmax.1, &[LABELS]),
        };
        SpacyTagger::new(c, cfg, weights).unwrap()
    }

    /// thinc `Maxout(normalize=True)` on one row, in plain loops.
    fn maxout_row(x: &[f32], m: &RawMaxout) -> Vec<f32> {
        let (n_in, w, b, g, nb) = m;
        let best: Vec<f64> = (0..W)
            .map(|o| {
                (0..P)
                    .map(|p| {
                        let row = &w[(o * P + p) * n_in..(o * P + p + 1) * n_in];
                        let dot: f64 = row.iter().zip(x).map(|(a, b)| f64::from(a * b)).sum();
                        dot + f64::from(b[o * P + p])
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        let mean = best.iter().sum::<f64>() / W as f64;
        let var = best.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / W as f64 + 1e-8;
        (0..W)
            .map(|o| ((best[o] - mean) / var.sqrt() * f64::from(g[o]) + f64::from(nb[o])) as f32)
            .collect()
    }

    /// The whole of `tok2vec` in plain loops, padding included.
    fn reference(cfg: &SpacyTaggerConfig, raw: &Raw, ids: &[u64]) -> Vec<Vec<f32>> {
        let n = ids.len() / 2;
        let mut h: Vec<Vec<f32>> = vec![vec![0.0; W]; cfg.pad];
        for tok in ids.chunks(2) {
            let mut concat = Vec::new();
            for (a, &id) in tok.iter().enumerate() {
                let (rows, table) = &raw.tables[a];
                let mut v = [0f32; W];
                for k in bucket_keys(id, cfg.seeds[a], *rows) {
                    for (o, x) in v.iter_mut().enumerate() {
                        *x += table[k * W + o];
                    }
                }
                concat.extend_from_slice(&v);
            }
            h.push(maxout_row(&concat, &raw.maxouts[0]));
        }
        h.extend(std::iter::repeat_n(vec![0.0; W], cfg.pad));
        for block in &raw.maxouts[1..] {
            let zero = vec![0.0; W];
            let next: Vec<Vec<f32>> = (0..h.len())
                .map(|i| {
                    let prev = if i == 0 { &zero } else { &h[i - 1] };
                    let after = h.get(i + 1).unwrap_or(&zero);
                    let window: Vec<f32> = [prev.as_slice(), &h[i], after].concat();
                    let y = maxout_row(&window, block);
                    h[i].iter().zip(y).map(|(a, b)| a + b).collect()
                })
                .collect();
            h = next;
        }
        h[cfg.pad..cfg.pad + n].to_vec()
    }

    #[test]
    fn tok2vec_matches_loop_reference() {
        let (c, d) = cpu_setup();
        let raw = raw(2);
        for pad in [0, 2] {
            let cfg = config(2, pad);
            let model = build(&c, &d, cfg.clone(), &raw);
            let ids: Vec<u64> = vec![0, 1, 42, 7, u64::MAX, 3, 101, 0, 9, 9];
            let got: Vec<f32> = model.tok2vec(&c, &ids).unwrap().to_vec();
            let want: Vec<f32> = reference(&cfg, &raw, &ids).concat();
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-4, "pad {pad} element {i}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn predict_is_argmax_of_logits() {
        let (c, d) = cpu_setup();
        let raw = raw(1);
        let model = build(&c, &d, config(1, 1), &raw);
        let ids: Vec<u64> = vec![5, 6, 7, 8, 9, 10];
        let logits: Vec<f32> = model.logits(&c, &ids).unwrap().to_vec();
        let want: Vec<usize> = logits
            .chunks(LABELS)
            .map(|row| {
                let mut best = 0;
                for (j, v) in row.iter().enumerate() {
                    if *v > row[best] {
                        best = j;
                    }
                }
                best
            })
            .collect();
        assert_eq!(model.predict(&c, &ids).unwrap(), want);
        assert_eq!(model.n_labels(), LABELS);
    }

    #[test]
    fn empty_and_ragged_input() {
        let (c, d) = cpu_setup();
        let model = build(&c, &d, config(1, 1), &raw(1));
        assert_eq!(model.tok2vec(&c, &[]).unwrap().shape(), &[0, W]);
        assert_eq!(model.logits(&c, &[]).unwrap().shape(), &[0, LABELS]);
        assert!(model.predict(&c, &[]).unwrap().is_empty());
        assert!(model.tok2vec(&c, &[1, 2, 3]).is_err());
    }

    #[test]
    fn rejects_mismatched_weights() {
        let (c, d) = cpu_setup();
        let raw = raw(2);
        let t = |v: &[f32], s: &[usize]| Tensor::<CpuRuntime>::from_slice(v, s, &d).unwrap();
        let mo = |n_in: usize| MaxoutWeights {
            w: t(&values(W * P * n_in, 1), &[W, P, n_in]),
            b: t(&values(W * P, 2), &[W, P]),
            norm_g: t(&values(W, 3), &[W]),
            norm_b: t(&values(W, 4), &[W]),
        };
        let weights = SpacyTaggerWeights {
            tables: raw.tables.iter().map(|(r, v)| t(v, &[*r, W])).collect(),
            embed_mix: mo(2 * W),
            // One block for a depth-2 config.
            encode: vec![mo(3 * W)],
            softmax_w: t(&raw.softmax.0, &[LABELS, W]),
            softmax_b: t(&raw.softmax.1, &[LABELS]),
        };
        assert!(SpacyTagger::new(&c, config(2, 1), weights).is_err());
    }
}
