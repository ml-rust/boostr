//! `MultiHashEmbed.v2` without static vectors: one `HashEmbed` per
//! attribute column, concatenated, then the Maxout mixing layer.

use super::config::SpacyTaggerConfig;
use super::hash::bucket_keys;
use super::maxout::{MaxoutNorm, check_shape};
use super::weights::MaxoutWeights;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Keys `HashEmbed` sums per ID.
const KEYS_PER_ID: usize = 4;

/// The hash-embedding stage of `tok2vec`.
pub(super) struct HashEmbedStack<R: Runtime> {
    /// Every table stacked on the row axis, `[sum(rows), width]`.
    table: Tensor<R>,
    /// First row of each table in `table`.
    offsets: Vec<usize>,
    /// Row count of each table.
    rows: Vec<usize>,
    seeds: Vec<u32>,
    width: usize,
    mix: MaxoutNorm<R>,
}

impl<R: Runtime<DType = DType>> HashEmbedStack<R> {
    pub(super) fn new<C>(
        client: &C,
        config: &SpacyTaggerConfig,
        tables: Vec<Tensor<R>>,
        mix: MaxoutWeights<R>,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        if tables.len() != config.n_attrs() {
            return Err(Error::ModelError {
                reason: format!(
                    "spaCy tagger has {} HashEmbed tables for {} seeds",
                    tables.len(),
                    config.n_attrs()
                ),
            });
        }
        let mut offsets = Vec::with_capacity(tables.len());
        let mut rows = Vec::with_capacity(tables.len());
        let mut total = 0usize;
        for (i, t) in tables.iter().enumerate() {
            let n = t.shape().first().copied().unwrap_or(0);
            if n == 0 {
                return Err(Error::ModelError {
                    reason: format!("spaCy tagger table embed.{i}.E has no rows"),
                });
            }
            check_shape(&format!("embed.{i}.E"), t, &[n, config.width])?;
            offsets.push(total);
            rows.push(n);
            total += n;
        }
        let refs: Vec<&Tensor<R>> = tables.iter().collect();
        let table = client.cat(&refs, 0)?;
        let mix = MaxoutNorm::new(
            "embed.maxout",
            mix,
            (config.width, config.embed_pieces, config.embed_in()),
            config.layer_norm_eps,
        )?;
        Ok(Self {
            table,
            offsets,
            rows,
            seeds: config.seeds.clone(),
            width: config.width,
            mix,
        })
    }

    /// Row indices into the stacked table for `ids`, ordered token, then
    /// attribute, then key.
    fn indices(&self, ids: &[u64]) -> Vec<i64> {
        let n_attrs = self.seeds.len();
        let mut out = Vec::with_capacity(ids.len() * KEYS_PER_ID);
        for (j, &id) in ids.iter().enumerate() {
            let a = j % n_attrs;
            let base = self.offsets[a];
            for key in bucket_keys(id, self.seeds[a], self.rows[a]) {
                out.push((base + key) as i64);
            }
        }
        out
    }

    /// `ids` is `[n_tokens, n_attrs]` row-major; returns `[n_tokens, width]`.
    /// `n_tokens` is non-zero and `ids.len() == n_tokens * n_attrs`.
    pub(super) fn forward<C>(&self, client: &C, ids: &[u64], n_tokens: usize) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let n_attrs = self.seeds.len();
        let index = self.indices(ids);
        let index = Tensor::<R>::from_slice(&index, &[index.len()], self.table.device())?;
        let rows = client.index_select(&self.table, 0, &index)?;
        let rows = rows.reshape(&[n_tokens, n_attrs, KEYS_PER_ID, self.width])?;
        let summed = client.sum(&rows, &[2], false)?;
        let concat = summed.reshape(&[n_tokens, n_attrs * self.width])?;
        self.mix.forward(client, &concat)
    }
}
