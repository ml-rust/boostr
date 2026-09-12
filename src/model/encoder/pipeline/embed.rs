//! Pipeline state, special-token wrapping, and single-text embedding.

use super::super::config::EncoderConfig;
use super::super::model::{Encoder, EncoderClient};
use crate::error::Result;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use splintr::AnyTokenizer;

/// End-to-end text → embedding pipeline.
///
/// Owns an `Encoder` and a tokenizer, providing a simple `embed_text()` API
/// that handles tokenization, forward pass, and pooling in one call.
///
/// The tokenizer is a [`splintr::AnyTokenizer`] whatever its source — a
/// `tokenizer.json`, a packaged vocabulary, or the vocabulary embedded in the
/// GGUF the weights came from ([`Self::from_gguf`]) — so every model takes the
/// same path through this module.
pub struct EmbeddingPipeline<R: Runtime> {
    pub(super) encoder: Encoder<R>,
    pub(super) tokenizer: AnyTokenizer,
    pub(super) device: R::Device,
}

impl<R: Runtime<DType = DType>> EmbeddingPipeline<R> {
    /// Create a new embedding pipeline from an encoder, tokenizer, and device.
    pub fn new(encoder: Encoder<R>, tokenizer: AnyTokenizer, device: R::Device) -> Self {
        Self {
            encoder,
            tokenizer,
            device,
        }
    }

    /// Wrap a freshly-encoded sequence in the model's `[CLS]` / `[SEP]` tokens
    /// and truncate it to `max_seq`.
    ///
    /// BERT-family encoders are trained on `[CLS] tokens [SEP]` and never on a
    /// bare token run, so feeding one is out-of-distribution input. It does not
    /// fail loudly — it quietly degrades the representation. Measured on
    /// `all-MiniLM-L6-v2`, mean-pooled embeddings of four unrelated sentences
    /// sat at ~0.029 cosine distance without the wrapper and spread to ~0.20
    /// with it: without this, dense retrieval ranks near-randomly while looking
    /// perfectly healthy end to end.
    ///
    /// The tokenizer's own [`splintr::SpecialPolicy`] owns which tokens those
    /// are and where they go — a reranker builds its own `[CLS] q [SEP] d [SEP]`
    /// through the pair template, and the single-segment caller here applies the
    /// single template via [`splintr::SpecialPolicy::apply_single`]. A tokenizer
    /// with no such convention has an identity template, zero
    /// [`splintr::SpecialPolicy::single_overhead`], and the sequence passes
    /// through unchanged.
    ///
    /// Truncation happens AFTER reserving the template's slots, so the result is
    /// never longer than `max_seq` and never loses its trailing `[SEP]`.
    pub(super) fn wrap_special_tokens(&self, ids: Vec<u32>, max_seq: usize) -> Vec<u32> {
        let policy = self.tokenizer.policy();
        let mut ids = ids;
        ids.truncate(max_seq.saturating_sub(policy.single_overhead()));
        policy.apply_single(ids)
    }

    /// Embed a single text string → `[hidden_size]` f32 vector.
    pub fn embed_text<C>(&self, client: &C, text: &str) -> Result<Vec<f32>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + DequantOps<R>,
    {
        let max_seq = self.encoder.config().max_position_embeddings;
        let token_ids = self.wrap_special_tokens(self.tokenizer.encode_raw(text), max_seq);
        let seq_len = token_ids.len();
        let input: Vec<i64> = token_ids.into_iter().map(|t| t as i64).collect();
        let input_tensor = Tensor::<R>::from_slice(&input, &[1, seq_len], &self.device)?;

        // Single input: no padding, no mask needed.
        let embedding = self.encoder.embed_inference(client, &input_tensor, None)?;
        Ok(embedding.to_vec())
    }

    /// Returns a reference to the underlying encoder model.
    pub fn encoder(&self) -> &Encoder<R> {
        &self.encoder
    }

    /// Returns a reference to the tokenizer.
    pub fn tokenizer(&self) -> &AnyTokenizer {
        &self.tokenizer
    }

    /// Returns the encoder's configuration.
    pub fn config(&self) -> &EncoderConfig {
        self.encoder.config()
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::error::Error;
    use crate::format::GgufMetadata;
    use crate::format::extract_gguf_vocab;
    use crate::format::gguf::value::GgufValue;
    use crate::model::Pooling;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use splintr::{AnyTokenizer, Tokenize};

    /// Build a minimal CPU pipeline — hidden=8, 2 heads, 1 layer, `max_pos`
    /// positions — around `tokenizer`, using `pos_emb` (`max_pos * 8` values) as the
    /// position-embedding table.
    ///
    /// Every weight but the position table is uniform, so a test that wants
    /// position-sensitive output supplies its own table and the rest stays shared.
    pub(in super::super) fn build_pipeline(
        tokenizer: AnyTokenizer,
        max_pos: usize,
        pos_emb: Vec<f32>,
    ) -> (EmbeddingPipeline<CpuRuntime>, numr::runtime::cpu::CpuClient) {
        let (client, device) = cpu_setup();
        let vocab_size = tokenizer.vocab_size();

        let config = EncoderConfig {
            vocab_size,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: max_pos,
            ..Default::default()
        };

        let d = &device;
        let encoder = Encoder::from_weights(config, Pooling::Mean, |name| match name {
            "embeddings.word_embeddings.weight" => {
                Ok(Tensor::from_slice(&vec![0.1f32; vocab_size * 8], &[vocab_size, 8], d).unwrap())
            }
            "embeddings.position_embeddings.weight" => {
                Ok(Tensor::from_slice(&pos_emb, &[max_pos, 8], d).unwrap())
            }
            "embeddings.layer_norm.weight" => {
                Ok(Tensor::from_slice(&[1.0f32; 8], &[8], d).unwrap())
            }
            "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d).unwrap()),
            n if n.ends_with("query.weight")
                || n.ends_with("key.weight")
                || n.ends_with("value.weight")
                || n.ends_with("attention.output.dense.weight") =>
            {
                Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], d).unwrap())
            }
            n if n.ends_with("query.bias")
                || n.ends_with("key.bias")
                || n.ends_with("value.bias")
                || n.ends_with("attention.output.dense.bias")
                || n.ends_with("output.dense.bias") =>
            {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d).unwrap())
            }
            n if n.ends_with("LayerNorm.weight") => {
                Ok(Tensor::from_slice(&[1.0f32; 8], &[8], d).unwrap())
            }
            n if n.ends_with("LayerNorm.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d).unwrap())
            }
            n if n.ends_with("intermediate.dense.weight") => {
                Ok(Tensor::from_slice(&vec![0.02f32; 16 * 8], &[16, 8], d).unwrap())
            }
            n if n.ends_with("intermediate.dense.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 16], &[16], d).unwrap())
            }
            n if n.ends_with("output.dense.weight") => {
                Ok(Tensor::from_slice(&vec![0.02f32; 8 * 16], &[8, 16], d).unwrap())
            }
            _ => Err(Error::ModelError {
                reason: format!("unknown weight: {name}"),
            }),
        })
        .unwrap();

        (EmbeddingPipeline::new(encoder, tokenizer, device), client)
    }

    pub(in super::super) fn make_test_pipeline()
    -> (EmbeddingPipeline<CpuRuntime>, numr::runtime::cpu::CpuClient) {
        // Use cl100k_base vocab size for a realistic tokenizer
        let tokenizer = splintr::from_pretrained("cl100k_base").unwrap();
        build_pipeline(tokenizer, 64, vec![0.01f32; 64 * 8])
    }

    /// Build a pipeline whose position embeddings are non-uniform so that
    /// unmasked padding produces a detectably different mean-pool output.
    pub(in super::super) fn make_pipeline_with_distinct_positions()
    -> (EmbeddingPipeline<CpuRuntime>, numr::runtime::cpu::CpuClient) {
        let tokenizer = splintr::from_pretrained("cl100k_base").unwrap();

        // Position embeddings: position i has value (i+1) * 0.1 in all 8 dims,
        // so different positions produce distinctly different hidden states.
        let mut pos_emb = vec![0.0f32; 64 * 8];
        for pos in 0..64usize {
            let v = (pos + 1) as f32 * 0.1;
            for dim in 0..8usize {
                pos_emb[pos * 8 + dim] = v;
            }
        }

        build_pipeline(tokenizer, 64, pos_emb)
    }

    #[test]
    fn test_embed_text_returns_hidden_size() {
        let (pipeline, client) = make_test_pipeline();
        let emb = pipeline.embed_text(&client, "hello").unwrap();
        assert_eq!(emb.len(), 8);
    }

    /// Regression guard: a single-text call (no padding path) must not regress.
    #[test]
    fn embed_text_single_is_stable() {
        let (pipeline, client) = make_pipeline_with_distinct_positions();
        let e1 = pipeline.embed_text(&client, "hello world").unwrap();
        let e2 = pipeline.embed_text(&client, "hello world").unwrap();
        assert_eq!(e1, e2);
    }

    /// Sequence length this BERT-family fixture is capped at — small enough that a
    /// handful of repeated words overruns it.
    const BERT_MAX_SEQ: usize = 16;

    /// A synthetic `tokenizer.ggml.*` block for a BERT (WordPiece) vocabulary,
    /// built through the same seam a real GGUF takes: `extract_gguf_vocab` lifts the
    /// metadata, `splintr::from_gguf_vocab` decides what it means.
    ///
    /// Synthetic rather than a checked-in model file so the invariant below runs in
    /// the ordinary test suite, with no external download to gate it on.
    fn bert_gguf_tokenizer() -> AnyTokenizer {
        let mut metadata = GgufMetadata::default();
        metadata.kv.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("bert".into()),
        );
        metadata.kv.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(
                ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the"]
                    .iter()
                    .map(|t| GgufValue::String((*t).to_string()))
                    .collect(),
            ),
        );

        let vocab = extract_gguf_vocab(&metadata).expect("extract vocab");
        splintr::from_gguf_vocab(vocab).expect("build tokenizer")
    }

    /// The `[CLS] … [SEP]` wrapper must survive truncation.
    ///
    /// BERT-family encoders never saw a bare token run in training, and they do not
    /// fail on one — they quietly return a degraded vector (see the note on
    /// `wrap_special_tokens`). The failure mode that hides best is length-driven:
    /// wrapping first and truncating afterwards produces a correct-looking sequence
    /// for every short input and silently drops the trailing `[SEP]` on every long
    /// one, so only an over-length case pins the ordering down.
    #[test]
    fn bert_gguf_sequence_keeps_cls_and_sep_through_truncation() {
        let tokenizer = bert_gguf_tokenizer();
        let cls = tokenizer.special_token_id("[CLS]").expect("[CLS] id");
        let sep = tokenizer.special_token_id("[SEP]").expect("[SEP] id");

        let (pipeline, client) =
            build_pipeline(tokenizer, BERT_MAX_SEQ, vec![0.01f32; BERT_MAX_SEQ * 8]);
        let max_seq = pipeline.config().max_position_embeddings;

        // Short input: wrapped, and nothing is dropped.
        let short =
            pipeline.wrap_special_tokens(pipeline.tokenizer().encode_raw("the the"), max_seq);
        assert_eq!(short.first(), Some(&cls), "sequence must open with [CLS]");
        assert_eq!(short.last(), Some(&sep), "sequence must close with [SEP]");
        assert_eq!(short.len(), 4, "[CLS] the the [SEP]");

        // Over-length input: truncated to exactly max_seq, still closed by [SEP].
        let long_text = "the ".repeat(max_seq * 4);
        let long =
            pipeline.wrap_special_tokens(pipeline.tokenizer().encode_raw(&long_text), max_seq);
        assert_eq!(long.len(), max_seq, "truncated sequence must fit max_seq");
        assert_eq!(long.first(), Some(&cls), "sequence must open with [CLS]");
        assert_eq!(
            long.last(),
            Some(&sep),
            "truncation must reserve the [SEP] slot, not overwrite it"
        );

        // And the whole path still runs end to end at exactly max_seq.
        let emb = pipeline.embed_text(&client, &long_text).unwrap();
        assert_eq!(emb.len(), 8);
    }
}
