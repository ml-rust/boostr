//! The forward of a joint-schema head over one record: memory and span
//! vectors, option routing, field summaries, the joint decoder, and the
//! per-option logits, with every intermediate optionally recorded under
//! the reference's stage names.

use super::head::JointSchemaHead;
use super::item::JointSchemaItem;
use super::layout::{Layout, cast_to, l2_normalize, row_dots, span_means};
use crate::error::Result;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The `eps` of torch `F.normalize` in the prior.
const NORMALIZE_EPS: f64 = 1e-12;

/// The `eps` of torch `F.cosine_similarity`.
const COSINE_EPS: f64 = 1e-8;

/// Named intermediates of one forward, in stage order.
pub type Stages<R> = Vec<(String, Tensor<R>)>;

/// Records stages when asked to.
struct Recorder<'a, R: Runtime> {
    stages: Option<&'a mut Stages<R>>,
}

impl<R: Runtime> Recorder<'_, R> {
    fn put(&mut self, name: impl Into<String>, t: &Tensor<R>) {
        if let Some(stages) = self.stages.as_mut() {
            stages.push((name.into(), t.clone()));
        }
    }
}

impl<R: Runtime<DType = DType>> JointSchemaHead<R> {
    /// Per-option logits `[N]` in the head dtype, options concatenated in
    /// field order, for `hidden` `[L, H]` (every position of the record)
    /// and `lexical` `[ids, H]` (the output-projection rows of every
    /// option's ids, in option order). Inputs are checked by the caller.
    pub(super) fn forward<C: ModelClient<R>>(
        &self,
        client: &C,
        hidden: &Tensor<R>,
        lexical: &Tensor<R>,
        items: &[JointSchemaItem],
        stages: Option<&mut Stages<R>>,
    ) -> Result<Tensor<R>> {
        let mut rec = Recorder { stages };
        let dt = self.dtype;
        let device = hidden.device();
        let layout = Layout::new(items);
        let (fields, total, widest) = (items.len(), layout.total(), layout.widest);
        let width = self.config.width;
        let len = hidden.shape()[0];
        let owner = Tensor::<R>::from_slice(&layout.owner, &[total], device)?;

        let normed = self
            .hidden_norm
            .forward(client, &cast_to(client, hidden, dt)?)?;
        rec.put("hidden_normed", &normed);
        let memory = self.memory_projection.forward(client, &normed)?;
        rec.put("memory", &memory);
        let global = normed.narrow(0, len - 1, 1)?.contiguous()?;
        rec.put("global_vector", &global);
        let questions = span_means(client, &normed, &layout.questions, dt)?;
        rec.put("question_vectors", &questions);
        let contexts = span_means(client, &normed, &layout.options, dt)?;
        rec.put("option_context_vectors", &contexts);
        let lexicals = span_means(client, lexical, &layout.lexical, dt)?;
        rec.put("option_lexical_vectors", &lexicals);

        let context_proj = self.option_context_projection.forward(client, &contexts)?;
        rec.put("option_context_proj", &context_proj);
        let lexical_proj = self.option_lexical_projection.forward(client, &lexicals)?;
        rec.put("option_lexical_proj", &lexical_proj);
        let question_proj = self
            .option_question_projection
            .forward(client, &questions)?;
        rec.put("option_question_proj", &question_proj);
        let mut routed = client.add(
            &client.add(&context_proj, &lexical_proj)?,
            &client.index_select(&question_proj, 0, &owner)?,
        )?;
        rec.put("option_queries", &routed);
        for (i, layer) in self.evidence_layers.iter().enumerate() {
            routed = layer.forward(client, &routed, &memory)?;
            rec.put(format!("routed_{}", i + 1), &routed);
        }

        // Field summaries: each field's options weighted by a softmax over
        // that field alone, the fields padded to `widest` with -inf scores
        // and zero rows.
        let base = self.question_projection.forward(client, &questions)?;
        rec.put("base_fields", &base);
        let scores = client.div_scalar(
            &row_dots(client, &routed, &client.index_select(&base, 0, &owner)?)?,
            (width as f64).sqrt(),
        )?;
        let (slots, real) = layout.padding();
        let slots = Tensor::<R>::from_slice(&slots, &[fields * widest], device)?;
        let real = Tensor::<R>::from_slice(&real, &[total], device)?;
        let sentinel = Tensor::<R>::full_scalar(&[1], dt, f64::NEG_INFINITY, device)?;
        let padded_scores = client
            .index_select(&client.cat(&[&scores, &sentinel], 0)?, 0, &slots)?
            .reshape(&[fields, widest])?;
        let weights = client.softmax(&padded_scores, -1)?;
        rec.put(
            "summary_weights",
            &client.index_select(&weights.reshape(&[fields * widest])?, 0, &real)?,
        );
        let zero_row = Tensor::<R>::zeros(&[1, width], dt, device)?;
        let padded_options = client
            .index_select(&client.cat(&[&routed, &zero_row], 0)?, 0, &slots)?
            .reshape(&[fields, widest, width])?;
        let summaries = client
            .matmul(&weights.reshape(&[fields, 1, widest])?, &padded_options)?
            .reshape(&[fields, width])?;
        rec.put("option_summaries", &summaries);
        let summaries = self.option_summary_norm.forward(client, &summaries)?;
        rec.put("option_summaries_normed", &summaries);
        let global_proj = self.global_projection.forward(client, &global)?;
        rec.put("global_projection", &global_proj);
        let types = Tensor::<R>::from_slice(&layout.types, &[fields], device)?;
        let type_rows = client.index_select(&self.type_embedding, 0, &types)?;
        rec.put("type_embedding", &type_rows);
        let mut x = client.add(
            &client.add(&client.add(&base, &summaries)?, &global_proj)?,
            &type_rows,
        )?;
        rec.put("fields_in", &x);
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(client, &x, &memory)?;
            rec.put(format!("decoder_{}", i + 1), &x);
        }
        let fields_normed = self.field_norm.forward(client, &x)?;
        rec.put("fields_normed", &fields_normed);
        let options = self.option_norm.forward(client, &routed)?;
        rec.put("options_normed", &options);

        // Scoring: a lexical prior plus the gated joint term.
        let anchors = l2_normalize(client, &client.add(&questions, &global)?, NORMALIZE_EPS)?;
        let prior = client.mul_scalar(
            &row_dots(
                client,
                &l2_normalize(client, &lexicals, NORMALIZE_EPS)?,
                &client.index_select(&anchors, 0, &owner)?,
            )?,
            self.prior_scale,
        )?;
        rec.put("prior", &prior);
        let field_rows = client.index_select(&fields_normed, 0, &owner)?;
        let cosine = row_dots(
            client,
            &l2_normalize(client, &field_rows, COSINE_EPS)?,
            &l2_normalize(client, &options, COSINE_EPS)?,
        )?;
        rec.put("cosine", &cosine);
        let features = client.cat(
            &[
                &field_rows,
                &options,
                &client.mul(&field_rows, &options)?,
                &client.abs(&client.sub(&field_rows, &options)?)?,
            ],
            -1,
        )?;
        let residual = self
            .residual_scorer
            .forward(client, &features)?
            .reshape(&[total])?;
        rec.put("residual", &residual);
        let joint = client.add(&client.mul_scalar(&cosine, self.joint_scale)?, &residual)?;
        let logits = client.add(&prior, &client.mul_scalar(&joint, self.gate)?)?;
        rec.put("logits", &logits);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::super::blocks::tests::{HostWeights, host_ff, host_linear, host_norm};
    use super::super::config::JointSchemaConfig;
    use super::super::decoder_layer::tests::host_decoder;
    use super::super::routing::tests::host_routing;
    use super::super::weights::tests::{
        SCALARS, tiny_config, tiny_head, tiny_host, tiny_inputs, tiny_items,
    };
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use std::ops::Range;

    fn means(x: &[f64], d: usize, spans: &[Range<usize>]) -> Vec<f64> {
        let mut out = Vec::with_capacity(spans.len() * d);
        for s in spans {
            for i in 0..d {
                out.push(s.clone().map(|p| x[p * d + i]).sum::<f64>() / s.len() as f64);
            }
        }
        out
    }

    fn unit(x: &[f64]) -> Vec<f64> {
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        x.iter().map(|v| v / norm).collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// The whole head on the host in f64, step by step as the reference.
    fn host_logits(
        m: &HostWeights,
        c: &JointSchemaConfig,
        hidden: &[f64],
        lexical: &[f64],
        items: &[JointSchemaItem],
    ) -> Vec<f64> {
        let (h, w) = (c.hidden_size, c.width);
        let len = hidden.len() / h;
        let layout = Layout::new(items);
        let (q_n, n) = (items.len(), layout.total());
        let owner = |j: usize| layout.owner[j] as usize;
        let normed = host_norm(m, hidden, len, "hidden_norm");
        let memory = host_linear(m, &normed, len, "memory_projection", h, false);
        let g = normed[(len - 1) * h..].to_vec();
        let qv = means(&normed, h, &layout.questions);
        let ctx = means(&normed, h, &layout.options);
        let lex = means(lexical, h, &layout.lexical);
        let a = host_linear(m, &ctx, n, "option_context_projection", h, false);
        let b = host_linear(m, &lex, n, "option_lexical_projection", h, false);
        let oq = host_linear(m, &qv, q_n, "option_question_projection", h, false);
        let mut r: Vec<f64> = (0..n * w)
            .map(|i| a[i] + b[i] + oq[owner(i / w) * w + i % w])
            .collect();
        for i in 0..c.routing_layers {
            host_routing(m, c, i, &mut r, &memory);
        }
        let base = host_linear(m, &qv, q_n, "question_projection", h, false);
        let mut summaries = vec![0.0; q_n * w];
        for (q, s_row) in summaries.chunks_mut(w).enumerate() {
            let mine: Vec<usize> = (0..n).filter(|&j| owner(j) == q).collect();
            let s: Vec<f64> = mine
                .iter()
                .map(|&j| {
                    dot(&r[j * w..(j + 1) * w], &base[q * w..(q + 1) * w]) / (w as f64).sqrt()
                })
                .collect();
            let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - top).exp()).sum();
            for (k, &j) in mine.iter().enumerate() {
                let weight = (s[k] - top).exp() / z;
                for (v, x) in s_row.iter_mut().zip(&r[j * w..(j + 1) * w]) {
                    *v += weight * x;
                }
            }
        }
        let sn = host_norm(m, &summaries, q_n, "option_summary_norm");
        let gp = host_linear(m, &g, 1, "global_projection", h, false);
        let te = &m["type_embedding.weight"];
        let mut x: Vec<f64> = (0..q_n * w)
            .map(|i| base[i] + sn[i] + gp[i % w] + te[layout.types[i / w] as usize * w + i % w])
            .collect();
        for i in 0..c.layers {
            host_decoder(m, c, i, &mut x, &memory);
        }
        let fields = host_norm(m, &x, q_n, "field_norm");
        let options = host_norm(m, &r, n, "option_norm");
        let scale = |v: f64| v.min(100f64.ln()).exp();
        let gate = 1.0 / (1.0 + (-SCALARS.residual_gate).exp());
        (0..n)
            .map(|j| {
                let q = owner(j);
                let anchor: Vec<f64> = (0..h).map(|i| qv[q * h + i] + g[i]).collect();
                let prior = scale(SCALARS.prior_logit_scale)
                    * dot(&unit(&anchor), &unit(&lex[j * h..(j + 1) * h]));
                let fr = &fields[q * w..(q + 1) * w];
                let or = &options[j * w..(j + 1) * w];
                let cos = dot(&unit(fr), &unit(or));
                let mut feat = fr.to_vec();
                feat.extend_from_slice(or);
                feat.extend(fr.iter().zip(or).map(|(a, b)| a * b));
                feat.extend(fr.iter().zip(or).map(|(a, b)| (a - b).abs()));
                let res = host_ff(m, &feat, 1, "residual_scorer.0", "residual_scorer.3", 4 * w)[0];
                prior + gate * (scale(SCALARS.joint_logit_scale) * cos + res)
            })
            .collect()
    }

    /// The device forward against the host evaluation of the same head,
    /// and the stage list in the reference's order.
    #[test]
    fn forward_matches_host_and_names_every_stage() {
        let (client, device) = cpu_setup();
        let h = 16;
        let head = tiny_head(&client, &device, h);
        let items = tiny_items();
        let (hidden, lexical) = tiny_inputs(&device, h, &items);
        let got = head.logits(&client, &hidden, &lexical, &items).unwrap();
        assert_eq!(got.counts, vec![3, 2, 1]);
        let got = got.flat.to_vec::<f32>();

        let c = tiny_config(h);
        let to64 = |t: &Tensor<CpuRuntime>| -> Vec<f64> {
            t.to_vec::<f32>().iter().map(|&x| f64::from(x)).collect()
        };
        let want = host_logits(&tiny_host(&c), &c, &to64(&hidden), &to64(&lexical), &items);
        // The cosine term carries a scale of 100 times a gate near 0.38, so
        // an F32 cosine error near 1e-6 reaches the logit near 4e-5.
        assert_eq!(got.len(), want.len());
        for (j, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((f64::from(*g) - w).abs() < 1e-3, "option {j}: {g} vs {w}");
        }

        let stages = head.stages(&client, &hidden, &lexical, &items).unwrap();
        let names: Vec<&str> = stages.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "hidden_normed",
                "memory",
                "global_vector",
                "question_vectors",
                "option_context_vectors",
                "option_lexical_vectors",
                "option_context_proj",
                "option_lexical_proj",
                "option_question_proj",
                "option_queries",
                "routed_1",
                "routed_2",
                "base_fields",
                "summary_weights",
                "option_summaries",
                "option_summaries_normed",
                "global_projection",
                "type_embedding",
                "fields_in",
                "decoder_1",
                "decoder_2",
                "fields_normed",
                "options_normed",
                "prior",
                "cosine",
                "residual",
                "logits",
            ]
        );
        let weights = stages[13].1.to_vec::<f32>();
        assert_eq!(weights.len(), 6);
        assert!((weights[..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((weights[3..5].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(weights[5], 1.0);
        assert_eq!(stages[26].1.to_vec::<f32>(), got);
    }
}
