//! Geometry of the spaCy CNN tagger.

use serde::Deserialize;

use crate::error::{Error, Result};

/// Hyperparameters of `spacy.Tok2Vec.v2` + `spacy.Tagger.v2`.
///
/// Table row counts and the label count come from the weights. The JSON
/// written by `convert_tagger.py` (its `config` object) deserializes into
/// this type.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SpacyTaggerConfig {
    /// Width of every embedding table, of the mixing layer and of the
    /// encoder.
    pub width: usize,
    /// Number of residual `expand_window >> Maxout >> LayerNorm` blocks.
    pub depth: usize,
    /// Tokens on each side that `expand_window` concatenates.
    pub window_size: usize,
    /// Maxout pieces of the embedding mixing layer.
    pub embed_pieces: usize,
    /// Maxout pieces of each encoder block.
    pub encode_pieces: usize,
    /// `HashEmbed` seed of each table, one table per attribute column.
    pub seeds: Vec<u32>,
    /// Zero rows `with_array` puts before and after the encoder input.
    pub pad: usize,
    /// thinc `LayerNorm` variance epsilon.
    pub layer_norm_eps: f32,
}

impl SpacyTaggerConfig {
    /// Number of attribute columns: one `HashEmbed` table each.
    pub fn n_attrs(&self) -> usize {
        self.seeds.len()
    }

    /// Input width of the embedding mixing layer.
    pub fn embed_in(&self) -> usize {
        self.width * self.n_attrs()
    }

    /// Input width of an encoder block: the window of `2 * window_size + 1`
    /// rows side by side.
    pub fn encode_in(&self) -> usize {
        self.width * (2 * self.window_size + 1)
    }

    /// Refuses a zero width, depth-free or piece-free geometry and an empty
    /// seed list.
    pub fn validate(&self) -> Result<()> {
        let zero = [
            ("width", self.width),
            ("embed_pieces", self.embed_pieces),
            ("encode_pieces", self.encode_pieces),
            ("seeds.len()", self.seeds.len()),
        ]
        .into_iter()
        .find(|(_, v)| *v == 0);
        if let Some((name, _)) = zero {
            return Err(Error::InvalidArgument {
                arg: "config",
                reason: format!("spaCy tagger config has {name} = 0"),
            });
        }
        if !(self.layer_norm_eps.is_finite() && self.layer_norm_eps > 0.0) {
            return Err(Error::InvalidArgument {
                arg: "config",
                reason: format!(
                    "spaCy tagger layer_norm_eps must be finite and positive, got {}",
                    self.layer_norm_eps
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sm() -> SpacyTaggerConfig {
        SpacyTaggerConfig {
            width: 96,
            depth: 4,
            window_size: 1,
            embed_pieces: 3,
            encode_pieces: 3,
            seeds: vec![8, 9, 10, 11, 12, 13],
            pad: 4,
            layer_norm_eps: 1e-8,
        }
    }

    #[test]
    fn derived_widths() {
        let c = sm();
        assert_eq!(c.n_attrs(), 6);
        assert_eq!(c.embed_in(), 576);
        assert_eq!(c.encode_in(), 288);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn rejects_degenerate_geometry() {
        let mut c = sm();
        c.seeds.clear();
        assert!(c.validate().is_err());
        let mut c = sm();
        c.layer_norm_eps = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn parses_converter_json() {
        let json = r#"{"width":96,"depth":4,"window_size":1,"embed_pieces":3,
            "encode_pieces":3,"seeds":[8,9,10,11,12,13],"pad":4,"layer_norm_eps":1e-08}"#;
        let c: SpacyTaggerConfig =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("config JSON: {e}"));
        assert_eq!(c, sm());
    }
}
