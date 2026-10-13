//! [`JointSchemaConfig`]: the sizes of a joint-schema head, as its
//! `joint_head_config.json` states them, and [`HeadDtype`], the dtype the
//! head computes in.

use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// The file name of the head's config inside a checkpoint directory.
pub const CONFIG_FILE: &str = "joint_head_config.json";

/// The file name of the head's weights inside a checkpoint directory.
pub const WEIGHTS_FILE: &str = "joint_head.safetensors";

/// The epsilon of every LayerNorm of the head: the torch default.
pub const LAYER_NORM_EPS: f32 = 1e-5;

/// The number of field types: noul, choice, score.
pub const FIELD_TYPES: usize = 3;

/// The sizes of a joint-schema head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSchemaConfig {
    /// The backbone hidden size `H` the head reads.
    pub hidden_size: usize,
    /// The head width `W` of option, field and memory rows.
    pub width: usize,
    /// The number of evidence-routing layers over the option rows.
    pub routing_layers: usize,
    /// The number of decoder layers over the field rows.
    pub layers: usize,
    /// The attention heads of every attention block. Divides `width`.
    pub heads: usize,
    /// The inner width of every feed-forward block.
    pub feedforward: usize,
}

/// The dtype the head computes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeadDtype {
    /// F32: every weight is cast to F32 once at load.
    #[default]
    F32,
    /// The dtype the checkpoint stores, as the release runs it: every
    /// weight is kept as stored and every op runs in that dtype. The
    /// checkpoint must store one float dtype for all tensors.
    Checkpoint,
}

fn config_error(path: &Path, reason: String) -> Error {
    Error::ModelError {
        reason: format!("{}: {reason}", path.display()),
    }
}

impl JointSchemaConfig {
    /// Reads and checks a `joint_head_config.json`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] naming the file when it cannot be read, does
    /// not parse, has an unknown or missing key, or fails
    /// [`Self::validate`].
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| config_error(path, format!("cannot read: {e}")))?;
        let config: Self = serde_json::from_str(&text)
            .map_err(|e| config_error(path, format!("does not parse: {e}")))?;
        config
            .validate()
            .map_err(|e| config_error(path, e.to_string()))?;
        Ok(config)
    }

    /// Checks that every size is at least 1 and that `heads` divides
    /// `width`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] naming the first size that fails.
    pub fn validate(&self) -> Result<()> {
        let sizes = [
            ("hidden_size", self.hidden_size),
            ("width", self.width),
            ("routing_layers", self.routing_layers),
            ("layers", self.layers),
            ("heads", self.heads),
            ("feedforward", self.feedforward),
        ];
        if let Some((name, _)) = sizes.iter().find(|(_, v)| *v == 0) {
            return Err(Error::InvalidArgument {
                arg: "config",
                reason: format!("{name} must be at least 1, got 0"),
            });
        }
        if !self.width.is_multiple_of(self.heads) {
            return Err(Error::InvalidArgument {
                arg: "config",
                reason: format!("heads {} must divide width {}", self.heads, self.width),
            });
        }
        Ok(())
    }

    /// The width of one attention head, `width / heads`.
    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASE: &str = r#"{"hidden_size": 4096, "width": 1024, "routing_layers": 2,
        "layers": 4, "heads": 16, "feedforward": 4096}"#;

    #[test]
    fn parses_the_release_config() {
        let c: JointSchemaConfig = serde_json::from_str(RELEASE).unwrap();
        c.validate().unwrap();
        assert_eq!(
            c,
            JointSchemaConfig {
                hidden_size: 4096,
                width: 1024,
                routing_layers: 2,
                layers: 4,
                heads: 16,
                feedforward: 4096,
            }
        );
        assert_eq!(c.head_dim(), 64);
    }

    #[test]
    fn bad_configs_are_refused() {
        let mut c: JointSchemaConfig = serde_json::from_str(RELEASE).unwrap();
        c.heads = 3;
        let e = c.validate().unwrap_err().to_string();
        assert!(e.contains("heads 3 must divide width 1024"), "{e}");
        c.heads = 16;
        c.layers = 0;
        let e = c.validate().unwrap_err().to_string();
        assert!(e.contains("layers must be at least 1"), "{e}");
        let extra = RELEASE.replace("\"heads\"", "\"dropout\": 0.1, \"heads\"");
        assert!(serde_json::from_str::<JointSchemaConfig>(&extra).is_err());
    }
}
