//! [`RnnConfig`]: the shape and behavior of a recurrent layer stack.
//!
//! Field names and defaults follow `torch.nn.LSTM`.

use crate::error::{Error, Result};

/// Config of a recurrent layer stack, as `torch.nn.LSTM` takes it.
#[derive(Debug, Clone, PartialEq)]
pub struct RnnConfig {
    /// Features per input step.
    pub input_size: usize,
    /// Features per hidden state.
    pub hidden_size: usize,
    /// Stacked layers. Layer `k > 0` reads the output of layer `k - 1`.
    pub num_layers: usize,
    /// Whether each layer carries `bias_ih` and `bias_hh`.
    pub bias: bool,
    /// `true`: input and output are `[B, T, F]`. `false`: `[T, B, F]`.
    pub batch_first: bool,
    /// Dropout probability on each layer output except the last one.
    pub dropout: f64,
    /// Whether each layer also runs a reverse-time direction.
    pub bidirectional: bool,
}

impl RnnConfig {
    /// One unidirectional layer with bias, time-major, no dropout.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            num_layers: 1,
            bias: true,
            batch_first: false,
            dropout: 0.0,
            bidirectional: false,
        }
    }

    /// Return an error for a zero size, zero layers or a dropout outside `[0, 1]`.
    pub fn validate(&self) -> Result<()> {
        if self.input_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "input_size",
                reason: "must be at least 1, got 0".into(),
            });
        }
        if self.hidden_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "hidden_size",
                reason: "must be at least 1, got 0".into(),
            });
        }
        if self.num_layers == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_layers",
                reason: "must be at least 1, got 0".into(),
            });
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(Error::InvalidArgument {
                arg: "dropout",
                reason: format!("must be in [0, 1], got {}", self.dropout),
            });
        }
        Ok(())
    }

    /// 2 for a bidirectional stack, else 1.
    pub fn num_directions(&self) -> usize {
        if self.bidirectional { 2 } else { 1 }
    }

    /// Rows of each weight and bias: `4 * hidden_size`, gates `[i, f, g, o]`.
    pub fn gate_size(&self) -> usize {
        4 * self.hidden_size
    }

    /// Features layer `layer` reads: `input_size` for layer 0, else `dirs * hidden_size`.
    pub fn layer_input_size(&self, layer: usize) -> usize {
        if layer == 0 {
            self.input_size
        } else {
            self.num_directions() * self.hidden_size
        }
    }

    /// Features of each output step: `dirs * hidden_size`.
    pub fn output_size(&self) -> usize {
        self.num_directions() * self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_pytorch() {
        let cfg = RnnConfig::new(3, 5);
        assert_eq!(cfg.num_layers, 1);
        assert!(cfg.bias);
        assert!(!cfg.batch_first);
        assert_eq!(cfg.dropout, 0.0);
        assert!(!cfg.bidirectional);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn sizes_follow_directions() {
        let cfg = RnnConfig {
            num_layers: 3,
            bidirectional: true,
            ..RnnConfig::new(7, 4)
        };
        assert_eq!(cfg.num_directions(), 2);
        assert_eq!(cfg.gate_size(), 16);
        assert_eq!(cfg.layer_input_size(0), 7);
        assert_eq!(cfg.layer_input_size(2), 8);
        assert_eq!(cfg.output_size(), 8);
    }

    #[test]
    fn rejects_bad_values() {
        let bad = [
            RnnConfig::new(0, 4),
            RnnConfig::new(4, 0),
            RnnConfig {
                num_layers: 0,
                ..RnnConfig::new(4, 4)
            },
            RnnConfig {
                dropout: -0.1,
                ..RnnConfig::new(4, 4)
            },
            RnnConfig {
                dropout: 1.5,
                ..RnnConfig::new(4, 4)
            },
            RnnConfig {
                dropout: f64::NAN,
                ..RnnConfig::new(4, 4)
            },
        ];
        for cfg in bad {
            assert!(cfg.validate().is_err(), "{cfg:?} must be an error");
        }
    }
}
