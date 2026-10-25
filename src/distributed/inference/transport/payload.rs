//! The payload of a request frame: token ids for the stage with the
//! embedding, an activation for every later stage.
//!
//! ```text
//! tokens:     [id: i64 LE] × seq_len
//! activation: [hidden_n: u32 LE][mlp_n: u32 LE][hidden: f32 LE × hidden_n][mlp: f32 LE × mlp_n]
//! ```
//!
//! `mlp` is the deferred MLP residual of the last layer the previous stage
//! ran. `mlp_n` of 0 means no residual.

use anyhow::{Result, anyhow, bail};

/// A hidden state on the wire, with the deferred MLP residual of the last
/// layer that ran.
#[derive(Debug, Clone, PartialEq)]
pub struct Activation {
    /// `[seq_len, hidden_size]` values in row-major order.
    pub hidden: Vec<f32>,
    /// `[seq_len, hidden_size]` values, or `None` before the first layer.
    pub mlp: Option<Vec<f32>>,
}

impl Activation {
    /// The wire bytes of this activation.
    ///
    /// # Errors
    ///
    /// More than `u32::MAX` values in `hidden` or `mlp`.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mlp = self.mlp.as_deref().unwrap_or(&[]);
        let count = |n: usize, what: &str| {
            u32::try_from(n).map_err(|_| anyhow!("an activation {what} of {n} values is too large"))
        };
        let hidden_n = count(self.hidden.len(), "hidden state")?;
        let mlp_n = count(mlp.len(), "MLP residual")?;
        let mut buf = Vec::with_capacity(8 + (self.hidden.len() + mlp.len()) * 4);
        buf.extend_from_slice(&hidden_n.to_le_bytes());
        buf.extend_from_slice(&mlp_n.to_le_bytes());
        for v in self.hidden.iter().chain(mlp) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Ok(buf)
    }

    /// The activation in `bytes`, which need no alignment.
    ///
    /// # Errors
    ///
    /// A buffer shorter than its counts say, or longer.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let Some((counts, values)) = bytes.split_first_chunk::<8>() else {
            bail!(
                "an activation of {} bytes has no 8-byte count header",
                bytes.len()
            );
        };
        let hidden_n = u32::from_le_bytes([counts[0], counts[1], counts[2], counts[3]]) as usize;
        let mlp_n = u32::from_le_bytes([counts[4], counts[5], counts[6], counts[7]]) as usize;
        let want = (hidden_n + mlp_n) * 4;
        if values.len() != want {
            bail!(
                "an activation of {hidden_n} hidden and {mlp_n} residual values needs {want} \
                 value bytes, and the frame holds {}",
                values.len()
            );
        }
        let (hidden, mlp) = values.split_at(hidden_n * 4);
        Ok(Self {
            hidden: f32s_le(hidden),
            mlp: (mlp_n > 0).then(|| f32s_le(mlp)),
        })
    }
}

fn f32s_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

/// What one request frame carries into a stage.
#[derive(Debug, Clone, PartialEq)]
pub enum StagePayload {
    /// Token ids, for the stage that holds the embedding.
    Tokens(Vec<i64>),
    /// The previous stage's output, for every later stage.
    Activation(Activation),
}

impl StagePayload {
    /// The wire bytes of this payload.
    ///
    /// # Errors
    ///
    /// As [`Activation::encode`].
    pub fn encode(&self) -> Result<Vec<u8>> {
        match self {
            Self::Tokens(ids) => Ok(ids.iter().flat_map(|id| id.to_le_bytes()).collect()),
            Self::Activation(a) => a.encode(),
        }
    }

    /// The payload in `bytes`: `seq_len` token ids when `tokens`, else an activation.
    ///
    /// # Errors
    ///
    /// Token bytes that are not `seq_len * 8`, or an activation
    /// [`Activation::decode`] refuses.
    pub fn decode(bytes: &[u8], seq_len: u32, tokens: bool) -> Result<Self> {
        if !tokens {
            return Activation::decode(bytes).map(Self::Activation);
        }
        let want = seq_len as usize * 8;
        if bytes.len() != want {
            bail!(
                "{seq_len} token ids need {want} bytes, and the frame holds {}",
                bytes.len()
            );
        }
        Ok(Self::Tokens(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&c| i64::from_le_bytes(c))
                .collect(),
        ))
    }

    /// Whether this payload is token ids.
    pub fn is_tokens(&self) -> bool {
        matches!(self, Self::Tokens(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_an_activation_round_trips() {
        let a = Activation {
            hidden: vec![1.0, -2.5, 3.25],
            mlp: Some(vec![0.5, 4.0, -0.0]),
        };
        assert_eq!(Activation::decode(&a.encode().unwrap()).unwrap(), a);
        let bare = Activation {
            hidden: vec![7.0],
            mlp: None,
        };
        assert_eq!(Activation::decode(&bare.encode().unwrap()).unwrap(), bare);
    }

    #[test]
    fn test_the_wire_is_little_endian_whatever_the_host() {
        let a = Activation {
            hidden: vec![1.0],
            mlp: None,
        };
        let mut want = vec![1, 0, 0, 0, 0, 0, 0, 0];
        want.extend_from_slice(&1.0f32.to_le_bytes());
        assert_eq!(a.encode().unwrap(), want);
    }

    #[test]
    fn test_a_misaligned_buffer_decodes() {
        let bytes = Activation {
            hidden: vec![7.0, 8.0],
            mlp: None,
        }
        .encode()
        .unwrap();
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&bytes);
        assert_eq!(
            Activation::decode(&shifted[1..]).unwrap().hidden,
            [7.0, 8.0]
        );
    }

    #[test]
    fn test_a_short_or_long_activation_is_refused() {
        let mut bytes = Activation {
            hidden: vec![1.0, 2.0],
            mlp: None,
        }
        .encode()
        .unwrap();
        bytes.push(0);
        assert!(Activation::decode(&bytes).is_err());
        bytes.truncate(bytes.len() - 2);
        assert!(Activation::decode(&bytes).is_err());
        assert!(Activation::decode(&bytes[..4]).is_err());
    }

    #[test]
    fn test_tokens_round_trip_and_check_their_count() {
        let p = StagePayload::Tokens(vec![3, -1, i64::MAX]);
        let bytes = p.encode().unwrap();
        assert_eq!(StagePayload::decode(&bytes, 3, true).unwrap(), p);
        let err = StagePayload::decode(&bytes, 2, true).unwrap_err();
        assert!(
            err.to_string().contains("2 token ids need 16 bytes"),
            "{err}"
        );
        assert!(p.is_tokens());
    }
}
