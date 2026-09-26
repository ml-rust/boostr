//! Options for [`super::greedy::generate_greedy`].

/// Controls one autoregressive generation run.
///
/// The KV cache's dtype is not here: `generate_greedy` reads it straight off
/// the model via `LoadedModel::dtype()`, which always matches the
/// checkpoint's activation dtype.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Maximum number of tokens to generate after the prompt.
    pub max_new_tokens: usize,
    /// Generation stops the step after any of these token ids is chosen.
    /// The stop token itself is included in the returned sequence.
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            stop_tokens: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_cap_at_256_tokens_with_no_stop_tokens() {
        let opts = GenerateOptions::default();
        assert_eq!(opts.max_new_tokens, 256);
        assert!(opts.stop_tokens.is_empty());
    }
}
