//! Phoneme-string to token-id lookup, the contract a G2P front end and a
//! phoneme-driven model agree on.
//!
//! Lives with the models rather than the G2P pipeline: `boostr-audio`'s
//! phonemizer produces the strings, a model's vocabulary (see
//! [`KokoroPhonemeVocab`](super::kokoro::KokoroPhonemeVocab)) owns the ids.

/// Trait implemented by Kokoro-specific phoneme vocabularies.
pub trait PhonemeVocab {
    fn lookup(&self, phoneme: &str) -> Option<u32>;
}

impl PhonemeVocab for std::collections::HashMap<String, u32> {
    fn lookup(&self, phoneme: &str) -> Option<u32> {
        self.get(phoneme).copied()
    }
}

/// Map an IPA phoneme token to its id in a user-supplied phoneme vocabulary.
///
/// Returns `None` for tokens missing from `vocab`. Callers typically fall back
/// to an `<unk>` id or skip silently.
pub fn phonemes_to_ids<V>(tokens: &[String], vocab: &V) -> Vec<Option<u32>>
where
    V: PhonemeVocab,
{
    tokens.iter().map(|t| vocab.lookup(t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn phonemes_to_ids_looks_up_vocab() {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("h".to_string(), 1);
        vocab.insert("ɛ".to_string(), 2);
        let ids = phonemes_to_ids(&["h".into(), "ɛ".into(), "?".into()], &vocab);
        assert_eq!(ids, vec![Some(1), Some(2), None]);
    }
}
