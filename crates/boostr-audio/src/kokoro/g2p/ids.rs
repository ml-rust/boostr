//! Phoneme-string → model input-id mapping, ported from reference Kokoro's
//! `KModel.forward`.

use boostr::model::audio::PhonemeVocab;
use boostr::model::audio::kokoro::KokoroPhonemeVocab;

/// Reference Kokoro's `input_ids = [0, *(vocab.get(p) for p in phonemes if
/// known), 0]`: one id per phoneme character the vocab knows (an unknown
/// character is dropped, not substituted), bracketed by the reserved
/// BOS/EOS id `0` the vocab table never assigns.
pub(crate) fn phonemes_to_input_ids(vocab: &KokoroPhonemeVocab, phonemes: &str) -> Vec<i64> {
    let mut ids = Vec::with_capacity(phonemes.chars().count() + 2);
    ids.push(0);
    for c in phonemes.chars() {
        let mut buf = [0u8; 4];
        if let Some(id) = vocab.lookup(c.encode_utf8(&mut buf)) {
            ids.push(i64::from(id));
        }
    }
    ids.push(0);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn vocab(pairs: &[(&str, u32)]) -> KokoroPhonemeVocab {
        let table: HashMap<String, u32> = pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        KokoroPhonemeVocab::new(table)
    }

    #[test]
    fn wraps_bos_eos_and_drops_unknown_characters() {
        let v = vocab(&[("h", 1), ("ɛ", 2), ("l", 3)]);
        let ids = phonemes_to_input_ids(&v, "hɛl?");
        assert_eq!(ids, vec![0, 1, 2, 3, 0]);
    }

    #[test]
    fn empty_phonemes_still_wraps_bos_eos() {
        let v = vocab(&[("h", 1)]);
        assert_eq!(phonemes_to_input_ids(&v, ""), vec![0, 0]);
    }
}
