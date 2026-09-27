//! The English dialect a Kokoro voice id selects.

use crate::error::{Error, Result};

/// English dialect Kokoro's `a*`/`b*` voice prefixes select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// `a*` voices: American English (misaki `british = false`).
    Us,
    /// `b*` voices: British English (misaki `british = true`).
    Gb,
}

impl Dialect {
    /// Reads a voice id's first character, exactly as reference Kokoro's
    /// `KPipeline` picks `en.G2P(british=lang_code=='b')` from `lang_code
    /// in 'ab'`. Errs naming `voice_id` for any other prefix: reference
    /// Kokoro routes `e/f/h/i/j/p/z` voices through espeak-ng or misaki's
    /// Japanese/Mandarin G2P, neither of which this crate links.
    pub fn from_voice_id(voice_id: &str) -> Result<Self> {
        match voice_id.chars().next() {
            Some('a') => Ok(Self::Us),
            Some('b') => Ok(Self::Gb),
            _ => Err(Error::InvalidArgument {
                arg: "voice_spec",
                reason: format!(
                    "voice {voice_id:?} is not an a*/b* English voice; only American \
                     (a*) and British (b*) Kokoro voices are supported in this build"
                ),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_voice_id_reads_prefix() {
        assert_eq!(Dialect::from_voice_id("af_heart").unwrap(), Dialect::Us);
        assert_eq!(Dialect::from_voice_id("bf_emma").unwrap(), Dialect::Gb);
    }

    #[test]
    fn from_voice_id_rejects_other_prefixes() {
        for voice in ["ef_dora", "jf_alpha", "zm_yunxi", "", "x"] {
            let err = Dialect::from_voice_id(voice).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains(voice), "{voice:?}: {msg}");
        }
    }
}
