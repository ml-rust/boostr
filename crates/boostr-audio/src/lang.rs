//! [`Lang`]: the language tag a [`crate::tts::Voice`] carries.
//!
//! Purely a tag — no G2P backend attaches to it. Kokoro's own English
//! voices phonemize through [`crate::g2p::misaki`] instead, which reads a
//! voice's dialect from its id prefix (`a*`/`b*`), not from `Lang`.

use crate::error::{Error, Result};

/// BCP-47-like language codes a [`crate::tts::Voice`] carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    EnUs,
    EnGb,
    Es,
    Fr,
    Ja,
    Hi,
    It,
    Pt,
    Zh,
    Ko,
    /// Bahasa Melayu: a multilingual model's zero-shot voice tag.
    Ms,
}

impl std::str::FromStr for Lang {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

impl Lang {
    /// Parse a BCP-47 or dash/underscore-separated code.
    pub fn parse(s: &str) -> Result<Self> {
        let normalized = s.to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "en" | "en-us" => Ok(Self::EnUs),
            "en-gb" | "en-uk" => Ok(Self::EnGb),
            "es" | "es-es" | "es-mx" => Ok(Self::Es),
            "fr" | "fr-fr" => Ok(Self::Fr),
            "ja" | "ja-jp" => Ok(Self::Ja),
            "hi" | "hi-in" => Ok(Self::Hi),
            "it" | "it-it" => Ok(Self::It),
            "pt" | "pt-br" | "pt-pt" => Ok(Self::Pt),
            "zh" | "zh-cn" | "zh-tw" => Ok(Self::Zh),
            "ko" | "ko-kr" => Ok(Self::Ko),
            "ms" | "ms-my" => Ok(Self::Ms),
            _ => Err(Error::InvalidArgument {
                arg: "lang",
                reason: format!("unsupported language code {s:?}"),
            }),
        }
    }

    /// BCP-47-style language tag, e.g. `"en-us"`. Used in API responses and
    /// voice-catalog language fields.
    pub fn bcp47(self) -> &'static str {
        match self {
            Self::EnUs => "en-us",
            Self::EnGb => "en-gb",
            Self::Es => "es",
            Self::Fr => "fr",
            Self::Ja => "ja",
            Self::Hi => "hi",
            Self::It => "it",
            Self::Pt => "pt",
            Self::Zh => "cmn",
            Self::Ko => "ko",
            Self::Ms => "ms",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_forms() {
        assert_eq!(Lang::parse("en").unwrap(), Lang::EnUs);
        assert_eq!(Lang::parse("en-US").unwrap(), Lang::EnUs);
        assert_eq!(Lang::parse("en_GB").unwrap(), Lang::EnGb);
        assert_eq!(Lang::parse("zh-CN").unwrap(), Lang::Zh);
        assert!(Lang::parse("xx").is_err());
    }

    #[test]
    fn tag_round_trips_through_parse() {
        for lang in [
            Lang::EnUs,
            Lang::EnGb,
            Lang::Es,
            Lang::Fr,
            Lang::Ja,
            Lang::Hi,
            Lang::It,
            Lang::Pt,
            Lang::Ko,
            Lang::Ms,
        ] {
            assert_eq!(Lang::parse(lang.bcp47()).unwrap(), lang);
        }
        // Mandarin's tag ("cmn") isn't itself a code `parse` accepts back;
        // "zh" is the accepted spelling for the same variant.
        assert_eq!(Lang::parse("zh").unwrap(), Lang::Zh);
    }
}
