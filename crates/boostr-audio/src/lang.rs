//! [`Lang`]: the language tag a [`crate::tts::Voice`] carries.
//!
//! Purely a tag — no G2P backend attaches to it. Kokoro's own English
//! voices phonemize through [`crate::g2p::misaki`] instead, which reads a
//! voice's dialect from its id prefix (`a*`/`b*`), not from `Lang`.

use crate::error::{Error, Result};

/// Language tag a [`crate::tts::Voice`] carries.
///
/// The named variants are the languages some engine in this crate treats
/// specially. [`Lang::Other`] holds any other well-formed ISO 639 tag, so a
/// checkpoint trained on one of MMS-TTS's many languages keeps its own tag.
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
    /// Any other language, by its normalized tag (e.g. `tha`, `sw`).
    Other(LangTag),
}

/// Longest tag [`LangTag`] holds: a primary subtag plus a region or script.
const MAX_TAG_LEN: usize = 16;

/// A normalized language tag outside [`Lang`]'s named variants: lowercase,
/// dash-separated, a primary subtag of 2 or 3 ASCII letters, then subtags
/// of 2 to 8 ASCII letters or digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LangTag {
    bytes: [u8; MAX_TAG_LEN],
    len: u8,
}

impl LangTag {
    /// Checks `normalized` (already lowercase and dash-separated) and stores
    /// it. `original` is the caller's spelling, echoed in the error.
    fn new(normalized: &str, original: &str) -> Result<Self> {
        let invalid = || Error::InvalidArgument {
            arg: "lang",
            reason: format!(
                "invalid language code {original:?}: expected an ISO 639 code such as \"tha\" \
                 or \"sw-ke\""
            ),
        };
        if normalized.len() > MAX_TAG_LEN {
            return Err(invalid());
        }
        let mut subtags = normalized.split('-');
        let primary = subtags.next().ok_or_else(invalid)?;
        if !(2..=3).contains(&primary.len()) || !primary.bytes().all(|b| b.is_ascii_lowercase()) {
            return Err(invalid());
        }
        for subtag in subtags {
            if !(2..=8).contains(&subtag.len())
                || !subtag
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
            {
                return Err(invalid());
            }
        }
        let mut bytes = [0u8; MAX_TAG_LEN];
        bytes[..normalized.len()].copy_from_slice(normalized.as_bytes());
        Ok(Self {
            bytes,
            len: normalized.len() as u8,
        })
    }

    /// The tag, e.g. `"tha"`.
    pub fn as_str(&self) -> &str {
        // `new` stores ASCII only, so the slice is always valid UTF-8.
        std::str::from_utf8(&self.bytes[..usize::from(self.len)]).unwrap_or_default()
    }
}

impl std::str::FromStr for Lang {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

impl Lang {
    /// Parses a BCP-47 tag or an ISO 639-1/639-3 code, dash- or
    /// underscore-separated, any case. A code outside the named variants
    /// becomes [`Lang::Other`]. A malformed code is an error.
    pub fn parse(s: &str) -> Result<Self> {
        let normalized = s.to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "en" | "en-us" | "eng" => Ok(Self::EnUs),
            "en-gb" | "en-uk" => Ok(Self::EnGb),
            "es" | "es-es" | "es-mx" | "spa" => Ok(Self::Es),
            "fr" | "fr-fr" | "fra" | "fre" => Ok(Self::Fr),
            "ja" | "ja-jp" | "jpn" => Ok(Self::Ja),
            "hi" | "hi-in" | "hin" => Ok(Self::Hi),
            "it" | "it-it" | "ita" => Ok(Self::It),
            "pt" | "pt-br" | "pt-pt" | "por" => Ok(Self::Pt),
            "zh" | "zh-cn" | "zh-tw" | "zho" | "chi" | "cmn" => Ok(Self::Zh),
            "ko" | "ko-kr" | "kor" => Ok(Self::Ko),
            "ms" | "ms-my" | "msa" | "may" | "zlm" | "zsm" => Ok(Self::Ms),
            other => LangTag::new(other, s).map(Self::Other),
        }
    }

    /// BCP-47-style language tag, e.g. `"en-us"`. Used in API responses and
    /// voice-catalog language fields.
    pub fn bcp47(&self) -> &str {
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
            Self::Other(tag) => tag.as_str(),
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
    }

    #[test]
    fn maps_iso_639_3_codes_to_named_variants() {
        assert_eq!(Lang::parse("eng").unwrap(), Lang::EnUs);
        assert_eq!(Lang::parse("zlm").unwrap(), Lang::Ms);
        assert_eq!(Lang::parse("cmn").unwrap(), Lang::Zh);
    }

    #[test]
    fn keeps_other_well_formed_codes() {
        let thai = Lang::parse("THA").unwrap();
        assert_eq!(thai.bcp47(), "tha");
        let swahili = Lang::parse("sw_KE").unwrap();
        assert_eq!(swahili.bcp47(), "sw-ke");
    }

    #[test]
    fn rejects_malformed_codes() {
        for bad in ["", "e", "english", "e1", "en-", "en-x", "abc-toolongsubtag"] {
            assert!(Lang::parse(bad).is_err(), "{bad:?} parsed");
        }
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
            Lang::Zh,
            Lang::Ko,
            Lang::Ms,
            Lang::parse("tha").unwrap(),
        ] {
            assert_eq!(Lang::parse(lang.bcp47()).unwrap(), lang);
        }
    }
}
