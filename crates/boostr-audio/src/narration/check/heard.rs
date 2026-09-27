//! A recognized word with its time in the audio, the input of the speech
//! check and of script alignment.

use serde::{Deserialize, Serialize};

/// A word as a recognizer heard it, or a script word placed on the audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeardWord {
    /// The word as written, punctuation included.
    pub text: String,
    /// Start in seconds.
    pub start: f64,
    /// End in seconds.
    pub end: f64,
}

#[cfg(feature = "whisper")]
impl From<&crate::whisper::WordTiming> for HeardWord {
    fn from(w: &crate::whisper::WordTiming) -> Self {
        Self {
            text: w.text.clone(),
            start: f64::from(w.start),
            end: f64::from(w.end),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_uses_javascript_field_names() -> Result<(), serde_json::Error> {
        let w: HeardWord = serde_json::from_str(r#"{"text":"Pada","start":6.2,"end":6.5}"#)?;
        assert_eq!(
            w,
            HeardWord {
                text: "Pada".into(),
                start: 6.2,
                end: 6.5
            }
        );
        assert_eq!(
            serde_json::to_string(&w)?,
            r#"{"text":"Pada","start":6.2,"end":6.5}"#
        );
        Ok(())
    }

    #[cfg(feature = "whisper")]
    #[test]
    fn from_word_timing_widens_times() {
        let t = crate::whisper::WordTiming {
            text: " hello".into(),
            start: 0.5,
            end: 0.75,
        };
        let w = HeardWord::from(&t);
        assert_eq!(w.text, " hello");
        assert_eq!((w.start, w.end), (0.5, 0.75));
    }
}
