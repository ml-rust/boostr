//! The token the spaCy tokenizer port returns.

/// One spaCy token with the lexical attributes `en_core_web_sm`'s
/// `tok2vec` embeds: `NORM`, `PREFIX`, `SUFFIX`, `SHAPE`, `SPACY` and
/// `IS_SPACE`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpacyToken {
    /// `Token.text` (`ORTH`).
    pub text: String,
    /// `Token.whitespace_ == " "` (`SPACY`): one space follows the token.
    pub spacy: bool,
    /// `Token.norm_`: a special case's `NORM` when it sets one, else the
    /// lexeme norm.
    pub norm: String,
    /// `Token.prefix_`: the first character.
    pub prefix: String,
    /// `Token.suffix_`: the last three characters.
    pub suffix: String,
    /// `Token.shape_`: spaCy `word_shape`.
    pub shape: String,
    /// `Token.is_space`: every character is whitespace.
    pub is_space: bool,
}

impl SpacyToken {
    /// `Token.whitespace_`: `" "` or `""`.
    pub fn whitespace(&self) -> &'static str {
        if self.spacy { " " } else { "" }
    }
}
