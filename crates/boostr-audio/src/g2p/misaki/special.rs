//! misaki `Lexicon.get_special_case`: symbols, dotted abbreviations, and
//! function words whose reading depends on tag or on the next word.

use super::chars::{PRIMARY_STRESS, SECONDARY_STRESS, char_len, is_alpha};
use super::lexicon::{Lexicon, Lookup, get_parent_tag, symbol_word};
use super::token::TokenContext;

/// misaki `ADD_SYMBOLS`, read when spaCy tags a token `ADD`.
const ADD_SYMBOLS: [(&str, &str); 2] = [(".", "dot"), ("/", "slash")];

/// `re.match(r'(?i)vs\.?$', word)`.
fn is_versus(word: &str) -> bool {
    let body = word.strip_suffix('\n').unwrap_or(word);
    let mut it = body.chars();
    let v = it.next();
    let s = it.next();
    let rest = it.as_str();
    matches!(v, Some('v' | 'V'))
        && matches!(s, Some('s' | 'S' | 'ſ'))
        && (rest.is_empty() || rest == ".")
}

/// misaki's dotted-abbreviation test: a dot inside the stripped word, only
/// letters besides dots, and every dot-separated part under three
/// characters.
fn is_dotted_abbreviation(word: &str) -> bool {
    word.trim_matches('.').contains('.')
        && is_alpha(&word.replace('.', ""))
        && word.split('.').map(char_len).max().unwrap_or(0) < 3
}

fn rated(ps: String) -> Lookup {
    (Some(ps), Some(4))
}

impl Lexicon {
    /// misaki `get_special_case`. `(None, None)` passes the word on to the
    /// general lookup.
    pub(crate) fn get_special_case(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f64>,
        ctx: &TokenContext,
    ) -> Lookup {
        let add_symbol = ADD_SYMBOLS.iter().find(|(s, _)| tag == "ADD" && *s == word);
        if let Some((_, w)) = add_symbol {
            return self.lookup(w, None, Some(-0.5), Some(ctx));
        }
        if let Some(w) = symbol_word(word) {
            return self.lookup(w, None, None, Some(ctx));
        }
        if is_dotted_abbreviation(word) {
            return self.get_nnp(word);
        }
        if word == "a" || word == "A" {
            return rated(if tag == "DT" {
                "ɐ".to_string()
            } else {
                "ˈA".to_string()
            });
        }
        if matches!(word, "am" | "Am" | "AM") {
            if tag.starts_with("NN") {
                return self.get_nnp(word);
            }
            if ctx.future_vowel.is_none() || word != "am" || stress.is_some_and(|s| s > 0.0) {
                return (self.gold_plain("am"), Some(4));
            }
            return rated("ɐm".to_string());
        }
        if matches!(word, "an" | "An" | "AN") {
            if word == "AN" && tag.starts_with("NN") {
                return self.get_nnp(word);
            }
            return rated("ɐn".to_string());
        }
        if word == "I" && tag == "PRP" {
            return rated(format!("{SECONDARY_STRESS}I"));
        }
        if matches!(word, "by" | "By" | "BY") && get_parent_tag(Some(tag)) == Some("ADV") {
            return rated("bˈI".to_string());
        }
        if matches!(word, "to" | "To") || (word == "TO" && matches!(tag, "TO" | "IN")) {
            return match ctx.future_vowel {
                None => (self.gold_plain("to"), Some(4)),
                Some(false) => rated("tə".to_string()),
                Some(true) => rated("tʊ".to_string()),
            };
        }
        if matches!(word, "in" | "In") || (word == "IN" && tag != "NNP") {
            let mark = if ctx.future_vowel.is_none() || tag != "IN" {
                PRIMARY_STRESS.to_string()
            } else {
                String::new()
            };
            return rated(format!("{mark}ɪn"));
        }
        if matches!(word, "the" | "The") || (word == "THE" && tag == "DT") {
            let ps = if ctx.future_vowel == Some(true) {
                "ði"
            } else {
                "ðə"
            };
            return rated(ps.to_string());
        }
        if tag == "IN" && is_versus(word) {
            return self.lookup("versus", None, None, Some(ctx));
        }
        if matches!(word, "used" | "Used" | "USED") {
            let key = if matches!(tag, "VBD" | "JJ") && ctx.future_to {
                "VBD"
            } else {
                "DEFAULT"
            };
            return (self.gold_tagged("used", key), Some(4));
        }
        (None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versus_pattern() {
        assert!(is_versus("vs") && is_versus("Vs.") && is_versus("VS"));
        assert!(!is_versus("vs..") && !is_versus("v") && !is_versus("vsx"));
    }

    #[test]
    fn dotted_abbreviations() {
        assert!(is_dotted_abbreviation("U.S."));
        assert!(is_dotted_abbreviation("e.g"));
        assert!(!is_dotted_abbreviation("Mr."));
        assert!(!is_dotted_abbreviation("abc.de"));
        assert!(!is_dotted_abbreviation("3.14"));
    }

    #[test]
    fn function_words_follow_context() {
        let lex = match Lexicon::new(false) {
            Ok(l) => l,
            Err(e) => panic!("lexicon: {e}"),
        };
        let before_vowel = TokenContext {
            future_vowel: Some(true),
            future_to: false,
        };
        let before_consonant = TokenContext {
            future_vowel: Some(false),
            future_to: true,
        };
        let the = |ctx: &TokenContext| lex.get_special_case("the", "DT", None, ctx).0;
        assert_eq!(the(&before_vowel).as_deref(), Some("ði"));
        assert_eq!(the(&before_consonant).as_deref(), Some("ðə"));
        let to = lex.get_special_case("to", "TO", None, &before_vowel).0;
        assert_eq!(to.as_deref(), Some("tʊ"));
        let used = lex
            .get_special_case("used", "VBD", None, &before_consonant)
            .0;
        assert_eq!(used.as_deref(), Some("jˈust"));
        let a = lex.get_special_case("a", "DT", None, &before_vowel).0;
        assert_eq!(a.as_deref(), Some("ɐ"));
    }
}
