//! The embedded misaki lexicons: decoding and `grow_dictionary`.
//!
//! `assets/misaki/build_lexicon.py` writes one zlib stream per dialect from
//! misaki 0.9.4's `{us,gb}_{gold,silver}.json` and documents the layout.

use std::collections::HashMap;
use std::io::Read;

use flate2::read::ZlibDecoder;

use crate::error::{Error, Result};

const US_LEXICON: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/misaki/en_us.lex.zlib"
));
const GB_LEXICON: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/misaki/en_gb.lex.zlib"
));
const MAGIC: &[u8; 4] = b"MSKL";
const FORMAT_VERSION: u8 = 1;

/// One lexicon value: a pronunciation, or pronunciations keyed by tag.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Entry {
    /// One pronunciation for every tag.
    Plain(String),
    /// Tag to pronunciation; a `None` value maps that tag to no
    /// pronunciation. Always holds `DEFAULT` in misaki's data.
    Tagged(Vec<(String, Option<String>)>),
}

impl Entry {
    /// `ps.get(tag)` on a tagged entry: `None` when absent, `Some(None)` when
    /// present with a null value.
    pub(crate) fn tagged_get(&self, tag: &str) -> Option<Option<&str>> {
        match self {
            Entry::Plain(_) => None,
            Entry::Tagged(pairs) => pairs
                .iter()
                .find(|(t, _)| t == tag)
                .map(|(_, v)| v.as_deref()),
        }
    }
}

/// Gold and silver dictionaries of one dialect.
pub(crate) type Dictionary = HashMap<String, Entry>;

fn corrupt(what: &str) -> Error {
    Error::DataError {
        reason: format!("embedded misaki lexicon is corrupt: {what}"),
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).ok_or_else(|| corrupt("truncated"))?;
        self.pos += 1;
        Ok(b)
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| corrupt("length overflow"))?;
        let out = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| corrupt("truncated"))?;
        self.pos = end;
        Ok(out)
    }

    fn u32_le(&mut self) -> Result<u32> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn varint(&mut self) -> Result<usize> {
        let mut out = 0usize;
        for shift in (0..35).step_by(7) {
            let b = self.u8()?;
            out |= usize::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
        }
        Err(corrupt("varint too long"))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.varint()?;
        let raw = self.bytes(n)?;
        String::from_utf8(raw.to_vec()).map_err(|_| corrupt("invalid UTF-8"))
    }
}

fn read_section(r: &mut Reader, tags: &[String]) -> Result<Dictionary> {
    let count = r.u32_le()? as usize;
    let mut dict = HashMap::with_capacity(count + count / 2);
    let mut prev: Vec<u8> = Vec::new();
    for _ in 0..count {
        let shared = usize::from(r.u8()?);
        if shared > prev.len() {
            return Err(corrupt("shared prefix longer than previous key"));
        }
        let n = r.varint()?;
        let suffix = r.bytes(n)?;
        prev.truncate(shared);
        prev.extend_from_slice(suffix);
        let key = String::from_utf8(prev.clone()).map_err(|_| corrupt("invalid UTF-8 key"))?;
        let entry = match r.u8()? {
            0 => Entry::Plain(r.string()?),
            1 => {
                let n = r.u8()?;
                let mut pairs = Vec::with_capacity(usize::from(n));
                for _ in 0..n {
                    let tag = tags
                        .get(usize::from(r.u8()?))
                        .ok_or_else(|| corrupt("tag index out of range"))?
                        .clone();
                    let value = match r.u8()? {
                        0 => None,
                        _ => Some(r.string()?),
                    };
                    pairs.push((tag, value));
                }
                Entry::Tagged(pairs)
            }
            _ => return Err(corrupt("unknown entry kind")),
        };
        dict.insert(key, entry);
    }
    Ok(dict)
}

/// ASCII `str.capitalize`: first character upper, the rest lower.
fn ascii_capitalize(k: &str) -> String {
    let mut out = k.to_ascii_lowercase();
    if let Some(first) = out.get_mut(..1) {
        first.make_ascii_uppercase();
    }
    out
}

/// misaki `Lexicon.grow_dictionary`: adds the capitalized form of each
/// lowercase key and the lowercase form of each capitalized key, without
/// overriding a key the data already has. Every key is ASCII.
fn grow_dictionary(dict: &mut Dictionary) {
    let mut extra: Vec<(String, Entry)> = Vec::new();
    for (k, v) in dict.iter() {
        if k.len() < 2 {
            continue;
        }
        let lower = k.to_ascii_lowercase();
        if *k == lower {
            let cap = ascii_capitalize(k);
            if cap != *k {
                extra.push((cap, v.clone()));
            }
        } else if *k == ascii_capitalize(&lower) {
            extra.push((lower, v.clone()));
        }
    }
    for (k, v) in extra {
        dict.entry(k).or_insert(v);
    }
}

/// Decodes the embedded gold and silver dictionaries for one dialect, grown
/// the way misaki's `Lexicon.__init__` grows them.
pub(crate) fn load(british: bool) -> Result<(Dictionary, Dictionary)> {
    let packed = if british { GB_LEXICON } else { US_LEXICON };
    let mut raw = Vec::new();
    ZlibDecoder::new(packed)
        .read_to_end(&mut raw)
        .map_err(|e| corrupt(&format!("zlib: {e}")))?;
    let mut r = Reader { buf: &raw, pos: 0 };
    if r.bytes(4)? != MAGIC.as_slice() {
        return Err(corrupt("bad magic"));
    }
    if r.u8()? != FORMAT_VERSION {
        return Err(corrupt("unsupported format version"));
    }
    let ntags = r.u8()?;
    let mut tags = Vec::with_capacity(usize::from(ntags));
    for _ in 0..ntags {
        tags.push(r.string()?);
    }
    let mut gold = read_section(&mut r, &tags)?;
    let mut silver = read_section(&mut r, &tags)?;
    if r.pos != raw.len() {
        return Err(corrupt("trailing bytes"));
    }
    if !gold.keys().chain(silver.keys()).all(|k| k.is_ascii()) {
        return Err(corrupt("non-ASCII key"));
    }
    grow_dictionary(&mut gold);
    grow_dictionary(&mut silver);
    Ok((gold, silver))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalize_is_ascii() {
        assert_eq!(ascii_capitalize("hello"), "Hello");
        assert_eq!(ascii_capitalize("HeLLo"), "Hello");
        assert_eq!(ascii_capitalize("'s"), "'s");
    }

    #[test]
    fn embedded_lexicons_decode_and_grow() {
        for british in [false, true] {
            let (gold, silver) = match load(british) {
                Ok(d) => d,
                Err(e) => panic!("load british={british}: {e}"),
            };
            assert!(gold.len() > 150_000 && silver.len() > 150_000);
            assert!(matches!(gold.get("used"), Some(Entry::Tagged(_))));
            // `grow_dictionary` adds the capitalized form of a lowercase key.
            assert_eq!(gold.get("Hello"), gold.get("hello"));
        }
    }
}
