//! spaCy's string hash.

/// spaCy `hash_string`: MurmurHash64A with seed 1 over the UTF-8 bytes.
///
/// spaCy's `lexeme_norm` table and its string IDs use this hash. A string
/// in spaCy's symbol table (`spacy.symbols.IDS`) gets a small symbol ID from
/// `StringStore` instead; this function never returns those.
pub fn hash_string(s: &str) -> u64 {
    murmur_hash_64a(s.as_bytes(), 1)
}

/// MurmurHash64A (Austin Appleby), as the `murmurhash` package builds it.
fn murmur_hash_64a(key: &[u8], seed: u64) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    const R: u32 = 47;
    let mut h = seed ^ (key.len() as u64).wrapping_mul(M);

    let (blocks, tail) = key.as_chunks::<8>();
    for block in blocks {
        let mut k = u64::from_le_bytes(*block);
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }

    if !tail.is_empty() {
        for (i, &b) in tail.iter().enumerate() {
            h ^= u64::from(b) << (8 * i);
        }
        h = h.wrapping_mul(M);
    }

    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_spacy_hash_string() {
        // Values from `spacy.strings.hash_string` 3.8.
        assert_eq!(hash_string("because"), 16950148841647037698);
        assert_eq!(hash_string("a"), 11901859001352538922);
        assert_eq!(hash_string("colour"), 2579953423244470672);
        assert_eq!(hash_string(" "), 8532415787641010193);
        assert_eq!(hash_string("héllo wörld!"), 15695761827416483309);
        assert_eq!(hash_string("12345678"), 9871996032675327904);
        assert_eq!(hash_string("123456789abcdefgh"), 4275926136390810747);
    }
}
