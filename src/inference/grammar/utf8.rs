//! Lowering of Unicode scalar ranges to UTF-8 byte-range sequences.
//!
//! A scalar range splits at each encoding-length boundary and at each
//! continuation-byte alignment. Each piece then encodes as one sequence of
//! byte ranges whose product is exactly the UTF-8 encodings of the piece.
//! Overlong forms and surrogate code points never occur in the output.

/// Largest Unicode scalar value.
const MAX_SCALAR: u32 = 0x10_FFFF;
/// First surrogate code point.
const SURROGATE_LO: u32 = 0xD800;
/// Last surrogate code point.
const SURROGATE_HI: u32 = 0xDFFF;
/// Last scalar value of each encoding length below 4 bytes.
const LENGTH_ENDS: [u32; 3] = [0x7F, 0x7FF, 0xFFFF];

/// One UTF-8 sequence: 1 to 4 inclusive byte ranges, one per byte position.
pub(super) type Utf8Seq = Vec<(u8, u8)>;

/// The sorted, merged, surrogate-free scalar ranges of a class.
///
/// A `negated` class gives the complement within all scalar values.
pub(super) fn class_scalars(negated: bool, ranges: &[(char, char)]) -> Vec<(u32, u32)> {
    let mut sorted: Vec<(u32, u32)> = ranges
        .iter()
        .filter(|(lo, hi)| lo <= hi)
        .map(|&(lo, hi)| (u32::from(lo), u32::from(hi)))
        .collect();
    sorted.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::with_capacity(sorted.len());
    for (lo, hi) in sorted {
        match merged.last_mut() {
            Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    let ranges = if negated { complement(&merged) } else { merged };
    without_surrogates(&ranges)
}

/// All scalar values as one surrogate-free range list.
pub(super) fn any_scalar() -> Vec<(u32, u32)> {
    without_surrogates(&[(0, MAX_SCALAR)])
}

/// The complement of sorted, merged `ranges` within `0..=MAX_SCALAR`.
fn complement(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(ranges.len().saturating_add(1));
    let mut next = 0u32;
    for &(lo, hi) in ranges {
        if lo > next {
            out.push((next, lo - 1));
        }
        next = hi.saturating_add(1);
    }
    if next <= MAX_SCALAR {
        out.push((next, MAX_SCALAR));
    }
    out
}

/// `ranges` with the surrogate block removed.
fn without_surrogates(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(ranges.len().saturating_add(1));
    for &(lo, hi) in ranges {
        let hi = hi.min(MAX_SCALAR);
        if lo > hi {
            continue;
        }
        if hi < SURROGATE_LO || lo > SURROGATE_HI {
            out.push((lo, hi));
            continue;
        }
        if lo < SURROGATE_LO {
            out.push((lo, SURROGATE_LO - 1));
        }
        if hi > SURROGATE_HI {
            out.push((SURROGATE_HI + 1, hi));
        }
    }
    out
}

/// The UTF-8 sequences of the scalar range `lo..=hi`, in ascending order.
///
/// Surrogates and values above `MAX_SCALAR` inside the range are skipped.
pub(super) fn utf8_sequences(lo: u32, hi: u32) -> Vec<Utf8Seq> {
    let mut out = Vec::new();
    let mut stack = vec![(lo, hi.min(MAX_SCALAR))];
    while let Some((mut start, mut end)) = stack.pop() {
        'piece: loop {
            if start > end {
                break;
            }
            if start < SURROGATE_LO && end > SURROGATE_HI {
                stack.push((SURROGATE_HI + 1, end));
                end = SURROGATE_LO - 1;
                continue;
            }
            if (SURROGATE_LO..=SURROGATE_HI).contains(&start) {
                start = SURROGATE_HI + 1;
                continue;
            }
            if (SURROGATE_LO..=SURROGATE_HI).contains(&end) {
                end = SURROGATE_LO - 1;
                continue;
            }
            for max in LENGTH_ENDS {
                if start <= max && max < end {
                    stack.push((max + 1, end));
                    end = max;
                    continue 'piece;
                }
            }
            for i in 1..4u32 {
                let mask = (1u32 << (6 * i)) - 1;
                if start & !mask != end & !mask {
                    if start & mask != 0 {
                        stack.push(((start | mask) + 1, end));
                        end = start | mask;
                        continue 'piece;
                    }
                    if end & mask != mask {
                        stack.push((end & !mask, end));
                        end = (end & !mask) - 1;
                        continue 'piece;
                    }
                }
            }
            let (a, len) = encode(start);
            let (b, _) = encode(end);
            out.push(
                a.iter()
                    .zip(b.iter())
                    .take(len)
                    .map(|(&x, &y)| (x, y))
                    .collect(),
            );
            break;
        }
    }
    out
}

/// The UTF-8 bytes of scalar `c` and their count.
fn encode(c: u32) -> ([u8; 4], usize) {
    // Each cast keeps the low bits on purpose: the masks select at most 8 bits.
    let cont = |shift: u32| 0x80 | ((c >> shift) & 0x3F) as u8;
    match c {
        0..=0x7F => ([c as u8, 0, 0, 0], 1),
        0x80..=0x7FF => ([0xC0 | (c >> 6) as u8, cont(0), 0, 0], 2),
        0x800..=0xFFFF => ([0xE0 | (c >> 12) as u8, cont(6), cont(0), 0], 3),
        _ => (
            [0xF0 | ((c >> 18) & 0x07) as u8, cont(12), cont(6), cont(0)],
            4,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(seqs: &[Utf8Seq], bytes: &[u8]) -> bool {
        seqs.iter().any(|seq| {
            seq.len() == bytes.len()
                && seq
                    .iter()
                    .zip(bytes)
                    .all(|(&(lo, hi), &b)| (lo..=hi).contains(&b))
        })
    }

    fn sequences(ranges: &[(u32, u32)]) -> Vec<Utf8Seq> {
        ranges
            .iter()
            .flat_map(|&(lo, hi)| utf8_sequences(lo, hi))
            .collect()
    }

    /// Scalar values to probe: every value near each boundary, plus a stride sample.
    fn probes() -> Vec<u32> {
        let mut out = Vec::new();
        for edge in [
            0, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x1_0000, MAX_SCALAR,
        ] {
            let lo = edge.saturating_sub(300);
            let hi = (edge + 300).min(MAX_SCALAR);
            out.extend(lo..=hi);
        }
        out.extend((0..=MAX_SCALAR).step_by(997));
        out.retain(|c| char::from_u32(*c).is_some());
        out
    }

    fn check_table(ranges: &[(u32, u32)]) {
        let scalars = without_surrogates(ranges);
        let seqs = sequences(&scalars);
        for seq in &seqs {
            assert!((1..=4).contains(&seq.len()), "{seq:?}");
            assert!(seq.iter().all(|(lo, hi)| lo <= hi), "{seq:?}");
        }
        for c in probes() {
            let Some(ch) = char::from_u32(c) else {
                continue;
            };
            let mut buf = [0u8; 4];
            let bytes = ch.encode_utf8(&mut buf).as_bytes();
            let want = scalars.iter().any(|&(lo, hi)| (lo..=hi).contains(&c));
            assert_eq!(accepts(&seqs, bytes), want, "{ranges:x?} at U+{c:04X}");
        }
        for bad in invalid_strings() {
            assert!(!accepts(&seqs, &bad), "{ranges:x?} accepts {bad:x?}");
        }
    }

    fn invalid_strings() -> Vec<Vec<u8>> {
        let mut out = vec![
            vec![0xC0, 0x80],
            vec![0xC1, 0xBF],
            vec![0xE0, 0x80, 0x80],
            vec![0xE0, 0x9F, 0xBF],
            vec![0xF0, 0x80, 0x80, 0x80],
            vec![0xF0, 0x8F, 0xBF, 0xBF],
            vec![0xED, 0xA0, 0x80],
            vec![0xED, 0xBF, 0xBF],
            vec![0xF4, 0x90, 0x80, 0x80],
            vec![0xF5, 0x80, 0x80, 0x80],
            vec![0xFF],
            vec![0xC3],
            vec![0xE4, 0xB8],
            vec![0xF0, 0x9F, 0x98],
        ];
        out.extend((0x80..=0xBFu8).map(|b| vec![b]));
        out
    }

    #[test]
    fn range_tables_accept_exactly_their_scalars() {
        let tables: Vec<Vec<(u32, u32)>> = vec![
            vec![(0x61, 0x61)],
            vec![(0xE9, 0xE9)],
            vec![(0x4E16, 0x4E16)],
            vec![(0x1F600, 0x1F600)],
            vec![(0x61, 0x7A)],
            vec![(0x80, 0x7FF)],
            vec![(0x800, 0xFFFF)],
            vec![(0x1_0000, MAX_SCALAR)],
            vec![(0, MAX_SCALAR)],
            vec![(0x70, 0x90)],
            vec![(0x7F0, 0x810)],
            vec![(0xFFF0, 0x1_0010)],
            vec![(0xD700, 0xE100)],
            vec![(0xD7FF, 0xD7FF), (0xE000, 0xE000)],
            vec![(0x10_FF00, MAX_SCALAR)],
            vec![
                (0x41, 0x5A),
                (0x3B1, 0x3C9),
                (0x3041, 0x3096),
                (0x2_0000, 0x2_A6DF),
            ],
            vec![(0x7F, 0x80), (0x7FF, 0x800), (0xFFFF, 0x1_0000)],
            vec![(0x123, 0x4567)],
        ];
        for table in &tables {
            check_table(table);
        }
    }

    #[test]
    fn pieces_never_cover_surrogates_or_overlong_forms() {
        let seqs = utf8_sequences(0, MAX_SCALAR);
        assert!(!accepts(&seqs, &[0xED, 0xA0, 0x80]));
        assert!(accepts(&seqs, &[0xED, 0x9F, 0xBF]));
        assert!(accepts(&seqs, &[0xF4, 0x8F, 0xBF, 0xBF]));
        assert!(utf8_sequences(SURROGATE_LO, SURROGATE_HI).is_empty());
        assert!(utf8_sequences(5, 4).is_empty());
    }

    #[test]
    fn classes_merge_complement_and_drop_surrogates() {
        let ranges = [('c', 'd'), ('a', 'b'), ('x', 'z')];
        assert_eq!(
            class_scalars(false, &ranges),
            vec![(0x61, 0x64), (0x78, 0x7A)]
        );
        let neg = class_scalars(true, &[('\0', 'a')]);
        assert_eq!(
            neg,
            vec![(0x62, SURROGATE_LO - 1), (SURROGATE_HI + 1, MAX_SCALAR)]
        );
        assert_eq!(class_scalars(true, &[('\0', char::MAX)]), vec![]);
        assert_eq!(class_scalars(false, &[]), vec![]);
        assert_eq!(class_scalars(true, &[]), any_scalar());
        let gap = class_scalars(false, &[('\u{D7FF}', '\u{E000}')]);
        assert_eq!(gap, vec![(0xD7FF, 0xD7FF), (0xE000, 0xE000)]);
    }

    #[test]
    fn encode_matches_the_standard_library() {
        for c in probes() {
            let Some(ch) = char::from_u32(c) else {
                continue;
            };
            let mut buf = [0u8; 4];
            let want = ch.encode_utf8(&mut buf).as_bytes();
            let (got, len) = encode(c);
            assert_eq!(got.get(..len), Some(want), "U+{c:04X}");
        }
    }
}
