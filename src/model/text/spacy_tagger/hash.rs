//! thinc `HashEmbed` bucket hashing.
//!
//! `HashEmbed` hashes each `u64` attribute ID with `ops.hash(ids, seed)` into
//! four `u32` keys, takes each key modulo the table's row count, and sums the
//! four rows. This module computes the row indices; the gather and the sum
//! run on tensors.

/// thinc `MurmurHash3_x86_128_uint64` (`thinc/backends/numpy_ops.pyx`): the
/// four 32-bit keys of one `u64` key under `seed`.
///
/// thinc's variant, not the reference MurmurHash3: it mixes the key as one
/// 64-bit block and finalizes with the 128-bit `fmix64` pair.
pub fn murmurhash3_x86_128_u64(val: u64, seed: u32) -> [u32; 4] {
    let seed = u64::from(seed);
    let mut h1 = val.wrapping_mul(0x87c3_7b91_1142_53d5);
    h1 = h1.rotate_left(31);
    h1 = h1.wrapping_mul(0x4cf5_ad43_2745_937f);
    h1 ^= seed;
    h1 ^= 8;
    let mut h2 = seed ^ 8;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    [h1 as u32, (h1 >> 32) as u32, h2 as u32, (h2 >> 32) as u32]
}

/// MurmurHash3 64-bit finalizer.
fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

/// Row indices `HashEmbed` reads for `id` from a table of `rows` rows under
/// `seed`: `ops.hash(ids, seed) % nV`. `rows` is non-zero.
pub fn bucket_keys(id: u64, seed: u32, rows: usize) -> [usize; 4] {
    murmurhash3_x86_128_u64(id, seed).map(|k| k as usize % rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_thinc_numpy_ops_hash() {
        // Values from `thinc.api.NumpyOps().hash(numpy.array([v], "uint64"), seed)`.
        assert_eq!(murmurhash3_x86_128_u64(0, 8), [0, 0, 0, 0]);
        assert_eq!(
            murmurhash3_x86_128_u64(1, 9),
            [2205482518, 2060344464, 1125017477, 2005428196]
        );
        assert_eq!(
            murmurhash3_x86_128_u64(101, 11),
            [1763378230, 3759297338, 627086936, 2342724702]
        );
        assert_eq!(
            murmurhash3_x86_128_u64(5097672513440128799, 8),
            [2032173550, 2625552531, 900776677, 3938328797]
        );
        assert_eq!(
            murmurhash3_x86_128_u64(u64::MAX, 13),
            [3199409384, 1062937581, 752279700, 3061442444]
        );
    }

    #[test]
    fn bucket_keys_reduce_modulo_rows() {
        let keys = bucket_keys(1, 9, 1000);
        assert_eq!(keys, [518, 464, 477, 196]);
        assert!(bucket_keys(u64::MAX, 13, 50).iter().all(|&k| k < 50));
    }
}
