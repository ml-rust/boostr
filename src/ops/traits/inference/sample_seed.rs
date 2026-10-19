//! The Philox key and counter one seeded sampling draw uses.

/// The position of one seeded sampling draw in a counter-based Philox stream.
///
/// The draw is a pure function of `(key, counter)`. The same pair gives the
/// same uniform on every call and every backend.
///
/// - `key`: the request seed. Each choice of one request offsets it.
/// - `counter`: the index of the token being sampled. Each decode step
///   must pass a new value, or every step draws the same uniform.
///
/// Key 0 is a valid stream: Philox adds its round constants to the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleSeed {
    /// The Philox key: the request seed.
    pub key: u64,
    /// The Philox counter: the token position within the request.
    pub counter: u64,
}

impl SampleSeed {
    /// The draw at `counter` in the stream `key` names.
    pub const fn new(key: u64, counter: u64) -> Self {
        Self { key, counter }
    }
}
