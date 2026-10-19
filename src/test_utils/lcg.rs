//! Deterministic 64-bit LCG for unit tests. No `rand` dependency.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Seeded linear congruential generator (Knuth MMIX constants).
///
/// Every draw advances the state once. The methods differ only in which
/// state bits they read, so a test keeps its exact stream.
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    fn step(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// The top 53 bits of the next state.
    pub(crate) fn bits(&mut self) -> u64 {
        self.step() >> 11
    }

    /// The top 31 bits of the next state.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.step() >> 33
    }

    /// A value in `0..n` from the top 31 bits.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A value in `0..n` from the top 31 bits. `n` below 1 counts as 1.
    pub(crate) fn below_u64(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }

    /// A value in `0..n` from the top 53 bits.
    pub(crate) fn index(&mut self, n: usize) -> usize {
        (self.bits() % n as u64) as usize
    }

    /// An element of `items`, chosen with [`Lcg::below_u64`].
    pub(crate) fn pick<T: Clone>(&mut self, items: &[T]) -> T {
        let i = self.below_u64(items.len() as u64) as usize;
        items[i].clone()
    }

    /// An F32 in `[0, 1)` from the top 24 bits.
    pub(crate) fn next_f32(&mut self) -> f32 {
        ((self.step() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// An F32 in `[lo, hi)` from [`Lcg::next_f32`].
    pub(crate) fn uniform_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// An F64 in `[0, 1)` from the top 53 bits.
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.bits() as f64) / ((1u64 << 53) as f64)
    }

    /// An F64 in `[lo, hi)` from [`Lcg::next_f64`].
    pub(crate) fn uniform_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// `n` F32 values in `[-scale, scale)`.
    pub(crate) fn data(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform_f32(-scale, scale)).collect()
    }

    /// `n` F64 values in `[-scale, scale)`.
    pub(crate) fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.uniform_f64(-scale, scale)).collect()
    }

    /// An F32 CPU tensor of `shape` with values in `[-scale, scale)`.
    pub(crate) fn tensor(
        &mut self,
        device: &CpuDevice,
        shape: &[usize],
        scale: f32,
    ) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&self.data(n, scale), shape, device).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_stream() {
        let (mut a, mut b) = (Lcg(7), Lcg(7));
        assert_eq!(a.vec(8, 1.0), b.vec(8, 1.0));
    }

    #[test]
    fn draws_stay_in_range() {
        let mut rng = Lcg(3);
        for _ in 0..200 {
            assert!((0.0..1.0).contains(&rng.next_f32()));
            assert!((0.0..1.0).contains(&rng.next_f64()));
            assert!(rng.below(5) < 5);
            assert_eq!(rng.below_u64(0), 0);
            assert!(rng.index(7) < 7);
        }
    }
}
