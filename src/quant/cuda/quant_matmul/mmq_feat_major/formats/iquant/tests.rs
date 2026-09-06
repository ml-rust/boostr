//! Assertions spanning both staging groups.

use super::codebook::IQ4_XS;
use super::grid::{IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS};

/// No i-quant asks for activation scratch; Q2_K remains the family's only
/// claimant. IQ1_S does carry a minimum term, but at the 32-element
/// granularity the activation record's own block sum already has, so it
/// needs no per-16 split. Asserted here as well as in `kquant.rs` so a new
/// i-quant descriptor that copies the wrong template fails a test rather
/// than silently enlarging every launch's request.
#[test]
fn no_iquant_asks_for_activation_scratch() {
    for f in [&IQ4_XS, &IQ2_XXS, &IQ2_XS, &IQ2_S, &IQ3_XXS, &IQ3_S, &IQ1_S] {
        assert_eq!(f.act_scratch_ints_per_token, 0);
        // The family's bank-padding rule, asserted in the kernel as well.
        assert_eq!(f.x_stride % 8, 4);
        assert_eq!(f.k_multiple, 256);
    }
}
