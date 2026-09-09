//! Detail carried by [`crate::error::Error::ActivationContractMismatch`].

use std::fmt;

use super::declared::ActivationContract;
use super::kernel::KernelContract;

/// Everything a caller needs to act on an activation-contract refusal.
///
/// Boxed inside [`crate::error::Error::ActivationContractMismatch`] so the
/// error type stays a small handle: [`ActivationContract`] and
/// [`KernelContract`] are each large enough on their own that carrying both
/// by value in the enum inflates every `Result<T>` in this crate.
#[derive(Debug, Clone)]
pub struct ActivationContractMismatchDetail {
    /// Provenance name of the weight, from the source file.
    pub tensor: String,
    /// Weight encoding the dispatch resolved on.
    pub encoding: String,
    /// The contract the file declares for this weight.
    pub declared: ActivationContract,
    /// What the kernel that would have run actually computes.
    pub kernel: KernelContract,
}

impl fmt::Display for ActivationContractMismatchDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "E_ACTIVATION_CONTRACT_MISMATCH: tensor '{}' encoded {} declares ({}), which the selected kernel does not satisfy ({})",
            self.tensor, self.encoding, self.declared, self.kernel
        )
    }
}
