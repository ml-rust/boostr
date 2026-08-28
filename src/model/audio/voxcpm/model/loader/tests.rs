//! Split out of `loader.rs` to keep that file under the crate's 500-line
//! hard limit for model-architecture files after
//! [`VoxCpm2Model::lora_projection_names`](super::VoxCpm2Model::lora_projection_names)
//! was added. `use super::*;` below reaches every item `loader.rs` itself
//! imported, exactly as if this module were still inline.

use super::*;
use numr::runtime::cpu::CpuRuntime;

#[test]
fn rejects_missing_checkpoint() {
    let device = <CpuRuntime as Runtime>::default_device();
    assert!(
        VoxCpm2Model::<CpuRuntime>::from_checkpoint(
            "/nonexistent/voxcpm2",
            "/nonexistent/audiovae.safetensors",
            &device,
            Some(DType::F32),
        )
        .is_err()
    );
}
