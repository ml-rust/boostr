//! Shared non-contiguous tensor fixtures for the CUDA sampling unit tests.

use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// A `[vocab_size]` view with stride 2, mirroring the CPU fixture: every
/// other element of a `[vocab_size, 2]` buffer.
pub(super) fn non_contiguous_vocab(
    device: &numr::runtime::cuda::CudaDevice,
) -> Tensor<CudaRuntime> {
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let base = Tensor::<CudaRuntime>::from_slice(&data, &[4, 2], device).expect("base");
    let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
    assert_eq!(view.shape(), &[4]);
    assert!(!view.is_contiguous());
    view
}

/// A `[1, seq_len, vocab_size]` view whose vocab axis is narrowed out of
/// a wider buffer, so it keeps the parent's stride instead of a
/// contiguous one.
pub(super) fn non_contiguous_batch_seq_vocab(
    device: &numr::runtime::cuda::CudaDevice,
) -> Tensor<CudaRuntime> {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let base = Tensor::<CudaRuntime>::from_slice(&data, &[1, 2, 8], device).expect("base");
    let view = base.narrow(2, 0, 4).expect("narrow");
    assert_eq!(view.shape(), &[1, 2, 4]);
    assert!(!view.is_contiguous());
    view
}

/// A `[len]` i64 view with stride 2, mirroring the CPU fixture: every
/// other element of a `[len, 2]` buffer of token ids.
pub(super) fn non_contiguous_ids(device: &numr::runtime::cuda::CudaDevice) -> Tensor<CudaRuntime> {
    let data: Vec<i64> = vec![0, 9, 1, 9];
    let base = Tensor::<CudaRuntime>::from_slice(&data, &[2, 2], device).expect("base");
    let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
    assert_eq!(view.shape(), &[2]);
    assert!(!view.is_contiguous());
    view
}

/// A `[len]` i32 view with stride 2, mirroring the CPU fixture: every
/// other element of a `[len, 2]` buffer of token counts.
pub(super) fn non_contiguous_counts(
    device: &numr::runtime::cuda::CudaDevice,
) -> Tensor<CudaRuntime> {
    let data: Vec<i32> = vec![1, 9, 2, 9];
    let base = Tensor::<CudaRuntime>::from_slice(&data, &[2, 2], device).expect("base");
    let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
    assert_eq!(view.shape(), &[2]);
    assert!(!view.is_contiguous());
    view
}
