//! Backend parity for greedy `logits_to_token` on rows full of exact ties.
//!
//! The rule matches numr's `argmax`:
//! - Equal logits resolve to the lowest id.
//! - NaN never wins. A row of only NaN returns id 0.
//!
//! A GPU thread scans ids tid, tid + width, ..., so the cases put the lowest
//! tied id in a higher thread than a later duplicate. The CPU result is checked
//! against a host reference, then every GPU backend is checked against CPU.

#![cfg_attr(
    not(any(feature = "wgpu", feature = "cuda")),
    allow(dead_code, unused_imports)
)]

use super::helpers::*;
use boostr::ops::traits::inference::sampling::SamplingOps;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// The Llama 3 vocabulary size.
const LARGE_VOCAB: usize = 128_256;

/// Greedy token for `[1, 2, vocab]` logits of `dtype` whose last row is `row`.
///
/// The first row peaks at id 0, so a read of the wrong position fails the case.
/// Penalties are off. The one-entry id tensor avoids a zero-size allocation.
fn greedy_on<R: Runtime<DType = DType>>(
    client: &R::Client,
    device: &R::Device,
    row: &[f32],
    dtype: DType,
) -> i64
where
    R::Client: SamplingOps<R> + TypeConversionOps<R>,
{
    let vocab = row.len();
    let mut data = vec![0.0f32; vocab];
    data[0] = 1.0e4;
    data.extend_from_slice(row);
    let logits = Tensor::<R>::from_slice(&data, &[1, 2, vocab], device).unwrap();
    let logits = if dtype == DType::F32 {
        logits
    } else {
        client.cast(&logits, dtype).unwrap()
    };
    let ids = Tensor::<R>::from_slice(&[0i64], &[1], device).unwrap();
    let counts = Tensor::<R>::from_slice(&[0i32], &[1], device).unwrap();
    let out = client
        .logits_to_token(
            &logits, &ids, &counts, 0, 1.0, 0.0, 0.0, 0.0, 0, 1.0, 0.0, None,
        )
        .unwrap();
    out.to_vec::<i64>()[0]
}

/// Lowest id of the largest non-NaN value, or 0 when every value is NaN.
fn reference(row: &[f32]) -> i64 {
    let mut best: Option<(f32, usize)> = None;
    for (i, &v) in row.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if best.is_none_or(|(b, _)| v > b) {
            best = Some((v, i));
        }
    }
    best.map_or(0, |(_, i)| i as i64)
}

/// `row` rounded through `dtype`, as the kernels read it.
fn stored(row: &[f32], dtype: DType) -> Vec<f32> {
    if dtype == DType::F32 {
        return row.to_vec();
    }
    let (client, device) = setup_cpu();
    let t = Tensor::<CpuRuntime>::from_slice(row, &[row.len()], &device).unwrap();
    let narrow = client.cast(&t, dtype).unwrap();
    client.cast(&narrow, DType::F32).unwrap().to_vec::<f32>()
}

fn check(name: &str, row: &[f32], dtype: DType) {
    let (cpu_client, cpu_device) = setup_cpu();
    let cpu = greedy_on::<CpuRuntime>(&cpu_client, &cpu_device, row, dtype);
    assert_eq!(
        cpu,
        reference(&stored(row, dtype)),
        "{name}: CPU vs reference, {dtype:?}, vocab {}",
        row.len()
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let gpu = greedy_on::<numr::runtime::cuda::CudaRuntime>(&client, &device, row, dtype);
        assert_eq!(
            gpu,
            cpu,
            "{name}: CUDA vs CPU, {dtype:?}, vocab {}",
            row.len()
        );
    });

    #[cfg(feature = "wgpu")]
    if dtype == DType::F32 {
        with_wgpu_backend(|client, device| {
            let gpu = greedy_on::<numr::runtime::wgpu::WgpuRuntime>(&client, &device, row, dtype);
            assert_eq!(gpu, cpu, "{name}: WGPU vs CPU, vocab {}", row.len());
        });
    }
}

/// Row of small values with 50 at every id in `peaks`.
fn spiked(vocab: usize, peaks: &[usize]) -> Vec<f32> {
    let mut row: Vec<f32> = (0..vocab).map(|i| (i % 7) as f32).collect();
    for &p in peaks {
        row[p] = 50.0;
    }
    row
}

#[test]
fn test_greedy_all_equal_rows() {
    check("all equal", &[3.5; 1000], DType::F32);
    check("all equal, vocab", &vec![-2.0; LARGE_VOCAB], DType::F32);
}

// The WebGPU workgroup has 256 threads and the CUDA block 1024. Each case puts
// a later tie in a lower thread than the lowest tie for the width it names.
#[test]
fn test_greedy_strided_ties() {
    // 256 wide: 1, 257 and 513 all sit in thread 1.
    check("1, 257, 513", &spiked(1000, &[1, 257, 513]), DType::F32);
    // 256 wide: 300 is thread 44, the lowest tie 45 is thread 45.
    check(
        "five ties",
        &spiked(1000, &[700, 300, 45, 513, 999]),
        DType::F32,
    );
    // Both widths: 1024 is thread 0, the lowest tie 3 is thread 3.
    check("3 behind 1024", &spiked(4000, &[1024, 1023, 3]), DType::F32);
    // Both widths: 1100 is thread 76, the lowest tie 77 is thread 77.
    check(
        "77 behind 1100",
        &spiked(4000, &[3000, 1100, 77, 2050]),
        DType::F32,
    );
    // 256 wide: 515 is thread 3. 1024 wide: 100355 is thread 3. The lowest tie,
    // 77, is thread 77 in both.
    let peaks = [128_000, 100_355, 515, 1000, 77];
    check("vocab row", &spiked(LARGE_VOCAB, &peaks), DType::F32);
}

#[test]
fn test_greedy_negative_ties() {
    // The max -3 sits at ids 4, 9, 14, ... 259 is thread 3 at 256 wide, and
    // 1024 is thread 0 at 1024 wide. Neither may beat id 4.
    let row: Vec<f32> = (0..3000).map(|i| -(((i + 1) % 5) as f32) - 3.0).collect();
    check("negative", &row, DType::F32);
}

#[test]
fn test_greedy_nonfinite_rows() {
    let n = 600;
    check("all -inf", &vec![f32::NEG_INFINITY; n], DType::F32);
    check("all NaN", &vec![f32::NAN; n], DType::F32);

    // NaN at id 0 never wins, so the first -inf (id 1) is the argmax.
    let mut nan_head = vec![f32::NEG_INFINITY; n];
    nan_head[0] = f32::NAN;
    nan_head[300] = f32::NAN;
    check("NaN head, -inf rest", &nan_head, DType::F32);

    // NaN spread through a tied row. The answer stays the lowest finite tie.
    let mut nan_ties = spiked(n, &[257, 2]);
    for i in [0, 1, 256, 512] {
        nan_ties[i] = f32::NAN;
    }
    check("NaN with ties", &nan_ties, DType::F32);
}

#[cfg(feature = "f16")]
#[test]
fn test_greedy_narrow_float_rounding_ties() {
    // 1 + i * 1e-4 collapses to a few F16 and BF16 values, so each repeats
    // across many threads.
    let row: Vec<f32> = (0..1000).map(|i| 1.0 + i as f32 * 1e-4).collect();
    let rev: Vec<f32> = row.iter().rev().copied().collect();
    for dtype in [DType::F16, DType::BF16] {
        check("rounding ties", &row, dtype);
        check("rounding ties, descending", &rev, dtype);
        check(
            "rounding ties, vocab",
            &spiked(LARGE_VOCAB, &[515, 77]),
            dtype,
        );
    }
}
