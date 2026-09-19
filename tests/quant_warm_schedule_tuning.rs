//! `warm_schedule_tuning` fills the small-M wave bound and the tile-parallel
//! pick for every format passed in, so a `tuned()` lookup made later inside
//! a CUDA graph capture (which refuses to probe) reads the warm cache
//! instead of falling back.
//!
//! Run with:
//!   cd boostr && cargo test --release --features cuda --test quant_warm_schedule_tuning

#![cfg(feature = "cuda")]

use boostr::quant::QuantFormat;
use boostr::quant::cuda::warm_schedule_tuning;
use numr::runtime::cuda::kernels::{SMALLM_MAX_WAVES_KEY, smallm_max_waves};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, tune};
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone()).ok()?;
    Some((client, device))
}

#[test]
fn warm_up_fills_the_tune_cache_before_any_probe_runs() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let device_index = device.id();

    warm_schedule_tuning(&client, &[QuantFormat::Q8_0, QuantFormat::PQ2_0]);

    let waves = tune::get::<usize>(device_index, SMALLM_MAX_WAVES_KEY);
    assert!(waves.is_some(), "small-M wave bound was not cached");

    let q8_0_key = "mmq_feat_major.q8_0.prefers_tile_parallel";
    let pq2_0_key = "mmq_feat_major.pq2_0.prefers_tile_parallel";
    assert!(
        tune::get::<bool>(device_index, q8_0_key).is_some(),
        "Q8_0 tile-parallel pick was not cached"
    );
    assert!(
        tune::get::<bool>(device_index, pq2_0_key).is_some(),
        "PQ2_0 tile-parallel pick was not cached"
    );

    // Inside a graph capture, `tuned` refuses to probe: a cache hit is the
    // only way `smallm_max_waves` can return anything but the fallback.
    let a = Tensor::<CudaRuntime>::from_slice(&vec![1.0f32; 64], &[64], &device).unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&vec![2.0f32; 64], &[64], &device).unwrap();
    let c = Tensor::<CudaRuntime>::from_slice(&vec![0.0f32; 64], &[64], &device).unwrap();

    use numr::ops::BinaryOps as _;
    use std::cell::Cell;
    let inside = Cell::new(None);
    let captured = CudaRuntime::capture_graph_into(&client, &[&a, &b], &[&c], |cc| {
        assert!(cc.is_capturing());
        inside.set(Some(smallm_max_waves(cc)));
        cc.add_into(&c, &a, &b)
    })
    .unwrap();
    captured.launch().unwrap();

    assert_eq!(
        inside.get(),
        waves,
        "capture read a different value than the warm cache holds"
    );
    assert_eq!(
        inside.get(),
        tune::get::<usize>(device_index, SMALLM_MAX_WAVES_KEY),
        "capture must not have changed the cached value"
    );
}
