//! Unit tests for [`super::Fsq`]'s mixed-radix index codec
//! (`codes_to_indices` / `decode_indices`), moved out of
//! `quantizer_tests.rs` alongside their implementations.

use super::*;
use crate::nn::fsq::config::FsqConfig;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

fn toy_fsq() -> (Fsq<CpuRuntime>, CpuClient, CpuDevice) {
    let (client, device) = cpu_setup();
    let config = FsqConfig::new(vec![4, 4], 2).unwrap();
    let fsq = Fsq::new(config, &device, None, None).unwrap();
    (fsq, client, device)
}

fn neucodec_fsq() -> (Fsq<CpuRuntime>, CpuClient, CpuDevice) {
    let (client, device) = cpu_setup();
    let config = FsqConfig::new(vec![4; 8], 8).unwrap();
    let fsq = Fsq::new(config, &device, None, None).unwrap();
    (fsq, client, device)
}

// --- round trips -------------------------------------------------------

#[test]
fn test_round_trip_toy_levels() {
    let (fsq, client, device) = toy_fsq();

    for index in [0i32, 1, 5, 8, 15] {
        let indices = Tensor::<CpuRuntime>::from_slice(&[index], &[1], &device).unwrap();
        let codes = fsq.decode_indices(&client, &indices).unwrap();
        let recovered = fsq.codes_to_indices(&client, &codes).unwrap();
        let recovered_val: Vec<i32> = recovered.contiguous().unwrap().to_vec();
        assert_eq!(recovered_val, vec![index], "round trip failed for {index}");
    }
}

#[test]
fn test_round_trip_neucodec_levels() {
    let (fsq, client, device) = neucodec_fsq();
    let max_index = fsq.config().codebook_size() as i32 - 1;
    assert_eq!(max_index, 65_535);

    for index in [0i32, 1, 100, max_index / 2, max_index - 1, max_index] {
        let indices = Tensor::<CpuRuntime>::from_slice(&[index], &[1], &device).unwrap();
        let codes = fsq.decode_indices(&client, &indices).unwrap();
        let recovered = fsq.codes_to_indices(&client, &codes).unwrap();
        let recovered_val: Vec<i32> = recovered.contiguous().unwrap().to_vec();
        assert_eq!(recovered_val, vec![index], "round trip failed for {index}");
    }
}

#[test]
fn test_round_trip_batched() {
    let (fsq, client, device) = toy_fsq();

    let all_indices: Vec<i32> = (0..16).collect();
    let indices = Tensor::<CpuRuntime>::from_slice(&all_indices, &[16], &device).unwrap();
    let codes = fsq.decode_indices(&client, &indices).unwrap();
    let recovered = fsq.codes_to_indices(&client, &codes).unwrap();
    let recovered_val: Vec<i32> = recovered.contiguous().unwrap().to_vec();
    assert_eq!(recovered_val, all_indices);
}
