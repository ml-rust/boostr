//! The Hadamard rotation fused into the feature-major MMQ activation
//! quantization forms the same bytes as the rotation and the quantization
//! run as two launches, and a rotated linear at one token forms the same
//! output bits through either.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test quant_fwht_quantize_cuda

#![cfg(feature = "cuda")]

use boostr::nn::hadamard::HadamardRotation;
use boostr::nn::{Linear, MaybeQuantLinear, MaybeRotatedLinear, QuantLinear, RotatedLinear};
use boostr::quant::cuda::quant_matmul::helpers::quantize_activation_q8_1_mmq;
use boostr::quant::cuda::quant_matmul::rotated::fwht_quantize_activation_q8_1_mmq;
use boostr::quant::{QuantFormat, QuantTensor};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::FwhtOps;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// Activation widths: two model widths and a multiple of 2048.
const DEPTHS: [usize; 3] = [5120, 17408, 6144];

/// Rotation widths: the record's k-group and a wide segment. Both divide
/// every depth above.
const BLOCK_SIZES: [usize; 2] = [128, 1024];

/// Token slots the batched record is padded to; the widest token tile.
const SLOTS: [usize; 2] = [1, 16];

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    // The feature-major kernels need sm_80; `caps.bf16` marks that floor.
    if !device.profile().caps.bf16 {
        eprintln!("skipping: device lacks the sm_80 MMQ family");
        return None;
    }
    let client = CudaClient::new(device.clone()).ok()?;
    Some((client, device))
}

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Values in `[-scale, scale)` with a few exact zeros and tiny values
    /// mixed in, so the scale and the rounding paths all run.
    fn row(&mut self, k: usize, scale: f32) -> Vec<f32> {
        (0..k)
            .map(|i| match i % 97 {
                0 => 0.0,
                1 => 1e-30,
                _ => (self.next_f32() * 2.0 - 1.0) * scale,
            })
            .collect()
    }

    fn signs(&mut self, k: usize) -> Vec<i8> {
        (0..k)
            .map(|_| if self.next_f32() < 0.5 { -1 } else { 1 })
            .collect()
    }
}

fn signs_tensor(signs: &[i8], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let data: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
    Tensor::<CudaRuntime>::from_slice(&data, &[signs.len()], device).expect("signs")
}

/// The two-launch record: numr's `fwht`, then the plain producer.
fn two_step_record(
    client: &CudaClient,
    x: &Tensor<CudaRuntime>,
    signs: Option<&Tensor<CudaRuntime>>,
    k: usize,
    block_size: usize,
    slots: usize,
) -> (Vec<u8>, u32) {
    let rotated = client.fwht(x, block_size, signs).expect("fwht");
    let (buf, ntok) = quantize_activation_q8_1_mmq(client, &rotated, 1, k, slots).expect("plain");
    (buf.to_vec::<u8>(), ntok)
}

fn fused_record(
    client: &CudaClient,
    x: &Tensor<CudaRuntime>,
    signs: Option<&Tensor<CudaRuntime>>,
    k: usize,
    block_size: usize,
    slots: usize,
) -> (Vec<u8>, u32) {
    let (buf, ntok) =
        fwht_quantize_activation_q8_1_mmq(client, x, signs, k, block_size, slots).expect("fused");
    (buf.to_vec::<u8>(), ntok)
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[test]
fn fused_record_matches_fwht_then_quantize_bytes() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let mut rng = Lcg(0x5eed_0001);
    for &k in &DEPTHS {
        for &block_size in &BLOCK_SIZES {
            for &slots in &SLOTS {
                let x = Tensor::<CudaRuntime>::from_slice(&rng.row(k, 3.0), &[1, k], &device)
                    .expect("x");
                let signs = signs_tensor(&rng.signs(k), &device);
                for signs in [Some(&signs), None] {
                    let (want, want_ntok) =
                        two_step_record(&client, &x, signs, k, block_size, slots);
                    let (got, got_ntok) = fused_record(&client, &x, signs, k, block_size, slots);
                    assert_eq!(got_ntok, want_ntok, "K={k} block_size={block_size}");
                    assert_eq!(got.len(), want.len(), "K={k} block_size={block_size}");
                    if let Some(at) = first_difference(&got, &want) {
                        let rec = at / 144;
                        panic!(
                            "K={k} block_size={block_size} slots={slots} signs={}: record {rec} \
                             byte {} differs, fused {:#04x} vs two-step {:#04x}",
                            signs.is_some(),
                            at % 144,
                            got[at],
                            want[at]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn fused_record_refuses_a_segment_narrower_than_a_k_group() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let x = Tensor::<CudaRuntime>::from_slice(&[0.5f32; 256], &[1, 256], &device).expect("x");
    assert!(fwht_quantize_activation_q8_1_mmq(&client, &x, None, 256, 64, 1).is_err());
    assert!(fwht_quantize_activation_q8_1_mmq(&client, &x, None, 256, 96, 1).is_err());
}

/// PQ2_0: 34-byte blocks of 128 elements, an f16 scale at byte 0 and 32
/// bytes of 2-bit codes at byte 2. Every byte is a valid code run, so the
/// block is built directly; `salt` varies the pattern between weights.
fn pq2_0_weight(n: usize, k: usize, salt: usize, device: &CudaDevice) -> QuantTensor<CudaRuntime> {
    let format = QuantFormat::PQ2_0;
    let block_bytes = format.block_bytes();
    let blocks = n * k / format.block_size();
    let mut bytes = vec![0u8; blocks * block_bytes];
    for block in 0..blocks {
        let base = block * block_bytes;
        let d = half::f16::from_f32(0.01 + ((block % 50) as f32) * 0.003 + salt as f32 * 0.02);
        bytes[base..base + 2].copy_from_slice(&d.to_le_bytes());
        for pos in 0..block_bytes - 2 {
            bytes[base + 2 + pos] = ((block * 131 + pos * 17 + salt * 7) % 251) as u8;
        }
    }
    QuantTensor::from_bytes(&bytes, format, &[n, k], device).expect("PQ2_0 weight")
}

fn rotated_quant(
    weight: QuantTensor<CudaRuntime>,
    rotation: HadamardRotation<CudaRuntime>,
) -> MaybeRotatedLinear<CudaRuntime> {
    RotatedLinear::new(
        MaybeQuantLinear::Quantized(QuantLinear::new(weight, None)),
        rotation,
    )
    .expect("rotated linear")
    .into()
}

/// The two-step reference for one rotated linear: the rotation as its own
/// op, then the base layer on the rotated activation.
fn two_step_forward(
    client: &CudaClient,
    layer: &MaybeRotatedLinear<CudaRuntime>,
    x: &Var<CudaRuntime>,
) -> Vec<f32> {
    let rotation = layer.rotation().expect("rotated");
    let rotated = rotation.forward(client, x.tensor()).expect("fwht");
    layer
        .base()
        .forward(client, &Var::new(rotated, false))
        .expect("base forward")
        .tensor()
        .to_vec::<f32>()
}

fn assert_same_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: output {i} is {g:e} ({:#010x}), two-step gives {w:e} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

#[test]
fn rotated_linear_forward_at_one_token_matches_two_step_bits() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let mut rng = Lcg(0x5eed_0002);
    for &k in &DEPTHS {
        for &block_size in &BLOCK_SIZES {
            let n = 256;
            let signs = rng.signs(k);
            let rotation =
                HadamardRotation::<CudaRuntime>::new(block_size, Some(&signs), DType::F32, &device)
                    .expect("rotation");
            let a = rotated_quant(pq2_0_weight(n, k, 1, &device), rotation.clone());
            let b = rotated_quant(pq2_0_weight(n, k, 2, &device), rotation);
            let x = Var::new(
                Tensor::<CudaRuntime>::from_slice(&rng.row(k, 2.0), &[1, 1, k], &device)
                    .expect("x"),
                false,
            );

            let want_a = two_step_forward(&client, &a, &x);
            let want_b = two_step_forward(&client, &b, &x);

            let got_a = a
                .forward(&client, &x)
                .expect("forward")
                .tensor()
                .to_vec::<f32>();
            assert_same_bits(
                &format!("forward K={k} block_size={block_size}"),
                &got_a,
                &want_a,
            );

            let batched =
                MaybeRotatedLinear::forward_batch(&[&a, &b], &client, &x).expect("forward_batch");
            assert_same_bits(
                &format!("forward_batch[0] K={k} block_size={block_size}"),
                &batched[0].tensor().to_vec::<f32>(),
                &want_a,
            );
            assert_same_bits(
                &format!("forward_batch[1] K={k} block_size={block_size}"),
                &batched[1].tensor().to_vec::<f32>(),
                &want_b,
            );
        }
    }
}

/// A rotated linear over a dense base has no quantization to fold the
/// rotation into and still rotates first; its output equals the manual
/// two-step product.
#[test]
fn rotated_dense_base_still_rotates_first() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let mut rng = Lcg(0x5eed_0003);
    let (n, k, block_size) = (64, 1024, 128);
    let signs = rng.signs(k);
    let rotation =
        HadamardRotation::<CudaRuntime>::new(block_size, Some(&signs), DType::F32, &device)
            .expect("rotation");
    let weight =
        Tensor::<CudaRuntime>::from_slice(&rng.row(n * k, 0.05), &[n, k], &device).expect("weight");
    let layer: MaybeRotatedLinear<CudaRuntime> = RotatedLinear::new(
        MaybeQuantLinear::Standard(Linear::new(weight, None, false)),
        rotation,
    )
    .expect("rotated linear")
    .into();
    let x = Var::new(
        Tensor::<CudaRuntime>::from_slice(&rng.row(k, 2.0), &[1, k], &device).expect("x"),
        false,
    );
    let want = two_step_forward(&client, &layer, &x);
    let got = layer
        .forward(&client, &x)
        .expect("forward")
        .tensor()
        .to_vec::<f32>();
    assert_same_bits("dense base", &got, &want);
}
