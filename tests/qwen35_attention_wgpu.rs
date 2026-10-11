//! A Qwen3.5-shaped gated attention block on WebGPU against the CPU.
//!
//! The block's constructor regroups the rows of a block-quantized `attn_q`.
//! That regroup is a strided copy of U8 block bytes. A Q8_0 row of 96
//! columns is 102 bytes, so no row starts on a 4-byte boundary. The test
//! also gathers token rows of a quantized embedding table, a U8
//! `index_select`.
//!
//! WebGPU dots dequantized F32 weights. The CPU reference holds the same
//! weights dequantized to F32 in dense linears, so both runtimes do the same
//! arithmetic.
//!
//! Run with:
//!   cd boostr && cargo test --features wgpu --test qwen35_attention_wgpu

#![cfg(feature = "wgpu")]

use boostr::autograd::Var;
use boostr::inference::KvCache;
use boostr::model::Qwen35AttentionConfig;
use boostr::model::hybrid::{Qwen35AttentionBlock, Qwen35AttentionWeights};
use boostr::nn::{Linear, MaybeQuantLinear, MaybeRotatedLinear, QuantEmbedding, QuantLinear};
use boostr::quant::cpu::kernels::quantize::quantize_q8_0;
use boostr::quant::{DequantOps, QuantFormat, QuantTensor};
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType, Runtime, Tensor};
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

const HIDDEN: usize = 96;
const H: usize = 4;
const H_KV: usize = 2;
const HD: usize = 32;
const ROPE_DIM: usize = 16;
const SEQ: usize = 5;
const MAX_POS: usize = 32;
const VOCAB: usize = 11;
/// Largest difference, relative to the CPU output's peak magnitude.
const TOLERANCE: f32 = 1e-4;

/// The WebGPU client, or `None` after one skip line.
fn wgpu_setup() -> Option<(WgpuClient, WgpuDevice)> {
    let device = WgpuDevice::new(0);
    match WgpuClient::new(device.clone()) {
        Ok(client) => Some((client, device)),
        Err(e) => {
            println!("skip: no WebGPU adapter: {e}");
            None
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn data(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((self.0 >> 40) as f32) / ((1u64 << 24) as f32);
                scale * (2.0 * unit - 1.0)
            })
            .collect()
    }
}

fn cfg() -> Qwen35AttentionConfig {
    Qwen35AttentionConfig {
        hidden_size: HIDDEN,
        num_heads: H,
        num_kv_heads: H_KV,
        head_dim: HD,
        rope_dim: ROPE_DIM,
        rope_sections: [3, 3, 2, 0],
        rope_theta: 10_000.0,
        rms_eps: 1e-6,
    }
}

/// Q8_0 block bytes of a `[rows, cols]` row-major matrix.
fn q8_0(values: &[f32]) -> Vec<u8> {
    let bytes = QuantFormat::Q8_0.storage_bytes(values.len()).unwrap();
    let mut out = vec![0u8; bytes];
    quantize_q8_0(values, &mut out);
    out
}

/// One projection's weights, quantized once and shared by both runtimes.
struct Projection {
    bytes: Vec<u8>,
    shape: [usize; 2],
}

impl Projection {
    fn new(rng: &mut Lcg, rows: usize, cols: usize) -> Self {
        let values = rng.data(rows * cols, 0.5 / (cols as f32).sqrt());
        Self {
            bytes: q8_0(&values),
            shape: [rows, cols],
        }
    }

    fn quant<R: Runtime<DType = DType>>(&self, device: &R::Device) -> QuantTensor<R> {
        QuantTensor::<R>::from_bytes(&self.bytes, QuantFormat::Q8_0, &self.shape, device).unwrap()
    }

    /// The quantized linear, as WebGPU runs it.
    fn quantized(&self, device: &WgpuDevice) -> MaybeRotatedLinear<WgpuRuntime> {
        let weight = self.quant::<WgpuRuntime>(device);
        MaybeRotatedLinear::Plain(MaybeQuantLinear::Quantized(QuantLinear::new(weight, None)))
    }

    /// The dequantized dense linear, the CPU reference.
    fn dense(&self, client: &CpuClient, device: &CpuDevice) -> MaybeRotatedLinear<CpuRuntime> {
        let weight = client
            .dequantize(&self.quant::<CpuRuntime>(device), DType::F32)
            .unwrap();
        MaybeRotatedLinear::Plain(MaybeQuantLinear::Standard(Linear::new(weight, None, false)))
    }
}

struct Parts {
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

fn parts() -> Parts {
    let mut rng = Lcg(0x3535_0001);
    let c = cfg();
    Parts {
        q: Projection::new(&mut rng, c.q_gate_dim(), HIDDEN),
        k: Projection::new(&mut rng, c.kv_dim(), HIDDEN),
        v: Projection::new(&mut rng, c.kv_dim(), HIDDEN),
        o: Projection::new(&mut rng, HIDDEN, c.q_dim()),
        q_norm: rng.data(HD, 0.5).iter().map(|x| x + 1.0).collect(),
        k_norm: rng.data(HD, 0.5).iter().map(|x| x + 1.0).collect(),
    }
}

/// `[4, seq]` text positions starting at `start` on `t`, `h`, `w`.
fn positions(start: usize, seq: usize) -> Vec<i32> {
    let mut data = Vec::with_capacity(4 * seq);
    for _ in 0..3 {
        data.extend((0..seq).map(|i| (start + i) as i32));
    }
    data.resize(4 * seq, 0);
    data
}

/// Prefill `SEQ - 1` tokens, then decode one, on runtime `R`. Returns both
/// outputs as host vectors.
fn prefill_then_decode<R>(
    client: &R::Client,
    device: &R::Device,
    block: &Qwen35AttentionBlock<R>,
    x: &[f32],
) -> Vec<f32>
where
    R: Runtime<DType = DType>,
    R::Client: boostr::model::ModelClient<R>
        + numr::ops::FwhtOps<R>
        + boostr::TensorOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::ShapeOps<R>
        + numr::ops::ActivationOps<R>
        + numr::ops::BinaryOps<R>
        + numr::ops::UnaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + DequantOps<R>,
{
    let rope = cfg().rope_table::<R>(MAX_POS, device).unwrap();
    let mut cache = KvCache::<R>::new(1, H_KV, MAX_POS, MAX_POS, HD, DType::F32, device).unwrap();
    let mut out = Vec::new();
    for (start, len) in [(0, SEQ - 1), (SEQ - 1, 1)] {
        let chunk = &x[start * HIDDEN..(start + len) * HIDDEN];
        let xt = Tensor::<R>::from_slice(chunk, &[1, len, HIDDEN], device).unwrap();
        let pos = Tensor::<R>::from_slice(&positions(start, len), &[4, len], device).unwrap();
        let y = block
            .forward(client, &Var::new(xt, false), &rope, &pos, &mut cache)
            .unwrap();
        out.extend(y.tensor().contiguous().unwrap().to_vec::<f32>());
    }
    out
}

fn relative_gap(reference: &[f32], other: &[f32]) -> f32 {
    assert_eq!(reference.len(), other.len());
    let peak = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let gap = reference
        .iter()
        .zip(other)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    gap / peak.max(f32::MIN_POSITIVE)
}

/// The block with Q8_0 projections on WebGPU equals the dequantized CPU
/// block, prefill and one decode step.
#[test]
fn quantized_attention_block_matches_cpu() {
    let Some((wgpu_client, wgpu_device)) = wgpu_setup() else {
        return;
    };
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuClient::new(cpu_device.clone());
    let p = parts();

    let wgpu_block = Qwen35AttentionBlock::new(
        cfg(),
        Qwen35AttentionWeights {
            attn_q: p.q.quantized(&wgpu_device),
            attn_k: p.k.quantized(&wgpu_device),
            attn_v: p.v.quantized(&wgpu_device),
            attn_output: p.o.quantized(&wgpu_device),
            attn_q_norm: Tensor::from_slice(&p.q_norm, &[HD], &wgpu_device).unwrap(),
            attn_k_norm: Tensor::from_slice(&p.k_norm, &[HD], &wgpu_device).unwrap(),
        },
    )
    .unwrap_or_else(|e| panic!("Qwen35AttentionBlock::new on wgpu: {e}"));
    let cpu_block = Qwen35AttentionBlock::new(
        cfg(),
        Qwen35AttentionWeights {
            attn_q: p.q.dense(&cpu_client, &cpu_device),
            attn_k: p.k.dense(&cpu_client, &cpu_device),
            attn_v: p.v.dense(&cpu_client, &cpu_device),
            attn_output: p.o.dense(&cpu_client, &cpu_device),
            attn_q_norm: Tensor::from_slice(&p.q_norm, &[HD], &cpu_device).unwrap(),
            attn_k_norm: Tensor::from_slice(&p.k_norm, &[HD], &cpu_device).unwrap(),
        },
    )
    .unwrap();

    let x = Lcg(0x3535_0002).data(SEQ * HIDDEN, 1.0);
    let want = prefill_then_decode::<CpuRuntime>(&cpu_client, &cpu_device, &cpu_block, &x);
    let got = prefill_then_decode::<WgpuRuntime>(&wgpu_client, &wgpu_device, &wgpu_block, &x);
    let gap = relative_gap(&want, &got);
    println!("qwen35 attention wgpu vs cpu: relative gap {gap:.3e}");
    assert!(gap < TOLERANCE, "relative gap {gap} exceeds {TOLERANCE}");
}

/// The regrouped Q8_0 `attn_q` bytes on WebGPU equal the CPU regroup.
#[test]
fn q8_0_regroup_matches_cpu() {
    let Some((_, wgpu_device)) = wgpu_setup() else {
        return;
    };
    let p = parts();
    let want =
        p.q.quant::<CpuRuntime>(&CpuDevice::new())
            .regroup_rows(H, 2)
            .unwrap()
            .to_bytes()
            .unwrap();
    let got =
        p.q.quant::<WgpuRuntime>(&wgpu_device)
            .regroup_rows(H, 2)
            .unwrap_or_else(|e| panic!("regroup_rows on wgpu: {e}"))
            .to_bytes()
            .unwrap();
    assert_eq!(got, want);
}

/// Quantized embedding rows gathered on WebGPU equal the CPU rows.
#[test]
fn q8_0_embedding_matches_cpu() {
    let Some((wgpu_client, wgpu_device)) = wgpu_setup() else {
        return;
    };
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuClient::new(cpu_device.clone());
    let table = Projection::new(&mut Lcg(0x3535_0003), VOCAB, HIDDEN);
    let ids = [3i32, 0, 10, 3, 7, 1];

    let cpu = QuantEmbedding::new(table.quant::<CpuRuntime>(&cpu_device));
    let cpu_ids = Tensor::<CpuRuntime>::from_slice(&ids, &[2, 3], &cpu_device).unwrap();
    let want = cpu.forward(&cpu_client, &cpu_ids).unwrap().to_vec::<f32>();

    let gpu = QuantEmbedding::new(table.quant::<WgpuRuntime>(&wgpu_device));
    let gpu_ids = Tensor::<WgpuRuntime>::from_slice(&ids, &[2, 3], &wgpu_device).unwrap();
    let got = gpu
        .forward(&wgpu_client, &gpu_ids)
        .unwrap_or_else(|e| panic!("QuantEmbedding::forward on wgpu: {e}"))
        .to_vec::<f32>();
    assert_eq!(got, want);
}
