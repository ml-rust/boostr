//! GGUF reader + dequant kernels against a real Ternary-Bonsai-2-27B checkpoint,
//! for the two PrismML fork formats `PQ2_0` and `PTQ1_0`.
//!
//! # Why this file exists
//!
//! Every other GGUF dequant test uses synthetic byte fixtures
//! (`tests/gguf_dequant_cpu_cuda_parity.rs`) or a cosine-similarity floor
//! against real weights (`tests/quant_vs_f16_matrix.rs`). The F16 file of
//! this checkpoint stores the same ternary values the PQ2_0 and PTQ1_0
//! files encode, so dequant must reproduce it BIT-EXACT. This file holds
//! boostr's reader and CPU/CUDA kernels to that bound on a real file.
//!
//! # Fixture files
//!
//! `BOOSTR_BONSAI2_DIR` must hold:
//! - `Ternary-Bonsai-2-27B-F16.gguf` — oracle
//! - `Ternary-Bonsai-2-27B-PQ2_0.gguf`
//! - `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
//!
//! ```bash
//! cargo nextest run -p boostr --test gguf_prism_real_file
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo nextest run --features cuda \
//!   -p boostr --test gguf_prism_real_file
//! ```
//!
//! # Memory
//!
//! `blk.0.attn_gate.weight` is 31,457,280 elements — 125 MB as f32. The
//! oracle load goes through `Gguf::load_tensor_f32_streaming`, which chunks
//! the dequant instead of allocating the full CPU-side buffer at once. The
//! quantized tensor itself is a few MB compressed, loaded whole via
//! `Gguf::load_tensor_quantized`.

use std::path::{Path, PathBuf};

use boostr::format::gguf::{GgmlType, Gguf, PrismHadamardConfig, SignMode};
use boostr::quant::DequantOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::{Runtime, RuntimeClient};
#[cfg(feature = "cuda")]
use std::sync::{Mutex, OnceLock};

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";

const F16_FILE: &str = "Ternary-Bonsai-2-27B-F16.gguf";
const PQ2_0_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const PTQ1_0_FILE: &str = "Ternary-Bonsai-2-27B-PTQ1_0.gguf";

/// The tensor checked for bit-exact dequant. Shape `(5120, 6144)` in GGUF
/// (innermost-first) dim order, 31,457,280 elements.
const TENSOR: &str = "blk.0.attn_gate.weight";
const EXPECTED_NUMEL: usize = 5120 * 6144;

/// Tensors checked in `reader_reports_prism_types` — one per major role
/// (attention, embedding, output head), so a format assignment that only
/// covers one role cannot pass by accident.
const TYPE_CHECK_TENSORS: &[&str] = &[TENSOR, "token_embd.weight", "output.weight"];

/// The base directory from `BOOSTR_BONSAI2_DIR`, or `None` when unset.
fn bonsai2_dir() -> Option<PathBuf> {
    std::env::var(ENV_DIR).ok().map(PathBuf::from)
}

/// Resolves and checks the three fixture files. Returns `None` after
/// printing one `skip: <path> not found` line for the first missing path
/// (the directory itself, or whichever file inside it is absent).
fn require_files() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let Some(dir) = bonsai2_dir() else {
        println!("skip: {ENV_DIR} not set");
        return None;
    };
    if !dir.is_dir() {
        println!("skip: {} not found", dir.display());
        return None;
    }
    let f16 = dir.join(F16_FILE);
    let pq2_0 = dir.join(PQ2_0_FILE);
    let ptq1_0 = dir.join(PTQ1_0_FILE);
    for path in [&f16, &pq2_0, &ptq1_0] {
        if !path.is_file() {
            println!("skip: {} not found", path.display());
            return None;
        }
    }
    Some((f16, pq2_0, ptq1_0))
}

/// Loads `TENSOR` as f32 via the streaming path — bounded CPU memory
/// regardless of tensor size.
fn load_f16_oracle(f16_path: &Path, device: &CpuDevice) -> Vec<f32> {
    let mut reader =
        Gguf::open(f16_path).unwrap_or_else(|e| panic!("open {}: {e}", f16_path.display()));
    reader
        .load_tensor_f32_streaming::<CpuRuntime>(TENSOR, device)
        .unwrap_or_else(|e| panic!("load {TENSOR} from {}: {e}", f16_path.display()))
        .to_vec::<f32>()
}

/// Asserts `oracle` and `got` agree element-wise, exactly. Reports the first
/// mismatch index and both values on failure — the fixture is bit-exact by
/// construction, so any diff at all is a decode defect, not rounding.
fn assert_bit_exact(oracle: &[f32], got: &[f32], label: &str) {
    assert_eq!(
        oracle.len(),
        got.len(),
        "{label}: element count differs: oracle has {}, dequant has {}",
        oracle.len(),
        got.len()
    );
    assert_eq!(
        oracle.len(),
        EXPECTED_NUMEL,
        "{label}: {TENSOR} has {} elements, expected {EXPECTED_NUMEL}",
        oracle.len()
    );
    for (i, (&a, &b)) in oracle.iter().zip(got.iter()).enumerate() {
        assert!(
            a == b,
            "{label}: first mismatch at index {i}: f16 oracle={a} dequant={b}"
        );
    }
}

/// CPU-side check shared by both formats: dequantize `quant_path`'s copy of
/// `TENSOR` with `CpuClient::dequantize` and compare against the F16 oracle.
fn cpu_matches_oracle(f16_path: &Path, quant_path: &Path, label: &str) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let oracle = load_f16_oracle(f16_path, &device);

    let mut reader =
        Gguf::open(quant_path).unwrap_or_else(|e| panic!("open {}: {e}", quant_path.display()));
    let qt = reader
        .load_tensor_quantized::<CpuRuntime>(TENSOR, &device)
        .unwrap_or_else(|e| panic!("load {TENSOR} from {}: {e}", quant_path.display()));
    let got = client
        .dequantize(&qt, DType::F32)
        .unwrap_or_else(|e| panic!("{label}: CpuClient::dequantize failed: {e}"))
        .to_vec::<f32>();

    assert_bit_exact(&oracle, &got, label);
}

#[test]
fn pq2_0_matches_f16_oracle() {
    let Some((f16_path, pq2_0_path, _)) = require_files() else {
        return;
    };
    cpu_matches_oracle(&f16_path, &pq2_0_path, "pq2_0_matches_f16_oracle");
}

#[test]
fn ptq1_0_matches_f16_oracle() {
    let Some((f16_path, _, ptq1_0_path)) = require_files() else {
        return;
    };
    cpu_matches_oracle(&f16_path, &ptq1_0_path, "ptq1_0_matches_f16_oracle");
}

#[test]
fn reader_reports_prism_types() {
    let Some((_, pq2_0_path, ptq1_0_path)) = require_files() else {
        return;
    };

    let pq2_0 =
        Gguf::open(&pq2_0_path).unwrap_or_else(|e| panic!("open {}: {e}", pq2_0_path.display()));
    for name in TYPE_CHECK_TENSORS {
        let info = pq2_0
            .tensor_info(name)
            .unwrap_or_else(|e| panic!("tensor_info({name}) in {}: {e}", pq2_0_path.display()));
        assert_eq!(
            info.ggml_type,
            GgmlType::PQ2_0,
            "{name} in {}: expected PQ2_0, got {:?}",
            pq2_0_path.display(),
            info.ggml_type
        );
    }
    assert_eq!(
        pq2_0.metadata().get_u32("prism.hadamard.version"),
        Some(1),
        "prism.hadamard.version in {}: expected u32 1",
        pq2_0_path.display()
    );

    let ptq1_0 =
        Gguf::open(&ptq1_0_path).unwrap_or_else(|e| panic!("open {}: {e}", ptq1_0_path.display()));
    for name in TYPE_CHECK_TENSORS {
        let info = ptq1_0
            .tensor_info(name)
            .unwrap_or_else(|e| panic!("tensor_info({name}) in {}: {e}", ptq1_0_path.display()));
        assert_eq!(
            info.ggml_type,
            GgmlType::PTQ1_0,
            "{name} in {}: expected PTQ1_0, got {:?}",
            ptq1_0_path.display(),
            info.ggml_type
        );
    }
    assert_eq!(
        ptq1_0.metadata().get_u32("prism.hadamard.version"),
        Some(1),
        "prism.hadamard.version in {}: expected u32 1",
        ptq1_0_path.display()
    );
}

/// Parses the full `prism.hadamard.*` contract out of the real PQ2_0 file
/// and checks it against the fixture's known-good values.
#[test]
fn prism_hadamard_config_parses_real_file() {
    let Some((_, pq2_0_path, _)) = require_files() else {
        return;
    };

    let pq2_0 =
        Gguf::open(&pq2_0_path).unwrap_or_else(|e| panic!("open {}: {e}", pq2_0_path.display()));
    let cfg = PrismHadamardConfig::from_metadata(pq2_0.metadata())
        .unwrap_or_else(|e| {
            panic!(
                "parse prism.hadamard metadata in {}: {e}",
                pq2_0_path.display()
            )
        })
        .unwrap_or_else(|| panic!("prism.hadamard.version missing in {}", pq2_0_path.display()));

    assert_eq!(cfg.block_size, 1024, "block_size");
    assert_eq!(cfg.sign_mode, SignMode::Explicit, "sign_mode");
    assert_eq!(cfg.weight_names().count(), 401, "weight_names count");
    assert!(
        cfg.rotates("blk.0.ssm_out.weight"),
        "blk.0.ssm_out.weight should rotate"
    );
    assert!(
        cfg.inverts("token_embd.weight"),
        "token_embd.weight should invert"
    );
    assert!(cfg.gdn_v_grouped, "gdn_v_grouped");

    for width in [5120usize, 6144, 17408] {
        cfg.signs_for_width(width)
            .unwrap_or_else(|e| panic!("signs_for_width({width}): {e}"))
            .unwrap_or_else(|| panic!("signs_for_width({width}) returned None"));
    }

    let signs_5120 = cfg
        .signs_for_width(5120)
        .unwrap_or_else(|e| panic!("signs_for_width(5120): {e}"))
        .unwrap_or_else(|| panic!("signs_for_width(5120) returned None"));
    assert_eq!(signs_5120.len(), 5120, "signs_for_width(5120) length");
    assert!(
        signs_5120.iter().all(|&s| s == 1 || s == -1),
        "signs_for_width(5120) entries must all be +1 or -1"
    );
}

// ── CUDA parity: same oracle, `CudaClient::dequantize` instead of CPU ──────

#[cfg(feature = "cuda")]
static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    numr::runtime::cuda::is_cuda_available()
}

/// CUDA-side check shared by both formats. Mirrors `cpu_matches_oracle`, but
/// dequantizes on `CudaClient` and skips (does not fail) when no CUDA device
/// is present, matching `tests/gguf_dequant_cpu_cuda_parity.rs`'s convention.
#[cfg(feature = "cuda")]
fn cuda_matches_oracle(f16_path: &Path, quant_path: &Path, label: &str) {
    if !cuda_available() {
        println!("skip: {label}: CUDA is not available on this machine");
        return;
    }
    let _lock = cuda_lock();

    let cpu_device = CpuDevice::new();
    let oracle = load_f16_oracle(f16_path, &cpu_device);

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);
    cuda_client.synchronize();

    let got = {
        let mut reader =
            Gguf::open(quant_path).unwrap_or_else(|e| panic!("open {}: {e}", quant_path.display()));
        let qt = reader
            .load_tensor_quantized::<CudaRuntime>(TENSOR, &cuda_device)
            .unwrap_or_else(|e| panic!("load {TENSOR} from {}: {e}", quant_path.display()));
        cuda_client
            .dequantize(&qt, DType::F32)
            .unwrap_or_else(|e| panic!("{label}: CudaClient::dequantize failed: {e}"))
            .to_vec::<f32>()
    };
    cuda_client.synchronize();

    assert_bit_exact(&oracle, &got, label);
}

#[cfg(feature = "cuda")]
#[test]
fn pq2_0_matches_f16_oracle_cuda() {
    let Some((f16_path, pq2_0_path, _)) = require_files() else {
        return;
    };
    cuda_matches_oracle(&f16_path, &pq2_0_path, "pq2_0_matches_f16_oracle_cuda");
}

#[cfg(feature = "cuda")]
#[test]
fn ptq1_0_matches_f16_oracle_cuda() {
    let Some((f16_path, _, ptq1_0_path)) = require_files() else {
        return;
    };
    cuda_matches_oracle(&f16_path, &ptq1_0_path, "ptq1_0_matches_f16_oracle_cuda");
}
