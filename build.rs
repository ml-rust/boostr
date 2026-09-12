//! Build script for boostr
//!
//! Compiles CUDA kernels to multi-arch fatbins when the cuda feature is
//! enabled, following numr's build.rs: real SASS for the local GPU(s) plus
//! forward-JIT PTX, never PTX alone. A PTX-only module JIT-compiles at every
//! `cuModuleLoad`, and the MMQ module's PTX runs to a hundred megabytes, so
//! every process paid that JIT before its first quantized matmul.
//!
//! Arch selection reads `BOOSTR_CUDA_ARCH`, falling back to `NUMR_CUDA_ARCH`
//! so one variable configures both crates: a comma-separated list (`86`,
//! `sm_86`, `8.6`, `86,89,90`), `all`/`portable` for every supported arch, or
//! unset to detect the local GPU(s) via `nvidia-smi` (portable when none is
//! found).

fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Kernel sets: (directory, filename, min arch, required, output_name_override)
    // Most kernels target sm_75 (Turing+); flash_v3 needs sm_90 (Hopper)
    // Optional kernels (required=false) warn on failure instead of panicking —
    // they require hardware-specific features (e.g. Hopper) that may not be
    // available on all build machines.
    // output_name_override: Some("name.fatbin") overrides the default (filename with .fatbin ext).
    // Needed when files in different subdirs share the same filename (e.g. gemv/q5_k.cu vs gemm/q5_k.cu).
    // Helper to build kernel entries concisely: (dir, file, arch, required, ptx_override=None)
    macro_rules! k {
        ($dir:expr, $file:expr, $arch:expr, $req:expr) => {
            (PathBuf::from($dir), $file.into(), $arch.into(), $req, None)
        };
    }

    let mut kernel_sets: Vec<(PathBuf, String, String, bool, Option<String>)> = vec![
        // Quantization kernels
        k!("src/quant/cuda/kernels", "dequant.cu", "sm_75", true),
        k!(
            "src/quant/cuda/kernels",
            "dequant_generic.cu",
            "sm_75",
            true
        ),
        k!(
            "src/quant/cuda/kernels",
            "quant_matmul_generic.cu",
            "sm_75",
            true
        ),
        k!("src/quant/cuda/kernels", "quant_matmul.cu", "sm_75", true),
        k!("src/quant/cuda/kernels", "quant_gemv.cu", "sm_75", true),
        k!("src/quant/cuda/kernels", "int4_gemm.cu", "sm_75", true),
        k!("src/quant/cuda/kernels", "int4_gemm_gptq.cu", "sm_75", true),
        k!("src/quant/cuda/kernels", "nf4_quant.cu", "sm_75", true),
        k!("src/quant/cuda/kernels", "marlin_gemm.cu", "sm_75", true),
        k!(
            "src/quant/cuda/kernels",
            "fused_int4_swiglu.cu",
            "sm_75",
            true
        ),
        k!("src/quant/cuda/kernels", "fused_int4_qkv.cu", "sm_75", true),
        k!("src/quant/cuda/kernels", "quant_act.cu", "sm_75", true),
        // TCF native quantized kernels: dequant, GEMV, GEMM in one module,
        // sharing the device decoder in tcf.cuh.
        k!("src/quant/cuda/kernels", "tcf.cu", "sm_75", true),
        // sm_80, not sm_75: `mma.sync.aligned.m16n8k32...s8.s8.s32` is an
        // Ampere+ instruction, unavailable at sm_75.
        k!("src/quant/cuda/kernels", "mma_int8_probe.cu", "sm_80", true),
        // sm_80, not sm_75: `mma.sync.aligned.m16n8k32...s8.s8.s32` is an
        // Ampere+ instruction, unavailable at sm_75.
        k!("src/quant/cuda/kernels", "quant_mmq_mma.cu", "sm_80", true),
    ];

    // Per-format GEMV + GEMM kernels: each format generates a gemv/ and gemm/ entry.
    // All target sm_75, are required, and use the naming convention gemv_{fmt}.fatbin / gemm_{fmt}.fatbin.
    let per_format_kernels: &[&str] = &[
        // K-quants
        "q5_k", "q3_k", "q2_k", // Simple quants
        "q5_0", "q4_1", "q5_1", "q8_1", "q8_k", // IQ quants
        "iq4_nl", "iq4_xs", "iq3_s", "iq2_xs", "iq1_s", "iq1_m", "iq2_xxs", "iq2_s", "iq3_xxs",
        // Ternary quants
        "tq1_0", "tq2_0",
    ];

    let gemv_dir = PathBuf::from("src/quant/cuda/kernels/gemv");
    let gemm_dir = PathBuf::from("src/quant/cuda/kernels/gemm");
    for fmt in per_format_kernels {
        let cu_file = format!("{}.cu", fmt);
        kernel_sets.push((
            gemv_dir.clone(),
            cu_file.clone(),
            "sm_75".to_string(),
            true,
            Some(format!("gemv_{}.fatbin", fmt)),
        ));
        kernel_sets.push((
            gemm_dir.clone(),
            cu_file,
            "sm_75".to_string(),
            true,
            Some(format!("gemm_{}.fatbin", fmt)),
        ));
    }

    // TCF's dp4a GEMV is outside the loop above: a TCF encoding has a GEMV
    // here but no gemm/ sibling — its large-batch path is the feature-major
    // MMQ kernel in quant_mmq_mma.cu, or the f32 tile in tcf.cu.
    kernel_sets.push((
        gemv_dir,
        "tcf_q4as32dt64.cu".to_string(),
        "sm_75".to_string(),
        true,
        Some("gemv_tcf_q4as32dt64.fatbin".to_string()),
    ));

    kernel_sets.extend([
        // Attention kernels
        k!(
            "src/ops/cuda/kernels/attention",
            "flash_v2.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "flash_v2_bwd.cu",
            "sm_75",
            true
        ),
        // sm_80, not sm_75: the FP8 kernels are guarded by
        // `#if __CUDA_ARCH__ >= 800`, so compiling them at sm_75 silently drops
        // every FP8 symbol while the launchers still accept `DType::F8E4M3` —
        // a runtime kernel-lookup failure. They live in their own translation
        // units because flash_v2.cu / flash_v2_bwd.cu also hold the general
        // flash kernels, which legitimately target Turing (sm_75).
        k!(
            "src/ops/cuda/kernels/attention",
            "flash_v2_fp8.cu",
            "sm_80",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "flash_v2_bwd_fp8.cu",
            "sm_80",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "paged_attention.cu",
            "sm_75",
            true
        ),
        // sm_80, not sm_75: the FP8 kernels need `__CUDA_ARCH__ >= 800` intrinsics.
        // Compiling them at sm_75 dropped every FP8 symbol while the launcher still
        // accepted FP8 dtypes, so the lookup failed on every device.
        k!(
            "src/ops/cuda/kernels/attention",
            "paged_attention_fp8.cu",
            "sm_80",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "paged_attention_bwd.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "varlen_attention.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "varlen_attention_fwd_fp16.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "varlen_attention_bwd.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "varlen_attention_bwd_fp16.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "mqa_gqa.cu",
            "sm_75",
            true
        ),
        // sm_80, not sm_75: the bf16 backward kernels are guarded by
        // `#if __CUDA_ARCH__ >= 800`, so compiling this at sm_75 silently drops
        // every bf16 symbol while the launcher still accepts `DType::BF16` —
        // a runtime kernel-lookup failure on the primary training dtype.
        // The forward (mqa_gqa.cu) has no native bf16 arithmetic and runs at
        // sm_75; `flash.rs` gates both call sites on `caps.bf16` so the two
        // stay on the same kernel family instead of pairing an untested mix.
        k!(
            "src/ops/cuda/kernels/attention",
            "mqa_gqa_bwd.cu",
            "sm_80",
            true
        ),
        k!("src/ops/cuda/kernels/attention", "sdpa.cu", "sm_75", true),
        k!(
            "src/ops/cuda/kernels/attention",
            "fused_qkv.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "decode_attention.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "kv_insert.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "paged_decode_attention.cu",
            "sm_75",
            true
        ),
        // Flash v3 — sm_90 (Hopper warp specialization, optional)
        k!(
            "src/ops/cuda/kernels/attention",
            "flash_v3.cu",
            "sm_90",
            false
        ),
        k!(
            "src/ops/cuda/kernels/attention",
            "flash_v3_bwd.cu",
            "sm_90",
            false
        ),
        // Cache kernels
        k!(
            "src/ops/cuda/kernels/cache",
            "kv_cache_update.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/cache",
            "kv_cache_int4.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/cache",
            "kv_cache_fp8.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/cache",
            "kv_cache_fp8_bwd.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/cache",
            "kv_cache_quant.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/cache",
            "reshape_and_cache.cu",
            "sm_75",
            true
        ),
        // Position kernels
        k!("src/ops/cuda/kernels/position", "alibi.cu", "sm_75", true),
        k!(
            "src/ops/cuda/kernels/position",
            "alibi_bwd.cu",
            "sm_75",
            true
        ),
        k!("src/ops/cuda/kernels/position", "rope.cu", "sm_75", true),
        k!(
            "src/ops/cuda/kernels/position",
            "rope_interleaved.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/position",
            "rope_yarn.cu",
            "sm_75",
            true
        ),
        // Fused optimizer kernels
        k!(
            "src/ops/cuda/kernels/training",
            "fused_adamw.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/training",
            "fused_sgd.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/training",
            "fused_adagrad.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/training",
            "fused_lamb.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/training",
            "fused_multi_tensor.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/training",
            "fused_grad_unscale_clip.cu",
            "sm_75",
            true
        ),
        // Architecture kernels (MoE, SSM)
        k!(
            "src/ops/cuda/kernels/architecture",
            "moe_routing.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/architecture",
            "moe_grouped_gemm.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/architecture",
            "ssd_state_passing.cu",
            "sm_75",
            true
        ),
        // Inference kernels (speculative decoding, sampling, prefix cache)
        k!(
            "src/ops/cuda/kernels/inference",
            "prefix_cache_lookup.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/inference",
            "speculative_verify.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/inference",
            "sampling_penalties.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/inference",
            "sampling.cu",
            "sm_75",
            true
        ),
        k!(
            "src/ops/cuda/kernels/inference",
            "logits_to_token.cu",
            "sm_75",
            true
        ),
        // Calibration kernels (quantization)
        k!(
            "src/ops/cuda/kernels/quantization",
            "calibration.cu",
            "sm_75",
            true
        ),
    ]);

    // Shared headers are not compilation units, so the `.cu` paths above do not
    // cover them. Emit every `.cuh` under the kernel source roots instead of
    // guessing one from the `.cu` stem: that guess matched only same-stem pairs
    // and left `dtype_traits.cuh`, `attention/atomics.cuh`, `gemm/common.cuh`
    // and `gemv/common.cuh` untracked. Scanning `#include` lines is not an
    // option either — it misses transitive includes.
    //
    // No stale-PTX window is open today: the loop below invokes `nvcc` for
    // every kernel on every run, so a header change is always picked up. This
    // makes the declared dependency honest, and is what keeps the build correct
    // if that unconditional rebuild ever becomes incremental.
    for root in ["src/ops/cuda/kernels", "src/quant/cuda/kernels"] {
        emit_header_deps(std::path::Path::new(root));
    }

    let nvcc = find_nvcc().unwrap_or_else(|| {
        eprintln!();
        eprintln!("=== CUDA COMPILATION ERROR ===");
        eprintln!();
        eprintln!("Could not find nvcc (NVIDIA CUDA Compiler).");
        eprintln!("Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads");
        eprintln!();
        panic!("nvcc not found - CUDA Toolkit must be installed for the 'cuda' feature");
    });

    // Re-run when the override changes — otherwise a stale fatbin survives
    // an `export BOOSTR_CUDA_ARCH=...` until something else invalidates OUT_DIR.
    println!("cargo:rerun-if-env-changed=BOOSTR_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NUMR_CUDA_ARCH");
    let (selected_arches, mode_desc) = select_arches();
    println!(
        "cargo:warning=boostr: compiling {} CUDA kernels into fatbins for {} \
         (+ per-kernel PTX floor and compute_120 JIT ceiling)",
        kernel_sets.len(),
        mode_desc
    );

    struct KernelOutcome {
        file: String,
        required: bool,
        arch: String,
        fatbin_path: PathBuf,
        success: bool,
        stdout: String,
        stderr: String,
        exec_error: Option<String>,
    }

    // Every kernel is compiled for each selected arch plus its PTX floor, so
    // the work is several times the old single-PTX build. A bounded pool
    // keeps nvcc's memory use in check; each output path is independent.
    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    let work_queue = std::sync::Mutex::new(kernel_sets.clone());
    let outcomes = std::sync::Mutex::new(Vec::<KernelOutcome>::new());

    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| {
                loop {
                    let (kernels_dir, kernel_file, arch, required, name_override) = {
                        let mut queue = work_queue.lock().unwrap();
                        match queue.pop() {
                            Some(entry) => entry,
                            None => break,
                        }
                    };
                    let cu_path = kernels_dir.join(&kernel_file);
                    let fatbin_name = name_override
                        .clone()
                        .unwrap_or_else(|| kernel_file.replace(".cu", ".fatbin"));
                    let fatbin_path = out_dir.join(&fatbin_name);

                    println!("cargo:rerun-if-changed={}", cu_path.display());
                    if !cu_path.exists() {
                        panic!(
                            "CUDA kernel source not found: {}\n\
                             Ensure kernel files exist in {}",
                            cu_path.display(),
                            kernels_dir.display()
                        );
                    }

                    // Include paths: kernel's own dir + root kernels dir for
                    // shared headers (dtype_traits.cuh).
                    let include_arg = format!("-I{}", kernels_dir.display());
                    let root_include_arg = "-Isrc/ops/cuda/kernels".to_string();

                    let mut args: Vec<String> = vec![
                        "-fatbin".to_string(),
                        "-O3".to_string(),
                        "--use_fast_math".to_string(),
                        include_arg,
                        root_include_arg,
                    ];
                    for gc in gencode_flags(&arch, &selected_arches) {
                        args.push("-gencode".to_string());
                        args.push(gc);
                    }
                    args.push("-o".to_string());
                    args.push(fatbin_path.to_str().unwrap().to_string());
                    args.push(cu_path.to_str().unwrap().to_string());

                    // Cargo reruns this script when ANY kernel source changes,
                    // and every kernel was then recompiled. A kernel whose
                    // fatbin is newer than its source, every header it can
                    // include, this script, and whose recorded nvcc arguments
                    // are unchanged, is already built.
                    let args_path = out_dir.join(format!("{fatbin_name}.args"));
                    let args_text = args.join("\n");
                    if fatbin_up_to_date(
                        &fatbin_path,
                        &args_path,
                        &args_text,
                        &cu_path,
                        &kernels_dir,
                    ) {
                        continue;
                    }
                    let _ = std::fs::remove_file(&args_path);

                    let outcome = match Command::new(&nvcc).args(&args).output() {
                        Ok(output) => KernelOutcome {
                            file: kernel_file.clone(),
                            required,
                            arch: arch.clone(),
                            fatbin_path: fatbin_path.clone(),
                            success: output.status.success(),
                            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                            exec_error: None,
                        },
                        Err(e) => KernelOutcome {
                            file: kernel_file.clone(),
                            required,
                            arch: arch.clone(),
                            fatbin_path: fatbin_path.clone(),
                            success: false,
                            stdout: String::new(),
                            stderr: String::new(),
                            exec_error: Some(e.to_string()),
                        },
                    };
                    if outcome.success {
                        // Written after a successful compile, so a failed one
                        // never reads as up to date.
                        let _ = std::fs::write(&args_path, &args_text);
                    }
                    outcomes.lock().unwrap().push(outcome);
                }
            });
        }
    });

    // All workers joined: report failures collected, not interleaved.
    let outcomes = outcomes.into_inner().unwrap();
    let mut failed_required: Vec<String> = Vec::new();
    for outcome in outcomes.iter().filter(|o| !o.success) {
        if let Some(e) = &outcome.exec_error {
            eprintln!();
            eprintln!("=== CUDA COMPILATION ERROR ===");
            eprintln!();
            eprintln!(
                "Failed to execute nvcc for kernel '{}': {}",
                outcome.file, e
            );
            eprintln!("Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads");
            eprintln!();
            panic!("nvcc execution failed for {}: {}", outcome.file, e);
        }
        if outcome.required {
            eprintln!();
            eprintln!("=== CUDA COMPILATION FAILED ===");
            eprintln!("Failed to compile: {}", outcome.file);
            if !outcome.stdout.is_empty() {
                eprintln!("stdout: {}", outcome.stdout);
            }
            if !outcome.stderr.is_empty() {
                eprintln!("stderr: {}", outcome.stderr);
            }
            failed_required.push(outcome.file.clone());
        } else {
            eprintln!(
                "cargo:warning=Optional kernel {} ({}) failed to compile — \
                 skipping (requires {} hardware). stderr: {}",
                outcome.file,
                outcome.arch,
                outcome.arch.to_uppercase(),
                outcome.stderr.lines().next().unwrap_or("unknown error")
            );
            // A placeholder keeps the path present; loading it fails, as
            // loading the kernel on hardware without the feature would.
            std::fs::write(&outcome.fatbin_path, "// Optional kernel not compiled\n")
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to write placeholder fatbin for {}: {}",
                        outcome.file, e
                    )
                });
        }
    }
    if !failed_required.is_empty() {
        panic!(
            "nvcc compilation failed for: {}",
            failed_required.join(", ")
        );
    }

    println!("cargo:rustc-env=CUDA_KERNEL_DIR={}", out_dir.display());
}

#[cfg(feature = "cuda")]
/// Whether `fatbin` was built from the current sources with the current
/// nvcc arguments.
///
/// Newer than the kernel source, every `.cuh` under its directory and the
/// shared header directory, and this script; and the recorded arguments equal
/// `args_text`. A header a kernel does not include still forces a rebuild,
/// which costs one compile and never a stale kernel.
fn fatbin_up_to_date(
    fatbin: &std::path::Path,
    args_path: &std::path::Path,
    args_text: &str,
    cu_path: &std::path::Path,
    kernels_dir: &std::path::Path,
) -> bool {
    use std::path::{Path, PathBuf};
    let Ok(built) = std::fs::metadata(fatbin).and_then(|m| m.modified()) else {
        return false;
    };
    if std::fs::read_to_string(args_path).ok().as_deref() != Some(args_text) {
        return false;
    }
    let mut inputs: Vec<PathBuf> = vec![cu_path.to_path_buf(), PathBuf::from("build.rs")];
    for dir in [kernels_dir, Path::new("src/ops/cuda/kernels")] {
        collect_headers(dir, &mut inputs);
    }
    inputs.into_iter().all(|input| {
        std::fs::metadata(&input)
            .and_then(|m| m.modified())
            .is_ok_and(|modified| modified < built)
    })
}

/// Every `.cuh` under `dir`, recursively — the same set `emit_header_deps`
/// registers, so the two views of "what a kernel can include" agree.
#[cfg(feature = "cuda")]
fn collect_headers(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_headers(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "cuh") {
            out.push(path);
        }
    }
}

fn find_nvcc() -> Option<String> {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
        if nvcc.exists() {
            return Some(nvcc.to_string_lossy().to_string());
        }
    }

    let common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ];

    for path in common_paths {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }

    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some("nvcc".to_string());
    }

    None
}

/// Emit `cargo:rerun-if-changed` for every `.cuh` under `dir`, recursively.
///
/// A missing or unreadable directory is skipped: a build script must not abort
/// the build over a directory that holds no compilation unit of its own.
#[cfg(feature = "cuda")]
fn emit_header_deps(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            emit_header_deps(&path);
        } else if path.extension().is_some_and(|ext| ext == "cuh") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Real SASS targets for hardware people own: sm_75 (Turing), sm_80 (A100),
/// sm_86 (Ampere consumer), sm_89 (Ada), sm_90 (Hopper), sm_100 (Blackwell
/// datacenter), sm_120 (Blackwell consumer). Same list as numr's.
#[cfg(feature = "cuda")]
const REAL_ARCHES: &[&str] = &["75", "80", "86", "89", "90", "100", "120"];

/// The arches to emit SASS for, and a description for the build log.
///
/// Four modes: `BOOSTR_CUDA_ARCH` (or `NUMR_CUDA_ARCH`) set to a list builds
/// exactly those; set to `all`/`portable` builds every arch in `REAL_ARCHES`;
/// unset with a GPU detected builds the local arch(es); unset with no GPU
/// builds the portable set and warns, since that is CI or a container.
#[cfg(feature = "cuda")]
fn select_arches() -> (Vec<String>, String) {
    use std::env;
    let requested = env::var("BOOSTR_CUDA_ARCH")
        .ok()
        .or_else(|| env::var("NUMR_CUDA_ARCH").ok());
    match requested.as_deref() {
        Some(v)
            if v.trim().eq_ignore_ascii_case("all")
                || v.trim().eq_ignore_ascii_case("portable") =>
        {
            (
                REAL_ARCHES.iter().map(|a| a.to_string()).collect(),
                format!("all {} portable archs (requested {v})", REAL_ARCHES.len()),
            )
        }
        Some(v) => {
            let mut archs: Vec<String> = v.split(',').map(parse_arch).collect();
            archs.sort();
            archs.dedup();
            let desc = format!(
                "{} arch(es) requested ({v}): {}",
                archs.len(),
                archs.join(",")
            );
            (archs, desc)
        }
        None => match detect_local_gpu_arches() {
            Some(archs) if !archs.is_empty() => {
                let desc = format!(
                    "{} arch(es) detected locally: {}",
                    archs.len(),
                    archs.join(",")
                );
                (archs, desc)
            }
            _ => {
                println!(
                    "cargo:warning=boostr: no GPU detected (nvidia-smi missing, failed, or \
                     reported nothing) — building portable fatbins for all {} archs; set \
                     BOOSTR_CUDA_ARCH to the local arch(s) to skip this cost",
                    REAL_ARCHES.len()
                );
                (
                    REAL_ARCHES.iter().map(|a| a.to_string()).collect(),
                    format!("all {} portable archs (no GPU detected)", REAL_ARCHES.len()),
                )
            }
        },
    }
}

/// `-gencode` values for one kernel whose minimum arch is `min_arch`
/// (`"sm_75"`, `"sm_80"`, `"sm_90"`).
///
/// Real SASS for every selected arch at or above the minimum, then two PTX
/// entries: the kernel's own floor, so any arch at or above it with no cubin
/// here still loads by forward JIT, and compute_120 for hardware newer than
/// this toolkit's SASS targets. A kernel whose minimum exceeds every selected
/// arch gets PTX only, exactly what the old PTX build shipped for it.
#[cfg(feature = "cuda")]
fn gencode_flags(min_arch: &str, selected: &[String]) -> Vec<String> {
    let floor = parse_arch(min_arch);
    let floor_n: u32 = floor.parse().expect("arch digits validated");
    let mut flags: Vec<String> = selected
        .iter()
        .filter(|a| a.parse::<u32>().expect("arch digits validated") >= floor_n)
        .map(|a| format!("arch=compute_{a},code=sm_{a}"))
        .collect();
    flags.push(format!("arch=compute_{floor},code=compute_{floor}"));
    if floor_n < 120 {
        flags.push("arch=compute_120,code=compute_120".to_string());
    }
    flags
}

/// One arch entry to bare digits (`86`), accepting `86`, `8.6`, `sm_86` and
/// `compute_86`, validated to the supported range.
#[cfg(feature = "cuda")]
fn parse_arch(entry: &str) -> String {
    let v = entry.trim();
    let bare = v.strip_prefix("sm_").or_else(|| v.strip_prefix("compute_"));
    let digits = match bare {
        Some(digits) => digits.to_string(),
        None => v.replace('.', ""),
    };
    assert!(
        !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()),
        "CUDA arch entry must be a compute capability such as `86`, `8.6`, `sm_86`, \
         `compute_86`, or `all`/`portable` — got {v:?}"
    );
    validate_arch_range(&digits, v);
    digits
}

/// Below the toolkit's floor or above its ceiling is a config error, named
/// explicitly, never a silent drop from the build.
#[cfg(feature = "cuda")]
fn validate_arch_range(digits: &str, original: &str) {
    let n: u32 = digits.parse().expect("digits already validated numeric");
    assert!(
        (75..=120).contains(&n),
        "compute capability {original:?} (compute_{digits}) is outside the range this \
         build supports: compute_75 (Turing) to compute_120 (Blackwell consumer)"
    );
}

/// Compute capabilities of the local GPUs via `nvidia-smi` (a subprocess,
/// never the driver API). `None` on any detection failure, so the caller
/// falls back to the portable build: no GPU at build time is normal in CI.
#[cfg(feature = "cuda")]
fn detect_local_gpu_arches() -> Option<Vec<String>> {
    use std::process::Command;
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut arches: Vec<String> = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let digits: String = line.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            continue;
        }
        validate_arch_range(&digits, line);
        arches.push(digits);
    }
    if arches.is_empty() {
        return None;
    }
    arches.sort();
    arches.dedup();
    Some(arches)
}
