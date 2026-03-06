//! Build script for boostr
//!
//! Compiles CUDA kernels to PTX when the cuda feature is enabled.
//! Follows the same pattern as numr's build.rs.

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

    // Kernel sets: (directory, filename, arch, required, ptx_name_override)
    // Most kernels target sm_75 (Turing+); flash_v3 needs sm_90 (Hopper)
    // Optional kernels (required=false) warn on failure instead of panicking —
    // they require hardware-specific features (e.g. Hopper) that may not be
    // available on all build machines.
    // ptx_name_override: Some("name.ptx") overrides the default (filename with .ptx ext).
    // Needed when files in different subdirs share the same filename (e.g. gemv/q5_k.cu vs gemm/q5_k.cu).
    let kernel_sets: Vec<(PathBuf, &str, &str, bool, Option<&str>)> = vec![
        // Quantization kernels
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "dequant.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "dequant_generic.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "quant_matmul_generic.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "quant_matmul.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "quant_gemv.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "int4_gemm.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "int4_gemm_gptq.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "nf4_quant.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "marlin_gemm.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "fused_int4_swiglu.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "fused_int4_qkv.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/quant/cuda/kernels"),
            "quant_act.cu",
            "sm_75",
            true,
            None,
        ),
        // Per-format GEMV kernels (subdirectory)
        (
            PathBuf::from("src/quant/cuda/kernels/gemv"),
            "q5_k.cu",
            "sm_75",
            true,
            Some("gemv_q5_k.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemv"),
            "q3_k.cu",
            "sm_75",
            true,
            Some("gemv_q3_k.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemv"),
            "q2_k.cu",
            "sm_75",
            true,
            Some("gemv_q2_k.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemv"),
            "q5_0.cu",
            "sm_75",
            true,
            Some("gemv_q5_0.ptx"),
        ),
        // Per-format IQ GEMV kernels (subdirectory)
        (
            PathBuf::from("src/quant/cuda/kernels/gemv"),
            "iq4_nl.cu",
            "sm_75",
            true,
            Some("gemv_iq4_nl.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemv"),
            "iq4_xs.cu",
            "sm_75",
            true,
            Some("gemv_iq4_xs.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemv"),
            "iq3_s.cu",
            "sm_75",
            true,
            Some("gemv_iq3_s.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemv"),
            "iq2_xs.cu",
            "sm_75",
            true,
            Some("gemv_iq2_xs.ptx"),
        ),
        // Per-format GEMM kernels (subdirectory)
        (
            PathBuf::from("src/quant/cuda/kernels/gemm"),
            "q5_k.cu",
            "sm_75",
            true,
            Some("gemm_q5_k.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemm"),
            "q3_k.cu",
            "sm_75",
            true,
            Some("gemm_q3_k.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemm"),
            "q2_k.cu",
            "sm_75",
            true,
            Some("gemm_q2_k.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemm"),
            "q5_0.cu",
            "sm_75",
            true,
            Some("gemm_q5_0.ptx"),
        ),
        // Per-format IQ GEMM kernels (subdirectory)
        (
            PathBuf::from("src/quant/cuda/kernels/gemm"),
            "iq4_nl.cu",
            "sm_75",
            true,
            Some("gemm_iq4_nl.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemm"),
            "iq4_xs.cu",
            "sm_75",
            true,
            Some("gemm_iq4_xs.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemm"),
            "iq3_s.cu",
            "sm_75",
            true,
            Some("gemm_iq3_s.ptx"),
        ),
        (
            PathBuf::from("src/quant/cuda/kernels/gemm"),
            "iq2_xs.cu",
            "sm_75",
            true,
            Some("gemm_iq2_xs.ptx"),
        ),
        // Attention kernels
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "flash_v2.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "flash_v2_bwd.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "paged_attention.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "paged_attention_bwd.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "varlen_attention.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "varlen_attention_bwd.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "mqa_gqa.cu",
            "sm_80",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "mqa_gqa_bwd.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "sdpa.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "fused_qkv.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "decode_attention.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "kv_insert.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "paged_decode_attention.cu",
            "sm_75",
            true,
            None,
        ),
        // Flash v3 — sm_90 (Hopper warp specialization, optional)
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "flash_v3.cu",
            "sm_90",
            false,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/attention"),
            "flash_v3_bwd.cu",
            "sm_90",
            false,
            None,
        ),
        // Cache kernels
        (
            PathBuf::from("src/ops/cuda/kernels/cache"),
            "kv_cache_update.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/cache"),
            "kv_cache_int4.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/cache"),
            "kv_cache_fp8.cu",
            "sm_80",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/cache"),
            "kv_cache_fp8_bwd.cu",
            "sm_80",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/cache"),
            "kv_cache_quant.cu",
            "sm_80",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/cache"),
            "reshape_and_cache.cu",
            "sm_75",
            true,
            None,
        ),
        // Position kernels
        (
            PathBuf::from("src/ops/cuda/kernels/position"),
            "alibi.cu",
            "sm_80",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/position"),
            "alibi_bwd.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/position"),
            "rope.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/position"),
            "rope_interleaved.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/position"),
            "rope_yarn.cu",
            "sm_75",
            true,
            None,
        ),
        // Fused optimizer kernels
        (
            PathBuf::from("src/ops/cuda/kernels/training"),
            "fused_adamw.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/training"),
            "fused_sgd.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/training"),
            "fused_adagrad.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/training"),
            "fused_lamb.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/training"),
            "fused_multi_tensor.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/training"),
            "fused_grad_unscale_clip.cu",
            "sm_75",
            true,
            None,
        ),
        // Architecture kernels (MoE)
        (
            PathBuf::from("src/ops/cuda/kernels/architecture"),
            "moe_routing.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/architecture"),
            "moe_grouped_gemm.cu",
            "sm_75",
            true,
            None,
        ),
        // Inference kernels (speculative decoding, sampling, prefix cache)
        (
            PathBuf::from("src/ops/cuda/kernels/inference"),
            "prefix_cache_lookup.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/inference"),
            "speculative_verify.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/inference"),
            "sampling_penalties.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/inference"),
            "sampling.cu",
            "sm_75",
            true,
            None,
        ),
        (
            PathBuf::from("src/ops/cuda/kernels/inference"),
            "logits_to_token.cu",
            "sm_75",
            true,
            None,
        ),
        // Architecture kernels (SSM / Mamba2)
        (
            PathBuf::from("src/ops/cuda/kernels/architecture"),
            "ssd_state_passing.cu",
            "sm_75",
            true,
            None,
        ),
        // Calibration kernels (quantization)
        (
            PathBuf::from("src/ops/cuda/kernels/quantization"),
            "calibration.cu",
            "sm_75",
            true,
            None,
        ),
    ];

    let nvcc = find_nvcc().unwrap_or_else(|| {
        eprintln!();
        eprintln!("=== CUDA COMPILATION ERROR ===");
        eprintln!();
        eprintln!("Could not find nvcc (NVIDIA CUDA Compiler).");
        eprintln!("Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads");
        eprintln!();
        panic!("nvcc not found - CUDA Toolkit must be installed for the 'cuda' feature");
    });

    for (kernels_dir, kernel_file, arch, required, ptx_override) in &kernel_sets {
        let cu_path = kernels_dir.join(kernel_file);
        let ptx_name = ptx_override
            .map(|s| s.to_string())
            .unwrap_or_else(|| kernel_file.replace(".cu", ".ptx"));
        let ptx_path = out_dir.join(&ptx_name);

        println!("cargo:rerun-if-changed={}", cu_path.display());

        if !cu_path.exists() {
            panic!(
                "CUDA kernel source not found: {}\n\
                 Ensure kernel files exist in {}",
                cu_path.display(),
                kernels_dir.display()
            );
        }

        // Include paths: kernel's own dir + root kernels dir for shared headers (dtype_traits.cuh)
        let include_arg = format!("-I{}", kernels_dir.display());
        let root_include_arg = "-Isrc/ops/cuda/kernels".to_string();
        let arch_arg = format!("-arch={}", arch);

        let output = Command::new(&nvcc)
            .args([
                "-ptx",
                "-O3",
                "--use_fast_math",
                &arch_arg,
                &include_arg,
                &root_include_arg,
                "-o",
                ptx_path.to_str().unwrap(),
                cu_path.to_str().unwrap(),
            ])
            .output();

        match output {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if *required {
                        eprintln!();
                        eprintln!("=== CUDA COMPILATION FAILED ===");
                        eprintln!("Failed to compile: {}", kernel_file);
                        if !stdout.is_empty() {
                            eprintln!("stdout: {}", stdout);
                        }
                        if !stderr.is_empty() {
                            eprintln!("stderr: {}", stderr);
                        }
                        panic!("nvcc compilation failed for {}", kernel_file);
                    } else {
                        eprintln!(
                            "cargo:warning=Optional kernel {} ({}) failed to compile — \
                             skipping (requires {} hardware). stderr: {}",
                            kernel_file,
                            arch,
                            arch.to_uppercase(),
                            stderr.lines().next().unwrap_or("unknown error")
                        );
                        // Write an empty PTX file so include_str! doesn't fail
                        std::fs::write(&ptx_path, "// Optional kernel not compiled\n")
                            .unwrap_or_else(|e| {
                                panic!("Failed to write placeholder PTX for {}: {}", kernel_file, e)
                            });
                    }
                }
            }
            Err(e) => {
                eprintln!();
                eprintln!("=== CUDA COMPILATION ERROR ===");
                eprintln!();
                eprintln!("Failed to execute nvcc for kernel '{}': {}", kernel_file, e);
                eprintln!("Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads");
                eprintln!();
                panic!("nvcc execution failed for {}: {}", kernel_file, e);
            }
        }
    }

    println!("cargo:rustc-env=CUDA_KERNEL_DIR={}", out_dir.display());
}

#[cfg(feature = "cuda")]
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
