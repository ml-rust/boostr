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
    let kernels_dir = PathBuf::from("src/quant/cuda/kernels");

    let kernel_files = vec!["dequant.cu", "quant_matmul.cu"];

    let nvcc = find_nvcc().unwrap_or_else(|| {
        eprintln!();
        eprintln!("=== CUDA COMPILATION ERROR ===");
        eprintln!();
        eprintln!("Could not find nvcc (NVIDIA CUDA Compiler).");
        eprintln!("Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads");
        eprintln!();
        panic!("nvcc not found - CUDA Toolkit must be installed for the 'cuda' feature");
    });

    for kernel_file in kernel_files {
        let cu_path = kernels_dir.join(kernel_file);
        let ptx_name = kernel_file.replace(".cu", ".ptx");
        let ptx_path = out_dir.join(&ptx_name);

        println!("cargo:rerun-if-changed={}", cu_path.display());

        if !cu_path.exists() {
            panic!(
                "CUDA kernel source not found: {}\n\
                 Ensure kernel files exist in src/quant/cuda/kernels/",
                cu_path.display()
            );
        }

        let output = Command::new(&nvcc)
            .args([
                "-ptx",
                "-O3",
                "--use_fast_math",
                "-arch=sm_75",
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
                }
            }
            Err(e) => {
                panic!("Failed to execute nvcc: {}", e);
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
