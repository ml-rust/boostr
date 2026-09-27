//! Shared CUDA fatbin embedding lookup and per-device module cache, used by
//! both `ops/cuda/kernels/loader.rs` and `quant/cuda/kernels/loader.rs`.

mod module_cache;

#[cfg(test)]
pub(crate) use module_cache::embedded_fatbin;
pub(crate) use module_cache::{get_kernel_function, get_or_load_module};
