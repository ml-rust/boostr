//! TCF vs GGUF throughput at matched size classes.
//!
//! CONFORMANCE.md Section 8.4 gates a v1 encoding on winning its size class on
//! effective cost. This target produces the cost half of that gate: how fast
//! each encoding dequantizes and how fast it runs a fused quantized matmul, on
//! CPU, CUDA and WebGPU.
//!
//! # Metric, and why it is not wall-clock
//!
//! The primary metric is RETIRED INSTRUCTIONS PER ITERATION, measured by
//! `perf stat -e instructions:u` over a child process that runs one case. A
//! second child runs the same case with zero iterations, so subtracting the two
//! removes setup, process start, and payload load exactly. Instruction counts
//! are deterministic; wall-clock on a loaded machine is not, and this machine is
//! frequently loaded.
//!
//! Allocation count and allocation bytes per iteration are measured in-process
//! by a counting global allocator. They are deterministic too.
//!
//! Elapsed nanoseconds and reference cycles are recorded as the MINIMUM over
//! iterations and reported in dimmed columns. They are advisory. The report
//! header states the machine's load average at run time, and every wall-clock
//! column is labelled load-sensitive. A CUDA or WebGPU row has no other choice:
//! host instructions there count kernel LAUNCH work, not kernel work, so those
//! rows must be read as device time and the machine must be quiet.
//!
//! # Harness choice
//!
//! `numr` benchmarks with `fluxbench`, and `boostr` had no benchmark target at
//! all. This target keeps `numr`'s convention — a `[[bench]]` target with
//! `harness = false` — and adds no dependency, because `fluxbench` reports
//! time, cycles and allocations but not retired instructions, and instructions
//! are the metric this project's rules mandate. The cycle and allocation
//! columns here are exactly the metrics `fluxbench` would have contributed.
//!
//! # Running
//!
//! ```text
//! cargo bench --bench quant_throughput -- --list
//! cargo bench --bench quant_throughput
//! cargo bench --features cuda --bench quant_throughput -- --backend cuda
//! ```

mod alloc_counter;
mod cases;
mod measure;
mod payload;
mod report;
mod worker;

use std::process::ExitCode;

#[global_allocator]
static ALLOCATOR: alloc_counter::CountingAllocator = alloc_counter::CountingAllocator;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match measure::dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("quant_throughput: {message}");
            ExitCode::FAILURE
        }
    }
}
