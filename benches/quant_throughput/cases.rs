//! The case matrix: three GGUF size classes, VoxCPM2 projection shapes, and
//! the two operations that dominate quantized inference.
//!
//! Enumeration is deterministic and depends only on which backends are compiled
//! in, so the parent process and a worker child agree on what index `n` names
//! without passing the case description over the command line.

use boostr::quant::QuantFormat;

/// A GGUF block format and the file size it spends.
///
/// Classes are keyed by BITS PER WEIGHT, not by bit width: a new encoding
/// competing in a class is compared at equal file size, or the comparison
/// measures the size and not the kernel.
pub struct SizeClass {
    /// Bits per weight: block bytes times 8 over block elements.
    pub bpw: f64,
    pub format: QuantFormat,
}

/// The three size classes a shipped model is quantized to.
pub static CLASSES: [SizeClass; 3] = [
    SizeClass {
        bpw: 4.5,
        format: QuantFormat::Q4K,
    },
    SizeClass {
        bpw: 6.5625,
        format: QuantFormat::Q6K,
    },
    SizeClass {
        bpw: 8.5,
        format: QuantFormat::Q8_0,
    },
];

/// A `[N, K]` weight shape, named after the projection it comes from.
pub struct WeightShape {
    pub label: &'static str,
    pub n: usize,
    pub k: usize,
}

/// VoxCPM2's `base_lm` (MiniCPM4): hidden 2048, FFN 6144, 16 heads of 128, 2 KV
/// heads. These are the real projection widths, not round numbers, so K is a
/// multiple of the GGUF 256-element super-block without any padding fiction.
pub static SHAPES: [WeightShape; 4] = [
    WeightShape {
        label: "q_proj",
        n: 2048,
        k: 2048,
    },
    WeightShape {
        label: "kv_proj",
        n: 256,
        k: 2048,
    },
    WeightShape {
        label: "gate_up",
        n: 6144,
        k: 2048,
    },
    WeightShape {
        label: "down_proj",
        n: 2048,
        k: 6144,
    },
];

/// Shapes dequantization is measured on. One square and one wide, which is
/// enough: dequantization cost is linear in elements and has no M dimension.
const DEQUANT_SHAPES: [&str; 2] = ["q_proj", "down_proj"];

/// Decode. `M = 1` is the GEMV case and is memory-bound on every backend.
const DECODE_M: [usize; 1] = [1];

/// Prefill batch sizes. 2 sits at or below every CUDA GEMV crossover (1 for
/// Q3_K/Q2_K, 2 for Q4_K/Q5_K, 4 for every other format), pinning the
/// small-batch side. 4 and 8 bracket the GEMV/MMQ crossover, so a kernel
/// change that moves it shows up here rather than silently costing
/// small-batch prefill. 32 and 256 are the continuous-batching and
/// full-prefill points; both exceed every crossover on CUDA and land on the
/// MMQ path, so that path stays covered too.
const PREFILL_M: [usize; 5] = [2, 4, 8, 32, 256];

/// Shapes the prefill sizes run on. Restricted to two, because a `M = 256`
/// GEMM does 256 times the arithmetic a GEMV does, which is minutes of work
/// per extra shape.
const PREFILL_SHAPES: [&str; 2] = ["q_proj", "down_proj"];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Wgpu,
}

impl Backend {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
        }
    }

    /// Backends this build can execute. A backend absent here is absent from
    /// the matrix, so its indices never shift the rest.
    pub fn compiled() -> Vec<Self> {
        // `mut` is unused when neither GPU feature is on.
        #[allow(unused_mut)]
        let mut backends = vec![Self::Cpu];
        #[cfg(feature = "cuda")]
        backends.push(Self::Cuda);
        #[cfg(feature = "wgpu")]
        backends.push(Self::Wgpu);
        backends
    }
}

#[derive(Clone, Copy)]
pub enum Op {
    /// Whole-tensor dequantization to F32.
    Dequant,
    /// Fused quantized matmul, `[M, K] x [N, K]^T`.
    Matmul { m: usize },
}

impl Op {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dequant => "dequant",
            Self::Matmul { m: 1 } => "gemv",
            Self::Matmul { .. } => "gemm",
        }
    }

    pub const fn m(self) -> usize {
        match self {
            Self::Dequant => 0,
            Self::Matmul { m } => m,
        }
    }
}

/// One measurable point: a size class, on a backend, running an operation at
/// a shape.
pub struct Case {
    pub backend: Backend,
    pub class: usize,
    pub shape: usize,
    pub op: Op,
}

impl Case {
    fn size_class(&self) -> &'static SizeClass {
        // `class` only ever comes from `enumerate`, which indexes `CLASSES`.
        &CLASSES[self.class % CLASSES.len()]
    }

    fn weight_shape(&self) -> &'static WeightShape {
        &SHAPES[self.shape % SHAPES.len()]
    }

    pub fn n(&self) -> usize {
        self.weight_shape().n
    }

    pub fn k(&self) -> usize {
        self.weight_shape().k
    }

    pub fn shape_label(&self) -> &'static str {
        self.weight_shape().label
    }

    pub fn bpw(&self) -> f64 {
        self.size_class().bpw
    }

    /// The block format this case's weight is packed in.
    pub fn format(&self) -> QuantFormat {
        self.size_class().format
    }

    pub fn encoding_name(&self) -> &'static str {
        self.format().name()
    }

    /// The work one iteration does, and the unit it is counted in.
    ///
    /// Dequantization is charged per output element. A matmul is charged per
    /// multiply-accumulate, so a `M = 1` row and a `M = 256` row normalize onto
    /// the same scale.
    pub fn work_units(&self) -> (u64, &'static str) {
        let n = self.n() as u64;
        let k = self.k() as u64;
        match self.op {
            Op::Dequant => (n * k, "elem"),
            Op::Matmul { m } => (m as u64 * n * k, "mac"),
        }
    }

    /// Stable human-readable identifier, also the `--filter` match target.
    pub fn id(&self) -> String {
        match self.op {
            Op::Dequant => format!(
                "{}/{}/{}/{}",
                self.backend.label(),
                self.op.label(),
                self.encoding_name(),
                self.shape_label(),
            ),
            Op::Matmul { m } => format!(
                "{}/{}/{}/{}/m{m}",
                self.backend.label(),
                self.op.label(),
                self.encoding_name(),
                self.shape_label(),
            ),
        }
    }
}

/// Every case this build can run, in a fixed order.
pub fn enumerate() -> Vec<Case> {
    let mut out = Vec::new();
    for backend in Backend::compiled() {
        for class in 0..CLASSES.len() {
            for (shape, weight) in SHAPES.iter().enumerate() {
                if DEQUANT_SHAPES.contains(&weight.label) {
                    out.push(Case {
                        backend,
                        class,
                        shape,
                        op: Op::Dequant,
                    });
                }
                let ms: Vec<usize> = if PREFILL_SHAPES.contains(&weight.label) {
                    DECODE_M.iter().chain(PREFILL_M.iter()).copied().collect()
                } else {
                    DECODE_M.to_vec()
                };
                for &m in &ms {
                    out.push(Case {
                        backend,
                        class,
                        shape,
                        op: Op::Matmul { m },
                    });
                }
            }
        }
    }
    out
}
