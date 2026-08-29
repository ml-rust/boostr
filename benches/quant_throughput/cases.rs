//! The case matrix: matched TCF/GGUF pairs, VoxCPM2 projection shapes, and the
//! two operations that dominate quantized inference.
//!
//! Enumeration is deterministic and depends only on which backends are compiled
//! in, so the parent process and a worker child agree on what index `n` names
//! without passing the case description over the command line.

use boostr::quant::{QuantFormat, QuantScheme, TcfEncoding};
use tcf_core::NativeEncoding;

/// A TCF encoding and the GGUF format it must be compared against.
///
/// The pairing is by BITS PER WEIGHT, not by bit width: `Q4AS32D_T64` and
/// `Q4_K` both spend 4.50 bpw, so a win on either side is a win at equal file
/// size. Comparing `Q4S32_T64` (4.00 bpw) against `Q4_K` would not be.
pub struct Pair {
    /// Bits per weight, both sides. Section 12.2 for TCF, block bytes for GGUF.
    pub bpw: f64,
    pub tcf: NativeEncoding,
    pub gguf: QuantFormat,
}

/// The three matched size classes. Section 8.4's target classes.
pub static PAIRS: [Pair; 3] = [
    Pair {
        bpw: 4.5,
        tcf: NativeEncoding::Q4AS32DT64,
        gguf: QuantFormat::Q4K,
    },
    Pair {
        bpw: 6.5625,
        tcf: NativeEncoding::Q6S16DT64,
        gguf: QuantFormat::Q6K,
    },
    Pair {
        bpw: 8.5,
        tcf: NativeEncoding::Q8S32T64,
        gguf: QuantFormat::Q8_0,
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
/// multiple of both the GGUF 256-element super-block and the TCF 64-element
/// tile without any padding fiction.
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

/// Prefill batch sizes. 32 stays inside the CUDA GEMV path's `M <= 64` window;
/// 256 crosses into the tiled GEMM path, so both kernels are covered.
const PREFILL_M: [usize; 2] = [32, 256];

/// Shapes the prefill sizes run on. Restricted to two, because a `M = 256`
/// GEMM does 256 times the arithmetic a GEMV does, which is minutes of work
/// per extra shape.
const PREFILL_SHAPES: [&str; 2] = ["q_proj", "down_proj"];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Tcf,
    Gguf,
}

impl Codec {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tcf => "tcf",
            Self::Gguf => "gguf",
        }
    }
}

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

/// One measurable point: an encoding, on a backend, running an operation at a
/// shape.
pub struct Case {
    pub backend: Backend,
    pub codec: Codec,
    pub pair: usize,
    pub shape: usize,
    pub op: Op,
}

impl Case {
    fn pair(&self) -> &'static Pair {
        // `pair` only ever comes from `enumerate`, which indexes `PAIRS`.
        &PAIRS[self.pair % PAIRS.len()]
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
        self.pair().bpw
    }

    /// The payload layout this case's weight is packed in.
    pub fn scheme(&self) -> QuantScheme {
        match self.codec {
            Codec::Tcf => QuantScheme::Tcf(TcfEncoding::new(self.pair().tcf)),
            Codec::Gguf => QuantScheme::Gguf(self.pair().gguf),
        }
    }

    pub fn encoding_name(&self) -> String {
        self.scheme().name()
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
        for pair in 0..PAIRS.len() {
            for codec in [Codec::Tcf, Codec::Gguf] {
                for (shape, weight) in SHAPES.iter().enumerate() {
                    if DEQUANT_SHAPES.contains(&weight.label) {
                        out.push(Case {
                            backend,
                            codec,
                            pair,
                            shape,
                            op: Op::Dequant,
                        });
                    }
                    let ms: &[usize] = if PREFILL_SHAPES.contains(&weight.label) {
                        &[DECODE_M[0], PREFILL_M[0], PREFILL_M[1]]
                    } else {
                        &DECODE_M
                    };
                    for &m in ms {
                        out.push(Case {
                            backend,
                            codec,
                            pair,
                            shape,
                            op: Op::Matmul { m },
                        });
                    }
                }
            }
        }
    }
    out
}
