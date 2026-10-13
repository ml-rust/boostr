//! [`PointerHead`]: the weights and fitted temperature of the Kev pointer
//! readout, built from tensors or read from a Kev `head.pt`, and
//! [`PointerItem`], the gathered rows one question reads.
//!
//! A question's option logit is `z_j = (k(h_j) . q(h_d)) / sqrt(dp)`: `h_d`
//! is the hidden row of its decide token, `h_j` that of option `j`'s
//! closing token, and `q`, `k` are affine maps from the hidden size `d` to
//! the pointer dimension `dp`.

use crate::error::{Error, Result};
use crate::format::{PtScalar, TorchStateDict};
use crate::inference::decision::GatherIndexed;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// The `head.pt` key of the query weight `[dp, d]`.
pub const Q_WEIGHT: &str = "head.q.weight";
/// The `head.pt` key of the query bias `[dp]`.
pub const Q_BIAS: &str = "head.q.bias";
/// The `head.pt` key of the key weight `[dp, d]`.
pub const K_WEIGHT: &str = "head.k.weight";
/// The `head.pt` key of the key bias `[dp]`.
pub const K_BIAS: &str = "head.k.bias";

/// One question of a pointer readout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointerItem {
    /// The gathered row of the question's decide token: the index of its
    /// gather in plan order.
    pub decide: usize,
    /// The gathered row of each option's closing token, in option order.
    /// Non-empty, no repeats, none equal to `decide`.
    pub options: Vec<usize>,
}

/// Merging requests moves `decide` and every option past the earlier
/// requests' gathers.
impl GatherIndexed for PointerItem {
    fn offset_gathers(mut self, by: usize) -> Self {
        self.decide += by;
        for o in &mut self.options {
            *o += by;
        }
        self
    }
}

/// The Kev pointer readout: query and key maps in F32 and the temperature
/// fitted for the checkpoint.
///
/// The head returns raw logits. The temperature is applied by the runner:
/// pass `TemperaturePolicy::Fixed(head.temperature())` to answer as the
/// checkpoint was calibrated.
pub struct PointerHead<R: Runtime> {
    /// Query weight, transposed: `[d, dp]` F32.
    pub(super) q_weight_t: Tensor<R>,
    /// Query bias `[dp]` F32.
    pub(super) q_bias: Tensor<R>,
    /// Key weight, transposed: `[d, dp]` F32.
    pub(super) k_weight_t: Tensor<R>,
    /// Key bias `[dp]` F32.
    pub(super) k_bias: Tensor<R>,
    hidden_size: usize,
    dim: usize,
    temperature: f64,
}

pub(super) fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

fn file_error(path: &Path, reason: String) -> Error {
    Error::ModelError {
        reason: format!("{}: {reason}", path.display()),
    }
}

/// `t` is an F32 tensor of shape `want`.
fn check_f32<R: Runtime<DType = DType>>(
    arg: &'static str,
    t: &Tensor<R>,
    want: &[usize],
) -> Result<()> {
    if t.shape() != want {
        return Err(invalid(
            arg,
            format!("must be {want:?}, got {:?}", t.shape()),
        ));
    }
    if t.dtype() != DType::F32 {
        return Err(invalid(arg, format!("must be F32, got {:?}", t.dtype())));
    }
    Ok(())
}

impl<R: Runtime<DType = DType>> PointerHead<R> {
    /// A head from its weights `[dp, d]`, biases `[dp]`, all F32 on one
    /// device, and the temperature its logits are divided by at answer
    /// time.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `q_weight` is not 2-D with both
    /// sides at least 1, when another tensor's shape disagrees with it,
    /// when a tensor is not F32, or when `temperature` is not finite and
    /// above 0; each names the tensor and the shapes. Device errors of the
    /// transposes propagate.
    pub fn new(
        q_weight: Tensor<R>,
        q_bias: Tensor<R>,
        k_weight: Tensor<R>,
        k_bias: Tensor<R>,
        temperature: f64,
    ) -> Result<Self> {
        let &[dim, hidden_size] = q_weight.shape() else {
            return Err(invalid(
                "q_weight",
                format!("must be [dp, d], got {:?}", q_weight.shape()),
            ));
        };
        if dim == 0 || hidden_size == 0 {
            return Err(invalid(
                "q_weight",
                format!("must be [dp >= 1, d >= 1], got [{dim}, {hidden_size}]"),
            ));
        }
        check_f32("q_weight", &q_weight, &[dim, hidden_size])?;
        check_f32("q_bias", &q_bias, &[dim])?;
        check_f32("k_weight", &k_weight, &[dim, hidden_size])?;
        check_f32("k_bias", &k_bias, &[dim])?;
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(invalid(
                "temperature",
                format!("must be finite and above 0, got {temperature}"),
            ));
        }
        Ok(Self {
            q_weight_t: q_weight.transpose(0, 1)?.contiguous()?,
            q_bias,
            k_weight_t: k_weight.transpose(0, 1)?.contiguous()?,
            k_bias,
            hidden_size,
            dim,
            temperature,
        })
    }

    /// The head of a Kev checkpoint's `head.pt`, for a backbone of
    /// `hidden_size`, on `device`.
    ///
    /// Reads [`Q_WEIGHT`], [`Q_BIAS`], [`K_WEIGHT`] and [`K_BIAS`],
    /// `temperature` (1 when absent, as Kev reads it), and checks
    /// `head_dim` against the weights when present. A file whose
    /// `option_isolation` is set is refused: its option spans need a block
    /// mask over one packed sequence, which a run of plain rows does not
    /// apply.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] naming the file when it does not parse, when a
    /// key is missing, when a tensor's shape disagrees with
    /// `[dp, hidden_size]` or `[dp]` (naming the tensor and both shapes),
    /// when `head_dim` disagrees with `dp`, when a tensor is not F32, when
    /// `temperature` is not a number above 0, or when `option_isolation` is
    /// set or is not a bool.
    pub fn from_pt(path: impl AsRef<Path>, hidden_size: usize, device: &R::Device) -> Result<Self> {
        let path = path.as_ref();
        let sd = TorchStateDict::open(path)?;
        match sd.scalar("option_isolation") {
            None | Some(PtScalar::Bool(false)) | Some(PtScalar::Int(0)) => {}
            Some(PtScalar::Bool(true)) | Some(PtScalar::Int(_)) => {
                return Err(file_error(
                    path,
                    "option_isolation is set: the head was trained with option spans \
                     isolated by a block mask over one packed sequence, and a run of \
                     plain rows does not apply that mask"
                        .into(),
                ));
            }
            Some(other) => {
                return Err(file_error(
                    path,
                    format!("option_isolation must be a bool, got {other:?}"),
                ));
            }
        }
        let temperature = match sd.scalar("temperature") {
            None => 1.0,
            Some(s) => s.as_f64().ok_or_else(|| {
                file_error(path, format!("temperature must be a number, got {s:?}"))
            })?,
        };
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(file_error(
                path,
                format!("temperature must be finite and above 0, got {temperature}"),
            ));
        }

        let shape = |key: &str| {
            sd.shape(key)
                .map(<[usize]>::to_vec)
                .ok_or_else(|| file_error(path, format!("no tensor {key}")))
        };
        let q_shape = shape(Q_WEIGHT)?;
        let dim = match q_shape.as_slice() {
            &[dim, d] if d == hidden_size && dim >= 1 => dim,
            other => {
                return Err(file_error(
                    path,
                    format!(
                        "{Q_WEIGHT} is {other:?} but the backbone's hidden size {hidden_size} \
                         needs [dp, {hidden_size}]"
                    ),
                ));
            }
        };
        for (key, want) in [
            (Q_BIAS, vec![dim]),
            (K_WEIGHT, vec![dim, hidden_size]),
            (K_BIAS, vec![dim]),
        ] {
            let got = shape(key)?;
            if got != want {
                return Err(file_error(
                    path,
                    format!("{key} is {got:?}, expected {want:?} beside {Q_WEIGHT} {q_shape:?}"),
                ));
            }
        }
        if let Some(s) = sd.scalar("head_dim")
            && s != &PtScalar::Int(dim as i64)
        {
            return Err(file_error(
                path,
                format!("head_dim is {s:?} but {Q_WEIGHT} is {q_shape:?}"),
            ));
        }

        let load = |key: &str| -> Result<Tensor<R>> {
            let t = sd.load_tensor::<R>(key, device)?;
            if t.dtype() != DType::F32 {
                return Err(file_error(
                    path,
                    format!("{key} is {:?}; a pointer head holds F32", t.dtype()),
                ));
            }
            Ok(t)
        };
        Self::new(
            load(Q_WEIGHT)?,
            load(Q_BIAS)?,
            load(K_WEIGHT)?,
            load(K_BIAS)?,
            temperature,
        )
    }

    /// The backbone hidden size `d` the head reads.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// The pointer dimension `dp`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The logit scale `1 / sqrt(dp)`.
    pub fn scale(&self) -> f64 {
        1.0 / (self.dim as f64).sqrt()
    }

    /// The temperature fitted for the checkpoint: the runner divides the
    /// raw logits by it when given `TemperaturePolicy::Fixed` of it.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use std::path::PathBuf;

    const D: usize = 8;
    const DP: usize = 4;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/kev_head")
            .join(name)
    }

    /// The F32 weights `[dp, d]` row-major and biases `[dp]` held by all
    /// three `tests/fixtures/kev_head` files. `tiny.pt` sets `temperature`
    /// 1.75, `tiny_untempered.pt` sets none, `tiny_isolated.pt` sets
    /// `option_isolation`.
    fn tiny_weights() -> [Vec<f32>; 4] {
        let qw = (0..DP * D)
            .map(|n| (((8 * (n / D) + n % D) % 11) as f32 - 5.0) / 16.0)
            .collect();
        let qb = (0..DP).map(|i| (i as f32 - 1.5) / 4.0).collect();
        let kw = (0..DP * D)
            .map(|n| (((5 * (n / D) + 3 * (n % D)) % 13) as f32 - 6.0) / 32.0)
            .collect();
        let kb = (0..DP).map(|i| (2.0 - i as f32) / 8.0).collect();
        [qw, qb, kw, kb]
    }

    /// `w` `[dp, d]` row-major, transposed to `[d, dp]`.
    fn transposed(w: &[f32], dp: usize, d: usize) -> Vec<f32> {
        (0..d * dp).map(|n| w[(n % dp) * d + n / dp]).collect()
    }

    fn from_pt_err(name: &str, hidden: usize) -> String {
        let device = CpuDevice::new();
        match PointerHead::<CpuRuntime>::from_pt(fixture(name), hidden, &device) {
            Ok(_) => panic!("{name} with hidden size {hidden} must be refused"),
            Err(e) => e.to_string(),
        }
    }

    /// A CPU head whose tensors read `values(i + offset)`, offsets 0,
    /// 1000, 2000, 3000 for `q_weight`, `q_bias`, `k_weight`, `k_bias`.
    pub(crate) fn built(
        dp: usize,
        d: usize,
        temperature: f64,
        values: impl Fn(usize) -> f32,
    ) -> PointerHead<CpuRuntime> {
        let device = CpuDevice::new();
        let t = |n: usize, off: usize, shape: &[usize]| {
            let v: Vec<f32> = (0..n).map(|i| values(i + off)).collect();
            Tensor::<CpuRuntime>::from_slice(&v, shape, &device).unwrap()
        };
        PointerHead::new(
            t(dp * d, 0, &[dp, d]),
            t(dp, 1000, &[dp]),
            t(dp * d, 2000, &[dp, d]),
            t(dp, 3000, &[dp]),
            temperature,
        )
        .unwrap()
    }

    #[test]
    fn reads_a_kev_head_pt() {
        let device = CpuDevice::new();
        let head = PointerHead::<CpuRuntime>::from_pt(fixture("tiny.pt"), D, &device).unwrap();
        assert_eq!(head.hidden_size(), D);
        assert_eq!(head.dim(), DP);
        assert_eq!(head.temperature(), 1.75);
        assert_eq!(head.scale(), 0.5);
        let [qw, qb, kw, kb] = tiny_weights();
        assert_eq!(head.q_weight_t.shape(), &[D, DP]);
        assert_eq!(head.q_weight_t.to_vec::<f32>(), transposed(&qw, DP, D));
        assert_eq!(head.q_bias.to_vec::<f32>(), qb);
        assert_eq!(head.k_weight_t.to_vec::<f32>(), transposed(&kw, DP, D));
        assert_eq!(head.k_bias.to_vec::<f32>(), kb);

        let plain =
            PointerHead::<CpuRuntime>::from_pt(fixture("tiny_untempered.pt"), D, &device).unwrap();
        assert_eq!(plain.temperature(), 1.0);
    }

    #[test]
    fn bad_head_pt_is_refused_with_its_values() {
        let e = from_pt_err("tiny_isolated.pt", D);
        assert!(
            e.contains("tiny_isolated.pt: option_isolation is set"),
            "{e}"
        );
        let e = from_pt_err("tiny.pt", 6);
        assert!(
            e.contains("head.q.weight is [4, 8] but the backbone's hidden size 6 needs [dp, 6]"),
            "{e}"
        );
        let e = from_pt_err("no_such_head.pt", D);
        assert!(e.contains("no_such_head.pt"), "{e}");
    }

    #[test]
    fn new_refuses_bad_shapes_dtypes_and_temperatures() {
        let device = CpuDevice::new();
        let f = |shape: &[usize]| Tensor::<CpuRuntime>::zeros(shape, DType::F32, &device).unwrap();
        let err =
            |q: &[usize], qb: &[usize], k: &[usize], kb: &[usize], t: f64| match PointerHead::new(
                f(q),
                f(qb),
                f(k),
                f(kb),
                t,
            ) {
                Ok(_) => panic!("{q:?} {qb:?} {k:?} {kb:?} at {t} must be refused"),
                Err(e) => e.to_string(),
            };
        let e = err(&[4, 8], &[4], &[4, 9], &[4], 1.0);
        assert!(e.contains("'k_weight': must be [4, 8], got [4, 9]"), "{e}");
        let e = err(&[4, 8], &[3], &[4, 8], &[4], 1.0);
        assert!(e.contains("'q_bias': must be [4], got [3]"), "{e}");
        let e = err(&[8], &[4], &[4, 8], &[4], 1.0);
        assert!(e.contains("'q_weight': must be [dp, d], got [8]"), "{e}");
        let e = err(&[0, 8], &[0], &[0, 8], &[0], 1.0);
        assert!(e.contains("must be [dp >= 1, d >= 1], got [0, 8]"), "{e}");
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let e = err(&[4, 8], &[4], &[4, 8], &[4], t);
            assert!(
                e.contains("'temperature': must be finite and above 0"),
                "{e}"
            );
        }
        let wide = Tensor::<CpuRuntime>::zeros(&[4, 8], DType::F64, &device).unwrap();
        let Err(e) = PointerHead::new(f(&[4, 8]), f(&[4]), wide, f(&[4]), 1.0) else {
            panic!("an F64 key weight must be refused");
        };
        assert!(
            e.to_string().contains("'k_weight': must be F32, got F64"),
            "{e}"
        );
    }

    #[test]
    fn merging_moves_every_gather() {
        let item = PointerItem {
            decide: 3,
            options: vec![0, 1, 2],
        };
        assert_eq!(
            item.offset_gathers(5),
            PointerItem {
                decide: 8,
                options: vec![5, 6, 7],
            }
        );
    }
}
