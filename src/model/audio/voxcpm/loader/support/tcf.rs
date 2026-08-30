//! [`TcfSource`]: a `.tcf` file presented as a [`WeightSource`].
//!
//! # Why a TCF needs no new format knowledge here
//!
//! Everything this module does is decided by the file's directory:
//! [`TcfLoader`] resolves it, [`crate::format::tcf::decode`] turns a payload
//! into f32 values, and `QuantTensor::from_bytes` places a packed payload on
//! a device unchanged. Plane offsets, bit positions and the Section 14.2
//! sub-plane split live in `tcf-core` and in the kernels, and
//! MIGRATION.md Section 4.5.3 forbids a second copy of them. There is none
//! below — this file only chooses BETWEEN those two paths.
//!
//! # The choice, per tensor
//!
//! Section 12 splits encodings into native quantized ones, which carry
//! codes plus scales, and raw ones, which carry literal values and no scale
//! at all. A native encoding therefore has a packed form worth keeping and
//! becomes [`Weight::Quantized`]; a raw one has none and becomes
//! [`Weight::Standard`].
//!
//! BOTH occur in a real file, and by a wide margin. A VoxCPM2 TCF written by
//! compressr holds 577 tensors of which 139 sit at a BF16 FALLBACK encoding
//! with reason `RankLt2` — Section 8.6 records why: a rank-1 tensor cannot be
//! tiled, so no native encoding applies to it. Assuming every tensor is
//! quantized would fail on every one of those 139.
//!
//! # Names repeat, and that is not an error in the FILE
//!
//! SPECIFICATION.md Section 6 makes a tensor name PROVENANCE, never
//! identity: a conforming file MAY carry the same name twice, and
//! [`TcfLoader::tensor_info`] resolves such a name to its FIRST occurrence.
//! A `WeightSource` looks tensors up by name and by nothing else, so on such
//! a file it would silently load one weight where another was meant.
//! [`TcfSource::new`] refuses the file instead, naming the collision.

use std::collections::HashSet;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use tcf_core::Encoding;

use super::weight_source::WeightSource;
use crate::error::{Error, Result};
use crate::format::tcf::{TcfLoader, TcfSession};
use crate::nn::Weight;

/// A `.tcf` file presented as a [`WeightSource`], with its directory bound
/// once for the whole load.
///
/// # Why it borrows the loader
///
/// A one-shot [`TcfLoader::load_tensor`] reparses and revalidates the entire
/// directory, which is O(directory) per tensor and quadratic over a whole
/// model. This holds a [`TcfSession`] instead, so the directory is validated
/// once when the source is built and every later lookup is a hash probe.
/// The session borrows the loader's mapping, which is why the caller opens
/// the [`TcfLoader`] first and hands out a reference.
pub struct TcfSource<'a> {
    session: TcfSession<'a>,
}

impl<'a> TcfSource<'a> {
    /// Bind `loader`'s directory for a run of name-addressed loads.
    ///
    /// # Errors
    /// [`Error::ModelError`] when the file declares one name twice — see the
    /// module docs — or when the mapped bytes no longer validate.
    pub fn new(loader: &'a TcfLoader) -> Result<Self> {
        if let Some(name) = first_repeated_name(loader.tensor_names()) {
            return Err(Error::ModelError {
                reason: format!(
                    "TCF file {} declares the tensor name '{name}' more than once; \
                     a name is provenance, not identity (SPECIFICATION.md Section 6), \
                     so loading by name would silently pick the first occurrence",
                    loader.path().display()
                ),
            });
        }
        Ok(Self {
            session: loader.session()?,
        })
    }

    /// The loader this source reads through.
    pub fn loader(&self) -> &'a TcfLoader {
        self.session.loader()
    }
}

/// The first name `names` yields twice, if any.
///
/// Split out because it is the whole of the collision rule and is worth
/// testing without a hand-built file that repeats a string reference.
fn first_repeated_name<'n>(names: impl Iterator<Item = &'n str>) -> Option<String> {
    let mut seen = HashSet::new();
    for name in names {
        if !seen.insert(name) {
            return Some(name.to_string());
        }
    }
    None
}

impl<R: Runtime<DType = DType>> WeightSource<R> for TcfSource<'_> {
    /// DEQUANTIZES, deliberately, exactly as the GGUF source does: the dense
    /// callers (norms, biases, conv kernels, the embedding tables) can
    /// consume nothing else. Section 15's digest and proof checks run first,
    /// per tensor, so a corrupted payload is named rather than decoded.
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        self.session.tensor::<R>(name, device)
    }

    /// Keeps a natively encoded tensor PACKED: its payload reaches the device
    /// verbatim as a `QuantTensor` carrying `QuantScheme::Tcf`, which
    /// `quant_matmul` consumes directly. A 1.2 GB Q4 file costs 1.2 GB, not
    /// the 10 GB its f32 expansion would.
    ///
    /// The ENCODING decides, never the name: a raw encoding has no packed
    /// form to hold (Section 12), so it takes the dense path. That is the
    /// BF16 `RankLt2` fallback the module docs describe, and the caller's
    /// `MaybeLoraLinear` runs its dense branch there unchanged.
    fn load_named_weight(&mut self, name: &str, device: &R::Device) -> Result<Weight<R>> {
        match self.session.loader().tensor_info(name)?.encoding() {
            Encoding::Native(_) => Ok(Weight::Quantized(
                self.session.quant_tensor::<R>(name, device)?,
            )),
            _ => Ok(Weight::Standard(self.session.tensor::<R>(name, device)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::tcf::fixtures;
    use crate::quant::{QuantScheme, TcfEncoding};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use tcf_core::NativeEncoding;

    #[test]
    fn a_repeated_name_is_reported_by_name() {
        assert_eq!(
            first_repeated_name(["a", "b", "a"].into_iter()),
            Some("a".to_string())
        );
        assert_eq!(first_repeated_name(["a", "b"].into_iter()), None);
        assert_eq!(first_repeated_name(std::iter::empty()), None);
    }

    /// The split the whole module exists for: a native encoding stays
    /// packed, a raw one arrives dense, and neither is decided by the name.
    #[test]
    fn native_stays_packed_and_raw_arrives_dense() {
        let file = fixtures::write_temp(&fixtures::good_file());
        let loader = TcfLoader::open(file.path()).expect("opens");
        let mut source = TcfSource::new(&loader).expect("binds");
        let (_client, device) = cpu_setup();

        let packed = WeightSource::<CpuRuntime>::load_named_weight(&mut source, "layer.w", &device)
            .expect("loads");
        match packed {
            Weight::Quantized(qt) => {
                assert_eq!(qt.shape(), &[1, 64]);
                assert_eq!(
                    qt.scheme(),
                    QuantScheme::Tcf(TcfEncoding::new(NativeEncoding::Q4S32T64))
                );
                // One Q4S32_T64 tile: 32 code bytes plus two binary16 scales.
                assert_eq!(qt.storage_bytes(), 36);
            }
            _ => panic!("layer.w is natively encoded, so it must arrive packed"),
        }

        let dense =
            WeightSource::<CpuRuntime>::load_named_weight(&mut source, "layer.bias", &device)
                .expect("loads");
        match dense {
            Weight::Standard(t) => {
                assert_eq!(t.shape(), &[4]);
                assert_eq!(t.to_vec::<f32>(), fixtures::RAW_F32_VALUES.to_vec());
            }
            _ => panic!("layer.bias is raw-encoded, so it must arrive dense"),
        }
    }

    /// `load_named` is the dense contract: a packed tensor dequantizes here,
    /// because every caller of it can consume nothing else.
    #[test]
    fn load_named_dequantizes_a_packed_tensor() {
        let file = fixtures::write_temp(&fixtures::good_file());
        let loader = TcfLoader::open(file.path()).expect("opens");
        let mut source = TcfSource::new(&loader).expect("binds");
        let (_client, device) = cpu_setup();

        let dense: Tensor<CpuRuntime> = source.load_named("layer.w", &device).expect("loads");
        assert_eq!(dense.shape(), &[1, 64]);
        assert_eq!(dense.to_vec::<f32>(), fixtures::expected_q4_values());
    }

    /// THE equivalence gate: a TCF written from a checkpoint must present
    /// the SAME tensor inventory that checkpoint has — every name, every
    /// shape — and must split that inventory into packed and dense weights
    /// rather than assuming one of the two.
    ///
    /// Both paths come from the environment because neither file is in the
    /// repo, and the test skips when either is absent rather than failing on
    /// a machine that has no checkpoint. Same idiom as
    /// `vae::loader::checkpoint`'s `.pth` gate.
    ///
    /// It also times the walk. A per-tensor directory reparse is quadratic
    /// over 577 names; the printed figure is what says the session avoided
    /// it.
    #[test]
    fn a_tcf_presents_the_same_inventory_as_its_safetensors() {
        use crate::format::safetensors_loader::SafeTensorsLoader;
        use std::path::Path;

        let (Ok(tcf), Ok(ckpt)) = (
            std::env::var("VOXCPM2_TCF"),
            std::env::var("VOXCPM2_SAFETENSORS"),
        ) else {
            return;
        };
        if !Path::new(&tcf).is_file() || !Path::new(&ckpt).is_file() {
            return;
        }

        let reference = SafeTensorsLoader::open(&ckpt).expect("open safetensors");
        let mut expected = reference.tensor_names();
        expected.sort();
        assert!(!expected.is_empty(), "reference checkpoint is empty");

        let opened = std::time::Instant::now();
        let loader = TcfLoader::open(&tcf).expect("open tcf");
        let open_secs = opened.elapsed().as_secs_f64();
        let mut source = TcfSource::new(&loader).expect("no repeated name");

        let mut names: Vec<String> = loader.tensor_names().map(str::to_string).collect();
        names.sort_unstable();
        assert_eq!(names, expected, "tensor inventory differs");

        let (_client, device) = cpu_setup();
        let started = std::time::Instant::now();
        let (mut packed, mut dense) = (0usize, 0usize);
        for name in &names {
            let want = reference
                .tensor_info(name)
                .expect("reference entry")
                .shape
                .clone();
            let weight = WeightSource::<CpuRuntime>::load_named_weight(&mut source, name, &device)
                .expect(name);
            match weight {
                Weight::Quantized(qt) => {
                    assert_eq!(qt.shape(), want.as_slice(), "{name}: shape");
                    assert!(want.len() >= 2, "{name}: a rank-1 tensor cannot be tiled");
                    packed += 1;
                }
                Weight::Standard(t) => {
                    assert_eq!(t.shape(), want.as_slice(), "{name}: shape");
                    dense += 1;
                }
                Weight::DecomposedQuant(_) => panic!("{name}: TCF has no decomposed form"),
            }
        }
        let walk_secs = started.elapsed().as_secs_f64();

        // Both cases must really occur: a source that assumed one of them
        // would still pass every shape assertion above.
        assert!(packed > 0 && dense > 0, "{packed} packed, {dense} dense");
        println!(
            "{} tensors: {packed} packed, {dense} dense; open {open_secs:.3}s, \
             walk {walk_secs:.3}s",
            names.len()
        );
    }

    #[test]
    fn an_unknown_name_is_named_in_the_error() {
        let file = fixtures::write_temp(&fixtures::good_file());
        let loader = TcfLoader::open(file.path()).expect("opens");
        let mut source = TcfSource::new(&loader).expect("binds");
        let (_client, device) = cpu_setup();

        let err = WeightSource::<CpuRuntime>::load_named(&mut source, "nope", &device)
            .expect_err("unknown name");
        assert!(err.to_string().contains("nope"), "{err}");
    }
}
