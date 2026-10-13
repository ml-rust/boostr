//! A `qwen35` decision model from a GGUF file, built the way blazr builds
//! one: config from the header, weights through `VarMap::from_gguf` (block
//! formats stay quantized), model through `LoadedModel::load`, on any
//! runtime.

use std::path::Path;

use boostr::format::gguf::Gguf;
use boostr::model::{LoadedModel, ModelClient, Qwen35Model, qwen35_config_from_gguf};
use boostr::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// The llama.cpp file types the decision tests hold tolerances for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// `general.file_type` 7: every matrix Q8_0.
    Q8_0,
    /// `general.file_type` 15: Q4_K matrices, a share of them Q6_K.
    Q4KM,
}

impl Quant {
    /// The quant `general.file_type` names, else the one the file name
    /// carries.
    ///
    /// # Panics
    ///
    /// When neither names Q8_0 or Q4_K_M.
    pub fn of(gguf: &Gguf, path: &Path) -> Self {
        let file_type = gguf.metadata().get_u32("general.file_type");
        match file_type {
            Some(7) => return Self::Q8_0,
            Some(15) => return Self::Q4KM,
            _ => {}
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_uppercase())
            .unwrap_or_default();
        if name.contains("Q8_0") {
            Self::Q8_0
        } else if name.contains("Q4_K_M") {
            Self::Q4KM
        } else {
            panic!(
                "{}: general.file_type is {file_type:?} and the file name names no quant; \
                 the decision tests hold tolerances for Q8_0 (file type 7) and Q4_K_M \
                 (file type 15) only",
                path.display()
            )
        }
    }
}

/// The model of the GGUF at `path` on `device`, and its quant.
///
/// Prints which head the file carries: its own `output.weight`, or a head
/// tied to `token_embd.weight`, with the storage type `candidate_logits`
/// reads rows from.
pub fn build_model<R>(path: &Path, device: &R::Device) -> (Box<Qwen35Model<R>>, Quant)
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    let gguf = Gguf::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let quant = Quant::of(&gguf, path);
    let config = qwen35_config_from_gguf(&gguf)
        .unwrap_or_else(|e| panic!("qwen35_config_from_gguf on {}: {e}", path.display()));
    let (head_name, relation) = if config.tie_word_embeddings {
        ("token_embd.weight", "tied to")
    } else {
        ("output.weight", "own")
    };
    let head_type = gguf
        .tensor_info(head_name)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .ggml_type;
    println!(
        "{}: {quant:?}, lm_head {relation} {head_name} ({head_type:?})",
        path.display()
    );
    drop(gguf);

    let mut map = VarMap::<R>::from_gguf(path, device)
        .unwrap_or_else(|e| panic!("VarMap::from_gguf on {}: {e}", path.display()));
    let mut vb = VarBuilder::new(&mut map, device);
    let loaded = LoadedModel::<R>::load(&config, &mut vb)
        .unwrap_or_else(|e| panic!("LoadedModel::load: {e}"));
    let model = match loaded {
        LoadedModel::Qwen35(model) => model,
        _ => panic!("LoadedModel::load returned a model other than Qwen35"),
    };
    assert_eq!(
        map.len(),
        0,
        "loader left {} tensors behind: {:?}",
        map.len(),
        map.names().take(8).collect::<Vec<_>>()
    );
    (model, quant)
}
