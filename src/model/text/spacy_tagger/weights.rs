//! Raw tensors of the spaCy CNN tagger and their SafeTensors names.
//!
//! The names are the ones `convert_tagger.py` writes; every tensor keeps
//! thinc's layout.

use crate::error::Result;
use crate::format::safetensors::SafeTensors;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One thinc `Maxout(normalize=True)`: the affine map, then `LayerNorm`.
pub struct MaxoutWeights<R: Runtime> {
    /// `W`, `[nO, nP, nI]`.
    pub w: Tensor<R>,
    /// `b`, `[nO, nP]`.
    pub b: Tensor<R>,
    /// `LayerNorm` gain `G`, `[nO]`.
    pub norm_g: Tensor<R>,
    /// `LayerNorm` bias `b`, `[nO]`.
    pub norm_b: Tensor<R>,
}

/// Every parameter of `tok2vec` and `tagger`.
pub struct SpacyTaggerWeights<R: Runtime> {
    /// `HashEmbed` table `E` per attribute column, `[rows_i, width]`.
    pub tables: Vec<Tensor<R>>,
    /// The `MultiHashEmbed` mixing layer.
    pub embed_mix: MaxoutWeights<R>,
    /// The `MaxoutWindowEncoder` blocks, in order.
    pub encode: Vec<MaxoutWeights<R>>,
    /// Tagger output layer `W`, `[n_labels, width]`.
    pub softmax_w: Tensor<R>,
    /// Tagger output layer `b`, `[n_labels]`.
    pub softmax_b: Tensor<R>,
}

impl<R: Runtime<DType = DType>> SpacyTaggerWeights<R> {
    /// Loads `n_tables` tables and `depth` encoder blocks from `st`.
    /// [`super::SpacyTagger::new`] checks the shapes.
    pub fn from_safetensors(
        st: &mut SafeTensors,
        n_tables: usize,
        depth: usize,
        device: &R::Device,
    ) -> Result<Self> {
        let mut tables = Vec::with_capacity(n_tables);
        for i in 0..n_tables {
            tables.push(st.load_tensor::<R>(&format!("embed.{i}.E"), device)?);
        }
        let embed_mix = maxout(st, "embed.maxout", "embed.norm", device)?;
        let mut encode = Vec::with_capacity(depth);
        for d in 0..depth {
            encode.push(maxout(
                st,
                &format!("encode.{d}.maxout"),
                &format!("encode.{d}.norm"),
                device,
            )?);
        }
        Ok(Self {
            tables,
            embed_mix,
            encode,
            softmax_w: st.load_tensor::<R>("softmax.W", device)?,
            softmax_b: st.load_tensor::<R>("softmax.b", device)?,
        })
    }
}

fn maxout<R: Runtime<DType = DType>>(
    st: &mut SafeTensors,
    maxout: &str,
    norm: &str,
    device: &R::Device,
) -> Result<MaxoutWeights<R>> {
    Ok(MaxoutWeights {
        w: st.load_tensor::<R>(&format!("{maxout}.W"), device)?,
        b: st.load_tensor::<R>(&format!("{maxout}.b"), device)?,
        norm_g: st.load_tensor::<R>(&format!("{norm}.G"), device)?,
        norm_b: st.load_tensor::<R>(&format!("{norm}.b"), device)?,
    })
}
