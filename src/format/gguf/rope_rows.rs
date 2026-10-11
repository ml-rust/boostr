//! llama.cpp's query/key row order for rotary attention.
//!
//! A HuggingFace Llama checkpoint rotates each head's dimensions as two
//! halves: dimension `i` pairs with dimension `i + head_dim / 2`. llama.cpp's
//! `llama` architecture rotates adjacent pairs instead: `2i` with `2i + 1`.
//! `convert_hf_to_gguf.py` reorders the rows of `q_proj` and `k_proj`, weight
//! and bias, so the same rotation lands on the same values:
//!
//! ```text
//! weights.reshape(n_head, 2, head_dim / 2, ...).swapaxes(1, 2).reshape(...)
//! ```
//!
//! A GGUF writer applies [`RopeRowOrder::ToGguf`] and a reader applies
//! [`RopeRowOrder::ToHf`]. A row of a block-quantized tensor is a whole number
//! of blocks, so the reorder moves whole rows and never touches a value.
//!
//! The reader reorders the raw bytes on the host, before any upload. The
//! file's bytes are already host memory, so the reorder costs no device copy.

use super::metadata::GgufMetadata;
use crate::error::{Error, Result};

/// The `general.architecture` values whose files store llama.cpp's row order.
///
/// Each one is an architecture whose `convert_hf_to_gguf.py` class sets
/// `undo_permute`. An architecture with half-split rotation in llama.cpp, such
/// as `qwen2`, stores the HuggingFace order and is absent here.
pub const ROPE_PERMUTED_ARCHS: &[&str] = &["llama"];

/// Which way a row reorder runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeRowOrder {
    /// HuggingFace order to llama.cpp order, for a writer.
    ToGguf,
    /// llama.cpp order to HuggingFace order, for a reader.
    ToHf,
}

/// The head count a GGUF tensor's rows are grouped by, when the file stores
/// llama.cpp's row order for it.
///
/// `gguf_name` is the GGUF tensor name, such as `blk.0.attn_q.weight`. The
/// query projection groups by `n_head` and the key projection by `n_head_kv`.
/// `None` for every other tensor, and for every architecture outside
/// [`ROPE_PERMUTED_ARCHS`].
pub fn rope_row_heads(
    arch: &str,
    gguf_name: &str,
    n_head: usize,
    n_head_kv: usize,
) -> Option<usize> {
    if !ROPE_PERMUTED_ARCHS.contains(&arch) {
        return None;
    }
    let rest = gguf_name.strip_prefix("blk.")?;
    let (_, suffix) = rest.split_once('.')?;
    match suffix {
        "attn_q.weight" | "attn_q.bias" => Some(n_head),
        "attn_k.weight" | "attn_k.bias" => Some(n_head_kv),
        _ => None,
    }
}

/// The head count [`rope_row_heads`] returns for `gguf_name`, with the
/// architecture and head counts read from the file's own metadata.
///
/// `head_count_kv` defaults to `head_count`, as llama.cpp reads it. `None`
/// when the file is outside [`ROPE_PERMUTED_ARCHS`], lacks `head_count`, or
/// `gguf_name` is not a query or key projection.
pub fn rope_row_heads_in(metadata: &GgufMetadata, gguf_name: &str) -> Option<usize> {
    let arch = metadata.architecture()?;
    if !ROPE_PERMUTED_ARCHS.contains(&arch) {
        return None;
    }
    let n_head = metadata.get_u32(&format!("{arch}.attention.head_count"))? as usize;
    let n_head_kv = metadata
        .get_u32(&format!("{arch}.attention.head_count_kv"))
        .map_or(n_head, |v| v as usize);
    rope_row_heads(arch, gguf_name, n_head, n_head_kv)
}

/// Reorder the rows of a tensor's raw bytes on the host.
///
/// `bytes` holds `rows` rows of equal byte length, in row-major order. A row
/// is opaque: an F32, F16 or BF16 row and a run of quantized blocks move
/// alike. `rows` is the leading axis of the row-major shape.
///
/// # Errors
///
/// `InvalidArgument` when `bytes` does not split into `rows` equal rows, or
/// as [`rope_row_sources`].
pub fn reorder_rope_rows(
    bytes: &[u8],
    rows: usize,
    heads: usize,
    order: RopeRowOrder,
) -> Result<Vec<u8>> {
    let row_bytes = bytes.len().checked_div(rows).unwrap_or(0);
    if row_bytes == 0 || row_bytes * rows != bytes.len() {
        return Err(Error::InvalidArgument {
            arg: "rows",
            reason: format!(
                "{} bytes do not split into {rows} equal rows. Check the tensor shape.",
                bytes.len()
            ),
        });
    }
    let sources = rope_row_sources(rows, heads, order)?;
    let mut out = Vec::with_capacity(bytes.len());
    for source in sources {
        out.extend_from_slice(&bytes[source * row_bytes..(source + 1) * row_bytes]);
    }
    Ok(out)
}

/// For each output row, the input row it is copied from.
///
/// # Errors
///
/// `InvalidArgument` when `heads` is zero or `rows` does not split into
/// `heads` heads of an even size.
pub fn rope_row_sources(rows: usize, heads: usize, order: RopeRowOrder) -> Result<Vec<usize>> {
    let head_dim = rows.checked_div(heads).unwrap_or(0);
    if heads == 0 || head_dim == 0 || head_dim * heads != rows || !head_dim.is_multiple_of(2) {
        return Err(Error::InvalidArgument {
            arg: "rows",
            reason: format!(
                "{rows} rows do not split into {heads} heads of an even size. \
                 Check the head count in the model config."
            ),
        });
    }
    let half = head_dim / 2;
    let mut sources = Vec::with_capacity(rows);
    for head in 0..heads {
        let base = head * head_dim;
        for out in 0..head_dim {
            // HF row `part * half + i` sits at GGUF row `2 * i + part`.
            let source = match order {
                RopeRowOrder::ToGguf => (out % 2) * half + out / 2,
                RopeRowOrder::ToHf => 2 * (out % half) + out / half,
            };
            sources.push(base + source);
        }
    }
    Ok(sources)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference order, from numpy:
    /// `np.arange(8).reshape(1, 2, 4).swapaxes(1, 2).reshape(8)`.
    #[test]
    fn to_gguf_matches_the_converter() {
        let sources = rope_row_sources(8, 1, RopeRowOrder::ToGguf).unwrap();
        assert_eq!(sources, [0, 4, 1, 5, 2, 6, 3, 7]);
        let two_heads = rope_row_sources(8, 2, RopeRowOrder::ToGguf).unwrap();
        assert_eq!(two_heads, [0, 2, 1, 3, 4, 6, 5, 7]);
    }

    #[test]
    fn to_hf_undoes_to_gguf() {
        let forward = rope_row_sources(24, 3, RopeRowOrder::ToGguf).unwrap();
        let back = rope_row_sources(24, 3, RopeRowOrder::ToHf).unwrap();
        let round_trip: Vec<usize> = back.iter().map(|&r| forward[r]).collect();
        assert_eq!(round_trip, (0..24).collect::<Vec<_>>());
    }

    #[test]
    fn only_llama_query_and_key_rows_are_permuted() {
        assert_eq!(
            rope_row_heads("llama", "blk.3.attn_q.weight", 32, 8),
            Some(32)
        );
        assert_eq!(rope_row_heads("llama", "blk.3.attn_k.bias", 32, 8), Some(8));
        assert_eq!(rope_row_heads("llama", "blk.3.attn_v.weight", 32, 8), None);
        assert_eq!(rope_row_heads("qwen2", "blk.3.attn_q.weight", 32, 8), None);
        assert_eq!(
            rope_row_heads("llama", "model.layers.3.self_attn.q_proj.weight", 32, 8),
            None
        );
    }

    /// Two-byte rows move whole. Byte pairs carry their row index.
    #[test]
    fn reorder_moves_whole_rows_of_bytes() {
        let bytes: Vec<u8> = (0..8u8).flat_map(|r| [r, r]).collect();
        let out = reorder_rope_rows(&bytes, 8, 1, RopeRowOrder::ToGguf).unwrap();
        let rows: Vec<[u8; 2]> = out.chunks(2).map(|c| [c[0], c[1]]).collect();
        let want: Vec<[u8; 2]> = [0u8, 4, 1, 5, 2, 6, 3, 7].map(|r| [r, r]).to_vec();
        assert_eq!(rows, want);
        let back = reorder_rope_rows(&out, 8, 1, RopeRowOrder::ToHf).unwrap();
        assert_eq!(back, bytes);
        assert!(reorder_rope_rows(&bytes[..15], 8, 1, RopeRowOrder::ToHf).is_err());
    }

    #[test]
    fn rejects_rows_that_do_not_split_into_heads() {
        assert!(rope_row_sources(10, 4, RopeRowOrder::ToHf).is_err());
        assert!(rope_row_sources(6, 2, RopeRowOrder::ToHf).is_err());
        assert!(rope_row_sources(8, 0, RopeRowOrder::ToHf).is_err());
    }
}
