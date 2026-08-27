//! The HuggingFace -> ggml-conventional tensor-name map for VoxCPM2, plus
//! the sentinel probe that decides which of the two conventions a GGUF uses.
//!
//! # Direction
//!
//! The map is written FORWARD (ours -> theirs) on purpose. Every VoxCPM2
//! sub-loader asks for the HuggingFace key it would ask a safetensors
//! checkpoint for, so the only useful moment to translate is just before the
//! lookup hits the file. A reverse map would have to be applied by walking
//! the whole tensor index up front, which buys nothing.
//!
//! # The stacks
//!
//! Four per-layer stacks share ONE suffix table — cstr's port gives every
//! transformer block the same ggml spelling regardless of which of our four
//! stacks it came from, and the layer index is identical on both sides. Only
//! the block prefix differs, so the per-layer half is a prefix rewrite plus
//! that shared table rather than 4 x 9 x N explicit entries.
//!
//! Everything else is flat and irregular enough that an explicit table is
//! the honest encoding.

use crate::error::{Error, Result};
use crate::format::gguf::Gguf;

/// Per-layer stack prefixes: ours -> cstr's. The layer index that follows
/// is carried through unchanged.
const LAYER_STACKS: &[(&str, &str)] = &[
    ("base_lm.layers.", "tslm.blk."),
    ("residual_lm.layers.", "ralm.blk."),
    ("feat_encoder.encoder.layers.", "locenc.blk."),
    ("feat_decoder.estimator.decoder.layers.", "locdit.blk."),
];

/// Within-layer suffixes, shared by all four stacks above.
const LAYER_SUFFIXES: &[(&str, &str)] = &[
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
];

/// Flat tensors, keyed by STEM — the name with a trailing `.weight`/`.bias`
/// removed, so a biased projection is one row instead of two.
///
/// `feat_encoder.special_token` carries neither suffix and is therefore its
/// own whole stem; that is the one entry whose stored SHAPE also differs —
/// see [`restore_leading_unit_dims`].
///
/// cstr's `time_mlp`/`dt_mlp` use 0-based member indices where ours use
/// `linear_1`/`linear_2`, i.e. their index + 1 == our linear number.
const FLAT_STEMS: &[(&str, &str)] = &[
    ("base_lm.embed_tokens", "tslm.token_embd"),
    ("base_lm.norm", "tslm.output_norm"),
    ("residual_lm.norm", "ralm.output_norm"),
    ("feat_encoder.encoder.norm", "locenc.output_norm"),
    ("feat_encoder.in_proj", "locenc.in_proj"),
    ("feat_encoder.special_token", "locenc.cls_token"),
    ("feat_decoder.estimator.decoder.norm", "locdit.output_norm"),
    ("feat_decoder.estimator.in_proj", "locdit.in_proj"),
    ("feat_decoder.estimator.out_proj", "locdit.out_proj"),
    ("feat_decoder.estimator.cond_proj", "locdit.cond_proj"),
    (
        "feat_decoder.estimator.time_mlp.linear_1",
        "locdit.time_mlp.0",
    ),
    (
        "feat_decoder.estimator.time_mlp.linear_2",
        "locdit.time_mlp.1",
    ),
    (
        "feat_decoder.estimator.delta_time_mlp.linear_1",
        "locdit.dt_mlp.0",
    ),
    (
        "feat_decoder.estimator.delta_time_mlp.linear_2",
        "locdit.dt_mlp.1",
    ),
    ("fsq_layer.in_proj", "fsq.in_proj"),
    ("fsq_layer.out_proj", "fsq.out_proj"),
    ("enc_to_lm_proj", "proj.enc_to_lm"),
    ("fusion_concat_proj", "proj.fusion"),
    ("lm_to_dit_proj", "proj.lm_to_dit"),
    ("res_to_dit_proj", "proj.res_to_dit"),
    ("stop_head", "stop.head"),
    ("stop_proj", "stop.proj"),
];

/// Trailing members a flat stem may carry. Empty last so a name that is
/// wholly a stem (`feat_encoder.special_token`) still matches.
const FLAT_MEMBERS: &[&str] = &[".weight", ".bias", ""];

/// Translate one of OUR HuggingFace tensor names to cstr's ggml-conventional
/// spelling.
///
/// `None` means "not a name this map knows" — the caller must then leave the
/// name alone rather than invent one, so an unmapped key fails as a missing
/// tensor naming the key we actually asked for.
pub fn hf_to_ggml_name(name: &str) -> Option<String> {
    for (ours, theirs) in LAYER_STACKS {
        if let Some(rest) = name.strip_prefix(ours)
            && let Some(dot) = rest.find('.')
        {
            let (index, suffix) = (&rest[..dot], &rest[dot + 1..]);
            // A non-numeric segment here is not a layer at all; refusing it
            // keeps `base_lm.layers.foo.bar` from becoming `tslm.blk.foo.bar`.
            if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let mapped = LAYER_SUFFIXES
                .iter()
                .find(|(from, _)| *from == suffix)
                .map(|(_, to)| *to)?;
            return Some(format!("{theirs}{index}.{mapped}"));
        }
    }

    for member in FLAT_MEMBERS {
        let Some(stem) = name.strip_suffix(member) else {
            continue;
        };
        if let Some((_, theirs)) = FLAT_STEMS.iter().find(|(ours, _)| *ours == stem) {
            return Some(format!("{theirs}{member}"));
        }
    }

    None
}

/// Our name for the sentinel that identifies a verbatim-HuggingFace GGUF —
/// the one `compressr convert --format gguf` writes.
pub(crate) const VERBATIM_SENTINEL: &str = "base_lm.embed_tokens.weight";

/// cstr's name for that same embedding table, identifying a
/// ggml-conventional GGUF.
pub const GGML_SENTINEL: &str = "tslm.token_embd.weight";

/// Which tensor-naming convention a VoxCPM2 GGUF uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GgufNaming {
    /// Keys are the checkpoint's HuggingFace names, written through
    /// unchanged by compressr. No translation.
    Verbatim,
    /// Keys follow llama.cpp's ggml convention, as in `cstr/voxcpm2-GGUF`.
    /// Translated through [`hf_to_ggml_name`].
    Ggml,
}

/// Decide the convention by probing for a sentinel tensor.
///
/// # Why not `general.architecture`
///
/// Because it cannot discriminate. cstr's file sets it to `voxcpm2`, and our
/// own compressr output ALSO sets it to `voxcpm2` while writing verbatim
/// HuggingFace keys. The metadata agrees on both files precisely where they
/// differ, so the only honest discriminator is the tensor index itself.
///
/// The embedding table is the sentinel because it is the one tensor both
/// conventions must carry exactly once, under a name unique to each.
pub(crate) fn probe_naming(gguf: &Gguf) -> Result<GgufNaming> {
    if gguf.tensor_info(GGML_SENTINEL).is_ok() {
        return Ok(GgufNaming::Ggml);
    }
    if gguf.tensor_info(VERBATIM_SENTINEL).is_ok() {
        return Ok(GgufNaming::Verbatim);
    }
    // Name what WAS there: an operator handed the wrong GGUF gets to see it
    // immediately instead of chasing a missing-tensor error per sub-loader.
    let mut found: Vec<&str> = gguf.tensor_names().take(8).collect();
    found.sort_unstable();
    Err(Error::ModelError {
        reason: format!(
            "not a VoxCPM2 GGUF: neither `{GGML_SENTINEL}` (ggml-conventional, as in \
             cstr/voxcpm2-GGUF) nor `{VERBATIM_SENTINEL}` (verbatim HuggingFace, as \
             written by `compressr convert --format gguf`) is present; the file holds \
             {} tensors, first few: {found:?}",
            gguf.len()
        ),
    })
}

/// HF names whose ggml-conventional counterpart is stored with its leading
/// unit dims squeezed away, paired with the RANK our loader demands.
///
/// One entry, and it stays one entry: `feat_encoder.special_token` is
/// `[1, 1, 1, hidden]` for us and `locenc.cls_token` is `[hidden]` in cstr's
/// file. Everything else agrees exactly once GGUF's reversed dim order is
/// undone.
const SQUEEZED_LEADING_UNIT_DIMS: &[(&str, usize)] = &[("feat_encoder.special_token", 4)];

/// Restore the leading unit dims a third-party writer squeezed off `name`.
///
/// Returns the shape to reshape to, or `None` for "leave it alone".
///
/// The rule is deliberately NARROW — it is not "reshape anything with the
/// same element count". Such a fallback would silently paper over a real
/// layout disagreement, which is exactly the class of bug this loader must
/// surface. Three conditions all hold before a reshape happens:
///
/// 1. `name` is in the table above, so the rank we demand is known.
/// 2. The stored rank is LOWER than that rank — an equal or higher rank is
///    either already right or genuinely different, and neither is ours to
///    rewrite.
/// 3. Dropping the stored shape's leading `1`s leaves at most ONE dim. The
///    expected shape's only non-unit dim is its last, so a stored shape with
///    two real dims (`[2, 512]` against an expected `[1, 1, 1, 1024]`) is a
///    different layout that happens to share an element count. It is
///    REJECTED here and fails the sub-loader's shape check with both shapes
///    named.
pub(crate) fn restore_leading_unit_dims(name: &str, actual: &[usize]) -> Option<Vec<usize>> {
    let rank = SQUEEZED_LEADING_UNIT_DIMS
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, rank)| *rank)?;
    if actual.len() >= rank {
        return None;
    }
    let core: &[usize] = match actual.iter().position(|&d| d != 1) {
        Some(first) => &actual[first..],
        None => &[],
    };
    if core.len() > 1 {
        return None;
    }
    let mut restored = vec![1usize; rank - core.len()];
    restored.extend_from_slice(core);
    Some(restored)
}
