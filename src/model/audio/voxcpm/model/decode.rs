//! Unit C1 of the VoxCPM2 end-to-end orchestrator: turn the patches
//! [`super::generate::GenerateState`] collected into a waveform.
//!
//! ```text
//! patches: Vec<Var<R>>, each [batch, patch_size, feat_dim], one row picked
//!   -> unfold_patches_row -> latent [1, feat_dim, n_patches * patch_size]
//!   -> chunked_decode::decode_latent_windowed (sr bucket 3, 48 kHz) -> waveform [1, 1, samples]
//! ```
//!
//! [`unfold_patches`] is the exact inverse of
//! [`super::patches::fold_patches`]: patch `t`'s slot `p` lands at time index
//! `t * patch_size + p`. Every other axis order is shape-valid and scrambles
//! time, so this file mirrors `fold_patches`'s reshape/permute rather than
//! reinventing the unfold.
//!
//! # Reference mode only
//!
//! In REFERENCE (voice-clone) mode `context_len == 0`, so the reference does
//! NOT trim the decoded waveform. A trim applies only in continuation mode,
//! which this orchestrator does not implement, so [`VoxCpm2Model::decode_patches`]
//! never trims for that reason. (`chunked_decode` does trim internally, per
//! window, to remove the overlap context added for chunking — an unrelated,
//! purely-internal trim that leaves the total sample count unchanged.)
//!
//! # The one device read
//!
//! There is none in this file. [`VoxCpm2Model::decode_patches`] returns the
//! decoded `Tensor<R>` as-is; converting it to `Vec<f32>` for a WAV file is a
//! later unit's job, at the boundary right before the write.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::model::chunked_decode::{
    CONTEXT_FRAMES, WINDOW_FRAMES, decode_latent_windowed, decode_latent_windowed_from,
};
use crate::model::audio::voxcpm::model::generate::GenerateState;
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use crate::model::audio::voxcpm::vae::decoder::HOP_LENGTH;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{ShapeOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Unfold generated patches back into an `AudioVAE` latent.
///
/// `patches[t]` must be `[1, patch_size, feat_dim]`, in generation order.
/// Produces `[1, feat_dim, patches.len() * patch_size]` such that
/// `folded[t * patch_size + p][c] == unfolded[c][t * patch_size + p]` — the
/// exact inverse of [`super::patches::fold_patches`]'s
/// `view(64, -1, 4).permute(1, 2, 0)`.
///
/// Errors on an empty slice, or when any patch is not `[1, patch_size,
/// feat_dim]`. [`unfold_patches_row`] is the batched form.
pub fn unfold_patches<R: Runtime<DType = DType>>(
    patches: &[Var<R>],
    patch_size: usize,
    feat_dim: usize,
) -> Result<Tensor<R>>
where
    R::Client: ShapeOps<R>,
{
    if let Some(first) = patches.first()
        && first.shape().first() != Some(&1)
    {
        return Err(Error::InvalidArgument {
            arg: "patches",
            reason: format!(
                "expected batch-1 patches [1, {patch_size}, {feat_dim}], got {:?}; \
                 use unfold_patches_row for a batch",
                first.shape()
            ),
        });
    }
    unfold_patches_row(patches, 0, patch_size, feat_dim)
}

/// [`unfold_patches`] for row `row` of `[batch, patch_size, feat_dim]`
/// patches: the same `[1, feat_dim, patches.len() * patch_size]` latent,
/// read off that row alone.
///
/// Errors on an empty slice, on a `row` outside the batch, or when any patch
/// is not `[batch, patch_size, feat_dim]`.
pub fn unfold_patches_row<R: Runtime<DType = DType>>(
    patches: &[Var<R>],
    row: usize,
    patch_size: usize,
    feat_dim: usize,
) -> Result<Tensor<R>>
where
    R::Client: ShapeOps<R>,
{
    let Some(first) = patches.first() else {
        return Err(Error::InvalidArgument {
            arg: "patches",
            reason: "expected at least 1 patch, got 0".to_string(),
        });
    };
    let batch = first.shape().first().copied().unwrap_or(0);
    if row >= batch {
        return Err(Error::InvalidArgument {
            arg: "row",
            reason: format!("expected a row below the batch {batch}, got {row}"),
        });
    }

    let expected = [batch, patch_size, feat_dim];
    let mut rows = Vec::with_capacity(patches.len());
    for (i, patch) in patches.iter().enumerate() {
        let shape = patch.shape();
        if shape != expected.as_slice() {
            return Err(Error::InvalidArgument {
                arg: "patches",
                reason: format!("patch {i}: expected {expected:?}, got {shape:?}"),
            });
        }
        rows.push(patch.tensor().narrow(0, row, 1)?);
    }
    let refs: Vec<&Tensor<R>> = rows.iter().collect();

    // [n_patches, patch_size, feat_dim], patches[t][p][c] preserved exactly
    // as fold_patches's own output would be.
    let stacked = Tensor::cat(&refs, 0)?;
    let frames = patches.len() * patch_size;

    // Inverse of fold_patches's `reshape([feat_dim, t_ref, patch_size])
    // .permute([1, 2, 0])`: permute back with the inverse permutation
    // ([1,2,0])^-1 = [2,0,1], then merge (t_ref, patch_size) back into one
    // frame axis exactly as the original reshape split it.
    Ok(stacked
        .permute(&[2, 0, 1])?
        .contiguous()?
        .reshape(&[feat_dim, frames])?
        .unsqueeze(0)?)
}

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Decode generated patches to a waveform.
    ///
    /// Unfolds `patches` with [`unfold_patches`] using `self.config`'s patch
    /// geometry, then runs [`super::loader::VoxCpm2Model::vae_decoder`] at
    /// the fixed 48 kHz sample-rate bucket
    /// ([`crate::model::audio::voxcpm::vae::decoder::DEFAULT_SR_BUCKET`]),
    /// windowed via [`super::chunked_decode::decode_latent_windowed`] so peak
    /// decoder activation memory is bounded by a fixed number of latent
    /// frames rather than growing linearly with the utterance length.
    /// Returns `[1, 1, samples]` at [`crate::model::audio::voxcpm::vae::decoder::SAMPLE_RATE`]
    /// Hz, un-trimmed (reference/voice-clone mode has no context to trim).
    pub fn decode_patches<C>(&self, client: &C, patches: &[Var<R>]) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
        R::Client: ShapeOps<R> + TypeConversionOps<R>,
    {
        let latent = unfold_patches(patches, self.config.patch_size, self.config.feat_dim)?;
        self.decode_latent(client, &latent)
    }

    /// [`decode_patches`](Self::decode_patches) for row `row` of a
    /// [`GenerateState`]: decodes that row's own patches
    /// ([`GenerateState::row_patches`]), read off row `row` of each
    /// `[batch, patch_size, feat_dim]` patch. For a one-row state this is
    /// `decode_patches(client, &state.patches)`.
    pub fn decode_row<C>(
        &self,
        client: &C,
        state: &GenerateState<R>,
        row: usize,
    ) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
        R::Client: ShapeOps<R> + TypeConversionOps<R>,
    {
        let patches = state.row_patches(row)?;
        let latent =
            unfold_patches_row(patches, row, self.config.patch_size, self.config.feat_dim)?;
        self.decode_latent(client, &latent)
    }

    /// Decode an unfolded latent `[1, feat_dim, frames]` to `[1, 1, samples]`
    /// at F32.
    fn decode_latent<C>(&self, client: &C, latent: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
        R::Client: ShapeOps<R> + TypeConversionOps<R>,
    {
        // The transformer stack runs at whatever dtype it was loaded at; the
        // decoder runs at its own, independently chosen `vae_decoder_dtype`
        // (the encoder has no such option — always F32, see
        // `AudioVaeEncoder::from_checkpoint`'s docs). This is the boundary
        // between the stack and the decoder, mirroring `prefill_inner`'s cast
        // of the encoder's reference features up into the stack's dtype.
        let latent = latent.to_dtype(self.vae_decoder.dtype())?;
        let decoded = decode_latent_windowed(client, &self.vae_decoder, &latent)?;
        // Cast back to F32 at the exit: every caller of `decode_patches`
        // (wav encoding, structural checks, `.to_vec::<f32>()`) expects a
        // stable F32 waveform contract, whatever dtype the decoder ran its
        // activations at internally.
        Ok(decoded.to_dtype(DType::F32)?)
    }

    /// Decode the waveform for `patches[from..]` only.
    ///
    /// `patches` is every patch generated so far, in order; the result is
    /// `[1, 1, (patches.len() - from) * patch_size * HOP_LENGTH]` at F32 and
    /// equals the corresponding tail of [`Self::decode_patches`] over the
    /// same slice. A streaming caller calls this once per emitted chunk with
    /// `from` at the previous chunk's end and concatenates the results.
    ///
    /// Mechanism: the decode runs on the same window grid
    /// [`Self::decode_patches`] uses, anchored at patch 0, so it unfolds from
    /// [`first_patch_for_frame`] — the start of the left context of the
    /// window containing `from` — rather than from `from` itself, decodes
    /// that suffix windowed, and drops everything before `from`. Every
    /// window complete at call time is therefore the identical decoder call
    /// the whole decode issues.
    ///
    /// Errors when `from >= patches.len()`.
    pub fn decode_patches_from<C>(
        &self,
        client: &C,
        patches: &[Var<R>],
        from: usize,
    ) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
        R::Client: ShapeOps<R> + TypeConversionOps<R>,
    {
        if let Some(first) = patches.first()
            && first.shape().first() != Some(&1)
        {
            return Err(Error::InvalidArgument {
                arg: "patches",
                reason: format!(
                    "expected batch-1 patches, got {:?}; use decode_row_from for a batch",
                    first.shape()
                ),
            });
        }
        self.decode_row_patches_from(client, patches, 0, from)
    }

    /// [`decode_patches_from`](Self::decode_patches_from) for row `row` of
    /// a [`GenerateState`]: the waveform for that row's patches `from..`.
    /// Errors when `from` is not below the row's own patch count.
    pub fn decode_row_from<C>(
        &self,
        client: &C,
        state: &GenerateState<R>,
        row: usize,
        from: usize,
    ) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
        R::Client: ShapeOps<R> + TypeConversionOps<R>,
    {
        self.decode_row_patches_from(client, state.row_patches(row)?, row, from)
    }

    fn decode_row_patches_from<C>(
        &self,
        client: &C,
        patches: &[Var<R>],
        row: usize,
        from: usize,
    ) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
        R::Client: ShapeOps<R> + TypeConversionOps<R>,
    {
        if from >= patches.len() {
            return Err(Error::InvalidArgument {
                arg: "from",
                reason: format!("expected a patch index below {}, got {from}", patches.len()),
            });
        }
        let patch_size = self.config.patch_size;
        let from_frame = from * patch_size;
        let start_patch = first_patch_for_frame(from_frame, patch_size);
        let latent = unfold_patches_row(
            &patches[start_patch..],
            row,
            patch_size,
            self.config.feat_dim,
        )?;
        let latent = latent.to_dtype(self.vae_decoder.dtype())?;
        let decoded = decode_latent_windowed_from(
            client,
            &self.vae_decoder,
            &latent,
            start_patch * patch_size,
            from_frame,
        )?;
        Ok(decoded.to_dtype(DType::F32)?)
    }

    /// Waveform samples one generated patch decodes to.
    pub const fn samples_per_patch(&self) -> usize {
        self.config.patch_size * HOP_LENGTH
    }
}

/// The first patch a suffix decode from latent frame `from_frame` must
/// unfold: the patch holding the context start of the window that contains
/// `from_frame`, on the grid anchored at frame 0. Rounds down to a whole
/// patch, so the unfolded latent starts at or before that context start.
pub(crate) fn first_patch_for_frame(from_frame: usize, patch_size: usize) -> usize {
    let window_start = (from_frame / WINDOW_FRAMES) * WINDOW_FRAMES;
    window_start.saturating_sub(CONTEXT_FRAMES) / patch_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::model::patches::fold_patches;
    use crate::test_utils::cpu_setup;
    use numr::ops::ScalarOps;
    use numr::runtime::cpu::CpuRuntime;

    /// Split a `fold_patches`-shaped tensor `[t_ref, patch_size, feat_dim]`
    /// into `t_ref` per-patch `Var`s of `[1, patch_size, feat_dim]`, mimicking
    /// how [`super::super::generate::GenerateState::patches`] is populated.
    fn split_into_patches(folded: &Tensor<CpuRuntime>, t_ref: usize) -> Vec<Var<CpuRuntime>> {
        (0..t_ref)
            .map(|t| {
                let row = folded.narrow(0, t, 1).expect("narrow");
                Var::new(row, false)
            })
            .collect()
    }

    /// `unfold_patches` inverts `fold_patches` exactly: every value encodes
    /// its own `(channel, frame)` index, so a transposed unfold (any axis
    /// order other than the one derived here) cannot pass.
    #[test]
    fn unfold_inverts_fold_for_self_indexing_values() {
        let (_client, device) = cpu_setup();
        let (feat_dim, patch_size, t_ref) = (3usize, 2usize, 4usize);
        let frames = patch_size * t_ref;

        let data: Vec<f32> = (0..feat_dim * frames)
            .map(|i| {
                let (c, t) = (i / frames, i % frames);
                (c * 100 + t) as f32
            })
            .collect();
        let latent =
            Tensor::<CpuRuntime>::from_slice(&data, &[1, feat_dim, frames], &device).expect("in");

        let folded = fold_patches(&latent, patch_size, feat_dim).expect("fold");
        let patches = split_into_patches(&folded, t_ref);

        let unfolded = unfold_patches(&patches, patch_size, feat_dim).expect("unfold");
        assert_eq!(unfolded.shape(), &[1, feat_dim, frames]);

        let want: Vec<f32> = latent.contiguous().expect("contig").to_vec();
        let got: Vec<f32> = unfolded.contiguous().expect("contig").to_vec();
        assert_eq!(got, want);
    }

    /// Row selection: two rows stacked per patch, the second a scaled copy of
    /// the first, unfold to the first latent and its scaled twin.
    #[test]
    fn unfold_row_picks_one_row_of_a_batch() {
        let (client, device) = cpu_setup();
        let (feat_dim, patch_size, t_ref) = (3usize, 2usize, 4usize);
        let frames = patch_size * t_ref;
        let data: Vec<f32> = (0..feat_dim * frames).map(|i| i as f32 + 1.0).collect();
        let latent =
            Tensor::<CpuRuntime>::from_slice(&data, &[1, feat_dim, frames], &device).expect("in");
        let folded = fold_patches(&latent, patch_size, feat_dim).expect("fold");
        let doubled = client.mul_scalar(&folded, 2.0).expect("scale");
        let patches: Vec<Var<CpuRuntime>> = (0..t_ref)
            .map(|t| {
                let a = folded.narrow(0, t, 1).expect("narrow");
                let b = doubled.narrow(0, t, 1).expect("narrow");
                Var::new(Tensor::cat(&[&a, &b], 0).expect("stack"), false)
            })
            .collect();

        let want: Vec<f32> = latent.contiguous().expect("contig").to_vec();
        let row0: Vec<f32> = unfold_patches_row(&patches, 0, patch_size, feat_dim)
            .expect("row 0")
            .to_vec();
        let row1: Vec<f32> = unfold_patches_row(&patches, 1, patch_size, feat_dim)
            .expect("row 1")
            .to_vec();
        assert_eq!(row0, want);
        assert_eq!(row1, want.iter().map(|v| v * 2.0).collect::<Vec<_>>());
        assert!(unfold_patches_row(&patches, 2, patch_size, feat_dim).is_err());
        assert!(
            unfold_patches(&patches, patch_size, feat_dim).is_err(),
            "the one-row entry point must refuse a batch"
        );
    }

    /// The unfold start never passes the containing window's context start,
    /// and never fetches a whole window more than needed.
    #[test]
    fn first_patch_for_frame_covers_the_window_context() {
        let patch_size = 4;
        for from_frame in (0..3 * WINDOW_FRAMES).step_by(patch_size) {
            let window_start = (from_frame / WINDOW_FRAMES) * WINDOW_FRAMES;
            let context_start = window_start.saturating_sub(CONTEXT_FRAMES);
            let origin = first_patch_for_frame(from_frame, patch_size) * patch_size;
            assert!(
                origin <= context_start,
                "from {from_frame}: origin {origin}"
            );
            assert!(origin + patch_size > context_start, "from {from_frame}");
        }
        assert_eq!(first_patch_for_frame(0, 4), 0);
        assert_eq!(first_patch_for_frame(60, 4), 0);
        assert_eq!(first_patch_for_frame(64, 4), 8);
        assert_eq!(first_patch_for_frame(3, 5), 0);
        assert_eq!(first_patch_for_frame(130, 5), 19);
    }

    #[test]
    fn unfold_rejects_an_empty_patch_list() {
        let patches: Vec<Var<CpuRuntime>> = Vec::new();
        let err = unfold_patches(&patches, 4, 64).expect_err("empty slice must error");
        let msg = err.to_string();
        assert!(msg.contains('0'), "{msg}");
    }

    #[test]
    fn unfold_rejects_a_patch_with_the_wrong_shape() {
        let (_client, device) = cpu_setup();
        let bad =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4 * 3], &[1, 4, 3], &device).expect("in");
        let patches = vec![Var::new(bad, false)];
        let err = unfold_patches(&patches, 4, 64).expect_err("feat_dim 3 != 64 must error");
        let msg = err.to_string();
        assert!(msg.contains("patch 0"), "{msg}");
        assert!(msg.contains("[1, 4, 64]"), "{msg}");
        assert!(msg.contains("[1, 4, 3]"), "{msg}");
    }
}
