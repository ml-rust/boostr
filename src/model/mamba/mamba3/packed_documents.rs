//! Mamba3 on packed rows: no state crosses a document start.
//!
//! The scan carries `h`, `prev_x` and `prev_b`. The decay and the
//! previous-token term are both masked at a start. The optional conv uses the
//! masked branch. Complex RoPE needs no mask: B and C rotate by the same
//! cumulative angle, so only angle differences inside a document matter.

#[cfg(test)]
mod tests {
    use crate::model::mamba::mamba3::config::Mamba3Config;
    use crate::model::mamba::mamba3::layer::{Mamba3, Mamba3Weights};
    use crate::model::mamba::packed_support::{
        check_no_leak, check_packed_vs_separate, check_single_document_equals_none, t64,
    };
    use crate::nn::segment_masks::SegmentMasks;
    use crate::nn::{Conv1d, Linear, RmsNorm};
    use crate::test_utils::Lcg;
    use numr::autograd::Var;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    const D_MODEL: usize = 8;
    const D_CONV: usize = 4;

    fn layer(rope: bool, conv: bool) -> Mamba3<CpuRuntime> {
        let config = Mamba3Config::new(D_MODEL)
            .with_nheads(2)
            .with_d_state(4)
            .with_expand(2)
            .with_d_conv(D_CONV)
            .with_complex_rope(rope)
            .with_use_conv(conv);
        let (d_inner, nheads, d_state) = (config.d_inner(), config.nheads, config.d_state);
        let mut rng = Lcg(0xb33d_f003);
        let in_proj = Linear::new(
            t64(&mut rng, &[config.proj_dim(), D_MODEL], 0.4),
            None,
            true,
        );
        let out_proj = Linear::new(t64(&mut rng, &[D_MODEL, d_inner], 0.4), None, true);
        let lambda_proj = Linear::new(t64(&mut rng, &[nheads, D_MODEL], 0.6), None, true);
        let theta_proj = rope.then(|| {
            Linear::new(
                t64(&mut rng, &[nheads * (d_state / 2), D_MODEL], 0.6),
                None,
                true,
            )
        });
        let channels = config.conv_channels();
        let conv1d = conv.then(|| {
            Conv1d::new(
                t64(&mut rng, &[channels, 1, D_CONV], 0.6),
                Some(t64(&mut rng, &[channels], 0.2)),
                1,
                PaddingMode::Custom(D_CONV - 1, 0, 0, 0),
                1,
                channels,
                true,
            )
        });
        let weights = Mamba3Weights {
            in_proj,
            out_proj,
            lambda_proj,
            theta_proj,
            b_bias: t64(&mut rng, &[nheads, d_state], 0.3),
            c_bias: t64(&mut rng, &[nheads, d_state], 0.3),
            dt_bias: Some(t64(&mut rng, &[nheads], 0.3)),
            a_log: t64(&mut rng, &[nheads], 0.5),
            d_param: Some(t64(&mut rng, &[nheads], 0.5)),
            bc_norm: RmsNorm::new(t64(&mut rng, &[d_state], 1.0), 1e-6, true),
            norm: RmsNorm::new(t64(&mut rng, &[d_inner], 1.0), 1e-6, true),
            conv1d,
            mimo_x_up: None,
            mimo_x_down: None,
        };
        Mamba3::new(config, weights, true)
    }

    fn fwd_of<'a>(
        m: &'a Mamba3<CpuRuntime>,
    ) -> impl Fn(
        &CpuClient,
        &Var<CpuRuntime>,
        Option<&SegmentMasks<CpuRuntime>>,
    ) -> crate::error::Result<Var<CpuRuntime>>
    + 'a {
        move |c, x, s| m.forward(c, x, s)
    }

    #[test]
    fn mamba3_packed_equals_separate_documents() {
        for (rope, conv) in [(false, false), (true, false), (false, true), (true, true)] {
            let m = layer(rope, conv);
            let params = m.parameters();
            check_packed_vs_separate(&fwd_of(&m), &params, D_MODEL, D_CONV, &[5, 4]);
        }
    }

    /// Without a conv, the previous-token term is the only path from the
    /// last token of A to the first token of B besides `h`.
    #[test]
    fn mamba3_previous_token_does_not_enter_next_document() {
        let m = layer(false, false);
        let params = m.parameters();
        let fwd = fwd_of(&m);
        check_packed_vs_separate(&fwd, &params, D_MODEL, D_CONV, &[1, 3]);
        check_packed_vs_separate(&fwd, &params, D_MODEL, D_CONV, &[3, 1]);
    }

    #[test]
    fn mamba3_earlier_document_does_not_leak() {
        for conv in [false, true] {
            let m = layer(false, conv);
            let params = m.parameters();
            check_no_leak(&fwd_of(&m), &params, D_MODEL, D_CONV, 4, 5);
        }
    }

    #[test]
    fn mamba3_three_documents_with_one_token_document() {
        let m = layer(true, true);
        let params = m.parameters();
        let fwd = fwd_of(&m);
        check_packed_vs_separate(&fwd, &params, D_MODEL, D_CONV, &[3, 1, 4]);
        check_packed_vs_separate(&fwd, &params, D_MODEL, D_CONV, &[1, 2, 1, 2]);
    }

    #[test]
    fn mamba3_single_document_segments_equal_none() {
        for (rope, conv) in [(false, false), (true, true)] {
            let m = layer(rope, conv);
            let params = m.parameters();
            check_single_document_equals_none(&fwd_of(&m), &params, D_MODEL, D_CONV);
        }
    }
}
