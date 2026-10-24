//! Mamba2 on packed rows: no state crosses a document start.

#[cfg(test)]
mod tests {
    use crate::model::mamba::mamba2::config::Mamba2Config;
    use crate::model::mamba::mamba2::layer::{Mamba2, Mamba2Weights};
    use crate::nn::segment_masks::SegmentMasks;
    use crate::nn::{Conv1d, Linear};
    use crate::test_utils::Lcg;
    use crate::test_utils::{
        check_no_leak, check_packed_vs_separate, check_single_document_equals_none, t64,
    };
    use numr::autograd::Var;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    const D_CONV: usize = 4;

    fn layer() -> Mamba2<CpuRuntime> {
        let config = Mamba2Config::new(8)
            .with_nheads(2)
            .with_d_state(3)
            .with_expand(2)
            .with_d_conv(D_CONV);
        let (d_inner, channels) = (config.d_inner(), config.conv_channels());
        let mut rng = Lcg(0xb22d_f002);
        let in_proj = Linear::new(
            t64(&mut rng, &[config.proj_dim(), config.d_model], 0.4),
            None,
            true,
        );
        let out_proj = Linear::new(t64(&mut rng, &[config.d_model, d_inner], 0.4), None, true);
        let conv1d = Conv1d::new(
            t64(&mut rng, &[channels, 1, D_CONV], 0.6),
            Some(t64(&mut rng, &[channels], 0.2)),
            1,
            PaddingMode::Custom(D_CONV - 1, 0, 0, 0),
            1,
            channels,
            true,
        );
        let weights = Mamba2Weights {
            in_proj,
            conv1d,
            out_proj,
            a_log: t64(&mut rng, &[config.nheads], 0.5),
            dt_bias: Some(t64(&mut rng, &[config.nheads], 0.3)),
            d_param: Some(t64(&mut rng, &[config.nheads], 0.5)),
            norm: None,
        };
        Mamba2::new(config, weights, true)
    }

    #[test]
    fn mamba2_packed_equals_separate_documents() {
        let m = layer();
        let params = m.parameters();
        let fwd = |c: &CpuClient, x: &Var<CpuRuntime>, s: Option<&SegmentMasks<CpuRuntime>>| {
            m.forward(c, x, s)
        };
        check_packed_vs_separate(&fwd, &params, 8, D_CONV, &[5, 4]);
    }

    #[test]
    fn mamba2_earlier_document_does_not_leak() {
        let m = layer();
        let params = m.parameters();
        let fwd = |c: &CpuClient, x: &Var<CpuRuntime>, s: Option<&SegmentMasks<CpuRuntime>>| {
            m.forward(c, x, s)
        };
        check_no_leak(&fwd, &params, 8, D_CONV, 4, 5);
    }

    #[test]
    fn mamba2_three_documents_with_one_token_document() {
        let m = layer();
        let params = m.parameters();
        let fwd = |c: &CpuClient, x: &Var<CpuRuntime>, s: Option<&SegmentMasks<CpuRuntime>>| {
            m.forward(c, x, s)
        };
        check_packed_vs_separate(&fwd, &params, 8, D_CONV, &[3, 1, 4]);
        check_packed_vs_separate(&fwd, &params, 8, D_CONV, &[1, 2, 1, 2]);
    }

    #[test]
    fn mamba2_single_document_segments_equal_none() {
        let m = layer();
        let params = m.parameters();
        let fwd = |c: &CpuClient, x: &Var<CpuRuntime>, s: Option<&SegmentMasks<CpuRuntime>>| {
            m.forward(c, x, s)
        };
        check_single_document_equals_none(&fwd, &params, 8, D_CONV);
    }
}
