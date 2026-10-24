//! Mamba1 on packed rows: no state crosses a document start.

#[cfg(test)]
mod tests {
    use crate::model::mamba::mamba1::config::Mamba1Config;
    use crate::model::mamba::mamba1::layer::{Mamba1, Mamba1Weights};
    use crate::model::mamba::packed_support::{
        check_no_leak, check_packed_vs_separate, check_single_document_equals_none, t64,
    };
    use crate::nn::segment_masks::SegmentMasks;
    use crate::nn::{Conv1d, Linear};
    use crate::test_utils::Lcg;
    use numr::autograd::Var;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    const D_CONV: usize = 4;

    fn layer() -> Mamba1<CpuRuntime> {
        let config = Mamba1Config::new(4)
            .with_expand(2)
            .with_d_state(3)
            .with_d_conv(D_CONV)
            .with_hidden_state_clamp(None);
        let (d_inner, d_state) = (config.d_inner(), config.d_state);
        let mut rng = Lcg(0xa11c_e001);
        let mut linear = |rows: usize, cols: usize, bias: bool| {
            let w = t64(&mut rng, &[rows, cols], 0.4);
            let b = bias.then(|| t64(&mut rng, &[rows], 0.2));
            Linear::new(w, b, true)
        };
        let in_proj = linear(config.in_proj_dim(), config.d_model, false);
        let x_proj = linear(config.x_proj_dim(), d_inner, false);
        let dt_proj = linear(d_inner, d_inner, true);
        let out_proj = linear(config.d_model, d_inner, false);
        let conv1d = Conv1d::new(
            t64(&mut rng, &[d_inner, 1, D_CONV], 0.6),
            Some(t64(&mut rng, &[d_inner], 0.2)),
            1,
            PaddingMode::Custom(D_CONV - 1, 0, 0, 0),
            1,
            d_inner,
            true,
        );
        let weights = Mamba1Weights {
            in_proj,
            conv1d,
            x_proj,
            dt_proj,
            out_proj,
            a_log: t64(&mut rng, &[d_inner, d_state], 0.5),
            d_param: Some(t64(&mut rng, &[d_inner], 0.5)),
        };
        Mamba1::new(config, weights, true)
    }

    #[test]
    fn mamba1_packed_equals_separate_documents() {
        let m = layer();
        let params = m.parameters();
        let fwd = |c: &CpuClient, x: &Var<CpuRuntime>, s: Option<&SegmentMasks<CpuRuntime>>| {
            m.forward(c, x, s)
        };
        check_packed_vs_separate(&fwd, &params, 4, D_CONV, &[5, 4]);
    }

    #[test]
    fn mamba1_earlier_document_does_not_leak() {
        let m = layer();
        let params = m.parameters();
        let fwd = |c: &CpuClient, x: &Var<CpuRuntime>, s: Option<&SegmentMasks<CpuRuntime>>| {
            m.forward(c, x, s)
        };
        check_no_leak(&fwd, &params, 4, D_CONV, 4, 5);
    }

    #[test]
    fn mamba1_three_documents_with_one_token_document() {
        let m = layer();
        let params = m.parameters();
        let fwd = |c: &CpuClient, x: &Var<CpuRuntime>, s: Option<&SegmentMasks<CpuRuntime>>| {
            m.forward(c, x, s)
        };
        check_packed_vs_separate(&fwd, &params, 4, D_CONV, &[3, 1, 4]);
        check_packed_vs_separate(&fwd, &params, 4, D_CONV, &[1, 2, 1, 2]);
    }

    #[test]
    fn mamba1_single_document_segments_equal_none() {
        let m = layer();
        let params = m.parameters();
        let fwd = |c: &CpuClient, x: &Var<CpuRuntime>, s: Option<&SegmentMasks<CpuRuntime>>| {
            m.forward(c, x, s)
        };
        check_single_document_equals_none(&fwd, &params, 4, D_CONV);
    }
}
