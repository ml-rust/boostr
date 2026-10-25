//! The router temperature and shared-expert scale that oxidizr trains a
//! [`MoeLayer`](super::MoeLayer) with. Inference reads the same values, so
//! a served layer routes and mixes exactly as it trained.

/// Softmax temperature of the trained router.
pub const MOE_ROUTER_TEMPERATURE: f32 = 2.0;

/// Weight of the shared expert's output: `1 / (top_k + 1)`, the average of
/// the shared expert with the `top_k` routed ones.
pub fn moe_shared_expert_scale(top_k: usize) -> f32 {
    1.0 / (top_k as f32 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_expert_scale_averages_with_the_routed_experts() {
        assert_eq!(moe_shared_expert_scale(1), 0.5);
        assert_eq!(moe_shared_expert_scale(3), 0.25);
    }
}
