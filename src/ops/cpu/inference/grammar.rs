//! CPU implementation of GrammarDfaOps.
//!
//! For each vocab token, walks the DFA from the current state using the token's
//! byte representation. If any byte leads to an invalid transition, the token's
//! logit is set to -inf.

use crate::error::Result;
use crate::ops::traits::inference::grammar::{DeviceGrammarDfa, GrammarDfaOps, INVALID_STATE};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl GrammarDfaOps<CpuRuntime> for CpuClient {
    fn grammar_dfa_mask_logits(
        &self,
        logits: &Tensor<CpuRuntime>,
        grammar: &DeviceGrammarDfa<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        let shape = logits.shape().to_vec();
        let vocab_size = grammar.vocab_size;

        // Read the transition table, accepting mask, vocab bytes, and offsets
        let tt_raw: Vec<f32> = grammar.transition_table.to_vec();
        let transition_table: Vec<i32> = tt_raw.iter().map(|&x| x as i32).collect();
        let am_raw: Vec<f32> = grammar.accepting_mask.to_vec();
        let accepting_mask: Vec<i32> = am_raw.iter().map(|&x| x as i32).collect();
        let vocab_bytes_raw: Vec<f32> = grammar.vocab_bytes.to_vec();
        let vocab_offsets_raw: Vec<f32> = grammar.vocab_offsets.to_vec();

        // Convert to proper types
        let vocab_bytes: Vec<u8> = vocab_bytes_raw.iter().map(|&x| x as u8).collect();
        let vocab_offsets: Vec<i32> = vocab_offsets_raw.iter().map(|&x| x as i32).collect();

        // Read logits and apply mask
        let mut logits_data: Vec<f32> = logits.to_vec();

        // Find the last-position logits (handle [1, seq_len, vocab] or [vocab])
        let total = logits_data.len();
        let offset = total.saturating_sub(vocab_size);
        let last_logits = &mut logits_data[offset..offset + vocab_size];

        let current_state = grammar.current_state as i32;
        let num_states = grammar.num_states;

        for token_id in 0..vocab_size {
            let byte_start = vocab_offsets[token_id] as usize;
            let byte_end = vocab_offsets[token_id + 1] as usize;

            let mut state = current_state;
            let mut valid = true;

            for &byte_val_raw in &vocab_bytes[byte_start..byte_end] {
                let byte_val = byte_val_raw as usize;
                let table_idx = (state as usize) * 256 + byte_val;

                if table_idx >= num_states * 256 {
                    valid = false;
                    break;
                }

                let next_state = transition_table[table_idx];
                if next_state == INVALID_STATE {
                    valid = false;
                    break;
                }
                state = next_state;
            }

            // Also check: if the token has zero bytes (empty), check if current state is accepting
            if valid && byte_start == byte_end {
                // Empty token: only allow if current state is accepting
                if current_state < 0 || (current_state as usize) >= accepting_mask.len() {
                    valid = false;
                } else {
                    valid = accepting_mask[current_state as usize] != 0;
                }
            }

            if !valid {
                last_logits[token_id] = f32::NEG_INFINITY;
            }
        }

        Ok(Tensor::from_slice(&logits_data, &shape, logits.device()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_grammar_dfa_mask_basic() {
        let device = CpuDevice::new();

        // 2 states: state 0 --'a'--> state 1 (accepting), all others invalid
        let num_states = 2;
        let mut transition_table = vec![INVALID_STATE as f32; num_states * 256];
        transition_table[b'a' as usize] = 1.0; // state 0 --'a'--> state 1

        let mut accepting_mask = vec![0.0f32; num_states];
        accepting_mask[1] = 1.0;

        // Vocab: token 0 = "a", token 1 = "b", token 2 = "ab"
        let vocab_bytes_data: Vec<f32> = vec![b'a' as f32, b'b' as f32, b'a' as f32, b'b' as f32];
        let vocab_offsets_data: Vec<f32> = vec![0.0, 1.0, 2.0, 4.0]; // 3 tokens + 1

        let transition_table_tensor =
            Tensor::from_slice(&transition_table, &[num_states * 256], &device);
        let accepting_mask_tensor = Tensor::from_slice(&accepting_mask, &[num_states], &device);
        let vocab_bytes_tensor =
            Tensor::from_slice(&vocab_bytes_data, &[vocab_bytes_data.len()], &device);
        let vocab_offsets_tensor =
            Tensor::from_slice(&vocab_offsets_data, &[vocab_offsets_data.len()], &device);

        let grammar = DeviceGrammarDfa {
            transition_table: transition_table_tensor,
            accepting_mask: accepting_mask_tensor,
            vocab_bytes: vocab_bytes_tensor,
            vocab_offsets: vocab_offsets_tensor,
            current_state: 0,
            num_states,
            vocab_size: 3,
        };

        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        let client = numr::runtime::cpu::CpuClient::new(device);
        let result = client.grammar_dfa_mask_logits(&logits, &grammar).unwrap();
        let result_data: Vec<f32> = result.to_vec();

        // Token "a" (id 0): state 0 --'a'--> state 1 → valid
        assert_eq!(result_data[0], 1.0);
        // Token "b" (id 1): state 0 --'b'--> INVALID → masked
        assert!(result_data[1].is_infinite() && result_data[1].is_sign_negative());
        // Token "ab" (id 2): state 0 --'a'--> state 1 --'b'--> INVALID → masked
        assert!(result_data[2].is_infinite() && result_data[2].is_sign_negative());
    }
}
