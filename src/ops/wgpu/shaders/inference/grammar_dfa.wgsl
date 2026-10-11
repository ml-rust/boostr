// Grammar DFA logit mask. F32 only (WebGPU limitation).
//
// One invocation per vocabulary token. Each walks the DFA from the current
// state over the token's bytes. A token that hits an invalid transition, or
// leaves the table, gets a logit of -inf. An empty token is valid only in an
// accepting state.
//
// The DFA tables arrive as F32 tensors holding integer values, the layout
// every backend shares.

struct GrammarParams {
    // Element offset of the last position's logits inside `logits`.
    logits_offset: u32,
    vocab_size: u32,
    current_state: i32,
    num_states: u32,
}

@group(0) @binding(0) var<storage, read> transition_table: array<f32>;
@group(0) @binding(1) var<storage, read> accepting_mask: array<f32>;
@group(0) @binding(2) var<storage, read> vocab_bytes: array<f32>;
@group(0) @binding(3) var<storage, read> vocab_offsets: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<uniform> params: GrammarParams;

const INVALID_STATE: i32 = -1;

fn neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

@compute @workgroup_size(256)
fn grammar_dfa_mask_logits(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    if (token >= params.vocab_size) {
        return;
    }
    let start = u32(vocab_offsets[token]);
    let end = u32(vocab_offsets[token + 1u]);
    let table_len = params.num_states * 256u;

    var state = params.current_state;
    var valid = true;
    for (var b = start; b < end; b = b + 1u) {
        let byte_val = u32(vocab_bytes[b]);
        let idx = u32(state) * 256u + byte_val;
        if (state < 0 || idx >= table_len) {
            valid = false;
            break;
        }
        let next = i32(transition_table[idx]);
        if (next == INVALID_STATE) {
            valid = false;
            break;
        }
        state = next;
    }

    if (valid && start == end) {
        let cur = params.current_state;
        if (cur < 0 || u32(cur) >= params.num_states) {
            valid = false;
        } else {
            valid = accepting_mask[u32(cur)] != 0.0;
        }
    }

    if (!valid) {
        logits[params.logits_offset + token] = neg_inf();
    }
}
