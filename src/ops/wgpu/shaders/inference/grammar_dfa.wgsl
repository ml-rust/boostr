// Grammar DFA logit mask. F32 only (WebGPU limitation).
//
// One invocation per vocabulary token. The rules match the CPU and CUDA backends:
// - An EOS token is allowed iff the current state accepts. Its bytes are never walked.
// - A non-EOS token is allowed iff each of its bytes has a transition.
// - A non-EOS token with zero bytes is never allowed.
// - No token is allowed from a state outside the table.
// A forbidden token gets a logit of -inf.
//
// The DFA tables arrive as F32 tensors holding exact integer values, the
// layout every backend shares. -1 marks a forbidden byte.

struct GrammarParams {
    // Element offset of the last position's logits inside `logits`.
    logits_offset: u32,
    vocab_size: u32,
    current_state: u32,
    num_states: u32,
    num_eos: u32,
    // Real byte count in `vocab_bytes`, without the padding element.
    vocab_bytes_len: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> transition_table: array<f32>;
@group(0) @binding(1) var<storage, read> accepting_mask: array<f32>;
@group(0) @binding(2) var<storage, read> vocab_bytes: array<f32>;
@group(0) @binding(3) var<storage, read> vocab_offsets: array<f32>;
@group(0) @binding(4) var<storage, read> eos_tokens: array<f32>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;
@group(0) @binding(6) var<uniform> params: GrammarParams;

fn neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

// True when the grammar allows `token` from the current state.
fn token_allowed(token: u32) -> bool {
    let cur = params.current_state;
    if (cur >= params.num_states || cur >= arrayLength(&accepting_mask)) {
        return false;
    }
    let token_f = f32(token);
    let num_eos = min(params.num_eos, arrayLength(&eos_tokens));
    for (var i = 0u; i < num_eos; i = i + 1u) {
        if (eos_tokens[i] == token_f) {
            return accepting_mask[cur] != 0.0;
        }
    }
    if (token + 1u >= arrayLength(&vocab_offsets)) {
        return false;
    }
    let start_f = vocab_offsets[token];
    let end_f = vocab_offsets[token + 1u];
    let bytes_len = min(params.vocab_bytes_len, arrayLength(&vocab_bytes));
    // A zero-length token makes no progress, so it is never allowed.
    if (!(start_f >= 0.0 && end_f > start_f && end_f <= f32(bytes_len))) {
        return false;
    }
    let table_len = arrayLength(&transition_table);
    let states_f = f32(params.num_states);
    var state = cur;
    for (var b = u32(start_f); b < u32(end_f); b = b + 1u) {
        let byte_f = vocab_bytes[b];
        if (!(byte_f >= 0.0 && byte_f <= 255.0)) {
            return false;
        }
        let idx = state * 256u + u32(byte_f);
        if (idx >= table_len) {
            return false;
        }
        let next = transition_table[idx];
        if (!(next >= 0.0 && next < states_f)) {
            return false;
        }
        state = u32(next);
    }
    return true;
}

@compute @workgroup_size(256)
fn grammar_dfa_mask_logits(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    if (token >= params.vocab_size) {
        return;
    }
    if (!token_allowed(token)) {
        logits[params.logits_offset + token] = neg_inf();
    }
}
