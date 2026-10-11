// Repetition, frequency and presence penalties, applied to logits in place.
// F32 only (WebGPU limitation).
//
// One invocation per unique token. Ids are unique, so each logit has one
// writer. `token_ids` is an I64 tensor read as u32 pairs, low word first.

struct PenaltyParams {
    // Element offset of the logits view inside its buffer.
    logits_offset: u32,
    vocab_size: u32,
    num_unique: u32,
    _pad0: u32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> token_counts: array<i32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> params: PenaltyParams;

@compute @workgroup_size(256)
fn apply_sampling_penalties(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.num_unique) {
        return;
    }
    let id = token_ids[2u * j];
    if (id >= params.vocab_size) {
        return;
    }
    let slot = params.logits_offset + id;
    var logit = logits[slot];
    if (params.repeat_penalty != 1.0) {
        if (logit > 0.0) {
            logit = logit / params.repeat_penalty;
        } else {
            logit = logit * params.repeat_penalty;
        }
    }
    logit = logit - params.frequency_penalty * f32(token_counts[j]);
    logit = logit - params.presence_penalty;
    logits[slot] = logit;
}
