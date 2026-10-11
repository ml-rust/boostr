// Fused logits-to-token kernel. F32 only (WebGPU limitation).
//
// One workgroup of 256 invocations runs the whole pipeline:
//   copy the last position's logits into `probs` → penalties →
//   greedy argmax, or temperature → softmax → top-k → min-p → top-p →
//   multinomial draw.
// `sample_token` dispatches this same entry point with no penalties.
//
// I64 tensors reach WGSL as pairs of u32 words, low word first. Token ids
// fit in the low word. The output token is written as `[id, 0]`.

struct LogitsToTokenParams {
    // Element offset of the last position's logits inside `logits`.
    logits_offset: u32,
    vocab_size: u32,
    num_unique: u32,
    top_k: u32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    // 1 = argmax, 0 = stochastic chain.
    greedy: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read> token_counts: array<i32>;
@group(0) @binding(3) var<storage, read> random_val: array<f32>;
@group(0) @binding(4) var<storage, read_write> probs: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_token: array<u32>;
@group(0) @binding(6) var<uniform> params: LogitsToTokenParams;

const WG: u32 = 256u;
const NEG_BIG: f32 = -3.402823e38;

var<workgroup> red_val: array<f32, 256>;
var<workgroup> red_idx: array<u32, 256>;
var<workgroup> bcast: f32;

// Workgroup max of `v`, returned to every invocation as a uniform value.
fn reduce_max(tid: u32, v: f32) -> f32 {
    red_val[tid] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            red_val[tid] = max(red_val[tid], red_val[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        bcast = red_val[0];
    }
    return workgroupUniformLoad(&bcast);
}

// Workgroup sum of `v`, returned to every invocation as a uniform value.
fn reduce_sum(tid: u32, v: f32) -> f32 {
    red_val[tid] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            red_val[tid] = red_val[tid] + red_val[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        bcast = red_val[0];
    }
    return workgroupUniformLoad(&bcast);
}

fn apply_penalty(logit: f32, count: i32) -> f32 {
    var out = logit;
    if (params.repeat_penalty != 1.0) {
        if (out > 0.0) {
            out = out / params.repeat_penalty;
        } else {
            out = out * params.repeat_penalty;
        }
    }
    out = out - params.frequency_penalty * f32(count);
    out = out - params.presence_penalty;
    return out;
}

@compute @workgroup_size(256)
fn logits_to_token(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.vocab_size;

    // Step 1: copy the last position's logits.
    for (var i = tid; i < n; i = i + WG) {
        probs[i] = logits[params.logits_offset + i];
    }
    storageBarrier();

    // Step 2: penalties. Ids are unique, so each slot has one writer.
    for (var j = tid; j < params.num_unique; j = j + WG) {
        let id = token_ids[2u * j];
        if (id < n) {
            probs[id] = apply_penalty(probs[id], token_counts[j]);
        }
    }
    storageBarrier();

    // Step 3: greedy argmax. The lowest index wins a tie, as on the CPU.
    if (params.greedy == 1u) {
        var best = NEG_BIG;
        var best_i = 0u;
        var seen = false;
        for (var i = tid; i < n; i = i + WG) {
            let v = probs[i];
            if (!seen || v > best) {
                best = v;
                best_i = i;
                seen = true;
            }
        }
        if (!seen) {
            best_i = 0xffffffffu;
        }
        red_val[tid] = best;
        red_idx[tid] = best_i;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s = s >> 1u) {
            if (tid < s) {
                let ov = red_val[tid + s];
                let oi = red_idx[tid + s];
                let mv = red_val[tid];
                let mi = red_idx[tid];
                let take = oi != 0xffffffffu
                    && (mi == 0xffffffffu || ov > mv || (ov == mv && oi < mi));
                if (take) {
                    red_val[tid] = ov;
                    red_idx[tid] = oi;
                }
            }
            workgroupBarrier();
        }
        if (tid == 0u) {
            out_token[0] = red_idx[0];
            out_token[1] = 0u;
        }
        return;
    }

    // Step 4: temperature, then the softmax max.
    var local_max = NEG_BIG;
    for (var i = tid; i < n; i = i + WG) {
        var v = probs[i];
        if (params.temperature != 1.0) {
            v = v / params.temperature;
        }
        probs[i] = v;
        local_max = max(local_max, v);
    }
    let global_max = reduce_max(tid, local_max);

    // Step 5: exponentiate and normalize.
    var local_sum = 0.0;
    for (var i = tid; i < n; i = i + WG) {
        let e = exp(probs[i] - global_max);
        probs[i] = e;
        local_sum = local_sum + e;
    }
    let total = reduce_sum(tid, local_sum);
    var local_pmax = 0.0;
    for (var i = tid; i < n; i = i + WG) {
        let p = probs[i] / total;
        probs[i] = p;
        local_pmax = max(local_pmax, p);
    }
    let max_prob = reduce_max(tid, local_pmax);

    // Step 6: top-k threshold by bisection on the probability value.
    var topk_threshold = 0.0;
    if (params.top_k > 0u && params.top_k < n) {
        var lo = 0.0;
        var hi = max_prob;
        for (var iter = 0u; iter < 32u; iter = iter + 1u) {
            let mid = (lo + hi) * 0.5;
            var count = 0.0;
            for (var i = tid; i < n; i = i + WG) {
                if (probs[i] >= mid) {
                    count = count + 1.0;
                }
            }
            let total_count = reduce_sum(tid, count);
            if (total_count > f32(params.top_k)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // `hi` keeps at most k tokens. `lo` always keeps more than k.
        topk_threshold = hi;
    }

    // Step 7: min-p threshold, combined with top-k.
    var threshold = topk_threshold;
    if (params.min_p > 0.0) {
        threshold = max(threshold, params.min_p * max_prob);
    }
    var local_kept = 0.0;
    for (var i = tid; i < n; i = i + WG) {
        if (probs[i] < threshold) {
            probs[i] = 0.0;
        } else {
            local_kept = local_kept + probs[i];
        }
    }
    var kept = reduce_sum(tid, local_kept);

    // Step 8: top-p threshold by bisection on the probability value.
    if (params.top_p < 1.0 && kept > 0.0) {
        let mass_goal = params.top_p * kept;
        var lo = 0.0;
        var hi = max_prob;
        for (var iter = 0u; iter < 32u; iter = iter + 1u) {
            let mid = (lo + hi) * 0.5;
            var above = 0.0;
            for (var i = tid; i < n; i = i + WG) {
                if (probs[i] >= mid) {
                    above = above + probs[i];
                }
            }
            let total_above = reduce_sum(tid, above);
            if (total_above > mass_goal) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        var local_new = 0.0;
        for (var i = tid; i < n; i = i + WG) {
            if (probs[i] < lo) {
                probs[i] = 0.0;
            } else {
                local_new = local_new + probs[i];
            }
        }
        kept = reduce_sum(tid, local_new);
    }
    storageBarrier();

    // Step 9: multinomial draw. One invocation scans the kept mass.
    if (tid == 0u) {
        let draw_point = random_val[0] * kept;
        var cumsum = 0.0;
        // Rounding at the tail falls back to the last kept token.
        var sampled = 0u;
        for (var i = 0u; i < n; i = i + 1u) {
            let p = probs[i];
            if (p > 0.0) {
                sampled = i;
            }
            cumsum = cumsum + p;
            if (cumsum > draw_point) {
                break;
            }
        }
        out_token[0] = sampled;
        out_token[1] = 0u;
    }
}
