// Fused optimizer kernels — F32 only (WebGPU is 32-bit by design)

struct OptimizerParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step_size: f32,
    momentum: f32,
    dampening: f32,
    nesterov: u32,
    has_buf: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read_write> state0: array<f32>;  // m or accum or momentum_buf
@group(0) @binding(3) var<storage, read_write> state1: array<f32>;  // v (for adamw/lamb)
@group(0) @binding(4) var<uniform> params: OptimizerParams;

@compute @workgroup_size(256)
fn fused_adamw_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }

    let gi = grad[idx];
    let mi = params.beta1 * state0[idx] + (1.0 - params.beta1) * gi;
    let vi = params.beta2 * state1[idx] + (1.0 - params.beta2) * gi * gi;

    let update = params.step_size * mi / (sqrt(vi) + params.eps);
    let decayed = param[idx] * (1.0 - params.lr * params.wd);

    param[idx] = decayed - update;
    state0[idx] = mi;
    state1[idx] = vi;
}

@compute @workgroup_size(256)
fn fused_sgd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }

    let gi = grad[idx];
    var grad_wd: f32;
    if (params.wd > 0.0) {
        grad_wd = gi + params.wd * param[idx];
    } else {
        grad_wd = gi;
    }

    var b: f32;
    if (params.momentum > 0.0) {
        if (params.has_buf != 0u) {
            b = params.momentum * state0[idx] + (1.0 - params.dampening) * grad_wd;
        } else {
            b = grad_wd;
        }
        state0[idx] = b;
    } else {
        b = grad_wd;
        state0[idx] = b;
    }

    var update: f32;
    if (params.nesterov != 0u && params.momentum > 0.0) {
        update = grad_wd + params.momentum * b;
    } else if (params.momentum > 0.0) {
        update = b;
    } else {
        update = grad_wd;
    }

    param[idx] = param[idx] - params.lr * update;
}

@compute @workgroup_size(256)
fn fused_adagrad_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }

    let gi = grad[idx];
    var grad_wd: f32;
    if (params.wd > 0.0) {
        grad_wd = gi + params.wd * param[idx];
    } else {
        grad_wd = gi;
    }

    let acc = state0[idx] + grad_wd * grad_wd;
    state0[idx] = acc;
    param[idx] = param[idx] - params.lr * grad_wd / (sqrt(acc) + params.eps);
}

@compute @workgroup_size(256)
fn fused_lamb_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }

    let gi = grad[idx];
    let mi = params.beta1 * state0[idx] + (1.0 - params.beta1) * gi;
    let vi = params.beta2 * state1[idx] + (1.0 - params.beta2) * gi * gi;

    state0[idx] = mi;
    state1[idx] = vi;

    // bias_corr1 stored in step_size, bias_corr2 stored in dampening (reusing uniform fields)
    let m_hat = mi / params.step_size;
    let v_hat = vi / params.dampening;
    let adam_update = m_hat / (sqrt(v_hat) + params.eps);

    // Store update in param buffer (caller reads it back)
    if (params.wd > 0.0) {
        param[idx] = adam_update + params.wd * param[idx];
    } else {
        param[idx] = adam_update;
    }
}
