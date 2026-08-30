//! The two TCF entry points, and the bindings they declare around the shared
//! decoder.
//!
//! Each generator emits its own uniform block, storage bindings and workgroup
//! shape, then pastes [`super::decoder::decoder`] verbatim, so a kernel owns
//! its dispatch shape and nothing else.

use crate::quant::tcf::TCF_TILE;

use super::decoder::decoder;

/// Tiles one dequantization workgroup owns: one super-block, so a workgroup's
/// group parameters come from one bit-packed sub-plane run.
pub const DEQUANT_TILES_PER_GROUP: u32 = 4;
/// Threads per dequantization workgroup: one per element of its four tiles.
pub const DEQUANT_WORKGROUP: u32 = DEQUANT_TILES_PER_GROUP * TCF_TILE as u32;
/// Output tile edge of the matmul kernel, matching the other WebGPU quantized
/// matmul shaders. Both matmul kernels dispatch a `MATMUL_TILE` square of
/// output elements per workgroup, so a caller picks between them without
/// recomputing the grid.
pub const MATMUL_TILE: u32 = 16;
/// Largest group count one tile can carry: group 16 over a 64-element tile.
const MAX_GROUPS_PER_TILE: u32 = 4;
/// Workgroup slots holding one super-block's resolved group parameters.
const DEQUANT_PARAM_SLOTS: u32 = DEQUANT_TILES_PER_GROUP * MAX_GROUPS_PER_TILE;

/// Invocations in a tiled matmul workgroup: one per output element of its
/// `MATMUL_TILE` square.
const MATMUL_TILED_THREADS: u32 = MATMUL_TILE * MATMUL_TILE;
/// Row stride of a staged 64-element run, padded by one f32 so the sixteen
/// invocations of a row that read the same element index of sixteen different
/// staged rows land on sixteen distinct memory banks. This is the same padding
/// the CUDA GEMM applies for the same reason.
const MATMUL_TILED_STRIDE: u32 = TCF_TILE as u32 + 1;
/// Length of one staged f32 plane: `MATMUL_TILE` rows at the padded stride.
const MATMUL_TILED_STAGE: u32 = MATMUL_TILE * MATMUL_TILED_STRIDE;
/// Staged values each invocation writes per K-tile and per plane:
/// `MATMUL_TILE * TCF_TILE / MATMUL_TILED_THREADS`.
const MATMUL_TILED_ROUNDS: u32 = MATMUL_TILE * TCF_TILE as u32 / MATMUL_TILED_THREADS;
/// Workgroup slots holding one K-tile's resolved group parameters, one per
/// (staged weight row, group).
const MATMUL_TILED_PARAM_SLOTS: u32 = MATMUL_TILE * MAX_GROUPS_PER_TILE;

/// Activation rows at or above which [`generate_tcf_matmul_tiled_shader`] is
/// dispatched instead of [`generate_tcf_matmul_shader`].
///
/// MEASURED, `q_proj` 2048x2048 on an RTX 3060 via Vulkan, minimum nanoseconds,
/// per-element kernel against tiled kernel at the same M:
///
/// | M | Q8 per-elem | Q8 tiled | Q6 per-elem | Q6 tiled |
/// |---|-------------|----------|-------------|----------|
/// | 1 |       457us |    412us |       582us |    458us |
/// | 4 |       469us |    409us |       662us |    456us |
/// | 8 |       526us |    403us |       923us |    456us |
///
/// The expectation was a crossover — that staging traffic and three barriers
/// per K-tile could not pay for themselves until a workgroup had enough
/// activation rows to fill its tile. There is no crossover on this adapter.
/// The tiled kernel wins at EVERY M including 1, because the decode it
/// amortizes costs far more than the staging does, and a partly filled tile
/// still decodes each weight once instead of `MATMUL_TILE` times.
///
/// So 1: the per-element kernel is currently never dispatched. It is kept, and
/// this stays a constant rather than becoming an unconditional call, because
/// the result above is from ONE desktop adapter. A mobile adapter has less
/// workgroup memory bandwidth and lower occupancy, which is exactly where a
/// staging cost could start to matter, and this crate targets those. Re-run
/// the sweep there before assuming the answer transfers.
pub const MATMUL_TILED_MIN_M: u32 = 1;

/// Entry point name of the dequantization shader.
pub const DEQUANT_ENTRY: &str = "tcf_dequant_f32";
/// Entry point name of the fused quantized matmul shader.
pub const MATMUL_ENTRY: &str = "tcf_quant_matmul_f32";
/// Entry point name of the workgroup-tiled fused quantized matmul shader.
pub const MATMUL_TILED_ENTRY: &str = "tcf_quant_matmul_tiled_f32";

/// The uniform block and storage bindings both TCF shaders declare.
///
/// One params struct serves both kernels so [`super::decoder::decoder`] reads
/// the same field names in either. `m`, `k` and `n` are zero in a
/// dequantization dispatch and unread there, and `zero_barrier` is always zero
/// in both.
fn bindings(payload_binding: u32, extra: &str) -> String {
    format!(
        r#"
struct TcfParams {{
    tiles: u32,
    code_high_off: u32,
    scale_off: u32,
    min_off: u32,
    super_off: u32,
    super_min_off: u32,
    bits: u32,
    group: u32,
    groups_per_tile: u32,
    symmetric: u32,
    scale_form: u32,
    sub_block_bytes: u32,
    m: u32,
    k: u32,
    n: u32,
    zero_barrier: u32,
}}
{extra}
@group(0) @binding({payload_binding}) var<storage, read_write> payload: array<u32>;
"#
    )
}

/// Dequantize a whole TCF payload to f32, in logical row-major order.
///
/// One workgroup per super-block of four tiles, 256 invocations, one element
/// each. The first `4 * groups_per_tile` invocations resolve the super-block's
/// group parameters into workgroup memory first, so the scale planes are read
/// once per group rather than once per weight — the same two-phase shape the
/// CUDA kernel uses.
#[must_use]
pub fn generate_tcf_dequant_shader() -> String {
    format!(
        r#"// Dequantize a TCF native quantized payload to f32
{bindings}
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: TcfParams;
{decoder}

const TCF_DEQUANT_TILES: u32 = {tiles_per_group}u;

var<workgroup> tcf_group_scale: array<f32, {slots}u>;
var<workgroup> tcf_group_min: array<f32, {slots}u>;

@compute @workgroup_size({workgroup})
fn {entry}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let base_tile = wid.x * TCF_DEQUANT_TILES;
    let groups_per_tile = params.groups_per_tile;
    let slots = TCF_DEQUANT_TILES * groups_per_tile;

    // Phase 1: resolve this super-block's group parameters once.
    if (lid.x < slots) {{
        let param_tile = base_tile + lid.x / groups_per_tile;
        if (param_tile < params.tiles) {{
            let values = tcf_group_values(param_tile, lid.x % groups_per_tile);
            tcf_group_scale[lid.x] = values.scale;
            tcf_group_min[lid.x] = values.min_value;
        }}
    }}
    workgroupBarrier();

    // Phase 2: stream the code plane, one element per invocation.
    let local_tile = lid.x / TCF_TILE;
    let tile = base_tile + local_tile;
    if (tile >= params.tiles) {{
        return;
    }}
    let e = lid.x % TCF_TILE;
    let slot = local_tile * groups_per_tile + e / params.group;
    let code = tcf_code(tile, e);
    output[tile * TCF_TILE + e] = tcf_value(code, tcf_group_scale[slot], tcf_group_min[slot]);
}}
"#,
        bindings = bindings(0, ""),
        decoder = decoder(),
        tiles_per_group = DEQUANT_TILES_PER_GROUP,
        slots = DEQUANT_PARAM_SLOTS,
        workgroup = DEQUANT_WORKGROUP,
        entry = DEQUANT_ENTRY,
    )
}

/// `activation [M, K] x weight [N, K]^T -> output [M, N]`, one invocation per
/// output element, matching the other WebGPU quantized matmul shaders.
///
/// The weight is never materialized as f32: each invocation walks its own
/// weight row tile by tile, resolves each tile's group parameters once, then
/// decodes that group's codes straight into the accumulator.
#[must_use]
pub fn generate_tcf_matmul_shader() -> String {
    format!(
        r#"// Fused matmul: activation [M,K] x TCF weight [N,K] -> output [M,N]
{bindings}
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: TcfParams;
{decoder}

@compute @workgroup_size({tile_edge}, {tile_edge})
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = gid.x;
    let m = gid.y;
    if (m >= params.m || n >= params.n) {{
        return;
    }}

    let tiles_per_row = params.k / TCF_TILE;
    let row_first_tile = n * tiles_per_row;
    let act_row_base = m * params.k;
    let groups_per_tile = params.groups_per_tile;

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < tiles_per_row; t = t + 1u) {{
        let tile = row_first_tile + t;
        let k_base = act_row_base + t * TCF_TILE;
        for (var g: u32 = 0u; g < groups_per_tile; g = g + 1u) {{
            let values = tcf_group_values(tile, g);
            let first = g * params.group;
            for (var i: u32 = 0u; i < params.group; i = i + 1u) {{
                let e = first + i;
                let w = tcf_value(tcf_code(tile, e), values.scale, values.min_value);
                acc = acc + activation[k_base + e] * w;
            }}
        }}
    }}

    output[m * params.n + n] = acc;
}}
"#,
        bindings = bindings(
            1,
            "\n@group(0) @binding(0) var<storage, read_write> activation: array<f32>;"
        ),
        decoder = decoder(),
        tile_edge = MATMUL_TILE,
        entry = MATMUL_ENTRY,
    )
}

/// `activation [M, K] x weight [N, K]^T -> output [M, N]`, workgroup-tiled:
/// a `MATMUL_TILE` square of output elements per workgroup, one invocation
/// each, with the weights decoded into workgroup memory once per K-tile.
///
/// # Why a second matmul kernel exists
///
/// [`generate_tcf_matmul_shader`] gives every invocation its own weight row and
/// lets it decode every element itself, so the sixteen invocations of a
/// workgroup that share a weight row decode that row sixteen times. A TCF
/// decode is not a table lookup — it is two code-plane reads for a 6-bit
/// encoding, a group-parameter resolution, and a two-level integer division —
/// so paying for it once per OUTPUT element rather than once per weight element
/// is what made this backend lose large-batch prefill to a GGUF block layout,
/// while winning dequantization outright.
///
/// This kernel is the CUDA `tcf_gemm_f32` shape: per K-tile the workgroup
/// resolves its sixteen weight rows' group parameters, decodes those rows'
/// sixty-four codes into workgroup memory, stages the sixteen matching
/// activation rows beside them, and only then accumulates. A weight element is
/// decoded once per workgroup.
///
/// # Workgroup memory budget
///
/// Two staged planes of `MATMUL_TILE` rows at a padded stride of `TCF_TILE + 1`
/// f32, plus one scale and one minimum per (staged weight row, group):
///
/// - weights: `16 * 65 * 4` = 4160 bytes
/// - activations: `16 * 65 * 4` = 4160 bytes
/// - group scales: `16 * 4 * 4` = 256 bytes
/// - group minima: `16 * 4 * 4` = 256 bytes
///
/// 8832 bytes total, against the 16384 WebGPU guarantees as
/// `maxComputeWorkgroupStorageSize`, so this kernel compiles on any conformant
/// adapter rather than only on a generous one.
///
/// # Why every barrier is uniformly reached
///
/// WGSL Section 14.7 makes a barrier in non-uniform control flow undefined
/// behaviour, so no invocation may leave before the last one. The kernel
/// therefore has NO early `return`: an invocation whose output element lies
/// past `M` or `N` runs the whole K loop and is masked at the single store.
/// The loop's trip count is `params.k / TCF_TILE`, read from the uniform block,
/// so it is workgroup-uniform by construction, and the three barriers sit
/// directly in the loop body rather than inside any `if`. A boundary tile's
/// out-of-range rows stage `0.0` instead of skipping their store, so the
/// accumulation reads defined values and contributes nothing.
#[must_use]
pub fn generate_tcf_matmul_tiled_shader() -> String {
    format!(
        r#"// Workgroup-tiled matmul: activation [M,K] x TCF weight [N,K] -> output [M,N]
{bindings}
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: TcfParams;
{decoder}

const TCF_GEMM_TILE: u32 = {tile_edge}u;
const TCF_GEMM_THREADS: u32 = {threads}u;
const TCF_GEMM_STRIDE: u32 = {stride}u;
const TCF_GEMM_ROUNDS: u32 = {rounds}u;

// 4160 + 4160 + 256 + 256 = 8832 bytes, inside the 16384-byte WebGPU minimum.
var<workgroup> tcf_stage_weight: array<f32, {stage}u>;
var<workgroup> tcf_stage_act: array<f32, {stage}u>;
var<workgroup> tcf_tile_scale: array<f32, {slots}u>;
var<workgroup> tcf_tile_min: array<f32, {slots}u>;

@compute @workgroup_size({tile_edge}, {tile_edge})
fn {entry}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    // No early return: every invocation below must reach every barrier, so a
    // workgroup overhanging the output is masked at the store instead.
    let n0 = wid.x * TCF_GEMM_TILE;
    let m0 = wid.y * TCF_GEMM_TILE;
    let n = n0 + lid.x;
    let m = m0 + lid.y;
    let tid = lid.y * TCF_GEMM_TILE + lid.x;

    // Uniform: read from the uniform block, so the loop bound and therefore
    // every barrier inside the loop is workgroup-uniform.
    let tiles_per_row = params.k / TCF_TILE;
    let groups_per_tile = params.groups_per_tile;
    let slots = TCF_GEMM_TILE * groups_per_tile;

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < tiles_per_row; t = t + 1u) {{
        // Phase 1: resolve this K-tile's group parameters, once per (row,
        // group) rather than once per weight.
        if (tid < slots) {{
            let param_row = tid / groups_per_tile;
            var scale: f32 = 0.0;
            var min_value: f32 = 0.0;
            if (n0 + param_row < params.n) {{
                let tile = (n0 + param_row) * tiles_per_row + t;
                let values = tcf_group_values(tile, tid % groups_per_tile);
                scale = values.scale;
                min_value = values.min_value;
            }}
            tcf_tile_scale[tid] = scale;
            tcf_tile_min[tid] = min_value;
        }}
        workgroupBarrier();

        // Phase 2: decode 16 weight rows x 64 elements once, and stage the 16
        // matching activation rows beside them. TCF_GEMM_ROUNDS values per
        // invocation per plane.
        for (var i: u32 = 0u; i < TCF_GEMM_ROUNDS; i = i + 1u) {{
            let index = tid + i * TCF_GEMM_THREADS;
            let row = index / TCF_TILE;
            let e = index % TCF_TILE;

            var w: f32 = 0.0;
            if (n0 + row < params.n) {{
                let tile = (n0 + row) * tiles_per_row + t;
                let slot = row * groups_per_tile + e / params.group;
                w = tcf_value(tcf_code(tile, e), tcf_tile_scale[slot], tcf_tile_min[slot]);
            }}
            tcf_stage_weight[row * TCF_GEMM_STRIDE + e] = w;

            var a: f32 = 0.0;
            if (m0 + row < params.m) {{
                a = activation[(m0 + row) * params.k + t * TCF_TILE + e];
            }}
            tcf_stage_act[row * TCF_GEMM_STRIDE + e] = a;
        }}
        workgroupBarrier();

        // Phase 3: consume. Element order matches the per-element kernel's, so
        // the two accumulate the same decoded values in the same order.
        for (var e: u32 = 0u; e < TCF_TILE; e = e + 1u) {{
            acc = acc + tcf_stage_act[lid.y * TCF_GEMM_STRIDE + e]
                      * tcf_stage_weight[lid.x * TCF_GEMM_STRIDE + e];
        }}
        // Read-before-overwrite: the next K-tile's phase 2 rewrites both
        // staged planes, so no invocation may start it while another is still
        // consuming this one.
        workgroupBarrier();
    }}

    if (m < params.m && n < params.n) {{
        output[m * params.n + n] = acc;
    }}
}}
"#,
        bindings = bindings(
            1,
            "\n@group(0) @binding(0) var<storage, read_write> activation: array<f32>;"
        ),
        decoder = decoder(),
        tile_edge = MATMUL_TILE,
        threads = MATMUL_TILED_THREADS,
        stride = MATMUL_TILED_STRIDE,
        rounds = MATMUL_TILED_ROUNDS,
        stage = MATMUL_TILED_STAGE,
        slots = MATMUL_TILED_PARAM_SLOTS,
        entry = MATMUL_TILED_ENTRY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every generated entry point. A property asserted below holds for the
    /// whole family, so adding a kernel without adding it here is the one way
    /// the family can drift.
    fn all_shaders() -> [String; 3] {
        [
            generate_tcf_dequant_shader(),
            generate_tcf_matmul_shader(),
            generate_tcf_matmul_tiled_shader(),
        ]
    }

    /// The decoder is generated once and pasted into every shader, so the
    /// kernels cannot drift on a bit position. MIGRATION.md Section 4.5.3
    /// forbids a second copy of a bit position, and a Q6_K release once shipped
    /// wrong because one existed.
    #[test]
    fn both_shaders_carry_the_same_decoder() {
        let decoder = decoder();
        for source in all_shaders() {
            assert!(source.contains(&decoder));
        }
    }

    /// The binary16 decode is written once, not per call site.
    #[test]
    fn the_binary16_decode_is_declared_once_per_shader() {
        for source in all_shaders() {
            assert_eq!(source.matches("fn tcf_binary16(").count(), 1);
            assert_eq!(source.matches("fn tcf_binary16_parts(").count(), 1);
        }
    }

    /// One helper divides every two-level scale and minimum, so no divisor can
    /// keep a float divide that WGSL specifies at 2.5 ULP.
    #[test]
    fn every_two_level_divisor_goes_through_the_integer_quotient() {
        for source in all_shaders() {
            assert_eq!(source.matches("fn tcf_scaled_quotient(").count(), 1);
            assert_eq!(source.matches("tcf_scaled_quotient(\n").count(), 3);
            // The only remaining `/` on a level count is the helper's own
            // fallback for a non-finite super-value.
            assert!(!source.contains("/ TCF_SUB_SCALE_LEVELS_U8"));
            assert!(!source.contains("/ TCF_SUB_SCALE_LEVELS_U6"));
            assert!(!source.contains("/ TCF_SUB_MIN_LEVELS_U6"));
        }
    }

    /// Every kernel declares the always-zero uniform field the decoder's
    /// contraction barrier reads, under the name the host writes.
    #[test]
    fn both_shaders_declare_the_zero_barrier_field() {
        for source in all_shaders() {
            assert_eq!(source.matches("zero_barrier: u32,").count(), 1);
        }
    }

    /// A dequantization workgroup covers exactly one super-block of tiles.
    #[test]
    fn a_dequant_workgroup_covers_one_super_block() {
        assert_eq!(DEQUANT_WORKGROUP, 256);
        assert_eq!(DEQUANT_PARAM_SLOTS, 16);
    }

    /// The tiled kernel's staged planes and parameter slots fit the 16384-byte
    /// `maxComputeWorkgroupStorageSize` WebGPU guarantees, with the budget the
    /// generator's comment states.
    #[test]
    fn the_tiled_workgroup_memory_fits_the_webgpu_minimum() {
        let bytes = (2 * MATMUL_TILED_STAGE + 2 * MATMUL_TILED_PARAM_SLOTS) * 4;
        assert_eq!(bytes, 8832);
        assert!(bytes <= 16384);
    }

    /// Each invocation stages a whole number of values per plane, so phase 2
    /// covers the tile exactly and no element is written twice or missed.
    #[test]
    fn the_tiled_stage_covers_its_tile_exactly() {
        assert_eq!(MATMUL_TILED_THREADS, 256);
        assert_eq!(MATMUL_TILED_ROUNDS * MATMUL_TILED_THREADS, MATMUL_TILE * 64);
        // A padded stride is what keeps sixteen invocations reading one element
        // index of sixteen staged rows off one memory bank.
        assert_eq!(MATMUL_TILED_STRIDE, 65);
    }

    /// The tiled kernel reaches every barrier from workgroup-uniform control
    /// flow: no invocation returns early, and the three barriers sit in the
    /// K loop's body rather than inside a conditional.
    #[test]
    fn the_tiled_kernel_has_no_early_return_before_a_barrier() {
        let source = generate_tcf_matmul_tiled_shader();
        let body = source
            .split_once(&format!("fn {MATMUL_TILED_ENTRY}("))
            .map(|(_, rest)| rest.to_string())
            .unwrap_or_default();
        assert!(!body.is_empty());
        assert!(!body.contains("return;"));
        assert_eq!(body.matches("workgroupBarrier();").count(), 3);
    }

    /// Both matmul kernels dispatch the same output grid, so the host picks
    /// between them without recomputing the workgroup count.
    #[test]
    fn both_matmul_kernels_share_an_output_grid() {
        for source in [
            generate_tcf_matmul_shader(),
            generate_tcf_matmul_tiled_shader(),
        ] {
            assert!(source.contains(&format!("@workgroup_size({MATMUL_TILE}, {MATMUL_TILE})")));
        }
    }
}
