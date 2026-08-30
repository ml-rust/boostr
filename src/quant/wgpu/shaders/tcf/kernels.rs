//! The four TCF entry points, and the bindings they declare around the shared
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

/// Invocations cooperating on one weight row in the GEMV kernel, and so the
/// width the final reduction folds. Thirty-two matches the CUDA kernel's warp
/// and is a power of two, which the reduction tree needs.
const MATMUL_GEMV_LANES: u32 = 32;
/// Weight rows (output columns) one GEMV workgroup owns, and so the width of
/// the one-dimensional grid the host dispatches over `N`.
pub const MATMUL_GEMV_COLS: u32 = 8;
/// Invocations in a GEMV workgroup: `MATMUL_GEMV_COLS` rows of
/// `MATMUL_GEMV_LANES` lanes.
const MATMUL_GEMV_THREADS: u32 = MATMUL_GEMV_LANES * MATMUL_GEMV_COLS;
/// Execution tiles one GEMV run covers, matching CUDA's `TCF_RUN_TILES`. The
/// unit of work is widened to a run so a group parameter is resolved once per
/// 512 weights rather than once per 64 — the cost the CUDA ablation found
/// dominant, not the width of a code load.
const MATMUL_GEMV_RUN_TILES: u32 = 8;
/// Run elements one lane owns: `8 * 64 / 32`. Sixteen elements from a multiple
/// of sixteen lie inside ONE tile and ONE quantization group at every group
/// width v1 defines (16, 32, 64), so a lane reads one resolved
/// `(scale, minimum)` pair for its whole run.
const MATMUL_GEMV_PER_LANE: u32 = MATMUL_GEMV_RUN_TILES * TCF_TILE as u32 / MATMUL_GEMV_LANES;
/// Elements one lane owns per tail tile: `64 / 32`.
const MATMUL_GEMV_TAIL_PER_LANE: u32 = TCF_TILE as u32 / MATMUL_GEMV_LANES;
/// Workgroup slots holding one run's resolved group parameters, one per
/// (owned weight row, run tile, group). Exactly `MATMUL_GEMV_THREADS` at the
/// widest group count, so the resolve phase is one slot per invocation and no
/// group is resolved twice.
const MATMUL_GEMV_PARAM_SLOTS: u32 = MATMUL_GEMV_COLS * MATMUL_GEMV_RUN_TILES * MAX_GROUPS_PER_TILE;

/// Activation rows at or below which [`generate_tcf_matmul_gemv_shader`] is
/// dispatched, in preference to [`generate_tcf_matmul_tiled_shader`].
///
/// It is a compile-time bound as well as a dispatch threshold — the kernel
/// keeps one accumulator register per activation row, and the WGSL array
/// holding them is sized by this constant — so raising it costs register
/// pressure in the M = 1 case that the kernel exists for.
///
/// MEASURED on an RTX 3060 via Vulkan, `q_proj` 2048x2048, minimum
/// nanoseconds, GEMV kernel against tiled kernel at the same M:
///
/// | M | Q8 gemv | Q8 tiled | Q6 gemv | Q6 tiled | Q4 gemv | Q4 tiled |
/// |---|---------|----------|---------|----------|---------|----------|
/// | 1 |   165us |    412us |   212us |    458us |   191us |    436us |
/// | 4 |   293us |    409us |   339us |    456us |   320us |    437us |
/// | 8 |   483us |    403us |   521us |    456us |   502us |    439us |
///
/// GEMV wins at 1 and 4 and LOSES at 8 on every encoding, so the ceiling sits
/// at 4: the band below it uses the kernel that wins there, and M = 8 is
/// dispatched to the tiled kernel instead.
pub const MATMUL_GEMV_MAX_M: u32 = 4;

/// Entry point name of the dequantization shader.
pub const DEQUANT_ENTRY: &str = "tcf_dequant_f32";
/// Entry point name of the workgroup-tiled fused quantized matmul shader.
pub const MATMUL_TILED_ENTRY: &str = "tcf_quant_matmul_tiled_f32";
/// Entry point name of the small-M fused quantized matmul shader.
pub const MATMUL_GEMV_ENTRY: &str = "tcf_quant_matmul_gemv_f32";

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

/// `activation [M, K] x weight [N, K]^T -> output [M, N]`, workgroup-tiled:
/// a `MATMUL_TILE` square of output elements per workgroup, one invocation
/// each, with the weights decoded into workgroup memory once per K-tile.
///
/// # Why the weight decode is amortized over a workgroup
///
/// An invocation given its own weight row and left to decode every element
/// itself would have the sixteen invocations of a workgroup that share a
/// weight row decode that row sixteen times. A TCF decode is not a table
/// lookup — it is two code-plane reads for a 6-bit encoding, a
/// group-parameter resolution, and a two-level integer division — so paying
/// for it once per OUTPUT element rather than once per weight element is what
/// made this backend lose large-batch prefill to a GGUF block layout, while
/// winning dequantization outright. A per-element kernel of that shape was
/// measured against this one, `q_proj` 2048x2048 on an RTX 3060 via Vulkan,
/// minimum nanoseconds, and lost at every M sampled including 1:
///
/// | M | Q8 per-elem | Q8 tiled |
/// |---|-------------|----------|
/// | 1 |       457us |    412us |
/// | 4 |       469us |    409us |
/// | 8 |       526us |    403us |
///
/// The expectation had been a crossover — that staging traffic and three
/// barriers per K-tile could not pay for themselves until a workgroup had
/// enough activation rows to fill its tile. There was no crossover on that
/// adapter: the amortized decode outweighed the staging cost even at M = 1.
/// The per-element kernel was removed rather than kept unreachable.
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

/// `activation [M, K] x weight [N, K]^T -> output [M, N]` for a small `M`: one
/// output column per group of `MATMUL_GEMV_LANES` invocations, eight columns
/// per workgroup, one dimension of workgroups over `N`.
///
/// # Why a third matmul kernel exists
///
/// Both kernels above are `@workgroup_size(16, 16)` over `(n, m)` and are
/// dispatched `(N/16, M/16)`. At `M = 1` that gives a workgroup 16 real
/// activation rows' worth of invocations out of 256, so 94% of it is masked
/// off, and the tiled kernel additionally stages 16 activation rows and
/// discards 15 of them. The result, measured, was a WebGPU matmul at `M = 1`
/// costing MORE than dequantizing the entire weight to f32 — the opposite of
/// the CUDA relationship on the same shapes. See [`MATMUL_GEMV_MAX_M`].
///
/// # Why every invocation is useful at M = 1
///
/// The grid is `(N / MATMUL_GEMV_COLS, 1)` and the workgroup is
/// `(MATMUL_GEMV_LANES, MATMUL_GEMV_COLS)`. `M` enters neither. A workgroup
/// owns eight weight ROWS, and its 32 lanes split each row's `K` elements
/// between them, so at `M = 1` every one of the 256 invocations decodes 16
/// distinct weights per run and accumulates them against a real activation
/// element. Nothing is masked but a partial trailing workgroup in `N`.
///
/// `M > 1` is handled inside the invocation rather than by a second grid
/// dimension: a lane decodes its 16 weights into registers ONCE and then walks
/// the `M` activation rows against them, so the total decode count stays `N*K`
/// however many rows there are, instead of `M*N*K`.
///
/// # Why a run of eight tiles
///
/// This is the CUDA GEMV's finding, restated. Its first form walked a tile at
/// a time and reached a fifth of the card's bandwidth; the ablation showed the
/// dominant cost was NOT narrow code loads but resolving a group's scale once
/// per 64-element tile. `TwoLevelU6M6` is the expensive form — four scattered
/// plane reads per group — and holding the scale constant alone roughly
/// halved the old kernel at every width. So the unit of work here is a RUN of
/// `MATMUL_GEMV_RUN_TILES` tiles, 512 elements, and the workgroup resolves the
/// whole run's `8 * 8 * groups_per_tile <= 256` group parameters in ONE step,
/// one slot per invocation, before any lane decodes a code.
///
/// A lane's 16 run elements start at a multiple of 16, so they lie inside one
/// tile and inside one group at every group width v1 defines, and the lane
/// reads exactly one resolved pair for the whole run.
///
/// # The tail
///
/// A row whose tile count is not a multiple of `MATMUL_GEMV_RUN_TILES`
/// finishes on a tile-at-a-time loop, which is also the only path when `K` is
/// under eight tiles. It is the same shape CUDA keeps for the same reason:
/// two elements per lane, the workgroup's `8 * groups_per_tile <= 32` group
/// parameters resolved in one step. Both paths call the SAME `tcf_code` and
/// `tcf_value`, so there is one decode, not two.
///
/// # Workgroup memory budget
///
/// One scale and one minimum per run slot, plus one reduction slot per
/// invocation:
///
/// - run scales: `256 * 4` = 1024 bytes
/// - run minima: `256 * 4` = 1024 bytes
/// - reduction: `256 * 4` = 1024 bytes
///
/// 3072 bytes, well inside the 16384 WebGPU guarantees as
/// `maxComputeWorkgroupStorageSize`. The reduction plane is one f32 per
/// invocation rather than one per `(invocation, activation row)` precisely so
/// the `M = 1` case this kernel exists for keeps its occupancy.
///
/// # Why every barrier is uniformly reached
///
/// WGSL Section 14.7 makes a barrier in non-uniform control flow undefined
/// behaviour. The kernel has NO `return`, and every barrier sits at statement
/// level in a loop whose trip count is workgroup-uniform:
///
/// - the run loop, `params.k / TCF_TILE / MATMUL_GEMV_RUN_TILES`, two
///   barriers — one after the resolve phase, one read-before-overwrite before
///   the next run rewrites the slots,
/// - the tail loop, over the same uniform tile count, two barriers for the
///   same two reasons,
/// - the reduction, `min(params.m, MATMUL_GEMV_MAX_M)` rows each with one
///   barrier after the accumulator store and one per tree step.
///
/// Every guard — `n < params.n`, `tid < slots`, `lane < offset` — masks a
/// WRITE and never encloses a barrier. An invocation whose column overhangs
/// `N` runs the whole loop and is masked at the single store.
///
/// The reduction needs no trailing barrier per row: the tree's last step is
/// followed by one, and the only slot an invocation reads after it is its own,
/// which no other invocation writes.
#[must_use]
pub fn generate_tcf_matmul_gemv_shader() -> String {
    format!(
        r#"// Small-M matmul: activation [M,K] x TCF weight [N,K] -> output [M,N]
{bindings}
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: TcfParams;
{decoder}

const TCF_GEMV_LANES: u32 = {lanes}u;
const TCF_GEMV_COLS: u32 = {cols}u;
const TCF_GEMV_RUN_TILES: u32 = {run_tiles}u;
const TCF_GEMV_PER_LANE: u32 = {per_lane}u;
const TCF_GEMV_TAIL_PER_LANE: u32 = {tail_per_lane}u;
const TCF_GEMV_MAX_M: u32 = {max_m}u;

// 1024 + 1024 + 1024 = 3072 bytes, inside the 16384-byte WebGPU minimum.
var<workgroup> tcf_run_scale: array<f32, {slots}u>;
var<workgroup> tcf_run_min: array<f32, {slots}u>;
var<workgroup> tcf_reduce: array<f32, {threads}u>;

@compute @workgroup_size({lanes}, {cols})
fn {entry}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    // No early return: every invocation below must reach every barrier, so a
    // workgroup overhanging N is masked at its stores instead.
    let lane = lid.x;
    let col = lid.y;
    let tid = col * TCF_GEMV_LANES + lane;
    let n0 = wid.x * TCF_GEMV_COLS;
    let n = n0 + col;
    let live = n < params.n;

    // Uniform: all four come from the uniform block, so every loop bound below
    // is workgroup-uniform and so is every barrier inside those loops.
    let tiles_per_row = params.k / TCF_TILE;
    let groups_per_tile = params.groups_per_tile;
    let runs = tiles_per_row / TCF_GEMV_RUN_TILES;
    let rows = min(params.m, TCF_GEMV_MAX_M);
    let run_slots = TCF_GEMV_RUN_TILES * groups_per_tile;
    let tail_slots = TCF_GEMV_COLS * groups_per_tile;

    // One accumulator per activation row, so a decoded weight is reused across
    // all of them instead of being decoded once per row.
    var acc: array<f32, {max_m}u>;
    for (var mi: u32 = 0u; mi < TCF_GEMV_MAX_M; mi = mi + 1u) {{
        acc[mi] = 0.0;
    }}

    // This lane's slice of a run. `e0` is a multiple of sixteen, so the slice
    // stays inside one tile and one group and `slot_in_tile` is constant.
    let e0 = lane * TCF_GEMV_PER_LANE;
    let run_tile = e0 / TCF_TILE;
    let e_base = e0 % TCF_TILE;
    let slot_in_tile = e_base / params.group;

    for (var r: u32 = 0u; r < runs; r = r + 1u) {{
        // Phase 1: resolve the whole run's group parameters, one slot per
        // invocation, so a scale plane is read once per 512 weights.
        let tile0 = r * TCF_GEMV_RUN_TILES;
        if (tid < TCF_GEMV_COLS * run_slots) {{
            let param_n = n0 + tid / run_slots;
            let within = tid % run_slots;
            var scale: f32 = 0.0;
            var min_value: f32 = 0.0;
            if (param_n < params.n) {{
                let tile = param_n * tiles_per_row + tile0 + within / groups_per_tile;
                let values = tcf_group_values(tile, within % groups_per_tile);
                scale = values.scale;
                min_value = values.min_value;
            }}
            tcf_run_scale[tid] = scale;
            tcf_run_min[tid] = min_value;
        }}
        workgroupBarrier();

        // Phase 2: decode this lane's sixteen weights once, then walk the M
        // activation rows against them.
        if (live) {{
            let tile = n * tiles_per_row + tile0 + run_tile;
            let slot = col * run_slots + run_tile * groups_per_tile + slot_in_tile;
            let scale = tcf_run_scale[slot];
            let min_value = tcf_run_min[slot];

            var w: array<f32, {per_lane}u>;
            for (var i: u32 = 0u; i < TCF_GEMV_PER_LANE; i = i + 1u) {{
                w[i] = tcf_value(tcf_code(tile, e_base + i), scale, min_value);
            }}

            let k_base = tile0 * TCF_TILE + e0;
            for (var mi: u32 = 0u; mi < rows; mi = mi + 1u) {{
                let act_base = mi * params.k + k_base;
                var sum = acc[mi];
                for (var i: u32 = 0u; i < TCF_GEMV_PER_LANE; i = i + 1u) {{
                    sum = sum + activation[act_base + i] * w[i];
                }}
                acc[mi] = sum;
            }}
        }}
        // Read-before-overwrite: the next run's phase 1 rewrites both slot
        // planes, so no invocation may start it while another still reads.
        workgroupBarrier();
    }}

    // The tiles a whole run cannot cover, one at a time, two elements per
    // lane. Same decode, narrower unit of work.
    for (var j: u32 = runs * TCF_GEMV_RUN_TILES; j < tiles_per_row; j = j + 1u) {{
        if (tid < tail_slots) {{
            let param_n = n0 + tid / groups_per_tile;
            var scale: f32 = 0.0;
            var min_value: f32 = 0.0;
            if (param_n < params.n) {{
                let values = tcf_group_values(
                    param_n * tiles_per_row + j, tid % groups_per_tile);
                scale = values.scale;
                min_value = values.min_value;
            }}
            tcf_run_scale[tid] = scale;
            tcf_run_min[tid] = min_value;
        }}
        workgroupBarrier();

        if (live) {{
            let tile = n * tiles_per_row + j;
            let k_base = j * TCF_TILE;
            for (var h: u32 = 0u; h < TCF_GEMV_TAIL_PER_LANE; h = h + 1u) {{
                let e = lane + h * TCF_GEMV_LANES;
                let slot = col * groups_per_tile + e / params.group;
                let w = tcf_value(tcf_code(tile, e), tcf_run_scale[slot], tcf_run_min[slot]);
                for (var mi: u32 = 0u; mi < rows; mi = mi + 1u) {{
                    acc[mi] = acc[mi] + activation[mi * params.k + k_base + e] * w;
                }}
            }}
        }}
        workgroupBarrier();
    }}

    // Reduce the lanes of each column, one activation row at a time. One f32
    // per invocation of workgroup memory rather than one per (invocation,
    // row), because M = 1 is the case this kernel exists for.
    for (var mi: u32 = 0u; mi < rows; mi = mi + 1u) {{
        tcf_reduce[tid] = acc[mi];
        workgroupBarrier();
        for (var offset: u32 = TCF_GEMV_LANES / 2u; offset > 0u; offset = offset >> 1u) {{
            if (lane < offset) {{
                tcf_reduce[tid] = tcf_reduce[tid] + tcf_reduce[tid + offset];
            }}
            workgroupBarrier();
        }}
        if (lane == 0u && live) {{
            output[mi * params.n + n] = tcf_reduce[tid];
        }}
    }}
}}
"#,
        bindings = bindings(
            1,
            "\n@group(0) @binding(0) var<storage, read_write> activation: array<f32>;"
        ),
        decoder = decoder(),
        lanes = MATMUL_GEMV_LANES,
        cols = MATMUL_GEMV_COLS,
        threads = MATMUL_GEMV_THREADS,
        run_tiles = MATMUL_GEMV_RUN_TILES,
        per_lane = MATMUL_GEMV_PER_LANE,
        tail_per_lane = MATMUL_GEMV_TAIL_PER_LANE,
        max_m = MATMUL_GEMV_MAX_M,
        slots = MATMUL_GEMV_PARAM_SLOTS,
        entry = MATMUL_GEMV_ENTRY,
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
            generate_tcf_matmul_tiled_shader(),
            generate_tcf_matmul_gemv_shader(),
        ]
    }

    /// The decoder is generated once and pasted into every shader, so the
    /// kernels cannot drift on a bit position. MIGRATION.md Section 4.5.3
    /// forbids a second copy of a bit position, and a Q6_K release once shipped
    /// wrong because one existed.
    #[test]
    fn every_shader_carries_the_same_decoder() {
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
    fn every_shader_declares_the_zero_barrier_field() {
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

    /// The tiled matmul kernel declares the `MATMUL_TILE` square workgroup
    /// its output grid is dispatched over.
    #[test]
    fn the_tiled_matmul_kernel_declares_its_output_grid() {
        let source = generate_tcf_matmul_tiled_shader();
        assert!(source.contains(&format!("@workgroup_size({MATMUL_TILE}, {MATMUL_TILE})")));
    }

    /// A GEMV workgroup's lanes cover a run exactly, and a lane's slice starts
    /// at a multiple of sixteen, which is what lets it read ONE resolved
    /// `(scale, minimum)` pair for its whole run at every group width v1
    /// defines.
    #[test]
    fn a_gemv_run_is_covered_by_its_lanes_exactly() {
        assert_eq!(MATMUL_GEMV_THREADS, 256);
        assert_eq!(
            MATMUL_GEMV_LANES * MATMUL_GEMV_PER_LANE,
            MATMUL_GEMV_RUN_TILES * TCF_TILE as u32
        );
        assert_eq!(MATMUL_GEMV_PER_LANE, 16);
        assert_eq!(
            MATMUL_GEMV_LANES * MATMUL_GEMV_TAIL_PER_LANE,
            TCF_TILE as u32
        );
        // Sixteen elements from a multiple of sixteen lie inside one group at
        // every group width v1 defines.
        for group in [16u32, 32, 64] {
            assert_eq!(group % MATMUL_GEMV_PER_LANE, 0);
        }
    }

    /// The resolve phase is one slot per invocation at the widest group count,
    /// so no group is resolved twice and no invocation idles through it.
    #[test]
    fn a_gemv_run_resolves_one_group_per_invocation() {
        assert_eq!(MATMUL_GEMV_PARAM_SLOTS, MATMUL_GEMV_THREADS);
        assert_eq!(
            MATMUL_GEMV_PARAM_SLOTS,
            MATMUL_GEMV_COLS * MATMUL_GEMV_RUN_TILES * MAX_GROUPS_PER_TILE
        );
    }

    /// The GEMV's workgroup memory is the two slot planes and one reduction
    /// f32 per invocation, with the budget the generator's comment states.
    #[test]
    fn the_gemv_workgroup_memory_fits_the_webgpu_minimum() {
        let bytes = (2 * MATMUL_GEMV_PARAM_SLOTS + MATMUL_GEMV_THREADS) * 4;
        assert_eq!(bytes, 3072);
        assert!(bytes <= 16384);
    }

    /// The GEMV reaches every barrier from workgroup-uniform control flow: no
    /// invocation returns early, and every barrier sits at statement level in
    /// a loop bounded by a uniform value rather than inside a conditional.
    #[test]
    fn the_gemv_kernel_has_no_early_return_before_a_barrier() {
        let source = generate_tcf_matmul_gemv_shader();
        let body = source
            .split_once(&format!("fn {MATMUL_GEMV_ENTRY}("))
            .map(|(_, rest)| rest.to_string())
            .unwrap_or_default();
        assert!(!body.is_empty());
        assert!(!body.contains("return;"));
        // Two per run, two per tail tile, one after the accumulator store and
        // one per reduction tree step.
        assert_eq!(body.matches("workgroupBarrier();").count(), 6);
    }

    /// The GEMV is one-dimensional over N, which is what makes every
    /// invocation useful at M = 1: `M` appears in neither the workgroup shape
    /// nor the grid, so no invocation is masked for want of an activation row.
    #[test]
    fn the_gemv_workgroup_is_one_column_group_per_lane_group() {
        let source = generate_tcf_matmul_gemv_shader();
        assert!(source.contains(&format!(
            "@workgroup_size({MATMUL_GEMV_LANES}, {MATMUL_GEMV_COLS})"
        )));
    }

    /// The GEMV ceiling is readable by the host, so `dispatch_matmul` picks
    /// between the two kernels without restating the bound, and it is also
    /// the WGSL accumulator array's compile-time size.
    #[test]
    fn the_gemv_ceiling_matches_the_accumulator_array() {
        // One accumulator register per activation row, so the dispatch
        // threshold and the WGSL array bound are the same constant.
        let source = generate_tcf_matmul_gemv_shader();
        assert!(source.contains(&format!("var acc: array<f32, {MATMUL_GEMV_MAX_M}u>;")));
        assert!(source.contains(&format!(
            "const TCF_GEMV_MAX_M: u32 = {MATMUL_GEMV_MAX_M}u;"
        )));
    }
}
