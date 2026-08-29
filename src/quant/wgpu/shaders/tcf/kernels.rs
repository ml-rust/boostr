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
/// matmul shaders.
pub const MATMUL_TILE: u32 = 16;
/// Largest group count one tile can carry: group 16 over a 64-element tile.
const MAX_GROUPS_PER_TILE: u32 = 4;
/// Workgroup slots holding one super-block's resolved group parameters.
const DEQUANT_PARAM_SLOTS: u32 = DEQUANT_TILES_PER_GROUP * MAX_GROUPS_PER_TILE;

/// Entry point name of the dequantization shader.
pub const DEQUANT_ENTRY: &str = "tcf_dequant_f32";
/// Entry point name of the fused quantized matmul shader.
pub const MATMUL_ENTRY: &str = "tcf_quant_matmul_f32";

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder is generated once and pasted into both shaders, so the two
    /// kernels cannot drift on a bit position.
    #[test]
    fn both_shaders_carry_the_same_decoder() {
        let decoder = decoder();
        assert!(generate_tcf_dequant_shader().contains(&decoder));
        assert!(generate_tcf_matmul_shader().contains(&decoder));
    }

    /// The binary16 decode is written once, not per call site.
    #[test]
    fn the_binary16_decode_is_declared_once_per_shader() {
        for source in [generate_tcf_dequant_shader(), generate_tcf_matmul_shader()] {
            assert_eq!(source.matches("fn tcf_binary16(").count(), 1);
            assert_eq!(source.matches("fn tcf_binary16_parts(").count(), 1);
        }
    }

    /// One helper divides every two-level scale and minimum, so no divisor can
    /// keep a float divide that WGSL specifies at 2.5 ULP.
    #[test]
    fn every_two_level_divisor_goes_through_the_integer_quotient() {
        for source in [generate_tcf_dequant_shader(), generate_tcf_matmul_shader()] {
            assert_eq!(source.matches("fn tcf_scaled_quotient(").count(), 1);
            assert_eq!(source.matches("tcf_scaled_quotient(\n").count(), 3);
            // The only remaining `/` on a level count is the helper's own
            // fallback for a non-finite super-value.
            assert!(!source.contains("/ TCF_SUB_SCALE_LEVELS_U8"));
            assert!(!source.contains("/ TCF_SUB_SCALE_LEVELS_U6"));
            assert!(!source.contains("/ TCF_SUB_MIN_LEVELS_U6"));
        }
    }

    /// Both kernels declare the always-zero uniform field the decoder's
    /// contraction barrier reads, under the name the host writes.
    #[test]
    fn both_shaders_declare_the_zero_barrier_field() {
        for source in [generate_tcf_dequant_shader(), generate_tcf_matmul_shader()] {
            assert_eq!(source.matches("zero_barrier: u32,").count(), 1);
        }
    }

    /// A dequantization workgroup covers exactly one super-block of tiles.
    #[test]
    fn a_dequant_workgroup_covers_one_super_block() {
        assert_eq!(DEQUANT_WORKGROUP, 256);
        assert_eq!(DEQUANT_PARAM_SLOTS, 16);
    }
}
