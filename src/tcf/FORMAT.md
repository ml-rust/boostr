# TCF v1 binary layout

TCF (Tensor Contract Format) stores GGUF block payloads in a stricter
container. This file is the layout reference for `boostr::tcf`. Section
numbers are stable: code comments cite them, and numbers of retired
sections are not reused.

What the container adds over GGUF:

- Fixed-size records at fixed offsets, decoded field by field.
- Every tensor declares an activation contract (Section 9).
- Every record that carries a digest is recomputed and checked on open.
- 64 proof values per block tensor, checked with the reader's own decoder.
- Placement metadata with provenance (Section 10, Section 10.5, Section 16).
- A directory a scheduler can plan from without touching a payload page.

## 1. Conventions

- Integers and floats are little-endian. Floats are IEEE binary32,
  binary16, or bfloat16, as the field names.
- A reader decodes each field at its offset. It never casts a struct over
  mapped bytes.
- `MUST` rules are enforced by `boostr::tcf::TcfFile::open` and the writer.

## 2. Scope

The container carries one model's tensors, their encodings, contracts,
and placement metadata. Training state, tokenizers, and architecture
config are out of scope.

## 3. The contract rule

A weight encoding does not define the operation on its own. Every tensor
names a `ContractRecord` through `activation_contract_id`, and every
block matmul entry point in boostr checks that record against the kernel
it runs. A file whose contract no kernel satisfies is refused: there is no
float fallback.

## 4. File-level rules

- Every top-level section starts at a multiple of 64.
- Unused and padding bytes are zero. A non-zero reserved field is refused.
- Before allocating from a file-supplied size, the reader checks every
  section range for overflow, overlap, alignment, `count x size`
  overflow, and file bounds.

### 4.1 Section order

```text
Header                    192 bytes at offset 0
ModuleRecord[]            128 bytes each
TensorRecord[]            256 bytes each
ContractRecord[]           64 bytes each
CalibrationRecord[]       128 bytes each
RelationRecord[]           64 bytes each
WorkloadProfileRecord[]   128 bytes each
String table
Proof section
Tensor data               from data_off
```

Everything before `Tensor data` is the directory and lies inside
`[192, data_off)`. A section with count zero is absent and its offset
carries no meaning. Offsets are authoritative: a reader locates sections
by offset, not by the listed order.

## 5. Header

192 bytes at offset 0.

| Offset | Size | Field |
| -----: | ---: | ----- |
| 0 | 8 | magic `54 43 46 00 00 00 00 00` |
| 8 | 2 | `major = 1` |
| 10 | 2 | `minor` |
| 12 | 4 | `header_bytes = 192` |
| 16 | 4 | `schema_id = 1` |
| 20 | 4 | `flags` |
| 24 | 4 | `tensor_count` |
| 28 | 4 | `module_count` |
| 32 | 4 | `contract_count` |
| 36 | 4 | `calibration_count` |
| 40 | 4 | `relation_count` |
| 44 | 4 | `workload_count` |
| 48 | 8 | `required_features` |
| 56 | 8 | `module_off` |
| 64 | 8 | `tensor_off` |
| 72 | 8 | `contract_off` |
| 80 | 8 | `calibration_off` |
| 88 | 8 | `relation_off` |
| 96 | 8 | `string_off` |
| 104 | 8 | `string_len` |
| 112 | 8 | `proof_off` |
| 120 | 8 | `proof_len` |
| 128 | 8 | `data_off` |
| 136 | 8 | `file_len` |
| 144 | 16 | `header_digest` |
| 160 | 16 | `directory_digest` |
| 176 | 8 | `workload_off` |
| 184 | 8 | reserved |

### 5.1 Header flags

| Bit | Name | Rule |
| --: | ---- | ---- |
| 0 | `LITTLE_ENDIAN` | Set in every v1 file |

### 5.2 Required features

| Bit | Name | Rule |
| --: | ---- | ---- |
| 0 | `ACTIVATION_CONTRACTS` | Set in every v1 file |
| 1 | `PLACEMENT_METADATA` | Set in every v1 file |
| 2 | `SEMANTIC_DIGESTS` | Set in every v1 file |
| 3 | `SOURCE_PROOFS` | Set in every v1 file |
| 4 | `RELATIONS` | Set exactly when `relation_count > 0` |
| 5 | `WORKLOAD_PROFILES` | Set exactly when `workload_count > 0` |
| 6 | `TWO_LEVEL_SCALES` | Retired. Named the removed native encodings. A file setting it is refused. |

Any other set bit is refused with `E_UNKNOWN_REQUIRED_FEATURE`.

### 5.3 Header digests

- `header_digest`: BLAKE3-128 over bytes `[0, 192)` with `[144, 160)`
  treated as zero.
- `directory_digest`: BLAKE3-128 over bytes `[192, data_off)`, padding
  included.

## 6. String table

UTF-8 bytes at `[string_off, string_off + string_len)`. A name is an
`(off, len)` pair relative to `string_off`. Names are provenance, never
identity: dispatch uses `tensor_id`, `module_id`, and `role`.

## 7. Module record

128 bytes.

| Offset | Size | Field |
| -----: | ---: | ----- |
| 0 | 4 | `module_id` |
| 4 | 4 | `parent_id`, `0xffffffff` = root |
| 8 | 8 | `name_off` |
| 16 | 4 | `name_len` |
| 20 | 2 | `module_role` |
| 22 | 2 | `fallback_encoding`, `0` = none |
| 24 | 8 | `preferred_encoding[4]`, four u16, `0` = unused |
| 32 | 4 | `activation_contract_id` |
| 36 | 4 | `policy_flags` |
| 40 | 4 | `min_quant_k` |
| 44 | 4 | reserved |
| 48 | 2 | `default_residency` |
| 50 | 2 | `state_dtype` |
| 52 | 4 | `state_flags` |
| 56 | 8 | reserved |
| 64 | 16 | `policy_digest` |
| 80 | 48 | reserved |

`policy_digest` is BLAKE3-128 over bytes `[0, 64)` with the digest field
zero, followed by the module's UTF-8 name bytes.

`preferred_encoding` is ordered highest preference first, filled from
the front. Every `TensorRecord.module_id` resolves to a module in the
file, else `E_SECTION_BOUNDS`.

`min_quant_k` is advisory producer policy. A reader never rejects a
tensor for being below it.

### 7.1 `module_role`

| Value | Name |
| ----: | ---- |
| 0 | `OTHER` |
| 1 | `EMBEDDING` |
| 2 | `ATTENTION` |
| 3 | `FFN` |
| 4 | `MOE` |
| 5 | `SSM` |
| 6 | `CONV_STACK` |
| 7 | `DIFFUSION` |
| 8 | `CODEC` |
| 9 | `NORM_GROUP` |

### 7.2 `policy_flags`

| Bit | Name |
| --: | ---- |
| 0 | `FORBID_REQUANT` |
| 1 | `FORBID_DEVICE_PLACEMENT` |

### 7.3 `state_dtype`

`0 NONE`, `1 F32`, `2 F16`, `3 BF16`.

## 8. Tensor record

256 bytes.

| Offset | Size | Field |
| -----: | ---: | ----- |
| 0 | 4 | `tensor_id` |
| 4 | 4 | `module_id` |
| 8 | 8 | `name_off` |
| 16 | 4 | `name_len` |
| 20 | 2 | `role` |
| 22 | 2 | `encoding` |
| 24 | 2 | `fallback_reason` |
| 26 | 2 | `residency_class` |
| 28 | 4 | `flags` |
| 32 | 4 | `rank` |
| 36 | 4 | `calibration_id` |
| 40 | 64 | `dims[8]`, eight u64 |
| 104 | 4 | `activation_contract_id` |
| 108 | 4 | `layout_id` |
| 112 | 8 | `data_offset`, absolute |
| 120 | 8 | `logical_payload_bytes` |
| 128 | 8 | `physical_span_bytes` |
| 136 | 8 | `resident_bytes` |
| 144 | 8 | `transfer_bytes` |
| 152 | 4 | `sensitivity_delta`, f32 |
| 156 | 4 | `sensitivity_ci95`, f32 |
| 160 | 4 | `accesses_per_generation`, f32 |
| 164 | 4 | `bytes_read_per_generation`, f32 |
| 168 | 4 | `sensitivity_samples` |
| 172 | 4 | `sensitivity_seed_count` |
| 176 | 4 | `access_profile_samples` |
| 180 | 4 | reserved |
| 184 | 2 | `execution_role` |
| 186 | 2 | reserved |
| 188 | 4 | `workload_profile_id` |
| 192 | 16 | `semantic_digest` |
| 208 | 16 | `payload_digest` |
| 224 | 8 | `proof_rel_off`, relative to `proof_off` |
| 232 | 4 | `proof_count` |
| 236 | 4 | `proof_format` |
| 240 | 16 | reserved |

`rank` is 1 through 8. Dimensions at index `>= rank` are zero.
`data_offset` is a multiple of 64.

### 8.0.1 Payload length is determined, never declared

- Block encoding: `rows x (K / block_elems) x block_bytes`, with
  `K = dims[rank - 1]`, `rank >= 2`, and `K` a multiple of `block_elems`.
- Raw encoding: `product(dims) x width`, widths per Section 12.

A stored `logical_payload_bytes` that disagrees is refused with
`E_INVALID_QUANT_SHAPE`.

### 8.1 Resident-byte invariant

`resident_bytes == physical_span_bytes` and
`transfer_bytes == physical_span_bytes`. A backend never holds a
permanent widened copy of a weight. A size-preserving, lossless,
verified-first, non-persistent repack into an ISA layout is permitted.

### 8.1.5 Flag fields and unknown bits

Every bit this document does not define is reserved and zero. An unknown
set bit in `required_features` is `E_UNKNOWN_REQUIRED_FEATURE`. In every
other flag field it is `E_NONZERO_RESERVED`. `ContractRecord`,
`CalibrationRecord`, `WorkloadProfileRecord`, `RelationRecord`, and
`ModuleRecord.state_flags` define no bits yet.

### 8.2 Tensor flags

| Bit | Name |
| --: | ---- |
| 0 | `SENSITIVITY_VALID` |
| 1 | `ACCESS_PROFILE_VALID` |
| 2 | `ALLOW_OFFLOAD` |
| 3 | `MUST_VERIFY_BEFORE_USE` |
| 4 | `TASK_CRITICAL` |

With `SENSITIVITY_VALID` clear, a consumer ignores the four sensitivity
fields. Zero never means "not measured".

### 8.3 Backend restriction

None in v1. Offset 180 is reserved.

### 8.4 `layout_id`

`0 ROW_MAJOR_DENSE`, the only v1 value.

### 8.5 `residency_class`

| Value | Name | Intent |
| ----: | ---- | ------ |
| 0 | `HOT` | Fastest compute memory if capacity permits |
| 1 | `WARM` | Prefer device, evict after hot |
| 2 | `COLD` | Host-resident or streamable |
| 3 | `HOST_ONLY` | Producer forbids device placement |
| 4 | `NO_MIGRATE` | Runtime picks the initial place, then it never moves |

### 8.6 `fallback_reason`

Mandatory whenever `encoding` differs from the module's highest-ranked
`preferred_encoding`, else `E_MISSING_FALLBACK_REASON`.

| Value | Name |
| ----: | ---- |
| 0 | `NONE` |
| 1 | `RANK_LT_2` |
| 2 | `SHAPE_INCOMPATIBLE_WITH_ENCODING` |
| 3 | `ROLE_FORBIDS_QUANT` |
| 4 | `TASK_SENSITIVITY` |
| 5 | `PHYSICAL_SIZE_NOT_BENEFICIAL` |
| 6 | `NUMERIC_RANGE` |
| 7 | `SOURCE_NONFINITE` |
| 8 | `UNSUPPORTED_ENCODING` |
| 9 | `USER_PINNED_PRECISION` |
| 10 | `PRODUCER_POLICY` |

A consumer never re-derives this decision.

### 8.6.1 `execution_role`

The dispatch key: how a kernel consumes the tensor.

| Value | Name |
| ----: | ---- |
| 0 | `MATMUL` |
| 1 | `LOOKUP` |
| 2 | `CONV1D` |
| 3 | `INDEXED` |
| 4 | `STATE_UPDATE` |
| 5 | `ELEMENTWISE` |

Dispatch resolves on `encoding`, `execution_role`, and the contract's
semantic fields. `role`, `calibration_id`, and `flags` take no part.

### 8.7 `role`

Semantic metadata for tooling, never a dispatch key.

| Value | Name | Value | Name |
| ----: | ---- | ----: | ---- |
| 0 | `OTHER` | 8 | `SSM_DYNAMICS` |
| 1 | `LINEAR_WEIGHT` | 9 | `MOE_EXPERT` |
| 2 | `EMBEDDING` | 10 | `MOE_ROUTER` |
| 3 | `CONV1D_WEIGHT` | 11 | `DIT_WEIGHT` |
| 4 | `BIAS` | 12 | `FSQ_PROJECTION` |
| 5 | `NORM_SCALE` | 13 | `FSQ_LEVELS` |
| 6 | `SNAKE_ALPHA` | 14 | `VAE_WEIGHT` |
| 7 | `SSM_PROJECTION` | 15 | `INDEXED_TABLE` |

Default `role` to `execution_role`: matrix roles (`LINEAR_WEIGHT`,
`MOE_EXPERT`, `MOE_ROUTER`, `DIT_WEIGHT`, `SSM_PROJECTION`,
`FSQ_PROJECTION`, `VAE_WEIGHT`) map to `MATMUL`, `EMBEDDING` to `LOOKUP`,
`CONV1D_WEIGHT` to `CONV1D`, `FSQ_LEVELS` and `INDEXED_TABLE` to
`INDEXED`, `SSM_DYNAMICS` to `STATE_UPDATE`, and `BIAS`, `NORM_SCALE`,
`SNAKE_ALPHA` to `ELEMENTWISE`.

### 8.8 Role quantization policy

Producer defaults. A producer overriding one records `fallback_reason`.
Quantizable: `LINEAR_WEIGHT`, `EMBEDDING`, large `CONV1D_WEIGHT`,
`MOE_EXPERT`, `SSM_PROJECTION`, `DIT_WEIGHT`, `FSQ_PROJECTION`,
`VAE_WEIGHT`. Raw by default: `MOE_ROUTER`, `SSM_DYNAMICS`, `NORM_SCALE`,
`BIAS`, `SNAKE_ALPHA`, `FSQ_LEVELS`, `INDEXED_TABLE`.

## 9. Contract record

64 bytes.

| Offset | Size | Field |
| -----: | ---: | ----- |
| 0 | 4 | `contract_id` |
| 4 | 2 | `input_representation` |
| 6 | 2 | `quant_group` |
| 8 | 2 | `quant_axis` |
| 10 | 2 | `rounding_mode` |
| 12 | 2 | `qmin`, signed |
| 14 | 2 | `qmax`, signed |
| 16 | 2 | `scale_compute_dtype` |
| 18 | 2 | `dot_accumulator` |
| 20 | 2 | `output_dtype` |
| 22 | 2 | `math_mode` |
| 24 | 4 | reserved |
| 28 | 4 | `calibration_id` |
| 32 | 4 | `flags` |
| 36 | 4 | reserved |
| 40 | 16 | `contract_digest` |
| 56 | 8 | reserved |

`contract_digest` is BLAKE3-128 over bytes `[0, 40)` with `contract_id`
treated as zero. The producer computes it and the reader recomputes it
(`E_CONTRACT_DIGEST_MISMATCH`). It is an integrity identity, not a
dispatch key.

### 9.1 Enumerations

| Field | Values |
| ----- | ------ |
| `input_representation` | `1 F32`, `2 F16`, `3 BF16`, `4 A8S32_DYNAMIC`, `5 GGML_REFERENCE` |
| `quant_axis` | `0 LAST` |
| `rounding_mode` | `1 RN_EVEN` |
| `scale_compute_dtype` | `1 F32` |
| `dot_accumulator` | `1 F32`, `2 I32_THEN_F32_SCALE`, `3 GGML_REFERENCE` |
| `output_dtype` | `1 F32`, `2 F16`, `3 BF16` |
| `math_mode` | `1 REASSOCIATION_ALLOWED`, `2 REASSOCIATION_FORBIDDEN` |

### 9.2 The `A8S32_DYNAMIC` contract

`d_a = max|a_i| / 127`, `q_i = RN_even(a_i / d_a)` clamped to
`[-127, 127]`, groups of 32 along K, `d_a` in F32, integer dot in I32,
each group rescaled by `d_w * d_a` into F32. Under
`REASSOCIATION_FORBIDDEN` a kernel reproduces strict left-to-right
accumulation. No boostr kernel does, so a file declaring it is refused.

### 9.3 The `GGML_REFERENCE` contract

The contract a block tensor declares. A GGML block type fixes its weight
bytes and not its activation path: `ggml-quants.c` runs K-quants against
Q8_K activations on the CPU, the CUDA kernels run the same weights
against Q8_1 activations, and a backend that dequantizes first uses f32.
The file therefore pins the kernel family, and the record's group,
range, and accumulator are nominal: `quant_group = 0`, `qmin = qmax = 0`,
`dot_accumulator = GGML_REFERENCE`. Every block matmul entry point in
boostr satisfies it. A kernel promising one specific representation
(`F32`, `A8S32_DYNAMIC`) does not, and the check refuses the pairing.

## 10. Calibration record

128 bytes.

| Offset | Size | Field |
| -----: | ---: | ----- |
| 0 | 4 | `calibration_id` |
| 4 | 2 | `primary_metric` |
| 6 | 2 | `flags` |
| 8 | 4 | `sample_count` |
| 12 | 4 | `seed_count` |
| 16 | 4 | `baseline_metric`, f32 |
| 20 | 4 | `acceptance_margin`, f32 |
| 24 | 8 | `dataset_name_off` |
| 32 | 4 | `dataset_name_len` |
| 36 | 4 | reserved |
| 40 | 8 | `evaluator_name_off` |
| 48 | 4 | `evaluator_name_len` |
| 52 | 4 | reserved |
| 56 | 32 | `dataset_digest`, BLAKE3-256 |
| 88 | 16 | `evaluator_config_digest` |
| 104 | 8 | `producer_timestamp`, Unix seconds |
| 112 | 16 | reserved |

`primary_metric`: `1 WER`, `2 CER`, `3 TASK_SPECIFIC`, `4 PERPLEXITY`.
`sensitivity_delta` on a tensor is a task delta under this record, never
an RMS score.

## 10.5 Workload profile record

128 bytes.

| Offset | Size | Field |
| -----: | ---: | ----- |
| 0 | 4 | `workload_id` |
| 4 | 2 | `workload_kind` |
| 6 | 2 | `flags` |
| 8 | 4 | `generation_count` |
| 12 | 4 | `avg_generated_tokens`, f32 |
| 16 | 4 | `avg_prompt_tokens`, f32 |
| 20 | 4 | reserved |
| 24 | 8 | `dataset_name_off` |
| 32 | 4 | `dataset_name_len` |
| 36 | 4 | reserved |
| 40 | 8 | `runtime_name_off` |
| 48 | 4 | `runtime_name_len` |
| 52 | 4 | reserved |
| 56 | 32 | `workload_digest`, BLAKE3-256 |
| 88 | 16 | `runtime_config_digest` |
| 104 | 8 | `producer_timestamp`, Unix seconds |
| 112 | 16 | reserved |

`workload_kind`: `1 GENERAL`, `2 CHAT`, `3 CODE`, `4 SUMMARIZATION`,
`5 SPEECH`, `6 TASK_SPECIFIC`.

### 10.5.1 Provenance

`accesses_per_generation`, `bytes_read_per_generation`, and
`access_profile_samples` are workload observations. `ACCESS_PROFILE_VALID`
is clear when `workload_profile_id` names no record
(`E_MISSING_WORKLOAD_PROFILE` otherwise). Re-profiling writes a new
record and reseals both header digests: there is no in-place mutable
region.

## 11. Relation record

64 bytes.

| Offset | Size | Field |
| -----: | ---: | ----- |
| 0 | 2 | `relation_type` |
| 2 | 2 | `flags` |
| 4 | 4 | `output_tensor_id` |
| 8 | 16 | `input_tensor_id[4]`, four u32, `0xffffffff` = unused |
| 24 | 4 | `rank_or_parameter` |
| 28 | 4 | `activation_contract_id` |
| 32 | 16 | `relation_digest` |
| 48 | 16 | reserved |

`relation_digest` is BLAKE3-128 over bytes `[0, 32)`.

### 11.1 `LOW_RANK_RESIDUAL`

`relation_type = 1`. Inputs: `[0]` quantized base, `[1]` U, `[2]` V.
Effective weight `dequant(base) + U V`, run as `dequant(base) x + U (V x)`.
Supported structurally, disabled by default.

## 12. Encoding registry

| Range | Kind |
| ----- | ---- |
| `0x0001`-`0x00FF` | Raw |
| `0x0100`-`0x01FF` | Retired native encodings. Unassigned. A reader refuses them. |
| `0x0200` + `ggml_type` | Block |

Raw encodings and widths:

| Value | Name | Width |
| ----: | ---- | ----: |
| `0x0001` | `F32` | 4 |
| `0x0002` | `F16` | 2 |
| `0x0003` | `BF16` | 2 |
| `0x0004` | `F8E4M3` | 1 |
| `0x0005` | `F8E5M2` | 1 |
| `0x0010` | `I8` | 1 |
| `0x0011` | `I16` | 2 |
| `0x0012` | `I32` | 4 |
| `0x0013` | `U8` | 1 |
| `0x0014` | `U16` | 2 |
| `0x0015` | `U32` | 4 |

Block encodings are `0x0200 + ggml_type` for every quantized `ggml_type`
boostr's `QuantFormat` names (Q4_0 through TQ2_0). `block_elems` and
`block_bytes` are the values in `ggml-common.h`. TCF stores the GGML
block stream verbatim and defines no layout of its own.

### 12.3 Payload length

`rows x (K / block_elems) x block_bytes`, per Section 8.0.1.

## 13. Quantization math

Retired with the native encodings. A block payload's math is
`ggml-quants.c`'s for its type.

## 14. Payload layouts

Retired with the native encodings, except:

### 14.4 Alignment

Each payload starts at a multiple of 64 and is zero-padded to
`physical_span_bytes`, the next multiple of 64. A non-zero padding byte
is `E_NONZERO_RESERVED`.

## 15. Verification

| Mechanism | Catches |
| --------- | ------- |
| `payload_digest` | Storage corruption of the stored bytes |
| `semantic_digest` | A stream that differs from what the producer meant |
| Proof vector | A decoder that reads the bytes differently from the producer |

### 15.1 `payload_digest`

BLAKE3-128 over exactly `logical_payload_bytes`, padding excluded.

### 15.2 `semantic_digest`

For a block tensor the stored stream is the logical stream, so
`semantic_digest == payload_digest`. A disagreement is
`E_SEMANTIC_DIGEST_MISMATCH`. A raw tensor carries zero.

### 15.3 Proof vectors

Every block tensor carries `proof_count = 64`, `proof_format = 1`: one
LE binary16 expected dequantized value per proof index, 128 bytes at
`proof_off + proof_rel_off`. A raw tensor carries `0 / 0 / 0`. For
`N = product(dims)`, `N >= 64`:

```text
entries  0..=15   index = entry
entries 16..=63   r = entry - 15   (1..=48)
                  index = 15 + floor(r * (N - 16) / 48)
```

The last index is `N - 1`. `N < 64` is `E_INVALID_QUANT_SHAPE`.

The producer computes each value with its own block decoder. The reader
checks them with its own (`TcfFile::verify_tensor_with`), so a producer
and a consumer that disagree on a block layout fail at first use.

### 15.3.1 Relationship between the two

The digest decides. The proof vector is a lazy screen at first use, so
header-only scheduling stays header-only. With `MUST_VERIFY_BEFORE_USE`
set, verification completes before the tensor's first dispatch.

## 16. Placement

A placement planner reads the directory only: `physical_span_bytes`,
`residency_class`, the sensitivity fields with their `calibration_id`,
and the access fields with their `workload_profile_id`. A measurement
without its provenance record carries no authority.

## 17. Error codes

Names are normative. `boostr::tcf::TcfError` carries them verbatim.

| Code | Raised when |
| ---- | ----------- |
| `E_BAD_MAGIC` | First 8 bytes are not the magic |
| `E_UNSUPPORTED_MAJOR` | `major != 1` |
| `E_UNKNOWN_REQUIRED_FEATURE` | Unknown or retired bit set, or a mandatory bit clear |
| `E_HEADER_DIGEST_MISMATCH` | `header_digest` fails |
| `E_DIRECTORY_DIGEST_MISMATCH` | `directory_digest` fails |
| `E_SECTION_BOUNDS` | A section overflows, overlaps, leaves `[192, data_off)`, `header_bytes != 192`, or a reference dangles |
| `E_MISALIGNED_SECTION` | A section or `data_offset` is not 64-aligned |
| `E_NONZERO_RESERVED` | A reserved field, unknown flag bit, or padding byte is non-zero |
| `E_INVALID_RANK` | `rank` outside 1..8, or a dimension at index `>= rank` non-zero |
| `E_INVALID_QUANT_SHAPE` | Block tensor with `rank < 2`, `K` not a block multiple, payload length disagreement, or proof fields inconsistent with the encoding |
| `E_MISSING_FALLBACK_REASON` | Encoding differs from module preference with `fallback_reason = NONE` |
| `E_PAYLOAD_DIGEST_MISMATCH` | `payload_digest` fails |
| `E_SEMANTIC_DIGEST_MISMATCH` | `semantic_digest` fails |
| `E_PROOF_MISMATCH` | A proof value differs from the decoder's value |
| `E_ACTIVATION_CONTRACT_MISMATCH` | No kernel satisfies the tensor's contract |
| `E_CONTRACT_DIGEST_MISMATCH` | A recomputed `contract_digest` differs, or one id resolves to two digests |
| `E_POLICY_DIGEST_MISMATCH` | A recomputed `policy_digest` differs |
| `E_RELATION_DIGEST_MISMATCH` | A recomputed `relation_digest` differs |
| `E_RESIDENT_BYTES_VIOLATION` | `resident_bytes` or `transfer_bytes` differs from `physical_span_bytes` |
| `E_UNSUPPORTED_ENCODING` | Encoding value not in the registry |
| `E_UNKNOWN_EXECUTION_ROLE` | `execution_role` not in the registry |
| `E_BAD_SCHEMA_ID` | `schema_id` names an unknown layout schema |
| `E_UNKNOWN_ENUM_VALUE` | Any other enumerated field holds an undefined value |
| `E_MISSING_WORKLOAD_PROFILE` | `ACCESS_PROFILE_VALID` set with no resolvable `workload_profile_id` |

## 18. Version policy

| Change | Mechanism |
| ------ | --------- |
| New optional record field | Consume a reserved range, no version bump |
| New encoding, role, or enum value | `minor` bump, readers reject unknown values |
| New behavior a reader must understand | New `required_features` bit |
| Changed byte layout of an existing encoding | `major` bump |

## 19. Implementation layout

| Path | Owns |
| ---- | ---- |
| `boostr/src/tcf/` | Container: records, digests, proofs, reader, writer |
| `boostr/src/format/tcf/` | Model loader: records to `QuantTensor`, block decoder for proofs |
| `compressr/src/formats/tcf/` | Producer: block writer, calibration and sensitivity records |
