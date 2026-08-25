//! Paged KV cache serialization for disaggregated prefill/decode transfer.
//!
//! # Wire format — `LayeredPagedKvCache`
//!
//! ```text
//! [magic:       u32 LE = 0xB0057B02]   — paged KV cache
//! [version:     u32 LE = 1]
//! [dtype_tag:   u32 LE]                — numr DType discriminant: 1 = f32, 2 = f16, 3 = bf16
//! [num_layers:  u32 LE]
//! [block_size:  u32 LE]
//! [seq_len:     u32 LE]
//! For each layer:
//!   [num_blocks:    u32 LE]
//!   [num_heads:     u32 LE]
//!   [head_dim:      u32 LE]
//!   [k_data:        num_blocks * block_size * num_heads * head_dim * E bytes (LE)]
//!   [v_data:        num_blocks * block_size * num_heads * head_dim * E bytes (LE)]
//!   [block_table_len: u32 LE]
//!   [block_ids:     block_table_len * 4 bytes (u32 LE)]   — BlockId = u32
//! ```
//!
//! `E` is the element width the dtype tag names: 4 bytes for f32, 2 for f16 and bf16.
//! Elements are little-endian on every host and are decoded element-wise, so a received
//! buffer needs no alignment.
//!
//! The magic differs from the flat cache's `0xB0057B01`, so a paged buffer handed to
//! [`super::kv_serialize::deserialize_kv_cache`] errors on the first four bytes instead
//! of reading block-table fields as layer dimensions.
//!
//! Both magics sit above `0xB0057B00` = 2_952_003_840. The pre-header format started
//! straight at `num_layers`, and no cache has billions of layers — every layer
//! allocates its own K and V tensors — so an old buffer can never present a matching
//! magic. It is rejected up front instead of being misparsed.

use crate::inference::{BlockTable, LayeredPagedKvCache};
use crate::{DType, Runtime, Tensor};
use anyhow::{Result, anyhow};

/// Wire magic for the flat (non-paged) KV cache format.
pub(super) const FLAT_MAGIC: u32 = 0xB005_7B01;

/// Wire magic for the paged KV cache format.
pub(super) const PAGED_MAGIC: u32 = 0xB005_7B02;

/// Version both KV cache wire formats currently write.
///
/// Bumped only when the byte layout changes such that this reader would misparse the
/// new bytes. A reader that meets any other version errors rather than guessing.
pub(super) const WIRE_VERSION: u32 = 1;

/// Byte length of the shared `[magic][version][dtype_tag]` prefix.
pub(super) const HEADER_LEN: usize = 12;

/// Encode a dtype as its wire tag, or refuse it by name.
///
/// The tag is numr's own `DType` discriminant, which numr documents as stable for
/// serialization, so neither side needs a private mapping table.
///
/// Only dtypes carried end to end are accepted: f32, f16 and bf16. Every other dtype is
/// refused here rather than reinterpreted or converted — reinterpreting corrupts the
/// cache, and converting changes the values the peer receives without telling it.
pub(super) fn dtype_to_tag(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(dtype as u32),
        other => Err(unsupported_dtype_err(other)),
    }
}

/// The refusal for a dtype no part of this format carries, named in the message.
///
/// One place builds it so the serializer's guard and every element codec below refuse
/// the same set with the same wording.
pub(super) fn unsupported_dtype_err(dtype: DType) -> anyhow::Error {
    anyhow!(
        "KV cache wire format cannot carry dtype {dtype}: it carries f32, f16 and bf16 \
         only. Refusing to transfer rather than reinterpreting {dtype} bytes or converting \
         them, since either hands the peer numbers it cannot know are wrong."
    )
}

/// Decode a wire dtype tag, naming the tag when this build does not carry it.
pub(super) fn dtype_from_tag(tag: u32) -> Result<DType> {
    if tag == DType::F32 as u32 {
        Ok(DType::F32)
    } else if tag == DType::F16 as u32 {
        Ok(DType::F16)
    } else if tag == DType::BF16 as u32 {
        Ok(DType::BF16)
    } else {
        Err(anyhow!(
            "KV cache header has unknown dtype tag {tag}; this build carries f32 (tag {}), \
             f16 (tag {}) and bf16 (tag {})",
            DType::F32 as u32,
            DType::F16 as u32,
            DType::BF16 as u32
        ))
    }
}

/// Write the `[magic][version][dtype_tag]` prefix, refusing an uncarryable dtype.
pub(super) fn write_header(buf: &mut Vec<u8>, magic: u32, dtype: DType) -> Result<()> {
    let tag = dtype_to_tag(dtype)?;
    buf.extend_from_slice(&magic.to_le_bytes());
    buf.extend_from_slice(&WIRE_VERSION.to_le_bytes());
    buf.extend_from_slice(&tag.to_le_bytes());
    Ok(())
}

/// Parse the `[magic][version][dtype_tag]` prefix, returning the element dtype.
///
/// `what` names the reader in the error text ("KV cache" / "Paged KV cache"). A magic
/// belonging to the sibling format is called out by name, so feeding a paged buffer to
/// the flat reader says exactly that instead of failing later on a dimension field.
pub(super) fn read_header(bytes: &[u8], expected_magic: u32, what: &str) -> Result<DType> {
    if bytes.len() < HEADER_LEN {
        return Err(anyhow!(
            "{what} buffer truncated in the header: need {HEADER_LEN} bytes, got {}",
            bytes.len()
        ));
    }

    let magic = read_u32_le(bytes, 0)?;
    if magic != expected_magic {
        let hint = if magic == FLAT_MAGIC {
            " — that is the flat KV cache magic"
        } else if magic == PAGED_MAGIC {
            " — that is the paged KV cache magic"
        } else {
            ""
        };
        return Err(anyhow!(
            "{what} buffer has magic 0x{magic:08X}, expected 0x{expected_magic:08X}{hint}. \
             A buffer from the pre-header format starts at its layer count, which can never \
             equal 0x{expected_magic:08X}, so it lands here as a clean error."
        ));
    }

    let version = read_u32_le(bytes, 4)?;
    if version != WIRE_VERSION {
        return Err(anyhow!(
            "{what} buffer has wire version {version}, but this build reads version \
             {WIRE_VERSION} only"
        ));
    }

    dtype_from_tag(read_u32_le(bytes, 8)?)
}

/// Read a little-endian `u32` at `offset`, erroring rather than panicking when the
/// buffer is too short.
///
/// The callers bounds-check before each read, so this is defence in depth: it keeps a
/// truncated or hostile payload from turning into a slice-index panic inside a server.
fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset.checked_add(4).ok_or_else(|| {
        anyhow!(
            "Paged KV cache read offset {} overflows the address space",
            offset
        )
    })?;
    let slice = bytes.get(offset..end).ok_or_else(|| {
        anyhow!(
            "Paged KV cache buffer too short: need {} bytes, got {}",
            end,
            bytes.len()
        )
    })?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Decode a little-endian `f32` run without requiring 4-byte alignment.
///
/// `len` is a multiple of 4 and already bounds-checked by the caller, so the trailing
/// partial chunk `chunks_exact` would leave is always empty.
pub(super) fn read_f32_le_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

/// Append a contiguous tensor's elements as little-endian bytes at the tensor's own
/// element width.
///
/// Each arm reads with a host type whose width equals the dtype width. Reading at a
/// wider type (`try_to_vec::<f32>()` on a bf16 tensor) copies `numel * 4` bytes out of a
/// `numel * 2` byte allocation and reinterprets them, which is a silent wrong answer at
/// the peer.
///
/// `to_le_bytes` per element keeps the wire little-endian on a big-endian host, where
/// `bytemuck::cast_slice` would emit native-endian bytes instead.
pub(super) fn append_le_elements<R>(buf: &mut Vec<u8>, tensor: &Tensor<R>) -> Result<()>
where
    R: Runtime<DType = DType>,
{
    match tensor.dtype() {
        DType::F32 => {
            for x in tensor.try_to_vec::<f32>()? {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        DType::F16 => {
            for x in tensor.try_to_vec::<half::f16>()? {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        DType::BF16 => {
            for x in tensor.try_to_vec::<half::bf16>()? {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        other => return Err(unsupported_dtype_err(other)),
    }
    Ok(())
}

/// Convert a little-endian element run off the wire into native-endian storage bytes.
///
/// Element-wise, so the input needs no alignment guarantee — `bytemuck::cast_slice`
/// panics outright on a misaligned slice — and the wire stays explicitly little-endian
/// whatever the host's byte order.
pub(super) fn le_wire_to_native_bytes(bytes: &[u8], dtype: DType) -> Result<Vec<u8>> {
    let out: Vec<u8> = match dtype {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&c| f32::from_le_bytes(c).to_ne_bytes())
            .collect(),
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&c| half::f16::from_le_bytes(c).to_ne_bytes())
            .collect(),
        DType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&c| half::bf16::from_le_bytes(c).to_ne_bytes())
            .collect(),
        other => return Err(unsupported_dtype_err(other)),
    };
    Ok(out)
}

/// Widen a little-endian element run off the wire to host `f32` values.
///
/// f16 and bf16 both widen to f32 exactly, so this loses nothing.
pub(super) fn read_le_f32_widened(bytes: &[u8], dtype: DType) -> Result<Vec<f32>> {
    let out: Vec<f32> = match dtype {
        DType::F32 => read_f32_le_vec(bytes),
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| half::f16::from_le_bytes(c).to_f32())
            .collect(),
        DType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| half::bf16::from_le_bytes(c).to_f32())
            .collect(),
        other => return Err(unsupported_dtype_err(other)),
    };
    Ok(out)
}

/// Build a contiguous tensor of `dtype` from a little-endian element run.
pub(super) fn tensor_from_le_wire<R>(
    bytes: &[u8],
    dtype: DType,
    shape: &[usize],
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
{
    let native = le_wire_to_native_bytes(bytes, dtype)?;
    let storage = numr::tensor::Storage::<R>::from_bytes(&native, dtype, device)?;
    Ok(Tensor::<R>::from_storage_contiguous(storage, shape))
}

/// Per-layer deserialized K/V data that could not be directly loaded into the
/// paged cache (the public API does not expose mutable raw-block writes).
///
/// The caller is responsible for feeding this data into the paged cache via
/// whatever mechanism is available (e.g. by using the data in model forward
/// passes directly, or by writing via `update()` with proper slot mappings).
///
/// `k_data` and `v_data` are always host `f32`: f16 and bf16 widen to f32 exactly, so
/// nothing is lost. `dtype` records the dtype the bytes arrived as — the same dtype the
/// returned cache was built with — so the caller narrows back before writing.
#[derive(Debug)]
pub struct PagedLayerData {
    pub k_data: Vec<f32>,
    pub v_data: Vec<f32>,
    pub block_ids: Vec<u32>,
    pub dtype: DType,
}

/// Serialize a `LayeredPagedKvCache` and its associated `BlockTable` into bytes.
///
/// # Errors
///
/// Returns an error naming the dtype if the cache's element type is not one the wire
/// format carries (f32, f16, bf16), or if the layers do not all share one dtype — the
/// header declares a single dtype for the whole cache.
pub fn serialize_paged_kv_cache<R>(
    cache: &LayeredPagedKvCache<R>,
    _block_table: &BlockTable,
) -> Result<Vec<u8>>
where
    R: Runtime<DType = DType>,
{
    let num_layers = cache.num_layers() as u32;
    let block_size = cache.block_size() as u32;
    let seq_len = cache.seq_len() as u32;

    // The header declares one dtype for the whole cache, taken from layer 0 and checked
    // against every other layer below. A cache with no layers carries no elements, so its
    // tag is the format's default rather than a property of absent data.
    let wire_dtype = if cache.num_layers() == 0 {
        DType::F32
    } else {
        cache.layer(0).dtype()
    };

    let mut buf: Vec<u8> = Vec::new();

    write_header(&mut buf, PAGED_MAGIC, wire_dtype)?;
    buf.extend_from_slice(&num_layers.to_le_bytes());
    buf.extend_from_slice(&block_size.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());

    for layer_idx in 0..num_layers as usize {
        let layer = cache.layer(layer_idx);
        let bt = cache.block_table(layer_idx);

        let num_blocks = layer.num_blocks() as u32;
        let num_heads = layer.num_heads() as u32;
        let head_dim = layer.head_dim() as u32;

        buf.extend_from_slice(&num_blocks.to_le_bytes());
        buf.extend_from_slice(&num_heads.to_le_bytes());
        buf.extend_from_slice(&head_dim.to_le_bytes());

        // The header declares one dtype for the whole cache. A layer that disagrees with
        // layer 0 would be written at layer 0's width and read back as the wrong numbers,
        // so refuse rather than emit a buffer no reader can parse.
        let layer_dtype = layer.dtype();
        if layer_dtype != wire_dtype {
            return Err(anyhow!(
                "Paged KV cache layer {layer_idx} has dtype {layer_dtype}, but the header \
                 declares {wire_dtype} from layer 0. The wire format carries one dtype for \
                 the whole cache; refusing to transfer a mixed-dtype cache."
            ));
        }

        append_le_elements(&mut buf, layer.k_cache())?;
        append_le_elements(&mut buf, layer.v_cache())?;

        let block_ids = &bt.blocks;
        let bt_len = block_ids.len() as u32;
        buf.extend_from_slice(&bt_len.to_le_bytes());
        for &id in block_ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }

    Ok(buf)
}

/// Deserialize bytes into a `LayeredPagedKvCache`, per-layer K/V data, and
/// per-layer block tables.
///
/// Returns `(cache, layer_data, block_tables)`. The `layer_data` contains the
/// deserialized K/V float data for each layer — the caller must write this into
/// the cache's backing tensors (the paged cache API does not expose raw block
/// writes, so this data is returned separately).
pub fn deserialize_paged_kv_cache<R>(
    bytes: &[u8],
    device: &R::Device,
) -> Result<(LayeredPagedKvCache<R>, Vec<PagedLayerData>, Vec<BlockTable>)>
where
    R: Runtime<DType = DType>,
{
    // Magic, version and dtype tag are checked before any dimension is read, so a foreign
    // or stale buffer errors on its first four bytes rather than on an absurd dimension.
    let wire_dtype = read_header(bytes, PAGED_MAGIC, "Paged KV cache")?;
    let elem_size = wire_dtype.size_in_bytes();

    if bytes.len() < HEADER_LEN + 12 {
        return Err(anyhow!(
            "Paged KV cache buffer too short: need {} bytes, got {}",
            HEADER_LEN + 12,
            bytes.len()
        ));
    }

    let num_layers = read_u32_le(bytes, HEADER_LEN)? as usize;
    let block_size = read_u32_le(bytes, HEADER_LEN + 4)? as usize;
    let seq_len = read_u32_le(bytes, HEADER_LEN + 8)? as usize;

    let mut cursor = HEADER_LEN + 12;

    struct RawLayerParams {
        num_blocks: usize,
        num_heads: usize,
        head_dim: usize,
        k_data: Vec<f32>,
        v_data: Vec<f32>,
        block_ids: Vec<u32>,
    }

    // `num_layers` is attacker-controlled: reserve lazily rather than trusting it.
    let mut raw_layers: Vec<RawLayerParams> = Vec::new();

    for layer_idx in 0..num_layers {
        if cursor + 12 > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} header",
                layer_idx
            ));
        }
        let num_blocks = read_u32_le(bytes, cursor)? as usize;
        let num_heads = read_u32_le(bytes, cursor + 4)? as usize;
        let head_dim = read_u32_le(bytes, cursor + 8)? as usize;
        cursor += 12;

        // Every factor here comes off the wire. In release builds `*` wraps, so an
        // unchecked product can land back inside the buffer and slip past the
        // truncation check below while the header still claims enormous dimensions.
        let data_bytes = num_blocks
            .checked_mul(block_size)
            .and_then(|n| n.checked_mul(num_heads))
            .and_then(|n| n.checked_mul(head_dim))
            .and_then(|n| n.checked_mul(elem_size))
            .ok_or_else(|| {
                anyhow!(
                    "Paged KV cache layer {} dimensions overflow: \
                     num_blocks={} block_size={} num_heads={} head_dim={}",
                    layer_idx,
                    num_blocks,
                    block_size,
                    num_heads,
                    head_dim
                )
            })?;

        let both = data_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(cursor))
            .ok_or_else(|| {
                anyhow!(
                    "Paged KV cache layer {} data size overflows the address space",
                    layer_idx
                )
            })?;
        if both > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} data",
                layer_idx
            ));
        }

        // Read element-wise rather than `bytemuck::cast_slice`: a received buffer
        // carries no alignment guarantee, and `cast_slice` panics when `&bytes[cursor..]`
        // is not aligned for the element type. This also makes the decode explicitly
        // little-endian, matching the documented wire format. f16 and bf16 widen to f32
        // exactly, so the returned host values are lossless whatever the wire dtype.
        let k_data = read_le_f32_widened(&bytes[cursor..cursor + data_bytes], wire_dtype)?;
        cursor += data_bytes;
        let v_data = read_le_f32_widened(&bytes[cursor..cursor + data_bytes], wire_dtype)?;
        cursor += data_bytes;

        if cursor + 4 > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} block table length",
                layer_idx
            ));
        }
        let bt_len = read_u32_le(bytes, cursor)? as usize;
        cursor += 4;

        let bt_end = bt_len
            .checked_mul(4)
            .and_then(|n| n.checked_add(cursor))
            .ok_or_else(|| {
                anyhow!(
                    "Paged KV cache layer {} block table length {} overflows",
                    layer_idx,
                    bt_len
                )
            })?;
        if bt_end > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} block table data",
                layer_idx
            ));
        }
        // `bt_len` is bounded by the buffer now, so reserving it cannot be used to
        // force a large allocation from a 4-byte field.
        let mut block_ids = Vec::with_capacity(bt_len);
        for i in 0..bt_len {
            block_ids.push(read_u32_le(bytes, cursor + i * 4)?);
        }
        cursor = bt_end;

        raw_layers.push(RawLayerParams {
            num_blocks,
            num_heads,
            head_dim,
            k_data,
            v_data,
            block_ids,
        });
    }

    if raw_layers.is_empty() {
        let cache = LayeredPagedKvCache::<R>::new(0, 0, block_size, 1, 64, wire_dtype, device)?;
        return Ok((cache, Vec::new(), Vec::new()));
    }

    let first = &raw_layers[0];
    let mut paged_cache = LayeredPagedKvCache::<R>::new(
        num_layers,
        first.num_blocks,
        block_size,
        first.num_heads,
        first.head_dim,
        wire_dtype,
        device,
    )?;
    paged_cache.set_seq_len(seq_len);

    // The header's dtype tag was passed to the constructor above. State the symmetry as a
    // check rather than leaving it implied: if the constructor ever stops honouring the
    // requested dtype, this errors instead of returning a cache the peer's tag misdescribes.
    let built_dtype = paged_cache.layer(0).dtype();
    if built_dtype != wire_dtype {
        return Err(anyhow!(
            "Paged KV cache header declares dtype {wire_dtype}, but the reconstructed \
             cache has dtype {built_dtype}"
        ));
    }

    let mut block_tables: Vec<BlockTable> = Vec::with_capacity(num_layers);
    let mut layer_data: Vec<PagedLayerData> = Vec::with_capacity(num_layers);

    for params in &raw_layers {
        let mut bt = BlockTable::new(block_size);
        bt.blocks = params.block_ids.clone();
        bt.num_tokens = seq_len;
        block_tables.push(bt);

        layer_data.push(PagedLayerData {
            k_data: params.k_data.clone(),
            v_data: params.v_data.clone(),
            block_ids: params.block_ids.clone(),
            dtype: wire_dtype,
        });
    }

    Ok((paged_cache, layer_data, block_tables))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the 12-byte `[magic][version][dtype_tag]` prefix a paged buffer starts with.
    fn paged_header(dtype: DType) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&PAGED_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(dtype as u32).to_le_bytes());
        bytes
    }

    /// A hostile header whose dimensions multiply past `usize::MAX` must be rejected.
    ///
    /// Without checked arithmetic the product wraps in release builds to a small
    /// `data_bytes`, the truncation check then passes, and the enormous `num_blocks` /
    /// `num_heads` / `head_dim` reach the cache constructor anyway. The assertion pins
    /// the error, not merely "did not panic".
    #[test]
    fn test_deserialize_rejects_dimension_overflow() {
        let mut bytes = paged_header(DType::F32);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // num_blocks
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // num_heads
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // head_dim

        // `expect_err` would require `Debug` on the success type, which the cache
        // does not implement.
        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("overflowing dimensions must be rejected");
        };
        assert!(
            err.to_string().contains("overflow"),
            "expected an overflow error, got: {err}"
        );
    }

    /// The reader must not require the input buffer to be 4-byte aligned.
    ///
    /// A buffer arriving off the wire carries no alignment guarantee, and
    /// `bytemuck::cast_slice::<u8, f32>` panics outright on a misaligned slice. Reading
    /// a deliberately offset copy reproduces that panic if the element-wise decode is
    /// reverted.
    #[test]
    fn test_deserialize_accepts_misaligned_buffer() {
        let mut payload = paged_header(DType::F32);
        payload.extend_from_slice(&1u32.to_le_bytes()); // num_layers
        payload.extend_from_slice(&1u32.to_le_bytes()); // block_size
        payload.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        payload.extend_from_slice(&1u32.to_le_bytes()); // num_blocks
        payload.extend_from_slice(&1u32.to_le_bytes()); // num_heads
        payload.extend_from_slice(&2u32.to_le_bytes()); // head_dim
        for x in [1.0f32, 2.0] {
            payload.extend_from_slice(&x.to_le_bytes()); // k_data
        }
        for x in [3.0f32, 4.0] {
            payload.extend_from_slice(&x.to_le_bytes()); // v_data
        }
        payload.extend_from_slice(&0u32.to_le_bytes()); // block_table_len

        // Shift by one byte so the f32 runs start at an odd address.
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&payload);

        let (_cache, layer_data, _tables) = deserialize_paged_kv_cache::<crate::CpuRuntime>(
            &shifted[1..],
            &crate::CpuDevice::new(),
        )
        .expect("a misaligned buffer must deserialize");
        assert_eq!(layer_data[0].k_data, vec![1.0, 2.0]);
        assert_eq!(layer_data[0].v_data, vec![3.0, 4.0]);
        assert_eq!(layer_data[0].dtype, DType::F32);
    }

    /// A dtype the format cannot carry must be refused by name at serialize time.
    ///
    /// This replaces the old `test_serialize_rejects_non_f32_cache`, which asserted that
    /// BF16 was refused. BF16 now round-trips (see `test_roundtrip_bf16_cache`), so the
    /// refusal is pinned on F64, which the format still does not carry.
    #[test]
    fn test_serialize_rejects_unsupported_dtype() {
        let device = crate::CpuDevice::new();
        let cache =
            LayeredPagedKvCache::<crate::CpuRuntime>::new(1, 2, 16, 2, 4, DType::F64, &device)
                .unwrap();
        let block_table = BlockTable::new(16);

        let Err(err) = serialize_paged_kv_cache(&cache, &block_table) else {
            panic!("an F64 paged KV cache must be refused, not written at another width");
        };
        let msg = err.to_string();
        // `DType` displays lowercase (`f64`, `bf16`), so compare case-insensitively
        // rather than against a spelling the type does not produce.
        let lower = msg.to_lowercase();
        assert!(lower.contains("f64"), "error must name the dtype: {msg}");
        assert!(
            lower.contains("f32") && lower.contains("f16") && lower.contains("bf16"),
            "error must list what the format does carry: {msg}"
        );
    }

    /// The declared dtype must survive the round trip: bf16 in, bf16 cache out, and the
    /// same bf16 values on the wire.
    #[test]
    fn test_roundtrip_bf16_cache() {
        let device = crate::CpuDevice::new();
        let cache =
            LayeredPagedKvCache::<crate::CpuRuntime>::new(1, 2, 4, 2, 4, DType::BF16, &device)
                .unwrap();
        let block_table = BlockTable::new(4);

        let bytes = serialize_paged_kv_cache(&cache, &block_table).unwrap();
        assert_eq!(read_u32_le(&bytes, 0).unwrap(), PAGED_MAGIC);
        assert_eq!(read_u32_le(&bytes, 4).unwrap(), WIRE_VERSION);
        assert_eq!(read_u32_le(&bytes, 8).unwrap(), DType::BF16 as u32);

        // 2 bytes per element, not 4: the layer holds 2 blocks * 4 tokens * 2 heads * 4 dims.
        let elems = 2 * 4 * 2 * 4;
        assert_eq!(bytes.len(), HEADER_LEN + 12 + 12 + elems * 2 * 2 + 4);

        let (restored, layer_data, _tables) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &device).unwrap();
        assert_eq!(restored.layer(0).dtype(), DType::BF16);
        assert_eq!(layer_data[0].dtype, DType::BF16);
        assert_eq!(layer_data[0].k_data, vec![0.0f32; elems]);
        assert_eq!(layer_data[0].v_data, vec![0.0f32; elems]);
    }

    /// f32 stays the 4-byte-per-element format it always was.
    #[test]
    fn test_roundtrip_f32_cache() {
        let device = crate::CpuDevice::new();
        let cache =
            LayeredPagedKvCache::<crate::CpuRuntime>::new(1, 2, 4, 2, 4, DType::F32, &device)
                .unwrap();
        let block_table = BlockTable::new(4);

        let bytes = serialize_paged_kv_cache(&cache, &block_table).unwrap();
        assert_eq!(read_u32_le(&bytes, 8).unwrap(), DType::F32 as u32);

        let elems = 2 * 4 * 2 * 4;
        assert_eq!(bytes.len(), HEADER_LEN + 12 + 12 + elems * 4 * 2 + 4);

        let (restored, layer_data, _tables) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &device).unwrap();
        assert_eq!(restored.layer(0).dtype(), DType::F32);
        assert_eq!(layer_data[0].dtype, DType::F32);
        assert_eq!(layer_data[0].k_data.len(), elems);
    }

    /// A buffer shorter than the 12-byte prefix errors in the header, naming the length.
    #[test]
    fn test_deserialize_too_short_buffer() {
        let bytes = [0u8; 4];
        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("a 4-byte buffer must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("truncated in the header"),
            "expected a header truncation error, got: {msg}"
        );
        assert!(
            msg.contains("need 12 bytes, got 4"),
            "error must name both lengths: {msg}"
        );
    }

    /// A flat buffer must be refused by the paged reader on its magic.
    #[test]
    fn test_deserialize_rejects_flat_magic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FLAT_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(DType::F32 as u32).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // num_layers
        bytes.extend_from_slice(&16u32.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len

        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("a flat KV cache buffer must be refused by the paged reader");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("magic 0xB0057B01") && msg.contains("expected 0xB0057B02"),
            "error must name both magics: {msg}"
        );
        assert!(
            msg.contains("that is the flat KV cache magic"),
            "error must name the sibling format: {msg}"
        );
    }

    /// An unknown wire version errors, naming the version it got.
    #[test]
    fn test_deserialize_rejects_unknown_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&PAGED_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&(DType::F32 as u32).to_le_bytes());

        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("an unknown wire version must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("wire version 7") && msg.contains("version 1 only"),
            "error must name the version it got and the one it reads: {msg}"
        );
    }

    /// An unknown dtype tag errors, naming the tag.
    #[test]
    fn test_deserialize_rejects_unknown_dtype_tag() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&PAGED_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&99u32.to_le_bytes());

        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("an unknown dtype tag must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("unknown dtype tag 99"),
            "error must name the tag: {msg}"
        );
        assert!(
            msg.contains("bf16 (tag 3)"),
            "error must name the tags it does carry: {msg}"
        );
    }

    #[test]
    fn test_deserialize_zero_layers() {
        let mut bytes = paged_header(DType::F32);
        bytes.extend_from_slice(&0u32.to_le_bytes()); // num_layers = 0
        bytes.extend_from_slice(&16u32.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len

        let (cache, layer_data, block_tables) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
                .unwrap();
        assert_eq!(cache.num_layers(), 0);
        assert!(layer_data.is_empty());
        assert!(block_tables.is_empty());
    }

    #[test]
    fn test_deserialize_truncated_layer_header() {
        let mut bytes = paged_header(DType::F32);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers = 1
        bytes.extend_from_slice(&16u32.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len
        // Missing layer header → should error

        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("a missing layer header must be refused");
        };
        assert!(
            err.to_string().contains("truncated at layer 0 header"),
            "expected a layer-header truncation error, got: {err}"
        );
    }
}
