//! Wire constants and element-level codec helpers for the paged KV cache format.
//!
//! See the module docs in `super` for the full wire layout.

use crate::{DType, Runtime, Tensor};
use anyhow::{Result, anyhow};

/// Wire magic for the flat (non-paged) KV cache format.
pub(in crate::distributed::inference) const FLAT_MAGIC: u32 = 0xB005_7B01;

/// Wire magic for the paged KV cache format.
pub(in crate::distributed::inference) const PAGED_MAGIC: u32 = 0xB005_7B02;

/// Version both KV cache wire formats currently write.
///
/// Bumped only when the byte layout changes such that this reader would misparse the
/// new bytes. A reader that meets any other version errors rather than guessing.
pub(in crate::distributed::inference) const WIRE_VERSION: u32 = 1;

/// Byte length of the shared `[magic][version][dtype_tag]` prefix.
pub(in crate::distributed::inference) const HEADER_LEN: usize = 12;

/// Encode a dtype as its wire tag, or refuse it by name.
///
/// The tag is numr's own `DType` discriminant, which numr documents as stable for
/// serialization, so neither side needs a private mapping table.
///
/// Only dtypes carried end to end are accepted: f32, f16 and bf16. Every other dtype is
/// refused here rather than reinterpreted or converted — reinterpreting corrupts the
/// cache, and converting changes the values the peer receives without telling it.
pub(in crate::distributed::inference) fn dtype_to_tag(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(dtype as u32),
        other => Err(unsupported_dtype_err(other)),
    }
}

/// The refusal for a dtype no part of this format carries, named in the message.
///
/// One place builds it so the serializer's guard and every element codec below refuse
/// the same set with the same wording.
pub(in crate::distributed::inference) fn unsupported_dtype_err(dtype: DType) -> anyhow::Error {
    anyhow!(
        "KV cache wire format cannot carry dtype {dtype}: it carries f32, f16 and bf16 \
         only. Refusing to transfer rather than reinterpreting {dtype} bytes or converting \
         them, since either hands the peer numbers it cannot know are wrong."
    )
}

/// Decode a wire dtype tag, naming the tag when this build does not carry it.
pub(in crate::distributed::inference) fn dtype_from_tag(tag: u32) -> Result<DType> {
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
pub(in crate::distributed::inference) fn write_header(
    buf: &mut Vec<u8>,
    magic: u32,
    dtype: DType,
) -> Result<()> {
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
pub(in crate::distributed::inference) fn read_header(
    bytes: &[u8],
    expected_magic: u32,
    what: &str,
) -> Result<DType> {
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
pub(super) fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
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
pub(in crate::distributed::inference) fn read_f32_le_vec(bytes: &[u8]) -> Vec<f32> {
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
pub(in crate::distributed::inference) fn append_le_elements<R>(
    buf: &mut Vec<u8>,
    tensor: &Tensor<R>,
) -> Result<()>
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
pub(in crate::distributed::inference) fn le_wire_to_native_bytes(
    bytes: &[u8],
    dtype: DType,
) -> Result<Vec<u8>> {
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
pub(in crate::distributed::inference) fn read_le_f32_widened(
    bytes: &[u8],
    dtype: DType,
) -> Result<Vec<f32>> {
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
pub(in crate::distributed::inference) fn tensor_from_le_wire<R>(
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
