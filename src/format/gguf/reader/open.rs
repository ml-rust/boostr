//! Opening a GGUF source (file, mmap, in-memory), header parsing, and raw byte access.

use super::super::GgufTensorInfo;
use super::super::io::{
    GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, align_offset, read_kv_pair, read_tensor_info, read_u32,
    read_u64,
};
use super::super::metadata::GgufMetadata;
use crate::error::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Backing storage for GGUF tensor data.
pub(super) enum GgufStorage {
    File(File),
    Mmap(Mmap),
    /// In-memory buffer (for wasm or data loaded via HTTP).
    InMemory(Vec<u8>),
}

/// GGUF file reader
pub struct Gguf {
    pub(super) storage: GgufStorage,
    pub(super) version: u32,
    pub(super) metadata: GgufMetadata,
    pub(super) tensors: HashMap<String, GgufTensorInfo>,
    pub(super) data_offset: u64,
}

impl Gguf {
    /// Open and parse a GGUF file using regular file I/O.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_impl(path, false)
    }

    /// Open a GGUF file with optional memory-mapping.
    ///
    /// When `use_mmap` is `true`, the tensor data region is memory-mapped for
    /// zero-copy reads. The header and metadata are still read via buffered I/O.
    /// Falls back to regular file I/O if mmap fails (e.g. on unsupported platforms).
    pub fn open_with_mmap<P: AsRef<Path>>(path: P, use_mmap: bool) -> Result<Self> {
        Self::open_impl(path, use_mmap)
    }

    /// Parse a GGUF model from an in-memory byte buffer.
    ///
    /// This is the primary entry point for wasm/browser usage where models are
    /// fetched via HTTP and provided as byte slices. The entire buffer is kept
    /// in memory — tensor reads are zero-copy slices into this buffer.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(&data);
        let (version, metadata, tensors, data_offset) = Self::parse_header(&mut cursor)?;

        Ok(Gguf {
            storage: GgufStorage::InMemory(data),
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Parse GGUF header (magic, version, metadata, tensor info) from any `Read + Seek`.
    fn parse_header<R: Read + Seek>(
        reader: &mut R,
    ) -> Result<(u32, GgufMetadata, HashMap<String, GgufTensorInfo>, u64)> {
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            return Err(Error::ModelError {
                reason: format!("invalid GGUF magic: 0x{magic:08x}"),
            });
        }

        let version = read_u32(reader)?;
        if !(1..=3).contains(&version) {
            return Err(Error::ModelError {
                reason: format!("unsupported GGUF version: {version}"),
            });
        }

        let tensor_count = read_u64(reader)?;
        let kv_count = read_u64(reader)?;

        let mut metadata = GgufMetadata::default();
        for _ in 0..kv_count {
            let (key, value) = read_kv_pair(reader, version)?;
            metadata.kv.insert(key, value);
        }

        let alignment = metadata
            .get_u32("general.alignment")
            .map(|v| v as usize)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let info = read_tensor_info(reader, version)?;
            tensors.insert(info.name.clone(), info);
        }

        let current_pos = reader.stream_position().map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        let data_offset = align_offset(current_pos, alignment);

        Ok((version, metadata, tensors, data_offset))
    }

    fn open_impl<P: AsRef<Path>>(path: P, use_mmap: bool) -> Result<Self> {
        let mut file = File::open(path.as_ref()).map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
        let mut reader = BufReader::new(&mut file);

        let (version, metadata, tensors, data_offset) = Self::parse_header(&mut reader)?;

        // Drop the BufReader so we can take ownership of `file` again.
        drop(reader);

        let storage = if use_mmap {
            // SAFETY: The file is read-only and we do not mutate the mapping.
            // The caller must not truncate or replace the file while the Gguf is live.
            match unsafe { Mmap::map(&file) } {
                Ok(mmap) => GgufStorage::Mmap(mmap),
                Err(_) => GgufStorage::File(file), // graceful fallback
            }
        } else {
            GgufStorage::File(file)
        };

        Ok(Gguf {
            storage,
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn tensor_info(&self, name: &str) -> Result<&GgufTensorInfo> {
        self.tensors.get(name).ok_or_else(|| Error::ModelError {
            reason: format!("GGUF tensor not found: {name}"),
        })
    }

    /// Read a byte slice from storage at the given absolute offset.
    pub(super) fn read_slice(
        &mut self,
        abs_offset: u64,
        size: usize,
        name: &str,
    ) -> Result<Vec<u8>> {
        match &mut self.storage {
            GgufStorage::Mmap(mmap) => {
                let start = abs_offset as usize;
                let end = start + size;
                if end > mmap.len() {
                    return Err(Error::ModelError {
                        reason: format!(
                            "tensor '{name}' data at [{start}..{end}) exceeds mmap length {}",
                            mmap.len()
                        ),
                    });
                }
                Ok(mmap[start..end].to_vec())
            }
            GgufStorage::InMemory(buf) => {
                let start = abs_offset as usize;
                let end = start + size;
                if end > buf.len() {
                    return Err(Error::ModelError {
                        reason: format!(
                            "tensor '{name}' data at [{start}..{end}) exceeds buffer length {}",
                            buf.len()
                        ),
                    });
                }
                Ok(buf[start..end].to_vec())
            }
            GgufStorage::File(file) => {
                let mut buf = vec![0u8; size];
                file.seek(SeekFrom::Start(abs_offset))
                    .map_err(|e| Error::ModelError {
                        reason: format!("IO seek error: {e}"),
                    })?;
                file.read_exact(&mut buf).map_err(|e| Error::ModelError {
                    reason: format!("IO read error: {e}"),
                })?;
                Ok(buf)
            }
        }
    }

    /// Read raw tensor data bytes.
    ///
    /// When backed by mmap or in-memory buffer, this copies from the backing store.
    /// When backed by a file, this seeks and reads.
    pub fn read_tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("GGUF tensor not found: {name}"),
            })?
            .clone();

        let abs_offset = self.data_offset + info.offset;
        let size = info.size_bytes();

        self.read_slice(abs_offset, size, name)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::io::{GGUF_MAGIC, align_offset};
    use super::super::super::types::{GgmlType, GgufValueType};
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: write a GGUF string (u64 length + bytes)
    fn write_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Build a minimal GGUF v3 byte buffer with one F32 tensor and one Q4_0 tensor.
    pub(in super::super) fn create_test_gguf_bytes() -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // 2 tensors
        buf.extend_from_slice(&2u64.to_le_bytes());
        // 2 KV pairs
        buf.extend_from_slice(&2u64.to_le_bytes());

        // KV 1: general.architecture = "test"
        write_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_str(&mut buf, "test");

        // KV 2: test.block_count = 4
        write_str(&mut buf, "test.block_count");
        buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());

        // Tensor 1: "weight_f32" F32 [4]
        write_str(&mut buf, "weight_f32");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
        buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Tensor 2: "weight_q4" Q4_0 [32]
        write_str(&mut buf, "weight_q4");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&(GgmlType::Q4_0 as u32).to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes()); // offset after 4 floats

        // Align to 32 bytes
        let aligned = buf.len().div_ceil(32) * 32;
        buf.resize(aligned, 0);

        // Data: weight_f32 = [1.0, 2.0, 3.0, 4.0]
        for f in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&f.to_le_bytes());
        }

        // Data: weight_q4 - Q4_0 block (scale=1.0, all nibbles=8 -> dequant to 0)
        let scale_bits = half::f16::from_f32(1.0).to_bits();
        buf.push((scale_bits & 0xFF) as u8);
        buf.push(((scale_bits >> 8) & 0xFF) as u8);
        buf.extend(std::iter::repeat_n(0x88u8, 16));

        buf
    }

    /// Create a minimal GGUF v3 file with one F32 tensor and one Q4_0 tensor
    pub(in super::super) fn create_test_gguf() -> NamedTempFile {
        let buf = create_test_gguf_bytes();
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_open_gguf() {
        let f = create_test_gguf();
        let gguf = Gguf::open(f.path()).unwrap();
        assert_eq!(gguf.version(), 3);
        assert_eq!(gguf.len(), 2);
        assert_eq!(gguf.metadata().architecture(), Some("test"));
        assert_eq!(gguf.metadata().block_count(), Some(4));
    }

    #[test]
    fn test_tensor_info_gguf() {
        let f = create_test_gguf();
        let gguf = Gguf::open(f.path()).unwrap();

        let f32_info = gguf.tensor_info("weight_f32").unwrap();
        assert_eq!(f32_info.shape, vec![4]);
        assert_eq!(f32_info.ggml_type, GgmlType::F32);
        assert_eq!(f32_info.size_bytes(), 16);

        let q4_info = gguf.tensor_info("weight_q4").unwrap();
        assert_eq!(q4_info.shape, vec![32]);
        assert_eq!(q4_info.ggml_type, GgmlType::Q4_0);
        assert_eq!(q4_info.size_bytes(), 18);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn test_tensor_not_found() {
        let f = create_test_gguf();
        let gguf = Gguf::open(f.path()).unwrap();
        assert!(gguf.tensor_info("nonexistent").is_err());
    }

    #[test]
    fn test_open_with_mmap() {
        let (_, device) = cpu_setup();
        let f = create_test_gguf();
        let mut gguf = Gguf::open_with_mmap(f.path(), true).unwrap();
        assert_eq!(gguf.version(), 3);
        let tensor = gguf
            .load_tensor_f32::<CpuRuntime>("weight_f32", &device)
            .unwrap();
        assert_eq!(tensor.shape(), &[4]);
        let data = tensor.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_open_without_mmap() {
        let (_, device) = cpu_setup();
        let f = create_test_gguf();
        // use_mmap=false should behave identically to open()
        let mut gguf = Gguf::open_with_mmap(f.path(), false).unwrap();
        let tensor = gguf
            .load_tensor_f32::<CpuRuntime>("weight_f32", &device)
            .unwrap();
        let data = tensor.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_bytes() {
        let buf = create_test_gguf_bytes();
        let gguf = Gguf::from_bytes(buf).unwrap();
        assert_eq!(gguf.version(), 3);
        assert_eq!(gguf.len(), 2);
        assert_eq!(gguf.metadata().architecture(), Some("test"));
    }
}
