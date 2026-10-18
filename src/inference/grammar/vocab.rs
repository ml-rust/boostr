//! `VocabBytes`: the byte string of every token id, uploaded once per device.

use std::any::{Any, TypeId};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use numr::dtype::DType;
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};

/// Upper bound on any count or id a device grammar table stores.
///
/// The device tables hold F32 values. F32 holds every integer up to 2^24 exactly.
pub const F32_EXACT_LIMIT: usize = 1 << 24;

/// The byte string of every token id of one tokenizer, in id order.
///
/// The caller decodes each id with its own tokenizer. Bytes stay raw, so a
/// piece that is not valid UTF-8 keeps its bytes, and the grammar DFA decides.
///
/// A clone shares the bytes and every device copy. The first
/// `DeviceGrammarDfa` built on a device uploads the vocabulary there. Each
/// later one on that device reuses the upload. A device copy is keyed by
/// runtime type and `Device::id`, and it lives until the last clone drops.
#[derive(Clone)]
pub struct VocabBytes {
    inner: Arc<Inner>,
}

struct Inner {
    bytes: Vec<u8>,
    /// `[vocab_size + 1]`: token `i` owns `bytes[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<u32>,
    uploads: Mutex<Vec<Upload>>,
}

/// One device copy, type-erased over the runtime.
struct Upload {
    runtime: TypeId,
    device: usize,
    tensors: Box<dyn Any + Send + Sync>,
}

/// The device copy of a `VocabBytes`. A clone shares the device storage.
///
/// `bytes` holds one F32 per byte. It holds one zero when every token is
/// empty, because a device binding needs one element. `offsets` holds
/// `[vocab_size + 1]` F32 values.
pub struct DeviceVocab<R: Runtime> {
    bytes: Tensor<R>,
    offsets: Tensor<R>,
    total_bytes: usize,
}

impl<R: Runtime> Clone for DeviceVocab<R> {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
            offsets: self.offsets.clone(),
            total_bytes: self.total_bytes,
        }
    }
}

impl<R: Runtime> DeviceVocab<R> {
    /// Concatenated token bytes, one F32 per byte.
    pub fn bytes(&self) -> &Tensor<R> {
        &self.bytes
    }

    /// Byte offsets of each token, `[vocab_size + 1]` F32.
    pub fn offsets(&self) -> &Tensor<R> {
        &self.offsets
    }

    /// Byte count of all tokens, without the padding element.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

impl VocabBytes {
    /// Build from the byte string of each token id, in id order.
    ///
    /// An empty vocabulary is an error. More than `F32_EXACT_LIMIT` tokens or
    /// bytes in total is an error, because the device tables store F32.
    pub fn new<I, T>(tokens: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<[u8]>,
    {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for token in tokens {
            bytes.extend_from_slice(token.as_ref());
            if bytes.len() > F32_EXACT_LIMIT {
                return Err(too_large("total token bytes", bytes.len()));
            }
            if offsets.len() > F32_EXACT_LIMIT {
                return Err(too_large("token count", offsets.len()));
            }
            let end = u32::try_from(bytes.len())
                .map_err(|_| too_large("total token bytes", bytes.len()))?;
            offsets.push(end);
        }
        if offsets.len() < 2 {
            return Err(Error::InvalidArgument {
                arg: "tokens",
                reason: "VocabBytes::new: the vocabulary is empty. Pass one byte string per \
                         token id"
                    .to_string(),
            });
        }
        Ok(Self {
            inner: Arc::new(Inner {
                bytes,
                offsets,
                uploads: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Number of token ids.
    pub fn len(&self) -> usize {
        self.inner.offsets.len().saturating_sub(1)
    }

    /// True when the vocabulary holds no token. `new` refuses that, so it is false.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Byte count of all tokens.
    pub fn total_bytes(&self) -> usize {
        self.inner.bytes.len()
    }

    /// The bytes of token `id`, or `None` past the vocabulary.
    ///
    /// The sampler passes them to `ByteDfa::walk` to advance its host state.
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        let at = usize::try_from(id).ok()?;
        let start = *self.inner.offsets.get(at)? as usize;
        let end = *self.inner.offsets.get(at.checked_add(1)?)? as usize;
        self.inner.bytes.get(start..end)
    }

    /// True when `self` and `other` share one allocation and one upload cache.
    pub fn ptr_eq(&self, other: &VocabBytes) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// The device copy on `device`. The first call per device uploads it.
    pub fn device<R: Runtime<DType = DType>>(&self, device: &R::Device) -> Result<DeviceVocab<R>> {
        let runtime = TypeId::of::<R>();
        let device_id = device.id();
        let mut uploads = self
            .inner
            .uploads
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let cached = uploads
            .iter()
            .filter(|u| u.runtime == runtime && u.device == device_id)
            .find_map(|u| u.tensors.downcast_ref::<DeviceVocab<R>>());
        if let Some(found) = cached {
            return Ok(found.clone());
        }
        let mut bytes: Vec<f32> = self.inner.bytes.iter().map(|&b| f32::from(b)).collect();
        if bytes.is_empty() {
            bytes.push(0.0);
        }
        // Each offset is at most F32_EXACT_LIMIT, so the F32 value is exact.
        let offsets: Vec<f32> = self.inner.offsets.iter().map(|&o| o as f32).collect();
        let uploaded = DeviceVocab {
            bytes: Tensor::<R>::from_slice(&bytes, &[bytes.len()], device)?,
            offsets: Tensor::<R>::from_slice(&offsets, &[offsets.len()], device)?,
            total_bytes: self.inner.bytes.len(),
        };
        uploads.push(Upload {
            runtime,
            device: device_id,
            tensors: Box::new(uploaded.clone()),
        });
        Ok(uploaded)
    }
}

impl fmt::Debug for VocabBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabBytes")
            .field("len", &self.len())
            .field("total_bytes", &self.total_bytes())
            .finish()
    }
}

fn too_large(what: &str, got: usize) -> Error {
    Error::InvalidArgument {
        arg: "tokens",
        reason: format!(
            "VocabBytes::new: {what} is {got}, above the limit of {F32_EXACT_LIMIT}. The device \
             tables store F32, which holds integers exactly only up to that limit"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn token_bytes_returns_each_raw_piece() {
        let tokens: [&[u8]; 4] = [b"ab", b"", &[0xFF, 0x80], b"c"];
        let vocab = VocabBytes::new(tokens).unwrap();
        assert_eq!(vocab.len(), 4);
        assert!(!vocab.is_empty());
        assert_eq!(vocab.total_bytes(), 5);
        assert_eq!(vocab.token_bytes(0), Some(&b"ab"[..]));
        assert_eq!(vocab.token_bytes(1), Some(&b""[..]));
        assert_eq!(vocab.token_bytes(2), Some(&[0xFF, 0x80][..]));
        assert_eq!(vocab.token_bytes(3), Some(&b"c"[..]));
        assert_eq!(vocab.token_bytes(4), None);
        assert_eq!(vocab.token_bytes(u32::MAX), None);
    }

    #[test]
    fn an_empty_vocabulary_is_an_error() {
        let err = VocabBytes::new(Vec::<Vec<u8>>::new())
            .err()
            .map(|e| e.to_string());
        assert!(err.is_some_and(|m| m.contains("vocabulary is empty")));
    }

    #[test]
    fn too_many_bytes_is_an_error() {
        let big = vec![b'a'; F32_EXACT_LIMIT + 1];
        let err = VocabBytes::new([big]).err().map(|e| e.to_string());
        assert!(err.is_some_and(|m| m.contains("total token bytes")));
    }

    #[test]
    fn one_upload_per_device_and_clones_share_it() {
        let device = CpuDevice::new();
        let vocab = VocabBytes::new([&b"a"[..], b"bc"]).unwrap();
        let clone = vocab.clone();
        assert!(vocab.ptr_eq(&clone));
        let first = vocab.device::<CpuRuntime>(&device).unwrap();
        let second = clone.device::<CpuRuntime>(&device).unwrap();
        assert!(first.bytes().shares_storage_with(second.bytes()));
        assert!(first.offsets().shares_storage_with(second.offsets()));
        assert_eq!(first.bytes().to_vec::<f32>(), vec![97.0, 98.0, 99.0]);
        assert_eq!(first.offsets().to_vec::<f32>(), vec![0.0, 1.0, 3.0]);
        assert_eq!(first.total_bytes(), 3);
        let other = VocabBytes::new([&b"a"[..], b"bc"]).unwrap();
        let third = other.device::<CpuRuntime>(&device).unwrap();
        assert!(!third.bytes().shares_storage_with(first.bytes()));
    }

    #[test]
    fn all_empty_tokens_upload_one_padding_byte() {
        let device = CpuDevice::new();
        let vocab = VocabBytes::new([&b""[..], b""]).unwrap();
        let up = vocab.device::<CpuRuntime>(&device).unwrap();
        assert_eq!(up.bytes().to_vec::<f32>(), vec![0.0]);
        assert_eq!(up.total_bytes(), 0);
        assert_eq!(up.offsets().to_vec::<f32>(), vec![0.0, 0.0, 0.0]);
    }
}
