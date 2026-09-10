//! Collecting the importance matrix: the arming switch, the DEVICE-side
//! accumulation, and the single host transfer at the end.
//!
//! # How capture attaches, and what it costs when off
//!
//! [`MaybeQuantLinear::forward`](crate::nn::MaybeQuantLinear::forward) opens
//! with [`is_armed`], a relaxed load of one process-wide `AtomicBool` followed
//! by a branch that is never taken in an ordinary run. Nothing else changes:
//! no field is added to any layer, no constructor is threaded, no allocation
//! happens, and no struct grows a byte. A never-taken, perfectly predicted
//! branch in front of a GEMM is the cheapest attachment available short of a
//! compile-time feature, and a feature would fracture the build — a binary
//! compiled without it could never collect, and every model file would need
//! `cfg` arms.
//!
//! # No bulk device-to-host transfers
//!
//! [`observe`] squares the activation and reduces it over the token axis ON
//! THE DEVICE, producing one `[in_features]` vector, then accumulates into a
//! device tensor held across the whole run. Activations never reach the host.
//! [`finish`] performs the ONLY transfer: one `[in_features]` vector per
//! measured weight, once, at the end.
//!
//! # One thread, one collection
//!
//! State lives in thread-local storage, so the collection belongs to the
//! thread that armed it. [`observe`] called from any other thread is an
//! ERROR, not a silent partial result — statistics split across threads would
//! look complete and be wrong. A collector is armed once per process;
//! [`arm`] refuses a second concurrent collection.
//!
//! # Naming
//!
//! Nothing is guessed. [`VarBuilder::take_maybe_quant_linear`](crate::nn::VarBuilder::take_maybe_quant_linear)
//! calls [`register_name`] with the weight's autograd `TensorId` and the
//! CHECKPOINT KEY it just loaded that weight from, so an entry's name is the
//! exact string the quantizer sees. A weight observed under an id no loader
//! registered makes [`finish`] fail, naming the count — a wrong name silently
//! produces a wrong importance vector, which is worse than none.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::ThreadId;

use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

use super::format::{ImportanceEntry, ImportanceMatrix};
use crate::error::{Error, Result};

/// Armed state. Read once per `MaybeQuantLinear::forward`; written twice per
/// process at most.
static ARMED: AtomicBool = AtomicBool::new(false);
/// The thread that called [`arm`]. Every observation must come from it.
static OWNER: Mutex<Option<ThreadId>> = Mutex::new(None);

thread_local! {
    /// Weight `TensorId` to checkpoint tensor name, filled at load time.
    static NAMES: RefCell<HashMap<TensorId, String>> = RefCell::new(HashMap::new());
    /// Per-runtime accumulators, type-erased because a `static` cannot be
    /// generic over `R`. The `TypeId` key is `TypeId::of::<R>()` and the value
    /// is always `HashMap<TensorId, Accumulator<R>>` for that same `R`.
    static SUMS: RefCell<HashMap<TypeId, Box<dyn Any>>> = RefCell::new(HashMap::new());
}

/// One weight's running column sums, resident on the device.
struct Accumulator<R: Runtime> {
    sums: Tensor<R>,
    rows: u64,
}

fn collect_error(reason: impl Into<String>) -> Error {
    Error::QuantError {
        reason: reason.into(),
    }
}

/// Whether an importance collection is running.
///
/// The hot-path guard. One relaxed atomic load, inlined into the caller.
#[inline(always)]
pub fn is_armed() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// Start a collection on the calling thread.
///
/// Every `MaybeQuantLinear::Standard` forward from here until [`finish`] or
/// [`disarm`] contributes to it. A second concurrent collection is refused:
/// two runs sharing one accumulator would sum statistics from different
/// corpora into one vector.
pub fn arm() -> Result<()> {
    let mut owner = OWNER
        .lock()
        .map_err(|_| collect_error("importance collector lock is poisoned"))?;
    if owner.is_some() {
        return Err(collect_error(
            "an importance collection is already running in this process",
        ));
    }
    *owner = Some(std::thread::current().id());
    NAMES.with(|names| names.borrow_mut().clear());
    SUMS.with(|sums| sums.borrow_mut().clear());
    ARMED.store(true, Ordering::Relaxed);
    Ok(())
}

/// Stop collecting and drop everything accumulated so far.
///
/// [`finish`] already does this after draining. Call this directly only to
/// abandon a collection.
pub fn disarm() {
    ARMED.store(false, Ordering::Relaxed);
    NAMES.with(|names| names.borrow_mut().clear());
    SUMS.with(|sums| sums.borrow_mut().clear());
    if let Ok(mut owner) = OWNER.lock() {
        *owner = None;
    }
}

/// Bind a weight's autograd id to the checkpoint key it was loaded from.
///
/// A no-op when no collection is running, so a loader can call it
/// unconditionally. Re-registering an id overwrites the name: a layer rebuilt
/// around a reshaped weight (a grown `lm_head`, say) registers again after
/// the rebuild, and the later call is the one that describes the live tensor.
pub fn register_name(id: TensorId, name: &str) {
    if !is_armed() {
        return;
    }
    NAMES.with(|names| names.borrow_mut().insert(id, name.to_string()));
}

/// Every checkpoint key a loader has bound to a weight in this collection.
///
/// A run compares this against what [`finish`] measured to tell a weight that
/// was LOADED but never exercised from one that was measured. Call it before
/// [`finish`], which clears the bindings along with everything else.
pub fn registered_names() -> Vec<String> {
    let mut names: Vec<String> = NAMES.with(|names| names.borrow().values().cloned().collect());
    names.sort_unstable();
    names
}

/// Accumulate `sum(x^2)` per input column of `activations`, on the device.
///
/// `activations` is the input to a linear layer, `[..., in_features]`. Every
/// leading axis is a token axis and is reduced away, leaving one
/// `[in_features]` vector that is added into this weight's running total.
/// Nothing is transferred to the host.
///
/// `weight_id` is the layer's weight `TensorId` — the key [`register_name`]
/// bound a checkpoint name to.
pub fn observe<R, C>(weight_id: TensorId, client: &C, activations: &Tensor<R>) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    if !is_armed() {
        return Ok(());
    }
    let owner = OWNER
        .lock()
        .map_err(|_| collect_error("importance collector lock is poisoned"))?
        .ok_or_else(|| collect_error("importance collector armed without an owning thread"))?;
    if owner != std::thread::current().id() {
        return Err(collect_error(
            "importance collection ran on a second thread; statistics split across threads \
             would look complete and be wrong. Run the calibration forward pass on the thread \
             that armed the collector",
        ));
    }

    let shape = activations.shape();
    let in_features = *shape
        .last()
        .ok_or_else(|| collect_error("importance collection: linear input has no dimensions"))?;
    let rows: usize = shape[..shape.len() - 1].iter().product();

    // F32 before the square. A BF16 running sum over a long corpus saturates
    // after a few thousand tokens and the tail of the calibration stops
    // contributing at all.
    let dense = if activations.dtype() == DType::F32 {
        activations.clone()
    } else {
        client.cast(activations, DType::F32)?
    };
    let squared = client.mul(&dense, &dense)?;
    let columns = if shape.len() == 1 {
        squared
    } else {
        let axes: Vec<usize> = (0..shape.len() - 1).collect();
        client.sum(&squared, &axes, false)?
    };
    if columns.numel() != in_features {
        return Err(collect_error(format!(
            "importance collection: column reduction produced {} value(s), expected {in_features}",
            columns.numel()
        )));
    }

    accumulate::<R, C>(weight_id, client, columns, rows as u64)
}

/// Add one `[in_features]` vector into this weight's running total, on device.
fn accumulate<R, C>(weight_id: TensorId, client: &C, columns: Tensor<R>, rows: u64) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    let mut outcome = Ok(());
    SUMS.with(|store| {
        let mut store = store.borrow_mut();
        let slot = store
            .entry(TypeId::of::<R>())
            .or_insert_with(|| Box::new(HashMap::<TensorId, Accumulator<R>>::new()));
        // The `TypeId` key and the map's value type are written together
        // above and nowhere else, so this downcast cannot fail.
        let Some(map) = slot.downcast_mut::<HashMap<TensorId, Accumulator<R>>>() else {
            outcome = Err(collect_error(
                "importance collector holds an accumulator of the wrong runtime type",
            ));
            return;
        };
        match map.get_mut(&weight_id) {
            Some(existing) => match client.add(&existing.sums, &columns) {
                Ok(sum) => {
                    existing.sums = sum;
                    existing.rows += rows;
                }
                Err(e) => outcome = Err(Error::Numr(e)),
            },
            None => {
                map.insert(
                    weight_id,
                    Accumulator {
                        sums: columns,
                        rows,
                    },
                );
            }
        }
    });
    outcome
}

/// Drain the collection into an [`ImportanceMatrix`] and disarm.
///
/// This is the ONLY device-to-host transfer in the whole collection: one
/// `[in_features]` `f32` vector per measured weight. `token_count` is the
/// tokens the run fed the model, recorded in the file so a consumer can tell
/// a small calibration from a large one.
///
/// A weight observed under an id no loader registered a name for is an error,
/// naming the count. Guessing a name would produce a plausible file whose
/// vectors belong to the wrong tensors.
pub fn finish<R: Runtime<DType = DType>>(token_count: u64) -> Result<ImportanceMatrix> {
    let measured: Vec<(TensorId, Vec<f32>, u64)> = SUMS.with(|store| {
        let mut store = store.borrow_mut();
        let Some(slot) = store.remove(&TypeId::of::<R>()) else {
            return Ok(Vec::new());
        };
        let map = *slot
            .downcast::<HashMap<TensorId, Accumulator<R>>>()
            .map_err(|_| {
                collect_error("importance collector holds an accumulator of the wrong runtime type")
            })?;
        map.into_iter()
            .map(|(id, acc)| {
                // `try_to_vec` refuses a strided tensor; a reduction output
                // is normally already contiguous and this makes it so
                // regardless of which backend produced it.
                let sums = acc.sums.contiguous().map_err(Error::Numr)?;
                let values = sums.try_to_vec::<f32>().map_err(Error::Numr)?;
                Ok((id, values, acc.rows))
            })
            .collect::<Result<Vec<_>>>()
    })?;

    // Names copied out, then the collector is torn down: everything below
    // works on owned data, so an error while assembling the matrix cannot
    // leave an armed collector behind for a later run in this process.
    let names = NAMES.with(|names| names.borrow().clone());
    disarm();

    let mut matrix = ImportanceMatrix::new(token_count);
    let mut unnamed = 0usize;
    for (id, values, rows) in measured {
        match names.get(&id) {
            Some(name) => matrix.insert(ImportanceEntry {
                name: name.clone(),
                rows,
                sums: values,
            })?,
            None => unnamed += 1,
        }
    }
    if unnamed > 0 {
        return Err(collect_error(format!(
            "{unnamed} measured weight(s) carry no checkpoint name. Every collected tensor must \
             be registered by the loader that read it, so the file's names are the ones the \
             quantizer sees"
        )));
    }
    Ok(matrix)
}
