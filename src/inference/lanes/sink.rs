//! The batched KV container [`LaneTable::admit`](super::LaneTable::admit) installs a lane into.
//!
//! A growable [`LayeredKvCache`] and a [`FrozenLayeredKvCache`] both take a
//! lane write, but they refuse differently: the growable cache grows to fit
//! a sequence that outgrows it, the frozen cache refuses instead. `admit`
//! is written once against this trait, so it never chooses between the two
//! itself.

use crate::error::Result;
use crate::inference::kv_cache::{FrozenLayeredKvCache, LayeredKvCache};
use crate::ops::traits::KvCacheOps;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

/// A batched KV container [`LaneTable::admit`](crate::inference::lanes::table::LaneTable::admit)
/// can install one sequence's lane into.
pub trait LaneKvSink<R: Runtime<DType = DType>>
where
    R::Client: IndexingOps<R>,
{
    /// Install `src`'s single sequence into lane `lane`.
    ///
    /// # Errors
    ///
    /// An implementation names the lane and, when it refuses rather than
    /// grows, the write end and the capacity it would have exceeded.
    fn install_lane<C: KvCacheOps<R>>(
        &mut self,
        src: &LayeredKvCache<R>,
        lane: usize,
        client: &C,
    ) -> Result<()>;
}

impl<R: Runtime<DType = DType>> LaneKvSink<R> for &mut LayeredKvCache<R>
where
    R::Client: IndexingOps<R>,
{
    fn install_lane<C: KvCacheOps<R>>(
        &mut self,
        src: &LayeredKvCache<R>,
        lane: usize,
        client: &C,
    ) -> Result<()> {
        LayeredKvCache::install_lane(self, src, lane, client)
    }
}

impl<R: Runtime<DType = DType>> LaneKvSink<R> for &mut FrozenLayeredKvCache<R>
where
    R::Client: IndexingOps<R>,
{
    fn install_lane<C: KvCacheOps<R>>(
        &mut self,
        src: &LayeredKvCache<R>,
        lane: usize,
        client: &C,
    ) -> Result<()> {
        FrozenLayeredKvCache::install_lane(self, src, lane, client)
    }
}
