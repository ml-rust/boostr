//! Prefix states kept across requests: [`PrefixStateCache`], bounded in
//! bytes with the least recently used state out first, and
//! [`cached_prefix`], the prefix a plan runs over, read from that cache or
//! prefilled and kept in it.
//!
//! A kept state is lent read-only to every later run with the same key:
//! its lanes attend to its KV cache without copying it, and each lane
//! copies its GDN state on the device as it does from a fresh prefix. A
//! hit costs no copy a miss does not make, and no host transfer.

use super::backbone::{DecisionBackbone, PrefixState};
use super::packing::{RunnerConfig, peak_lane_bytes};
use super::plan::DecisionPlan;
use super::runner::prefill_prefix;
use crate::error::Result;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

/// What a [`PrefixStateCache`] did, and what it holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StateCacheStats {
    /// Lookups that found a state.
    pub hits: u64,
    /// Lookups that found none.
    pub misses: u64,
    /// States removed to make room, or by [`PrefixStateCache::clear`].
    pub evictions: u64,
    /// States held.
    pub entries: usize,
    /// Bytes the held states take, as [`PrefixState::bytes`] counts them.
    pub bytes: usize,
    /// The byte bound. 0: the cache is off.
    pub capacity: usize,
}

struct Entry<R: Runtime> {
    state: Arc<PrefixState<R>>,
    bytes: usize,
    used: u64,
}

/// Prefix states kept across requests, keyed by the caller.
///
/// A key names everything a state depends on: the model's weights, an
/// adapter included, and the prefix token ids. Equal keys must give equal
/// states. The cache never checks it, so a key that leaves out the weights
/// hands one model's state to another.
///
/// Only a state that keeps no hidden rows is kept: kept rows depend on the
/// plan's prefix gathers, which no key names. An off cache (capacity 0)
/// keeps nothing and counts nothing.
pub struct PrefixStateCache<K, R: Runtime> {
    capacity: usize,
    bytes: usize,
    clock: u64,
    entries: HashMap<K, Entry<R>>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<K, R> PrefixStateCache<K, R>
where
    K: Eq + Hash + Clone,
    R: Runtime<DType = DType>,
    R::Client: IndexingOps<R>,
{
    /// An empty cache that holds at most `capacity` bytes. 0 is off.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            bytes: 0,
            clock: 0,
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Whether the cache keeps states.
    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }

    /// Bytes the held states take.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// States held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no state is held.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The counts and the holdings now.
    pub fn stats(&self) -> StateCacheStats {
        StateCacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            entries: self.entries.len(),
            bytes: self.bytes,
            capacity: self.capacity,
        }
    }

    /// The state under `key`, marked most recently used. Counts a hit or a
    /// miss; an off cache returns `None` and counts nothing.
    pub fn get(&mut self, key: &K) -> Option<Arc<PrefixState<R>>> {
        if !self.is_enabled() {
            return None;
        }
        self.clock += 1;
        match self.entries.get_mut(key) {
            Some(entry) => {
                entry.used = self.clock;
                self.hits += 1;
                Some(Arc::clone(&entry.state))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Keep `state` under `key`, replacing what the key held, and remove
    /// the least recently used states until the cache fits its capacity.
    /// Returns whether it was kept: an off cache, a state that keeps
    /// hidden rows, and a state larger than the capacity are not.
    pub fn insert(&mut self, key: K, state: Arc<PrefixState<R>>) -> bool {
        let bytes = state.bytes();
        if !self.is_enabled() || !state.rows.is_empty() || bytes > self.capacity {
            return false;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.bytes -= old.bytes;
        }
        self.shrink_to(self.capacity - bytes, None);
        self.clock += 1;
        self.bytes += bytes;
        let used = self.clock;
        self.entries.insert(key, Entry { state, bytes, used });
        true
    }

    /// Remove the least recently used states, never `keep`, until the
    /// cache holds at most `limit` bytes or only `keep` is left. Returns
    /// how many it removed.
    pub fn shrink_to(&mut self, limit: usize, keep: Option<&K>) -> usize {
        let mut removed = 0;
        while self.bytes > limit {
            let victim = self
                .entries
                .iter()
                .filter(|(k, _)| keep != Some(*k))
                .min_by_key(|(_, e)| e.used)
                .map(|(_, e)| (e.used, e.bytes));
            let Some((used, bytes)) = victim else {
                break;
            };
            // Every state takes a fresh clock tick, so `used` names one entry
            // and the key, a token vector, is not cloned to remove it.
            self.entries.retain(|_, e| e.used != used);
            self.bytes -= bytes;
            removed += 1;
        }
        self.evictions += removed as u64;
        removed
    }

    /// Remove every state, as when the model it was built on unloads.
    /// Returns how many it removed.
    pub fn clear(&mut self) -> usize {
        let removed = self.entries.len();
        self.entries.clear();
        self.bytes = 0;
        self.evictions += removed as u64;
        removed
    }
}

/// The prefix state a run reads, and whether a cache held it.
pub struct CachedPrefix<R: Runtime> {
    /// The state. Pass `Some(&*state)` to `run_decision`.
    pub state: Arc<PrefixState<R>>,
    /// `true` when the cache held it, so its prefix was not run.
    pub hit: bool,
}

/// The prefix state `plan` runs over: the one `cache` holds under `key`,
/// else the prefix run now and kept under `key`.
///
/// `None` when `plan` has no prefix, when the cache is off, or when a
/// gather of `plan` reads the prefix: `run_decision` then runs the prefix
/// itself, as with no cache.
///
/// When `cfg.max_lane_bytes` is set, the cached states and the run share
/// it. Before the run, the least recently used states go until the cache,
/// the state this run reads counted once, plus the largest pass of the
/// run ([`peak_lane_bytes`]) fits the budget. The state this run reads is
/// never removed for it, and a run whose own state and lanes pass the
/// budget runs as it would with no cache. `cfg` is read, not changed, so a
/// hit packs the same passes a miss does.
///
/// A prefix that fails to run keeps nothing, a cancel included: `cfg.cancel`
/// is read before the prefix runs. A prefix that ran to its end is kept
/// even when the run is cancelled afterwards, so a caller that gave up and
/// sends the request again finds it.
///
/// # Errors
///
/// Plan errors as [`DecisionPlan::validate`]; budget errors as
/// [`peak_lane_bytes`]; `Error::Cancelled` when `cfg.cancel` is set before
/// the prefix; backbone errors propagate.
pub fn cached_prefix<R, C, B, K>(
    cache: &mut PrefixStateCache<K, R>,
    client: &C,
    backbone: &B,
    plan: &DecisionPlan,
    key: K,
    cfg: &RunnerConfig,
) -> Result<Option<CachedPrefix<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: IndexingOps<R>,
    C: ModelClient<R>,
    B: DecisionBackbone<R>,
    K: Eq + Hash + Clone,
{
    if plan.prefix.is_empty() || !cache.is_enabled() {
        return Ok(None);
    }
    plan.validate()?;
    if !plan.prefix_positions().is_empty() {
        return Ok(None);
    }
    let peak = peak_lane_bytes(plan, cfg, |width| backbone.lane_bytes(width))?;
    if let Some(state) = cache.get(&key) {
        if let Some(budget) = cfg.max_lane_bytes {
            cache.shrink_to(budget.saturating_sub(peak), Some(&key));
        }
        return Ok(Some(CachedPrefix { state, hit: true }));
    }
    if let Some(budget) = cfg.max_lane_bytes {
        let fresh = backbone.lane_bytes(plan.prefix_len());
        cache.shrink_to(budget.saturating_sub(peak).saturating_sub(fresh), None);
    }
    cfg.check_cancel(|| "before the prefix".into())?;
    let state = Arc::new(prefill_prefix(client, backbone, plan)?);
    cache.insert(key, Arc::clone(&state));
    Ok(Some(CachedPrefix { state, hit: false }))
}

#[cfg(test)]
mod tests {
    use super::super::cancel::CancelToken;
    use super::super::plan::{DecisionPlan, PrefixInput};
    use super::super::runner::tests::{PREFIX, at, close, model, plan, run, tokens};
    use super::*;
    use crate::error::Error;
    use crate::model::qwen35::Qwen35Model;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    /// A key as a server builds one: the weights it runs, then the ids.
    type Key = (u64, Vec<u32>);
    type Cache = PrefixStateCache<Key, CpuRuntime>;

    /// The test plan with its prefix tokens drawn with `salt`.
    fn plan_with(salt: usize) -> DecisionPlan {
        let mut p = plan(PREFIX);
        p.prefix = PrefixInput::Tokens(tokens(PREFIX, salt));
        p.gathers.retain(|g| g.row.is_some());
        p
    }

    fn key(model: u64, p: &DecisionPlan) -> Key {
        (model, p.prefix.tokens().to_vec())
    }

    fn lookup(
        cache: &mut Cache,
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        p: &DecisionPlan,
        cfg: &RunnerConfig,
    ) -> Result<Option<CachedPrefix<CpuRuntime>>> {
        cached_prefix(cache, client, model, p, key(1, p), cfg)
    }

    /// The state `lookup` gives for a plan with a cacheable prefix.
    fn kept(
        cache: &mut Cache,
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        p: &DecisionPlan,
        cfg: &RunnerConfig,
    ) -> CachedPrefix<CpuRuntime> {
        let found = lookup(cache, client, model, p, cfg).unwrap();
        found.expect("a plan with a prefix and no prefix gathers is cached")
    }

    /// The prefix of `p` run alone, keeping the hidden rows at `rows`.
    fn prefilled(
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        p: &DecisionPlan,
        rows: &[usize],
    ) -> Arc<PrefixState<CpuRuntime>> {
        Arc::new(
            model
                .prefill_prefix(client, p.prefix.tokens(), rows)
                .unwrap(),
        )
    }

    /// A hit lends the kept state, and its answers equal a run that
    /// prefills, for the plan that kept it and for other rows after the
    /// same prefix.
    #[test]
    fn hit_equals_a_fresh_run() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let cfg = RunnerConfig::default();
        let mut cache = Cache::new(usize::MAX);
        let first = plan_with(0);
        let miss = kept(&mut cache, &client, &model, &first, &cfg);
        assert!(!miss.hit);
        assert_eq!(miss.state.bytes(), model.lane_bytes(PREFIX));
        let mut other = first.clone();
        for (r, row) in other.rows.iter_mut().enumerate() {
            *row = tokens(row.len(), r + 7);
        }
        for p in [&first, &other] {
            let hit = kept(&mut cache, &client, &model, p, &cfg);
            assert!(hit.hit);
            assert!(Arc::ptr_eq(&hit.state, &miss.state));
            let cached = run(&client, &model, p, Some(&*hit.state), &cfg).unwrap();
            let fresh = run(&client, &model, p, None, &cfg).unwrap();
            for (a, b) in cached.raw_logits.iter().zip(&fresh.raw_logits) {
                close(a, b, 1e-6, "cached prefix");
            }
        }
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (2, 1, 1));
        assert_eq!(stats.bytes, model.lane_bytes(PREFIX));
    }

    /// Keys differ by weights or by ids; the least recently used state
    /// goes first when the bytes run out.
    #[test]
    fn keys_and_lru_by_bytes() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let one = model.lane_bytes(PREFIX);
        let mut cache = Cache::new(2 * one);
        let cfg = RunnerConfig::default();
        let (a, b, c) = (plan_with(0), plan_with(1), plan_with(2));
        for p in [&a, &b] {
            assert!(!kept(&mut cache, &client, &model, p, &cfg).hit);
        }
        let variant = cached_prefix(&mut cache, &client, &model, &a, key(2, &a), &cfg);
        assert!(!variant.unwrap().unwrap().hit, "another adapter's weights");
        assert_eq!(cache.stats().evictions, 1);
        assert!(
            cache.get(&key(1, &a)).is_none(),
            "a was least recently used"
        );
        assert!(cache.get(&key(1, &b)).is_some());
        assert!(!kept(&mut cache, &client, &model, &c, &cfg).hit);
        assert!(cache.get(&key(2, &a)).is_none(), "b was read after it");
        assert!(cache.get(&key(1, &b)).is_some());
        let stats = cache.stats();
        assert_eq!(
            (stats.entries, stats.bytes, stats.evictions),
            (2, 2 * one, 2)
        );
        assert_eq!(cache.clear(), 2);
        assert_eq!(
            (cache.len(), cache.bytes(), cache.stats().evictions),
            (0, 0, 4)
        );
        let mut small = Cache::new(one - 1);
        let state = prefilled(&client, &model, &a, &[]);
        assert!(!small.insert(key(1, &a), state), "larger than the capacity");
    }

    /// Before a run the cache gives up states so that it, the run's own
    /// state and its largest pass fit the lane budget; the state the run
    /// reads stays.
    #[test]
    fn lane_budget_evicts_before_a_run() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let one = model.lane_bytes(PREFIX);
        let free = RunnerConfig::default();
        let (a, b, c) = (plan_with(0), plan_with(1), plan_with(2));
        let peak = peak_lane_bytes(&a, &free, |w| model.lane_bytes(w)).unwrap();
        let mut cache = Cache::new(usize::MAX);
        for p in [&a, &b] {
            lookup(&mut cache, &client, &model, p, &free).unwrap();
        }
        let room_for_one = RunnerConfig {
            max_lane_bytes: Some(peak + 2 * one),
            ..RunnerConfig::default()
        };
        assert!(!kept(&mut cache, &client, &model, &c, &room_for_one).hit);
        assert_eq!((cache.len(), cache.stats().evictions), (2, 1));
        assert!(cache.get(&key(1, &a)).is_none(), "a went first");
        let only_the_run = RunnerConfig {
            max_lane_bytes: Some(peak),
            ..RunnerConfig::default()
        };
        let hit = kept(&mut cache, &client, &model, &b, &only_the_run);
        assert!(hit.hit);
        assert_eq!(cache.len(), 1, "the state the run reads stays");
        let out = run(&client, &model, &b, Some(&*hit.state), &only_the_run).unwrap();
        let fresh = run(&client, &model, &b, None, &only_the_run).unwrap();
        for (x, y) in out.raw_logits.iter().zip(&fresh.raw_logits) {
            close(x, y, 1e-6, "under a budget");
        }
    }

    /// A cancel before the prefix and a prefix that fails keep nothing.
    #[test]
    fn failed_or_cancelled_prefix_keeps_nothing() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let mut cache = Cache::new(usize::MAX);
        let token = CancelToken::new();
        token.cancel();
        let cancelled = RunnerConfig {
            cancel: Some(token),
            ..RunnerConfig::default()
        };
        let a = plan_with(0);
        let err = lookup(&mut cache, &client, &model, &a, &cancelled).err();
        assert!(matches!(err, Some(Error::Cancelled { .. })), "{err:?}");
        let mut long = plan_with(0);
        long.prefix = PrefixInput::Tokens(tokens(model.max_positions() + 1, 0));
        let cfg = RunnerConfig::default();
        assert!(lookup(&mut cache, &client, &model, &long, &cfg).is_err());
        assert!(cache.is_empty());
        assert_eq!(cache.stats().misses, 1);
    }

    /// An off cache, a plan with no prefix and a plan that gathers in its
    /// prefix leave the prefix to the run and count nothing.
    #[test]
    fn off_cache_and_prefix_gathers_are_left_to_the_run() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let cfg = RunnerConfig::default();
        let a = plan_with(0);
        let mut off = Cache::new(0);
        assert!(
            lookup(&mut off, &client, &model, &a, &cfg)
                .unwrap()
                .is_none()
        );
        assert!(!off.insert(key(1, &a), prefilled(&client, &model, &a, &[])));
        assert_eq!(off.stats(), StateCacheStats::default());

        let mut cache = Cache::new(usize::MAX);
        let gathers = plan(PREFIX);
        assert!(gathers.gathers.contains(&at(None, 0)));
        assert!(
            lookup(&mut cache, &client, &model, &gathers, &cfg)
                .unwrap()
                .is_none()
        );
        let bare = plan(0);
        assert!(
            lookup(&mut cache, &client, &model, &bare, &cfg)
                .unwrap()
                .is_none()
        );
        let rows = prefilled(&client, &model, &a, &[1]);
        assert!(!cache.insert(key(1, &a), rows), "a state that keeps rows");
        assert_eq!(cache.stats().hits + cache.stats().misses, 0);
    }
}
