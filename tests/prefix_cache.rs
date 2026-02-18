use boostr::inference::memory::CpuBlockAllocator;
use boostr::inference::prefix_cache::{
    CacheResult, PrefixCache, PrefixCacheConfig, SequencePrefixState,
};

fn create_cache() -> PrefixCache<CpuBlockAllocator> {
    let allocator = CpuBlockAllocator::new(100, 16);
    let config = PrefixCacheConfig {
        enabled: true,
        max_cached_blocks: 50,
        min_prefix_tokens: 4,
        block_size: 16,
    };
    PrefixCache::new(allocator, config)
}

#[test]
fn test_prefix_cache_create() {
    let cache = create_cache();
    assert!(cache.config().enabled);
    assert_eq!(cache.config().block_size, 16);
}

#[test]
fn test_cache_miss() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    let result = cache.get_or_allocate_blocks(1, &tokens).unwrap();

    assert!(!result.is_hit());
    assert_eq!(result.blocks().len(), 2);

    let stats = cache.stats();
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.blocks_allocated, 2);
}

#[test]
fn test_cache_hit() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    let result1 = cache.get_or_allocate_blocks(1, &tokens).unwrap();
    assert!(!result1.is_hit());

    let result2 = cache.get_or_allocate_blocks(2, &tokens).unwrap();
    assert!(result2.is_hit());
    assert_eq!(result2.cached_count(), 2);

    assert_eq!(result1.blocks(), result2.blocks());

    let stats = cache.stats();
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.blocks_from_cache, 2);
}

#[test]
fn test_partial_cache_hit() {
    let mut cache = create_cache();

    let tokens1: Vec<u32> = (0..32).collect();
    cache.get_or_allocate_blocks(1, &tokens1).unwrap();

    let tokens2: Vec<u32> = (0..48).collect();
    let result = cache.get_or_allocate_blocks(2, &tokens2).unwrap();

    assert!(result.is_hit());
    assert_eq!(result.blocks().len(), 3);
    match result {
        CacheResult::Hit {
            cached_blocks,
            new_blocks,
            ..
        } => {
            assert_eq!(cached_blocks, 2);
            assert_eq!(new_blocks, 1);
        }
        _ => panic!("Expected hit"),
    }
}

#[test]
fn test_different_prefix_no_hit() {
    let mut cache = create_cache();

    let tokens1: Vec<u32> = (0..32).collect();
    cache.get_or_allocate_blocks(1, &tokens1).unwrap();

    let tokens2: Vec<u32> = (100..132).collect();
    let result = cache.get_or_allocate_blocks(2, &tokens2).unwrap();

    assert!(!result.is_hit());
}

#[test]
fn test_release_blocks() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    let result = cache.get_or_allocate_blocks(1, &tokens).unwrap();
    let blocks = result.blocks().to_vec();

    cache.release_blocks(1, &blocks).unwrap();
    assert!(cache.has_cached_prefix(&tokens));
}

#[test]
fn test_force_free_blocks() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    let result = cache.get_or_allocate_blocks(1, &tokens).unwrap();
    let blocks = result.blocks().to_vec();

    cache.force_free_blocks(&blocks).unwrap();
    assert!(!cache.has_cached_prefix(&tokens));
}

#[test]
fn test_has_cached_prefix() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    assert!(!cache.has_cached_prefix(&tokens));
    cache.get_or_allocate_blocks(1, &tokens).unwrap();
    assert!(cache.has_cached_prefix(&tokens));
}

#[test]
fn test_cached_block_count() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    assert_eq!(cache.cached_block_count(&tokens), 0);
    cache.get_or_allocate_blocks(1, &tokens).unwrap();
    assert_eq!(cache.cached_block_count(&tokens), 2);

    let longer: Vec<u32> = (0..48).collect();
    assert_eq!(cache.cached_block_count(&longer), 2);
}

#[test]
fn test_min_prefix_tokens() {
    let mut cache = create_cache();

    let short_tokens: Vec<u32> = vec![1, 2, 3];
    let result1 = cache.get_or_allocate_blocks(1, &short_tokens).unwrap();
    let result2 = cache.get_or_allocate_blocks(2, &short_tokens).unwrap();

    assert!(!result1.is_hit());
    assert!(!result2.is_hit());
}

#[test]
fn test_cache_disabled() {
    let allocator = CpuBlockAllocator::new(100, 16);
    let config = PrefixCacheConfig {
        enabled: false,
        ..Default::default()
    };
    let mut cache = PrefixCache::new(allocator, config);

    let tokens: Vec<u32> = (0..32).collect();
    cache.get_or_allocate_blocks(1, &tokens).unwrap();
    let result = cache.get_or_allocate_blocks(2, &tokens).unwrap();
    assert!(!result.is_hit());
}

#[test]
fn test_eviction() {
    let allocator = CpuBlockAllocator::new(10, 16);
    let config = PrefixCacheConfig {
        enabled: true,
        max_cached_blocks: 10,
        min_prefix_tokens: 4,
        block_size: 16,
    };
    let mut cache = PrefixCache::new(allocator, config);

    for i in 0..4 {
        let tokens: Vec<u32> = ((i * 100)..((i * 100) + 32)).map(|x| x as u32).collect();
        let result = cache.get_or_allocate_blocks(i as u64, &tokens).unwrap();
        cache.release_blocks(i as u64, result.blocks()).unwrap();
    }

    let tokens: Vec<u32> = (1000..1032).collect();
    let result = cache.get_or_allocate_blocks(100, &tokens);
    assert!(result.is_ok());
}

#[test]
fn test_stats() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    cache.get_or_allocate_blocks(1, &tokens).unwrap();
    cache.get_or_allocate_blocks(2, &tokens).unwrap();
    cache.get_or_allocate_blocks(3, &tokens).unwrap();

    let stats = cache.stats();
    assert_eq!(stats.total_lookups, 3);
    assert_eq!(stats.cache_hits, 2);
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.blocks_from_cache, 4);
    assert_eq!(stats.blocks_allocated, 2);
    assert!(stats.hit_rate > 0.6);
}

#[test]
fn test_reset() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    let result = cache.get_or_allocate_blocks(1, &tokens).unwrap();
    let blocks = result.blocks().to_vec();
    cache.release_blocks(1, &blocks).ok();

    assert!(cache.has_cached_prefix(&tokens));
    cache.reset().unwrap();
    assert!(!cache.has_cached_prefix(&tokens));
    assert_eq!(cache.stats().total_lookups, 0);
}

#[test]
fn test_sequence_prefix_state() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    let result1 = cache.get_or_allocate_blocks(1, &tokens).unwrap();
    let state1 = SequencePrefixState::from_cache_result(&result1, tokens.len());
    assert_eq!(state1.cached_count(), 0);
    assert_eq!(state1.new_count(), 2);

    let result2 = cache.get_or_allocate_blocks(2, &tokens).unwrap();
    let state2 = SequencePrefixState::from_cache_result(&result2, tokens.len());
    assert_eq!(state2.cached_count(), 2);
    assert_eq!(state2.new_count(), 0);
}

#[test]
fn test_refcount_tracking() {
    let mut cache = create_cache();
    let tokens: Vec<u32> = (0..32).collect();

    let result1 = cache.get_or_allocate_blocks(1, &tokens).unwrap();
    cache.get_or_allocate_blocks(2, &tokens).unwrap();
    cache.get_or_allocate_blocks(3, &tokens).unwrap();

    let blocks = result1.blocks().to_vec();

    cache.release_blocks(1, &blocks).unwrap();
    assert!(cache.has_cached_prefix(&tokens));

    cache.release_blocks(2, &blocks).unwrap();
    assert!(cache.has_cached_prefix(&tokens));

    cache.release_blocks(3, &blocks).unwrap();
    assert!(cache.has_cached_prefix(&tokens));
}
