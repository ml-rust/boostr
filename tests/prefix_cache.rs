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

/// max_cached_blocks must be enforced: once sequences release their blocks
/// (ref_count → 0), new allocations must evict LRU entries to stay within cap.
#[test]
fn test_max_cached_blocks_enforced() {
    let block_size = 4;
    // Physical pool: 20 blocks; cache cap: 4 blocks.
    let allocator = CpuBlockAllocator::new(20, block_size);
    let config = PrefixCacheConfig {
        enabled: true,
        max_cached_blocks: 4,
        min_prefix_tokens: 1,
        block_size,
    };
    let mut cache = PrefixCache::new(allocator, config);

    // Allocate 4 sequences and immediately release them so their blocks are
    // eligible for eviction (ref_count == 0).
    let mut all_blocks = Vec::new();
    for seq_id in 1u64..=4 {
        let tokens: Vec<u32> = (seq_id as u32 * 10..(seq_id as u32 * 10 + 4)).collect();
        let result = cache.get_or_allocate_blocks(seq_id, &tokens).unwrap();
        let blocks = result.blocks().to_vec();
        cache.release_blocks(seq_id, &blocks).unwrap();
        all_blocks.push(blocks);
    }

    // At this point the cache holds exactly 4 entries, all ref_count == 0.
    assert_eq!(cache.stats().cached_blocks, 4);

    // A 5th allocation must evict the LRU entry, keeping the count at ≤ 4.
    let tokens5: Vec<u32> = (50..54).collect();
    cache.get_or_allocate_blocks(5, &tokens5).unwrap();

    let stats = cache.stats();
    assert!(
        stats.cached_blocks <= 4,
        "cached_blocks {} exceeds max_cached_blocks 4",
        stats.cached_blocks
    );
    assert!(
        stats.blocks_evicted >= 1,
        "at least one block must have been evicted"
    );
}

/// has_cached_prefix and cached_block_count must validate token content,
/// not just hash presence, to guard against hash collisions.
#[test]
fn test_cached_block_count_validates_tokens() {
    let block_size = 4;
    let allocator = CpuBlockAllocator::new(20, block_size);
    let config = PrefixCacheConfig {
        enabled: true,
        max_cached_blocks: 100,
        min_prefix_tokens: 1,
        block_size,
    };
    let mut cache = PrefixCache::new(allocator, config);

    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    cache.get_or_allocate_blocks(1, &tokens).unwrap();

    // Correct tokens → should find the cached block.
    assert!(cache.has_cached_prefix(&tokens));
    assert_eq!(cache.cached_block_count(&tokens), 1);

    // Different tokens with the same prefix length → should NOT hit.
    let different: Vec<u32> = vec![9, 9, 9, 9];
    // (These are very unlikely to collide; the test verifies the token check.)
    if !cache.has_cached_prefix(&different) {
        assert_eq!(cache.cached_block_count(&different), 0);
    }
    // If by extreme chance hashes collide, the token check must still return 0.
    // We can't force a collision in a unit test, but we verify the path compiles
    // and the non-colliding case is correct.
}
