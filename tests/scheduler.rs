use boostr::inference::memory::CpuBlockAllocator;
use boostr::inference::scheduler::{
    ScheduledBatch, SchedulerConfig, SchedulingPriority, SequenceRequest, SequenceScheduler,
    SequenceState,
};
use std::collections::HashMap;

fn create_scheduler() -> SequenceScheduler<CpuBlockAllocator> {
    let allocator = CpuBlockAllocator::new(100, 16);
    let config = SchedulerConfig {
        max_batch_size: 4,
        max_batch_tokens: 1024,
        max_seq_len: 512,
        block_size: 16,
        enable_preemption: true,
        max_preempt_per_step: 2,
    };
    SequenceScheduler::new(allocator, config)
}

#[test]
fn test_scheduler_create() {
    let scheduler = create_scheduler();
    assert!(!scheduler.has_work());
    assert_eq!(scheduler.stats().total_requests, 0);
}

#[test]
fn test_scheduler_add_request() {
    let mut scheduler = create_scheduler();
    let request = SequenceRequest::new(1, vec![1, 2, 3, 4, 5]);
    scheduler.add_request(request).unwrap();
    assert!(scheduler.has_work());
    assert_eq!(scheduler.stats().total_requests, 1);
    assert_eq!(scheduler.stats().waiting_count, 1);
}

#[test]
fn test_scheduler_add_duplicate_request() {
    let mut scheduler = create_scheduler();
    let request1 = SequenceRequest::new(1, vec![1, 2, 3]);
    scheduler.add_request(request1).unwrap();

    let request2 = SequenceRequest::new(1, vec![4, 5, 6]);
    let result = scheduler.add_request(request2);
    assert!(result.is_err());
}

#[test]
fn test_scheduler_schedule_prefill() {
    let mut scheduler = create_scheduler();
    let request = SequenceRequest::new(1, vec![1, 2, 3, 4, 5]);
    scheduler.add_request(request).unwrap();

    let batch = scheduler.schedule().unwrap().unwrap();
    assert_eq!(batch.prefill_sequences, vec![1]);
    assert!(batch.decode_sequences.is_empty());
    assert!(batch.block_tables.contains_key(&1));
    assert_eq!(scheduler.stats().waiting_count, 0);
    assert_eq!(scheduler.stats().running_count, 1);
}

#[test]
fn test_scheduler_schedule_decode() {
    let mut scheduler = create_scheduler();
    let request = SequenceRequest::new(1, vec![1, 2, 3, 4, 5]);
    scheduler.add_request(request).unwrap();

    scheduler.schedule().unwrap();
    scheduler.prefill_complete(1).unwrap();

    let batch = scheduler.schedule().unwrap().unwrap();
    assert!(batch.prefill_sequences.is_empty());
    assert_eq!(batch.decode_sequences, vec![1]);
}

#[test]
fn test_scheduler_append_token() {
    let mut scheduler = create_scheduler();
    let request = SequenceRequest::new(1, vec![1, 2, 3]).with_max_tokens(5);
    scheduler.add_request(request).unwrap();
    scheduler.schedule().unwrap();

    for i in 0..5 {
        let finished = scheduler.append_token(1, 100 + i).unwrap();
        if i < 4 {
            assert!(!finished);
        } else {
            assert!(finished);
        }
    }

    assert_eq!(scheduler.get_generated_tokens(1).unwrap().len(), 5);
    assert_eq!(
        scheduler.get_sequence_state(1),
        Some(SequenceState::Finished)
    );
}

#[test]
fn test_scheduler_finish_sequence() {
    let mut scheduler = create_scheduler();
    let request = SequenceRequest::new(1, vec![1, 2, 3]);
    scheduler.add_request(request).unwrap();
    scheduler.schedule().unwrap();
    scheduler.finish_sequence(1).unwrap();

    assert_eq!(
        scheduler.get_sequence_state(1),
        Some(SequenceState::Finished)
    );
    assert_eq!(scheduler.stats().finished_count, 1);
    assert_eq!(scheduler.stats().running_count, 0);
}

#[test]
fn test_scheduler_abort_sequence() {
    let mut scheduler = create_scheduler();
    let request = SequenceRequest::new(1, vec![1, 2, 3]);
    scheduler.add_request(request).unwrap();
    scheduler.abort_sequence(1).unwrap();

    assert_eq!(
        scheduler.get_sequence_state(1),
        Some(SequenceState::Finished)
    );
    assert_eq!(scheduler.stats().waiting_count, 0);
}

#[test]
fn test_scheduler_multiple_sequences() {
    let mut scheduler = create_scheduler();
    for i in 1..=3 {
        let request = SequenceRequest::new(i, vec![1, 2, 3]);
        scheduler.add_request(request).unwrap();
    }

    let batch = scheduler.schedule().unwrap().unwrap();
    assert_eq!(batch.prefill_sequences.len(), 3);
    assert_eq!(scheduler.stats().running_count, 3);
}

#[test]
fn test_scheduler_batch_size_limit() {
    let mut scheduler = create_scheduler();
    for i in 1..=10 {
        let request = SequenceRequest::new(i, vec![1, 2, 3]);
        scheduler.add_request(request).unwrap();
    }

    let batch = scheduler.schedule().unwrap().unwrap();
    assert_eq!(batch.prefill_sequences.len(), 4);
    assert_eq!(scheduler.stats().waiting_count, 6);
}

#[test]
fn test_scheduler_block_allocation() {
    let mut scheduler = create_scheduler();
    let request = SequenceRequest::new(1, vec![0; 17]);
    scheduler.add_request(request).unwrap();
    scheduler.schedule().unwrap();

    let block_table = scheduler.get_block_table(1).unwrap();
    assert_eq!(block_table.num_blocks(), 2);
}

#[test]
fn test_scheduler_no_work() {
    let mut scheduler = create_scheduler();
    assert!(scheduler.schedule().unwrap().is_none());
}

#[test]
fn test_scheduler_cleanup_finished() {
    let mut scheduler = create_scheduler();
    let request = SequenceRequest::new(1, vec![1, 2, 3]);
    scheduler.add_request(request).unwrap();
    scheduler.schedule().unwrap();
    scheduler.finish_sequence(1).unwrap();

    assert!(scheduler.has_sequence(1));
    scheduler.cleanup_finished();
    assert!(!scheduler.has_sequence(1));
}

#[test]
fn test_scheduler_stats() {
    let mut scheduler = create_scheduler();
    for i in 1..=3 {
        let request = SequenceRequest::new(i, vec![1, 2, 3]).with_max_tokens(2);
        scheduler.add_request(request).unwrap();
    }
    scheduler.schedule().unwrap();

    for i in 1..=3 {
        scheduler.append_token(i, 100).unwrap();
        scheduler.append_token(i, 101).unwrap();
    }

    let stats = scheduler.stats();
    assert_eq!(stats.total_requests, 3);
    assert_eq!(stats.total_tokens_generated, 6);
    assert_eq!(stats.finished_count, 3);
}

#[test]
fn test_sequence_request_builder() {
    let request = SequenceRequest::new(1, vec![1, 2, 3])
        .with_max_tokens(100)
        .with_priority(SchedulingPriority::High);

    assert_eq!(request.max_new_tokens, 100);
    assert_eq!(request.priority, SchedulingPriority::High);
}

/// Verify that append_token returns an error rather than silently advancing
/// the token count when no physical block can be allocated for the new token.
#[test]
fn test_append_token_returns_error_when_blocks_exhausted() {
    // Tiny pool: 1 block of size 2 — can hold at most 2 tokens.
    let allocator = CpuBlockAllocator::new(1, 2);
    let config = SchedulerConfig {
        max_batch_size: 4,
        max_batch_tokens: 1024,
        max_seq_len: 512,
        block_size: 2,
        enable_preemption: false,
        max_preempt_per_step: 0,
    };
    let mut scheduler = SequenceScheduler::new(allocator, config);

    // 1 token prompt, block size 2 → one block is enough for prefill (1 token)
    // and still leaves slot 1 free within that block.
    let req = SequenceRequest::new(1, vec![10]);
    scheduler.add_request(req).unwrap();
    scheduler.schedule().unwrap();

    // Token 1 (pos 1) still fits in the first block.
    let done = scheduler.append_token(1, 20).unwrap();
    assert!(!done);

    // Token 2 (pos 2) would need a second block, but the pool is exhausted.
    let result = scheduler.append_token(1, 30);
    assert!(result.is_err(), "must fail when no block available");

    // The sequence total_tokens must NOT have been incremented past 2.
    let bt = scheduler.get_block_table(1).unwrap();
    assert!(
        bt.num_tokens <= 2,
        "num_tokens must not advance beyond available blocks: got {}",
        bt.num_tokens
    );
}

/// Double-abort of the same sequence must not panic, corrupt stats, or
/// double-free blocks.
#[test]
fn test_abort_sequence_is_idempotent() {
    let mut scheduler = create_scheduler();
    let req = SequenceRequest::new(42, vec![1, 2, 3, 4]);
    scheduler.add_request(req).unwrap();
    scheduler.schedule().unwrap();

    // First abort.
    scheduler.abort_sequence(42).unwrap();
    let stats_after_first = scheduler.stats();

    // Second abort on the same (now-finished) sequence must succeed without
    // corrupting counters or double-freeing blocks.
    scheduler.abort_sequence(42).unwrap();
    let stats_after_second = scheduler.stats();

    assert_eq!(
        stats_after_first.finished_count, stats_after_second.finished_count,
        "finished_count must not change on second abort"
    );
    assert_eq!(
        stats_after_first.running_count, stats_after_second.running_count,
        "running_count must not underflow on second abort"
    );
}

#[test]
fn test_scheduled_batch_methods() {
    let batch = ScheduledBatch {
        prefill_sequences: vec![1, 2],
        decode_sequences: vec![3, 4, 5],
        block_tables: HashMap::new(),
        preempted_sequences: vec![],
    };

    assert!(!batch.is_empty());
    assert_eq!(batch.len(), 5);
    assert_eq!(batch.all_sequences(), vec![1, 2, 3, 4, 5]);
}
