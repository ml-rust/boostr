//! Sequence lifecycle transitions: prefill completion, token append,
//! finish, abort, and preemption.

use crate::error::{Error, Result};
use crate::inference::memory::BlockAllocator;

use super::core::SequenceScheduler;
use super::types::{SequenceId, SequenceState};

impl<A: BlockAllocator> SequenceScheduler<A> {
    pub fn prefill_complete(&mut self, seq_id: SequenceId) -> Result<()> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if seq.state != SequenceState::Running {
                return Err(Error::SchedulerError {
                    reason: format!("Sequence {} is not running", seq_id),
                });
            }
            Ok(())
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }

    pub fn append_token(&mut self, seq_id: SequenceId, token: u32) -> Result<bool> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if seq.state != SequenceState::Running {
                return Err(Error::SchedulerError {
                    reason: format!("Sequence {} is not running", seq_id),
                });
            }

            // Ensure a physical block exists for the slot where the new token lands
            // before committing any state change.  The new token occupies position
            // `total_tokens` (0-indexed), which belongs to block
            // `total_tokens / block_size`.
            let new_token_block_idx = seq.total_tokens / seq.block_table.block_size;
            if new_token_block_idx >= seq.block_table.blocks.len() {
                if !self.allocator.can_allocate(1) {
                    return Err(Error::SchedulerError {
                        reason: format!(
                            "Cannot append token to sequence {}: no free blocks available",
                            seq_id
                        ),
                    });
                }
                let blk = self.allocator.allocate(1)?;
                seq.block_table.append_blocks(blk);
            }

            seq.generated_tokens.push(token);
            seq.total_tokens += 1;
            seq.block_table.num_tokens = seq.total_tokens;
            self.stats.total_tokens_generated += 1;

            // Best-effort preallocation for the token after next (non-fatal if it fails).
            let blocks_needed = seq.blocks_needed_for_next_token();
            if blocks_needed > 0
                && self.allocator.can_allocate(blocks_needed)
                && let Ok(new_blocks) = self.allocator.allocate(blocks_needed)
            {
                seq.block_table.append_blocks(new_blocks);
            }

            if seq.is_finished() || seq.total_tokens >= self.config.max_seq_len {
                self.finish_sequence(seq_id)?;
                return Ok(true);
            }

            Ok(false)
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }

    pub fn finish_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if !seq.block_table.blocks.is_empty() {
                Self::release_blocks(
                    &mut self.prefix_cache,
                    &self.allocator,
                    seq_id,
                    &seq.block_table.blocks,
                )?;
            }

            seq.state = SequenceState::Finished;
            seq.block_table.blocks.clear();

            self.running_set.retain(|&id| id != seq_id);
            self.stats.running_count -= 1;
            self.stats.finished_count += 1;

            Ok(())
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }

    pub fn abort_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            // Idempotent: already terminal, nothing to do.
            if seq.state == SequenceState::Finished {
                return Ok(());
            }

            // Free physical blocks first, then clear to prevent double-free on
            // a second call before cleanup_finished() runs.
            if !seq.block_table.blocks.is_empty() {
                Self::release_blocks(
                    &mut self.prefix_cache,
                    &self.allocator,
                    seq_id,
                    &seq.block_table.blocks,
                )?;
                seq.block_table.blocks.clear();
            }

            match seq.state {
                SequenceState::Waiting => {
                    self.waiting_queue.retain(|&id| id != seq_id);
                    self.stats.waiting_count -= 1;
                }
                SequenceState::Running => {
                    self.running_set.retain(|&id| id != seq_id);
                    self.stats.running_count -= 1;
                }
                SequenceState::Preempted => {
                    self.preempted_set.retain(|&id| id != seq_id);
                    self.stats.preempted_count -= 1;
                }
                SequenceState::Finished => unreachable!("guarded above"),
            }

            seq.state = SequenceState::Finished;
            self.stats.finished_count += 1;

            Ok(())
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }

    pub(super) fn preempt_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if !seq.block_table.blocks.is_empty() {
                Self::release_blocks(
                    &mut self.prefix_cache,
                    &self.allocator,
                    seq_id,
                    &seq.block_table.blocks,
                )?;
            }

            seq.state = SequenceState::Preempted;
            seq.block_table.blocks.clear();

            self.running_set.retain(|&id| id != seq_id);
            self.preempted_set.push(seq_id);
            self.stats.running_count -= 1;
            self.stats.preempted_count += 1;
            self.stats.total_preemptions += 1;

            Ok(())
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }
}
