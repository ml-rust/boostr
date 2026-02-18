//! Block table - maps logical block indices to physical block IDs

use super::BlockId;

/// Block table - maps logical block indices to physical block IDs
#[derive(Debug, Clone)]
pub struct BlockTable {
    pub blocks: Vec<BlockId>,
    pub num_tokens: usize,
    pub block_size: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks_needed(num_tokens: usize, block_size: usize) -> usize {
        num_tokens.div_ceil(block_size)
    }

    pub fn additional_blocks_needed(&self, additional_tokens: usize) -> usize {
        let total_tokens = self.num_tokens + additional_tokens;
        let total_blocks_needed = Self::blocks_needed(total_tokens, self.block_size);
        total_blocks_needed.saturating_sub(self.blocks.len())
    }

    pub fn append_blocks(&mut self, new_blocks: Vec<BlockId>) {
        self.blocks.extend(new_blocks);
    }

    pub fn set_num_tokens(&mut self, num_tokens: usize) {
        self.num_tokens = num_tokens;
    }

    pub fn get_block(&self, logical_idx: usize) -> Option<BlockId> {
        self.blocks.get(logical_idx).copied()
    }

    pub fn get_slot(&self, token_pos: usize) -> Option<(BlockId, usize)> {
        let logical_block = token_pos / self.block_size;
        let slot_in_block = token_pos % self.block_size;
        self.get_block(logical_block)
            .map(|block_id| (block_id, slot_in_block))
    }

    pub fn to_device_format(&self) -> Vec<i32> {
        self.blocks.iter().map(|&b| b as i32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_table_create() {
        let table = BlockTable::new(16);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens, 0);
        assert_eq!(table.block_size, 16);
    }

    #[test]
    fn test_block_table_blocks_needed() {
        assert_eq!(BlockTable::blocks_needed(0, 16), 0);
        assert_eq!(BlockTable::blocks_needed(1, 16), 1);
        assert_eq!(BlockTable::blocks_needed(16, 16), 1);
        assert_eq!(BlockTable::blocks_needed(17, 16), 2);
        assert_eq!(BlockTable::blocks_needed(32, 16), 2);
        assert_eq!(BlockTable::blocks_needed(33, 16), 3);
    }

    #[test]
    fn test_block_table_additional_blocks() {
        let mut table = BlockTable::new(16);
        table.blocks = vec![0, 1];
        table.num_tokens = 20;

        assert_eq!(table.additional_blocks_needed(12), 0);
        assert_eq!(table.additional_blocks_needed(13), 1);
        assert_eq!(table.additional_blocks_needed(30), 2);
    }

    #[test]
    fn test_block_table_get_slot() {
        let mut table = BlockTable::new(16);
        table.blocks = vec![5, 10, 15];

        assert_eq!(table.get_slot(0), Some((5, 0)));
        assert_eq!(table.get_slot(15), Some((5, 15)));
        assert_eq!(table.get_slot(16), Some((10, 0)));
        assert_eq!(table.get_slot(32), Some((15, 0)));
        assert_eq!(table.get_slot(48), None);
    }

    #[test]
    fn test_block_table_to_device_format() {
        let mut table = BlockTable::new(16);
        table.blocks = vec![5, 10, 15];
        let device_format = table.to_device_format();
        assert_eq!(device_format, vec![5i32, 10, 15]);
    }
}
