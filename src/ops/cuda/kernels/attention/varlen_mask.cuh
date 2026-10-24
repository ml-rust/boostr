// Mask predicate and K-tile range shared by every varlen attention kernel.
//
// Positions are absolute within one segment. Local query row r sits at
// q_pos = key_offset + r, with key_offset = max(0, seq_len_k - seq_len_q).
// Key j is masked for that row when
//   (causal && j > q_pos) || (window_size > 0 && j + window_size <= q_pos).
// window_size == 0 turns the window off. With causal == 0 the window still
// anchors at q_pos. Same rule as `ops/cpu/attention/varlen_attention/mask.rs`
// and the window in `flash_v2.cu`.
//
// The forward and both backward phases call the same predicate. The backward
// recomputes P from the forward LSE, so any drift between them corrupts the
// gradients.

#pragma once

// True when key `key` is masked for the query at absolute position `q_pos`.
// The window test is `key < q_pos - window_size + 1`: it cannot overflow for
// any non-negative window_size, unlike `key + window_size <= q_pos`.
__device__ __forceinline__ bool varlen_masked(
    const int q_pos, const int key, const int causal, const int window_size
) {
    return (causal && key > q_pos)
        || (window_size > 0 && key < q_pos - window_size + 1);
}

// K tiles [*first_tile, *end_tile) that hold at least one key visible to a
// query block whose absolute positions span [first_q_pos, last_q_pos]. The
// window start of the first row sets the first tile. The causal limit of the
// last row sets the end. Every skipped tile is masked for every row of the
// block, so skipping it changes no result. Cost per block is O(window).
__device__ __forceinline__ void varlen_k_tile_range(
    const int first_q_pos, const int last_q_pos, const int seq_len_k,
    const int block_n, const int causal, const int window_size,
    int* first_tile, int* end_tile
) {
    const int num_k_blocks = (seq_len_k + block_n - 1) / block_n;
    const int min_key = window_size > 0 ? max(0, first_q_pos - window_size + 1) : 0;
    int end = num_k_blocks;
    if (causal) {
        // Last visible key is last_q_pos. Tiles past it hold no visible key.
        end = min(end, last_q_pos / block_n + 1);
    }
    *first_tile = min(min_key / block_n, end);
    *end_tile = end;
}
