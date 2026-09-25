//! Slot seating: [`GenerateState::open`], [`GenerateState::admit`] and
//! [`GenerateState::release`].
//!
//! A state opened here is a fixed number of SLOTS, all free: both caches and
//! both hidden-state rows are allocated at the full slot width, and a
//! request is seated into one slot at a time from its own single-row
//! prefill. Every slot runs every sub-step of the loop, a free one included;
//! what a free slot changes is its key range, which the lane table holds at
//! empty.

mod open;
mod seat;
