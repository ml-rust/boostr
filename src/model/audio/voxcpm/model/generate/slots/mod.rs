//! Slot seating: [`GenerateState::open`], [`GenerateState::admit`] and
//! [`GenerateState::release`].
//!
//! A state opened here is a fixed number of SLOTS, all free: both caches and
//! both hidden-state rows are allocated at the full slot width, and a
//! request is seated into one slot at a time from its own single-row
//! prefill. A seated slot runs every sub-step of the loop, a finished one
//! included; what a free slot changes is its key range, which the lane table
//! holds at empty.
//!
//! The loop steps a PREFIX of the slots, not all of them. Slots are seated
//! lowest-free-first, so the occupied ones are a prefix and the rows above
//! it hold nothing; seating and releasing recompute how wide that prefix is
//! and the rows past it are then not stepped at all.
//!
//! `open` opens the state, `seat` seats and releases one request, and
//! `active` decides the stepped width.

mod active;
mod open;
mod seat;
