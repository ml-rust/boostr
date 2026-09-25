//! Slot seating: [`GenerateState::open`], [`GenerateState::admit`] and
//! [`GenerateState::release`].
//!
//! A state opened here is a CEILING of SLOTS, all free, and a request is
//! seated into one slot at a time from its own single-row prefill. A seated
//! slot runs every sub-step of the loop, a finished one included; what a
//! free slot changes is its key range, which the lane table holds at empty.
//!
//! The ceiling is not a reservation. Every device tensor — both caches, both
//! hidden-state rows, the prefix condition and the lane table — is allocated
//! for ONE row and widened when a slot above the allocation is seated. The
//! allocation never shrinks.
//!
//! The loop steps a PREFIX of the slots, not all of them. Slots are seated
//! lowest-free-first, so the occupied ones are a prefix and the rows above
//! it hold nothing; seating and releasing recompute how wide that prefix is
//! and the rows past it are then not stepped at all.
//!
//! `open` opens the state, `seat` seats and releases one request, `grow`
//! widens the allocation, and `active` decides the stepped width.

mod active;
mod grow;
mod open;
mod seat;
