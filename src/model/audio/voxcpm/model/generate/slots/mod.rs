//! Slot seating: [`GenerateState::open`], [`GenerateState::admit`] and
//! [`GenerateState::release`].
//!
//! A state opened here is a CEILING of SLOTS, all free. A request is seated
//! into one slot from its own single-row prefill. A seated slot runs every
//! sub-step, a finished one included. A free slot differs only in its key
//! range, which the lane table holds at empty.
//!
//! The ceiling is not a reservation. Every device tensor — both caches, both
//! hidden-state rows, the prefix condition and the lane table — is allocated
//! for ONE row and widened when a slot above the allocation is seated. The
//! allocation never shrinks.
//!
//! The loop steps a PREFIX of the slots. Slots are seated lowest-free-first,
//! so the occupied ones are a prefix. Seating and releasing recompute that
//! prefix, and no step touches the rows past it.
//!
//! `open` opens the state, `seat` seats and releases one request, `grow`
//! widens the allocation, and `active` decides the stepped width.

mod active;
mod grow;
mod open;
mod seat;
