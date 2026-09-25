//! Continuous batching: ONE batch stays open and queued requests join its
//! free slots between steps, so a request arriving mid-generation waits one
//! step rather than the batch it missed.
//!
//! [`owner`] holds the loop and the batch it drives, [`slots`] the slot
//! table and the non-blocking queue drain.

mod owner;
mod slots;

pub(super) use owner::run_continuous;
