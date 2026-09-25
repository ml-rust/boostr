//! The slot table a persistent batch keeps its rows in, and the
//! non-blocking queue drain that feeds it.
//!
//! Both are free of device work and of the runtime type, so what lands where
//! is decided — and tested — without a model.

use std::sync::mpsc::{Receiver, TryRecvError};

use crate::voxcpm::engine::batch::Job;

/// A fixed-width table of slots, each free or holding one `T`.
///
/// The width never changes while the table is in use: it is the batch width
/// the generation loop runs at, and the DiT's per-shape graph cache is keyed
/// on it.
pub(super) struct Slots<T> {
    entries: Vec<Option<T>>,
}

impl<T> Slots<T> {
    /// `width` free slots.
    pub(super) fn new(width: usize) -> Self {
        Self {
            entries: (0..width).map(|_| None).collect(),
        }
    }

    /// Slots in the table.
    pub(super) fn width(&self) -> usize {
        self.entries.len()
    }

    /// The lowest free slot, or `None` when every slot holds a row.
    ///
    /// Lowest rather than round-robin: a slot's index is nothing but a row of
    /// the batch, and a stable choice keeps a quiet engine using the same few
    /// slots.
    pub(super) fn free(&self) -> Option<usize> {
        self.entries.iter().position(Option::is_none)
    }

    /// Slots holding no row.
    pub(super) fn free_count(&self) -> usize {
        self.entries.iter().filter(|slot| slot.is_none()).count()
    }

    /// Whether any slot holds a row.
    pub(super) fn any_occupied(&self) -> bool {
        self.entries.iter().any(Option::is_some)
    }

    /// Put `value` in slot `slot`. A slot outside the width holds nothing
    /// and takes nothing; [`free`](Self::free) never names one.
    pub(super) fn seat(&mut self, slot: usize, value: T) {
        if let Some(held) = self.entries.get_mut(slot) {
            *held = Some(value);
        }
    }

    /// Empty slot `slot` and hand back what it held.
    pub(super) fn take(&mut self, slot: usize) -> Option<T> {
        self.entries.get_mut(slot).and_then(Option::take)
    }

    /// Slot `slot`'s row.
    pub(super) fn get_mut(&mut self, slot: usize) -> Option<&mut T> {
        self.entries.get_mut(slot).and_then(Option::as_mut)
    }

    /// Empty every slot, handing back each row it held with its slot index.
    pub(super) fn drain_all(&mut self) -> Vec<(usize, T)> {
        self.entries
            .iter_mut()
            .enumerate()
            .filter_map(|(slot, held)| held.take().map(|row| (slot, row)))
            .collect()
    }
}

/// Take at most `free` queued jobs, blocking on nothing.
///
/// The second value is `true` once the queue has closed, which is what
/// [`Drop`] on the engine does; the jobs taken before that are still
/// returned.
///
/// This is the whole of the steady-state wait: a request that arrives after
/// this returns is taken by the next call, one step later, rather than after
/// the batch it missed has drained.
pub(super) fn drain(jobs: &Receiver<Job>, free: usize) -> (Vec<Job>, bool) {
    let mut taken = Vec::new();
    while taken.len() < free {
        match jobs.try_recv() {
            Ok(job) => taken.push(job),
            Err(TryRecvError::Empty) => return (taken, false),
            Err(TryRecvError::Disconnected) => return (taken, true),
        }
    }
    (taken, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxcpm::engine::batch::Reply;
    use std::sync::mpsc::channel;

    fn job() -> Job {
        let (tx, rx) = channel();
        // The receiver is kept alive by the job it belongs to in the engine;
        // here nothing reads it, so it is dropped and the send simply fails.
        drop(rx);
        Job {
            voice: None,
            text_token_ids: vec![1, 2],
            max_len: 22,
            seq_len: 2,
            reply: Reply::Buffered(tx),
        }
    }

    /// Seating fills the lowest free slot, taking frees it again, and the
    /// counts follow both.
    #[test]
    fn slots_track_what_they_hold() {
        let mut slots = Slots::<u32>::new(4);
        assert_eq!(slots.width(), 4);
        assert_eq!(slots.free_count(), 4);
        assert!(!slots.any_occupied());
        assert_eq!(slots.free(), Some(0));

        for value in 0..4u32 {
            let slot = slots.free().expect("a free slot");
            assert_eq!(slot, value as usize, "the lowest free slot is taken");
            slots.seat(slot, value);
        }
        assert_eq!(slots.free(), None, "a full table offers no slot");
        assert_eq!(slots.free_count(), 0);
        assert!(slots.any_occupied());

        assert_eq!(slots.take(2), Some(2));
        assert_eq!(slots.free(), Some(2), "a released slot is offered again");
        assert_eq!(slots.free_count(), 1);
        assert_eq!(slots.take(2), None, "an empty slot holds nothing");
    }

    /// A released slot takes the next request, and the rows beside it stay
    /// where they are.
    #[test]
    fn a_released_slot_is_reseated_without_moving_the_others() {
        let mut slots = Slots::<u32>::new(3);
        for value in 0..3u32 {
            slots.seat(value as usize, value);
        }
        slots.take(1);
        let slot = slots.free().expect("slot 1 is free");
        assert_eq!(slot, 1);
        slots.seat(slot, 99);

        assert_eq!(slots.get_mut(0).copied(), Some(0));
        assert_eq!(slots.get_mut(1).copied(), Some(99));
        assert_eq!(slots.get_mut(2).copied(), Some(2));
    }

    #[test]
    fn drain_all_empties_every_slot_and_names_it() {
        let mut slots = Slots::<u32>::new(3);
        slots.seat(0, 7);
        slots.seat(2, 9);
        assert_eq!(slots.drain_all(), vec![(0, 7), (2, 9)]);
        assert!(!slots.any_occupied());
    }

    /// The drain never blocks and never takes more than the free slots hold;
    /// the rest stays queued for the next step.
    #[test]
    fn drain_takes_what_fits_and_leaves_the_rest_queued() {
        let (tx, rx) = channel();
        for _ in 0..5 {
            tx.send(job()).expect("queue");
        }
        let (taken, closed) = drain(&rx, 2);
        assert_eq!(taken.len(), 2);
        assert!(!closed);

        let (taken, closed) = drain(&rx, 8);
        assert_eq!(taken.len(), 3, "an empty queue ends the drain at once");
        assert!(!closed);
    }

    /// An empty queue and a full table both return nothing, without
    /// blocking.
    #[test]
    fn drain_of_an_empty_queue_or_a_full_table_takes_nothing() {
        let (tx, rx) = channel();
        let (taken, closed) = drain(&rx, 4);
        assert!(taken.is_empty());
        assert!(!closed);

        tx.send(job()).expect("queue");
        let (taken, closed) = drain(&rx, 0);
        assert!(taken.is_empty(), "no free slot takes no job");
        assert!(!closed);
    }

    /// A closed queue is reported, and whatever was still in it comes back
    /// rather than being dropped on the floor.
    #[test]
    fn drain_reports_a_closed_queue_after_taking_what_was_left() {
        let (tx, rx) = channel();
        tx.send(job()).expect("queue");
        drop(tx);
        let (taken, closed) = drain(&rx, 4);
        assert_eq!(taken.len(), 1);
        assert!(closed, "the queue closed");
    }
}
