//! Cancelled and finished request ids a worker remembers.

use std::collections::{HashSet, VecDeque};
use std::sync::Mutex;

/// Ids each set holds before it drops its oldest.
const CAPACITY: usize = 4096;

/// A set that drops its oldest id past [`CAPACITY`].
#[derive(Default)]
struct BoundedSet {
    ids: HashSet<u64>,
    order: VecDeque<u64>,
}

impl BoundedSet {
    fn insert(&mut self, id: u64) {
        if !self.ids.insert(id) {
            return;
        }
        self.order.push_back(id);
        while self.order.len() > CAPACITY {
            if let Some(old) = self.order.pop_front() {
                self.ids.remove(&old);
            }
        }
    }

    fn remove(&mut self, id: u64) {
        if self.ids.remove(&id) {
            self.order.retain(|&x| x != id);
        }
    }
}

#[derive(Default)]
struct Inner {
    cancelled: BoundedSet,
    finished: BoundedSet,
}

/// The cancel state of one worker.
///
/// A cancel can arrive before the request it names. The book keeps it until
/// the worker meets that request. A cancel of a finished request is dropped.
/// Both sets are bounded: an evicted cancel only lets that request run to its
/// end, and the router drops events of a request it no longer tracks.
#[derive(Default)]
pub(crate) struct CancelBook {
    inner: Mutex<Inner>,
}

impl CancelBook {
    pub(crate) fn cancel(&self, id: u64) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !g.finished.ids.contains(&id) {
            g.cancelled.insert(id);
        }
    }

    pub(crate) fn is_cancelled(&self, id: u64) -> bool {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.cancelled.ids.contains(&id)
    }

    /// Marks `id` done: its cancel entry goes, and later cancels of it are dropped.
    pub(crate) fn finish(&self, id: u64) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.cancelled.remove(id);
        g.finished.insert(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_before_and_after_finish() {
        let book = CancelBook::default();
        book.cancel(5);
        assert!(book.is_cancelled(5));
        book.finish(5);
        assert!(!book.is_cancelled(5));
        book.cancel(5);
        assert!(
            !book.is_cancelled(5),
            "a cancel of a finished id is dropped"
        );
    }

    #[test]
    fn sets_stay_bounded() {
        let book = CancelBook::default();
        for id in 0..(CAPACITY as u64 + 10) {
            book.cancel(id);
        }
        assert!(!book.is_cancelled(0), "the oldest cancel is evicted");
        assert!(book.is_cancelled(CAPACITY as u64 + 9));
        let g = book.inner.lock().unwrap();
        assert_eq!(g.cancelled.ids.len(), CAPACITY);
        assert_eq!(g.cancelled.order.len(), CAPACITY);
    }
}
