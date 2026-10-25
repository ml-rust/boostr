//! Per-request state of one stage: a cache per request id, the position
//! each request reached, and the cap on requests in flight.
//!
//! A forward checks a request's cache out, runs, and checks it back in.
//! A forward that fails never checks it back in, so an error frees the
//! request at this stage.

use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

use super::super::transport::{GenRequestHeader, StageOp};

struct Entry<C> {
    cache: C,
    /// The position the next frame of this request must start at.
    next_position: u32,
}

/// One cache per open request, at most `max_requests` of them.
pub(super) struct RequestBook<C> {
    entries: HashMap<u64, Entry<C>>,
    max_requests: usize,
    max_seq_len: usize,
}

impl<C> RequestBook<C> {
    pub(super) fn new(max_requests: usize, max_seq_len: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_requests,
            max_seq_len,
        }
    }

    /// The cache of `header`'s request, taken out of the book for one forward.
    ///
    /// A prefill opens the request with `open`. A decode takes the open
    /// request. Check the result back in with [`Self::checkin`].
    ///
    /// # Errors
    ///
    /// Each frees the request:
    /// - a release header
    /// - an end position past `max_seq_len`
    /// - a prefill of an open id, at a position other than 0, or past the cap
    /// - a decode of an id that is not open, or at another position than the
    ///   one the request reached
    /// - the error of `open`
    pub(super) fn checkout(
        &mut self,
        header: &GenRequestHeader,
        open: impl FnOnce() -> Result<C>,
    ) -> Result<C> {
        let id = header.request_id;
        let taken = self.entries.remove(&id);
        let end = header.end()?;
        if end as usize > self.max_seq_len {
            bail!(
                "request {id} reaches position {end}, past this stage's limit of {}",
                self.max_seq_len
            );
        }
        match header.op {
            StageOp::Release => bail!("request {id}: a release frame has no forward"),
            StageOp::Prefill => {
                if taken.is_some() {
                    bail!("request {id} is already open on this stage. Use a new id per request");
                }
                if header.position != 0 {
                    bail!(
                        "request {id}: a prefill starts at position 0, and this one starts at {}",
                        header.position
                    );
                }
                if self.entries.len() >= self.max_requests {
                    bail!(
                        "request {id}: this stage holds {} requests, its cap. Release a \
                         finished request first",
                        self.entries.len()
                    );
                }
                open()
            }
            StageOp::Decode => {
                let entry = taken.ok_or_else(|| {
                    anyhow!("request {id} is not open on this stage. Send its prefill first")
                })?;
                if entry.next_position != header.position {
                    bail!(
                        "request {id} is at position {}, and the frame starts at {}",
                        entry.next_position,
                        header.position
                    );
                }
                Ok(entry.cache)
            }
        }
    }

    /// Returns the cache of a successful forward of `header` to the book.
    ///
    /// The request's next frame starts at `header.position + header.seq_len`.
    pub(super) fn checkin(&mut self, header: &GenRequestHeader, cache: C) {
        let next_position = header.position.saturating_add(header.seq_len);
        self.entries.insert(
            header.request_id,
            Entry {
                cache,
                next_position,
            },
        );
    }

    /// Frees `request_id`. Returns whether it was open.
    pub(super) fn release(&mut self, request_id: u64) -> bool {
        self.entries.remove(&request_id).is_some()
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(request_id: u64, op: StageOp, position: u32, seq_len: u32) -> GenRequestHeader {
        GenRequestHeader {
            request_id,
            op,
            position,
            seq_len,
        }
    }

    /// Runs one forward that leaves `cache + seq_len` in the book.
    fn step(book: &mut RequestBook<u32>, h: GenRequestHeader) -> Result<u32> {
        let cache = book.checkout(&h, || Ok(0))?;
        let cache = cache + h.seq_len;
        book.checkin(&h, cache);
        Ok(cache)
    }

    #[test]
    fn test_positions_advance_per_request() {
        let mut book = RequestBook::new(4, 16);
        assert_eq!(
            step(&mut book, frame(1, StageOp::Prefill, 0, 3)).unwrap(),
            3
        );
        assert_eq!(
            step(&mut book, frame(2, StageOp::Prefill, 0, 5)).unwrap(),
            5
        );
        assert_eq!(step(&mut book, frame(1, StageOp::Decode, 3, 1)).unwrap(), 4);
        assert_eq!(step(&mut book, frame(2, StageOp::Decode, 5, 2)).unwrap(), 7);
        assert_eq!(book.len(), 2);
    }

    #[test]
    fn test_a_wrong_position_fails_and_frees_the_request() {
        let mut book = RequestBook::new(4, 16);
        step(&mut book, frame(1, StageOp::Prefill, 0, 3)).unwrap();
        let err = step(&mut book, frame(1, StageOp::Decode, 2, 1)).unwrap_err();
        assert!(err.to_string().contains("is at position 3"), "{err}");
        assert_eq!(book.len(), 0);
        let err = step(&mut book, frame(1, StageOp::Decode, 3, 1)).unwrap_err();
        assert!(err.to_string().contains("not open"), "{err}");
    }

    #[test]
    fn test_a_prefill_needs_a_new_id_position_0_and_room() {
        let mut book = RequestBook::new(1, 16);
        let err = step(&mut book, frame(1, StageOp::Prefill, 2, 1)).unwrap_err();
        assert!(err.to_string().contains("starts at 2"), "{err}");
        step(&mut book, frame(1, StageOp::Prefill, 0, 1)).unwrap();
        let err = step(&mut book, frame(2, StageOp::Prefill, 0, 1)).unwrap_err();
        assert!(err.to_string().contains("its cap"), "{err}");
        let err = step(&mut book, frame(1, StageOp::Prefill, 0, 1)).unwrap_err();
        assert!(err.to_string().contains("already open"), "{err}");
    }

    #[test]
    fn test_release_frees_room_and_the_limit_holds() {
        let mut book = RequestBook::new(1, 4);
        step(&mut book, frame(1, StageOp::Prefill, 0, 4)).unwrap();
        assert!(book.release(1));
        assert!(!book.release(1));
        step(&mut book, frame(2, StageOp::Prefill, 0, 2)).unwrap();
        let err = step(&mut book, frame(2, StageOp::Decode, 2, 3)).unwrap_err();
        assert!(err.to_string().contains("position 5, past"), "{err}");
        step(&mut book, frame(3, StageOp::Prefill, 0, 1)).unwrap();
        book.clear();
        assert_eq!(book.len(), 0);
    }

    #[test]
    fn test_a_failed_open_leaves_nothing() {
        let mut book: RequestBook<u32> = RequestBook::new(2, 8);
        let h = frame(1, StageOp::Prefill, 0, 1);
        assert!(book.checkout(&h, || Err(anyhow!("no memory"))).is_err());
        assert_eq!(book.len(), 0);
        assert!(
            book.checkout(&GenRequestHeader::release(1), || Ok(0))
                .is_err()
        );
    }
}
