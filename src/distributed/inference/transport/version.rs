//! The version byte that opens every versioned swarm frame.
//!
//! ```text
//! [version: u8 = PROTOCOL_VERSION][body]
//! ```
//!
//! `GEN_REQUEST`, `LOGITS`, `STAGE_ERROR`, `LAYER_ASSIGNMENT` and
//! `MODEL_PATH` frames start with it, and so does an encoded [`Activation`].
//! A reader refuses any other version with an error that names both
//! versions. Token ids carry no byte of their own: they travel only inside a
//! `GEN_REQUEST` frame, after its versioned header. `WORKER_READY` and
//! `SHUTDOWN` are one-byte notices whose content no reader reads.
//!
//! [`Activation`]: super::Activation

use anyhow::{Result, bail};

/// Version every swarm frame writes. A reader refuses any other version.
pub const PROTOCOL_VERSION: u8 = 2;

/// An empty frame buffer that holds the version byte, with room for `body` more bytes.
pub(crate) fn versioned(body: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + body);
    buf.push(PROTOCOL_VERSION);
    buf
}

/// Checks a version byte read from a `what` frame.
///
/// # Errors
///
/// `version` is not [`PROTOCOL_VERSION`].
pub(crate) fn check_version(version: u8, what: &str) -> Result<()> {
    if version != PROTOCOL_VERSION {
        bail!(
            "a swarm {what} of protocol version {version} reached a reader of version \
             {PROTOCOL_VERSION}. Run the same boostr release on every rank"
        );
    }
    Ok(())
}

/// The body of a `what` frame after its version byte.
///
/// # Errors
///
/// An empty frame, or a version [`check_version`] refuses.
pub(crate) fn body<'a>(frame: &'a [u8], what: &str) -> Result<&'a [u8]> {
    let Some((&version, body)) = frame.split_first() else {
        bail!("a swarm {what} is empty and has no version byte");
    };
    check_version(version, what)?;
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_byte_round_trips_and_another_version_is_refused() {
        let mut frame = versioned(2);
        frame.extend_from_slice(&[7, 8]);
        assert_eq!(body(&frame, "test frame").unwrap(), [7, 8]);
        frame[0] = PROTOCOL_VERSION + 1;
        let err = body(&frame, "test frame").unwrap_err().to_string();
        assert!(
            err.contains(&format!(
                "test frame of protocol version {}",
                PROTOCOL_VERSION + 1
            )),
            "{err}"
        );
        let err = body(&[], "test frame").unwrap_err().to_string();
        assert!(err.contains("no version byte"), "{err}");
    }
}
