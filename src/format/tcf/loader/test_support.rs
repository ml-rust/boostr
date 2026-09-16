//! Shared test fixture for the loader test modules.

use super::directory::TcfLoader;
use crate::error::Result;
use crate::format::tcf::fixtures;

pub(super) fn open_fixture(bytes: &[u8]) -> Result<(tempfile::NamedTempFile, TcfLoader)> {
    let file = fixtures::write_temp(bytes);
    let loader = TcfLoader::open(file.path())?;
    Ok((file, loader))
}
