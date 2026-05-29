//! Pickle stream primitive readers and the value stack helpers.

use super::pickle::PValue;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::io::{Cursor, Read};

pub(super) fn pop(stack: &mut Vec<PValue>) -> Result<PValue> {
    stack.pop().ok_or_else(|| Error::ModelError {
        reason: "pickle stack underflow".into(),
    })
}

pub(super) fn read_exact(cur: &mut Cursor<&[u8]>, buf: &mut [u8]) -> Result<()> {
    cur.read_exact(buf).map_err(|e| Error::ModelError {
        reason: format!("truncated pickle: {e}"),
    })
}

pub(super) fn read_u16(cur: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut b = [0u8; 2];
    read_exact(cur, &mut b)?;
    Ok(u16::from_le_bytes(b))
}

pub(super) fn read_u32(cur: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(cur, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub(super) fn read_i32(cur: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut b = [0u8; 4];
    read_exact(cur, &mut b)?;
    Ok(i32::from_le_bytes(b))
}

pub(super) fn read_u64(cur: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut b = [0u8; 8];
    read_exact(cur, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

pub(super) fn read_signed_long(cur: &mut Cursor<&[u8]>, len: usize) -> Result<i64> {
    if len == 0 {
        return Ok(0);
    }
    if len > 8 {
        return Err(Error::ModelError {
            reason: format!("LONG with {len} bytes exceeds i64"),
        });
    }
    let mut buf = vec![0u8; len];
    read_exact(cur, &mut buf)?;
    let sign_extend = if buf[len - 1] & 0x80 != 0 { 0xff } else { 0x00 };
    let mut full = [sign_extend; 8];
    full[..len].copy_from_slice(&buf);
    Ok(i64::from_le_bytes(full))
}

pub(super) fn read_line(cur: &mut Cursor<&[u8]>) -> Result<String> {
    let mut out = Vec::new();
    loop {
        let mut b = [0u8; 1];
        read_exact(cur, &mut b)?;
        if b[0] == b'\n' {
            break;
        }
        out.push(b[0]);
    }
    from_utf8(out)
}

pub(super) fn from_utf8(bytes: Vec<u8>) -> Result<String> {
    String::from_utf8(bytes).map_err(|e| Error::ModelError {
        reason: format!("invalid utf-8 in pickle: {e}"),
    })
}

pub(super) fn memo_get(memo: &HashMap<u32, PValue>, idx: u32) -> Result<PValue> {
    memo.get(&idx).cloned().ok_or_else(|| Error::ModelError {
        reason: format!("memo miss: {idx}"),
    })
}
