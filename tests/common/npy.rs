//! A minimal `.npy` reader: format 1.0 to 3.0, C order, little-endian
//! `f4` or `u2` data. A `u2` array holds BF16 bits and reads as F32.

use std::path::Path;

/// One array: its shape and its values as F32.
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

fn fail(path: &Path, why: &str) -> ! {
    panic!("{}: {why}", path.display())
}

/// Reads `path`, panicking with the path and the reason when the file is
/// not an array this reader handles.
pub fn read(path: &Path) -> Npy {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        fail(path, "no .npy magic");
    }
    let (header_len, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 if bytes.len() >= 12 => {
            let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (n as usize, 12)
        }
        v => fail(path, &format!("format version {v} is not 1, 2 or 3")),
    };
    let end = start + header_len;
    if bytes.len() < end {
        fail(path, "header runs past the file");
    }
    let header = String::from_utf8_lossy(&bytes[start..end]).to_string();
    let field = |key: &str| -> String {
        let at = header
            .find(&format!("'{key}':"))
            .unwrap_or_else(|| fail(path, &format!("header has no {key}: {header}")));
        header[at + key.len() + 3..].trim_start().to_string()
    };
    let descr = field("descr");
    let fortran = field("fortran_order");
    if !fortran.starts_with("False") {
        fail(path, "Fortran order is not handled");
    }
    let shape_text = field("shape");
    let close = shape_text
        .find(')')
        .unwrap_or_else(|| fail(path, &format!("bad shape in {header}")));
    let shape: Vec<usize> = shape_text[1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .unwrap_or_else(|_| fail(path, &format!("bad dim {s}")))
        })
        .collect();
    let count: usize = shape.iter().product();
    let body = &bytes[end..];
    let data: Vec<f32> = if descr.starts_with("'<f4'") {
        if body.len() != count * 4 {
            fail(path, "f4 payload size disagrees with the shape");
        }
        let (words, _) = body.as_chunks::<4>();
        words.iter().map(|c| f32::from_le_bytes(*c)).collect()
    } else if descr.starts_with("'<u2'") {
        if body.len() != count * 2 {
            fail(path, "u2 payload size disagrees with the shape");
        }
        let (words, _) = body.as_chunks::<2>();
        words
            .iter()
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
            .collect()
    } else {
        fail(path, &format!("dtype {descr} is not '<f4' or '<u2'"))
    };
    Npy { shape, data }
}
