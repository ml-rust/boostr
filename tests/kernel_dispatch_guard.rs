//! Fails when a CUDA kernel compiled into the fatbin has no Rust dispatch site.
//!
//! This scans `.cu` source text for `extern "C" __global__` kernel symbols
//! and checks whether each one is ever referenced from Rust — literally, or
//! assembled at runtime via `format!`. A kernel that nothing calls is
//! invisible: it never runs, so a latent bug in it (wrong math, an inverted
//! scale, the wrong mask) never surfaces. This test turns "nobody calls
//! this" into a build failure instead of a silent trap.
//!
//! It does NOT require a GPU or the `cuda` feature — it is pure text
//! scanning, so it runs on a plain CPU build and protects contributors
//! without CUDA hardware.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Allowlist
// ---------------------------------------------------------------------------
//
// An entry here is a deliberate, TEMPORARY exception — not a place to
// silence this test. Unreferenced code in this repo is routinely UNFINISHED
// WORK, not dead code: the fix is to wire the kernel, never to delete it or
// add it here just to get the test green. Every entry must carry a written
// reason explaining why the kernel cannot be wired right now.
const ALLOWED_UNWIRED: &[(&str, &str)] = &[
    // (kernel_name, reason)
    (
        "quantize_kv_fp8_per_token_fp32",
        "No dequantize_kv_fp8_per_token_fp32 kernel exists, so an F32 quantize \
         has no matching-precision inverse. The CUDA dispatch casts F32 to F16 \
         and uses the F16 pair instead. Wire this once an F32 dequant exists.",
    ),
    (
        "quant_gemv_q4_k_q8_1",
        "Superseded by quant_gemv_q4_k_q8_1_mwr, which dispatch_gemv selects for \
         Q4_K on every path. The _mwr rewrite splits K across 4 warps; this is \
         the older single-warp-per-column form and covers no shape _mwr misses.",
    ),
    (
        "quant_gemv_q6_k_q8_1",
        "Superseded by quant_gemv_q6_k_q8_1_mwr, selected by dispatch_gemv for \
         Q6_K on every path. Same relationship as the Q4_K pair above.",
    ),
    (
        "quant_gemv_q4_k_fused",
        "Unfinished, and wiring it as-is would CHANGE NUMERICS. It computes the \
         Q8_1 activation scale in full f32, while quantize_f32_q8_1 rounds that \
         scale through __half; the live GEMV reads the rounded value. It is also \
         built on the pre-MWR single-warp-per-column layout. Finishing it means \
         matching the __half rounding and porting to the MWR K-split, then \
         re-validating tests/gguf_conformance_llama_cpp.rs.",
    ),
    (
        "quant_matmul_q4_k_f32",
        "Superseded by quant_mmq_q4_k_q8_1 / _mma, which dispatch_matmul selects \
         for Q4_K above the GEMV crossover. This is a scalar f32 accumulate with \
         no dp4a or tensor cores, and its arithmetic differs from the shipping \
         path, so it is not a drop-in.",
    ),
    (
        "quant_matmul_q6_k_f32",
        "Superseded by quant_mmq_q6_k_q8_1 / _mma. One thread per output element, \
         re-reading the full weight row from global memory each time — the exact \
         pattern quant_matmul.cu's own header calls out as far slower at large M.",
    ),
];

/// Shortest static head accepted as a `format!` dispatch prefix. Set from a
/// real case: `format!("clip_scale_{}", suffix)` has an 11-byte head, and a
/// threshold of 12 reported all three `clip_scale_*` kernels as unwired when
/// they are dispatched. The match is anchored on a following `{` or `_{`, so
/// a short prefix still cannot match arbitrary text.
const MIN_PREFIX_LEN: usize = 8;

// ---------------------------------------------------------------------------
// Low-level text scanning helpers
// ---------------------------------------------------------------------------

fn skip_ws(text: &str, mut pos: usize) -> usize {
    let bytes = text.as_bytes();
    while pos < bytes.len() && (bytes[pos] as char).is_whitespace() {
        pos += 1;
    }
    pos
}

/// Given the byte index of an opening `(`, returns the index just past its
/// matching closing `)`. Byte-level scanning is safe here: UTF-8 multi-byte
/// continuation bytes always have the high bit set and can never equal the
/// ASCII bytes for `(` or `)`.
fn match_balanced_parens(text: &str, open_pos: usize) -> usize {
    let bytes = text.as_bytes();
    debug_assert_eq!(bytes[open_pos], b'(');
    let mut depth = 1i32;
    let mut i = open_pos + 1;
    while depth > 0 {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    i
}

fn line_bounds(text: &str, pos: usize) -> (usize, usize) {
    let start = text[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let end = text[pos..]
        .find('\n')
        .map(|i| pos + i)
        .unwrap_or(text.len());
    (start, end)
}

fn line_number(text: &str, byte_pos: usize) -> usize {
    text[..byte_pos].bytes().filter(|&b| b == b'\n').count() + 1
}

fn is_valid_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// After the byte offset just past `__global__`, skips whitespace and any
/// `__launch_bounds__(...)` call (which may appear either side of `void` in
/// this codebase), then expects `void`. Returns the offset just past `void`.
fn skip_to_name_start(text: &str, mut pos: usize) -> Option<usize> {
    loop {
        pos = skip_ws(text, pos);
        if text[pos..].starts_with("__launch_bounds__") {
            pos += "__launch_bounds__".len();
            pos = skip_ws(text, pos);
            if text.as_bytes().get(pos) != Some(&b'(') {
                return None;
            }
            pos = match_balanced_parens(text, pos);
            continue;
        }
        break;
    }
    if text[pos..].starts_with("void") {
        Some(pos + "void".len())
    } else {
        None
    }
}

/// True if the parameter list closing at `pos` (just past its `)`) is
/// followed by a function body (`{`) rather than a bare declaration (`;`).
fn is_definition(text: &str, mut pos: usize) -> bool {
    let bytes = text.as_bytes();
    while pos < bytes.len() {
        match bytes[pos] {
            b' ' | b'\t' | b'\r' | b'\n' | b'\\' => pos += 1,
            _ => break,
        }
    }
    pos < bytes.len() && bytes[pos] == b'{'
}

fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut last = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[last..i].to_string());
                last = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(s[last..].to_string());
    parts
}

/// Cleans raw text captured between `void` and the kernel's parameter-list
/// `(` — strips whitespace and macro line-continuation backslashes, leaving
/// either a plain identifier or a `##`-joined token-paste template.
fn clean_name_text(raw: &str) -> String {
    raw.chars()
        .filter(|c| !c.is_whitespace() && *c != '\\')
        .collect()
}

// ---------------------------------------------------------------------------
// Macro-aware kernel extraction
// ---------------------------------------------------------------------------
//
// Several kernels are not written out by hand: a `#define` builds one or
// more `extern "C" __global__ void ...` definitions from macro parameters,
// using `##` token-pasting (e.g. `flash_attention_bwd_##HEAD_DIM##_##SUFFIX`)
// or a whole parameter as the name (e.g. `void NAME(...)`), and the macro is
// then invoked once per concrete instantiation. To get real, checkable
// kernel symbol names we expand those invocations rather than reporting the
// macro template text itself.

struct MacroDef {
    params: Vec<String>,
    body: String,
    /// Byte range in the file `[define_start, body_end)`, used to exclude
    /// this macro's own body from the direct (non-macro) kernel scan.
    range: (usize, usize),
}

fn find_macro_defs(text: &str) -> HashMap<String, MacroDef> {
    let mut macros = HashMap::new();
    let len = text.len();
    let mut pos = 0usize;
    while pos < len {
        let (line_start, line_end) = line_bounds(text, pos);
        let line = &text[line_start..line_end];
        let trimmed = line.trim_start();
        if trimmed.starts_with("#define") {
            let define_start = line_start + (line.len() - trimmed.len());
            let mut p = define_start + "#define".len();
            p = skip_ws(text, p);
            let name_start = p;
            let bytes = text.as_bytes();
            while p < len {
                let c = bytes[p];
                if c.is_ascii_alphanumeric() || c == b'_' {
                    p += 1;
                } else {
                    break;
                }
            }
            let name = text[name_start..p].to_string();
            if !name.is_empty() && bytes.get(p) == Some(&b'(') {
                let open = p;
                let close = match_balanced_parens(text, open);
                let params_text = if close > open + 1 {
                    &text[open + 1..close - 1]
                } else {
                    ""
                };
                let params: Vec<String> = params_text
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                let body_start = close;
                let mut cur = close;
                let body_end;
                loop {
                    let (ls, le) = line_bounds(text, cur);
                    let cur_line = &text[ls..le];
                    if cur_line.trim_end().ends_with('\\') {
                        if le + 1 >= len {
                            body_end = len;
                            break;
                        }
                        cur = le + 1;
                    } else {
                        body_end = le;
                        break;
                    }
                }

                let body = text[body_start..body_end].to_string();
                macros.insert(
                    name,
                    MacroDef {
                        params,
                        body,
                        range: (line_start, body_end),
                    },
                );
                pos = body_end + 1;
                continue;
            }
        }
        pos = line_end + 1;
    }
    macros
}

/// Finds every `extern "C" __global__ void <name-template>(` inside a macro
/// body and returns each as a list of `##`-split segments (a segment is
/// either a literal fragment or the name of one of the macro's parameters).
fn extract_name_templates(body: &str) -> Vec<Vec<String>> {
    let mut templates = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = body[search_from..].find("__global__") {
        let gpos = search_from + rel;
        search_from = gpos + "__global__".len();
        let Some(name_start) = skip_to_name_start(body, search_from) else {
            continue;
        };
        let Some(paren_rel) = body[name_start..].find('(') else {
            continue;
        };
        let paren_pos = name_start + paren_rel;
        let cleaned = clean_name_text(&body[name_start..paren_pos]);
        if cleaned.is_empty() {
            continue;
        }
        let segs: Vec<String> = if cleaned.contains("##") {
            cleaned.split("##").map(|s| s.to_string()).collect()
        } else {
            vec![cleaned]
        };
        templates.push(segs);
    }
    templates
}

/// Finds every invocation `MACRO(arg0, arg1, ...)` of `macro_name` in the
/// file (excluding the `#define` line itself) and returns each invocation's
/// line number and argument list.
fn find_macro_invocations(text: &str, macro_name: &str) -> Vec<(usize, Vec<String>)> {
    let mut out = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find(macro_name) {
        let mpos = search_from + rel;
        search_from = mpos + macro_name.len();

        if let Some(prev) = text[..mpos].chars().next_back()
            && (prev.is_alphanumeric() || prev == '_')
        {
            continue;
        }
        let after = mpos + macro_name.len();
        if let Some(next_char) = text[after..].chars().next()
            && (next_char.is_alphanumeric() || next_char == '_')
        {
            continue;
        }
        let p = skip_ws(text, after);
        if text.as_bytes().get(p) != Some(&b'(') {
            continue;
        }
        let (ls, _) = line_bounds(text, mpos);
        if text[ls..].trim_start().starts_with("#define") {
            continue;
        }
        let close = match_balanced_parens(text, p);
        let args_text = if close > p + 1 {
            &text[p + 1..close - 1]
        } else {
            ""
        };
        let args: Vec<String> = split_top_level_commas(args_text)
            .into_iter()
            .map(|s| s.trim().to_string())
            .collect();
        let line = line_number(text, mpos);
        out.push((line, args));
    }
    out
}

/// Direct (non-macro) kernel definitions: `extern "C" __global__ void name(`
/// or a bare `__global__ void name(` inside an `extern "C" { ... }` block.
/// Skips forward declarations (no `{` body) and anything inside a macro
/// body, which is handled separately by the macro-expansion path.
fn extract_direct_kernels(text: &str, macro_ranges: &[(usize, usize)]) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("__global__") {
        let gpos = search_from + rel;
        search_from = gpos + "__global__".len();
        if macro_ranges.iter().any(|&(s, e)| gpos >= s && gpos < e) {
            continue;
        }
        let Some(name_start) = skip_to_name_start(text, search_from) else {
            continue;
        };
        let Some(paren_rel) = text[name_start..].find('(') else {
            continue;
        };
        let paren_pos = name_start + paren_rel;
        let cleaned = clean_name_text(&text[name_start..paren_pos]);
        if cleaned.is_empty() || cleaned.contains("##") || !is_valid_ident(&cleaned) {
            continue;
        }
        let close_paren = match_balanced_parens(text, paren_pos);
        if !is_definition(text, close_paren) {
            continue;
        }
        let line = line_number(text, gpos);
        out.push((cleaned, line));
    }
    out
}

/// Extracts every kernel symbol defined in one `.cu` file: direct
/// definitions plus every concrete instantiation of every macro-defined
/// kernel template, with the line number of the site that produced it.
fn extract_kernels_from_file(text: &str) -> Vec<(String, usize)> {
    let macros = find_macro_defs(text);
    let macro_ranges: Vec<(usize, usize)> = macros.values().map(|m| m.range).collect();
    let mut out = extract_direct_kernels(text, &macro_ranges);

    for (macro_name, def) in &macros {
        let templates = extract_name_templates(&def.body);
        if templates.is_empty() {
            continue;
        }
        for (line, args) in find_macro_invocations(text, macro_name) {
            for segs in &templates {
                let mut name = String::new();
                for seg in segs {
                    match def.params.iter().position(|p| p == seg) {
                        Some(idx) => {
                            name.push_str(args.get(idx).map(String::as_str).unwrap_or(seg.as_str()))
                        }
                        None => name.push_str(seg),
                    }
                }
                out.push((name, line));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// The longest prefix of `name` (at least `MIN_PREFIX_LEN` bytes) that opens
/// a string literal in `rs_text` and is immediately followed by `{` or `_{`
/// — the static head of a `format!("prefix_{}", ...)` dispatch string.
///
/// Anchored on the opening quote for the same reason as the exact match: an
/// unanchored search would also hit prose in comments.
fn find_dynamic_prefix(name: &str, rs_text: &str) -> Option<String> {
    if name.len() < MIN_PREFIX_LEN {
        return None;
    }
    for len in (MIN_PREFIX_LEN..=name.len()).rev() {
        let prefix = &name[..len];
        if rs_text.contains(format!("\"{prefix}{{").as_str())
            || rs_text.contains(format!("\"{prefix}_{{").as_str())
        {
            return Some(prefix.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

fn collect_files_with_ext(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_with_ext(&path, ext, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
fn kernel_dispatch_guard() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");
    let tests_dir = manifest_dir.join("tests");

    let mut cu_files = Vec::new();
    collect_files_with_ext(&src_dir, "cu", &mut cu_files);
    cu_files.sort();

    let mut all_kernels: Vec<(String, String, usize)> = Vec::new();
    for path in &cu_files {
        let text = fs::read_to_string(path).unwrap();
        let relpath = path
            .strip_prefix(&manifest_dir)
            .unwrap()
            .to_string_lossy()
            .to_string();
        for (name, line) in extract_kernels_from_file(&text) {
            all_kernels.push((name, relpath.clone(), line));
        }
    }

    // Dedupe by kernel symbol name, keeping the first occurrence (some
    // kernels are forward-declared then defined, or, in a couple of cases,
    // the same name is reused across two files).
    let mut sites: HashMap<String, (String, usize)> = HashMap::new();
    let mut ordered_names: Vec<String> = Vec::new();
    for (name, relpath, line) in &all_kernels {
        if !sites.contains_key(name) {
            sites.insert(name.clone(), (relpath.clone(), *line));
            ordered_names.push(name.clone());
        }
    }

    // Rot-detector, not a target: if `.cu` discovery or the `__global__`
    // scan silently breaks and starts matching nothing, this test must not
    // pass vacuously. The real count is in the high hundreds; 200 is a
    // generous floor with headroom below the current true count.
    const MIN_PLAUSIBLE_KERNELS: usize = 200;
    assert!(
        ordered_names.len() >= MIN_PLAUSIBLE_KERNELS,
        "kernel_dispatch_guard found only {} kernel symbols across {} .cu file(s) under {:?} \
         (expected at least {MIN_PLAUSIBLE_KERNELS}). This means the .cu file discovery or the \
         __global__ scan broke, not that kernels were genuinely removed — if kernels really were \
         deleted in bulk, lower this floor deliberately.",
        ordered_names.len(),
        cu_files.len(),
        src_dir,
    );

    let mut rs_files = Vec::new();
    collect_files_with_ext(&src_dir, "rs", &mut rs_files);
    collect_files_with_ext(&tests_dir, "rs", &mut rs_files);
    let mut rs_text = String::new();
    for path in &rs_files {
        rs_text.push_str(&fs::read_to_string(path).unwrap());
        rs_text.push('\n');
    }

    let allowed: HashMap<&str, &str> = ALLOWED_UNWIRED.iter().copied().collect();
    let mut unwired: Vec<(String, String, usize)> = Vec::new();
    for name in &ordered_names {
        // Anchored on the quotes: a kernel name only ever reaches CUDA as a
        // string literal. An unanchored search also matches COMMENTS, so
        // documenting an unwired kernel would mark it wired and hide it —
        // the exact failure this test exists to prevent.
        if rs_text.contains(&format!("\"{name}\"")) {
            continue; // wired: exact string-literal match
        }
        if find_dynamic_prefix(name, &rs_text).is_some() {
            continue; // dynamically wired: static prefix feeds a format! lookup
        }
        if allowed.contains_key(name.as_str()) {
            continue; // deliberate, documented exception
        }
        let (relpath, line) = &sites[name];
        unwired.push((name.clone(), relpath.clone(), *line));
    }
    unwired.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));

    assert!(
        unwired.is_empty(),
        "{} CUDA kernel(s) are compiled into the fatbin but have no Rust dispatch site:\n\n{}\n\
         Three options:\n\
         1. Wire the kernel — call it from Rust dispatch code (a match arm, a format! lookup, \
         or a direct load_function call).\n\
         2. Add it to ALLOWED_UNWIRED at the top of tests/kernel_dispatch_guard.rs with a \
         written reason it cannot be wired right now.\n\
         3. Deleting the kernel is NOT an option — unreferenced code in this repo is routinely \
         unfinished work, not dead code.\n",
        unwired.len(),
        unwired
            .iter()
            .map(|(name, relpath, line)| format!("  {name}  ({relpath}:{line})"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}
