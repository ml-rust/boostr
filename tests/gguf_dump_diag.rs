//! Diagnostic dump of a GGUF file's architecture metadata and tensor shapes.
//!
//! `#[ignore]` by default — needs a real model file.
//!
//! ```bash
//! BOOSTR_GGUF_DUMP=/path/to/model.gguf \
//!     cargo nextest run -p boostr --test gguf_dump_diag --no-capture --run-ignored all
//! ```

use boostr::format::gguf::Gguf;

#[test]
#[ignore]
fn dump_gguf_metadata_and_tensors() {
    let path = std::env::var("BOOSTR_GGUF_DUMP").expect("set BOOSTR_GGUF_DUMP");
    let gguf = Gguf::open(&path).expect("open gguf");
    let md = gguf.metadata();

    eprintln!("=== ARCH METADATA ===");
    let arch = md.get_string("general.architecture").unwrap_or("<none>");
    eprintln!("  general.architecture = {arch}");
    for suffix in [
        "block_count",
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.layer_norm_epsilon",
        "attention.causal",
        "rope.freq_base",
        "rope.dimension_count",
        "rope.scaling.factor",
        "pooling_type",
    ] {
        let key = format!("{arch}.{suffix}");
        if let Some(v) = md.get(&key) {
            eprintln!("  {key} = {v:?}");
        }
    }

    eprintln!("=== TENSORS (non-block + blk.0) ===");
    let mut names: Vec<String> = gguf.tensor_names().map(str::to_owned).collect();
    names.sort();
    for n in &names {
        if n.starts_with("blk.") && !n.starts_with("blk.0.") {
            continue;
        }
        let info = gguf.tensor_info(n).expect("tensor info");
        eprintln!(
            "  {n}  shape={:?}  ggml_type={:?}",
            info.shape, info.ggml_type
        );
    }
    eprintln!("  (total tensors: {})", names.len());
}

/// Does the GGUF tokenizer actually distinguish different texts?
///
/// A tokenizer that emits UNK (or nothing) for every word makes every input
/// encode to nearly the same id sequence, which collapses embeddings to a
/// near-constant vector — cosine ~0.999 between unrelated texts.
#[test]
#[ignore]
fn dump_tokenizer_output() {
    use boostr::format::gguf_tokenizer::GgufTokenizer;

    let path = std::env::var("BOOSTR_GGUF_DUMP").expect("set BOOSTR_GGUF_DUMP");
    let gguf = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&gguf).expect("tokenizer");

    for text in [
        "storage flush persists the in-memory index state to disk",
        "kangaroos hop across the australian outback at dusk",
        "hello",
    ] {
        let ids = tok.encode(text);
        eprintln!("  {:?}\n    -> {} ids: {:?}", text, ids.len(), ids);
    }
}

/// Round-trip check: does `encode` → `decode` reproduce the input, and do
/// well-known bert-base-uncased ids map to their expected strings?
#[test]
#[ignore]
fn tokenizer_round_trip_and_vocab_probe() {
    use boostr::format::gguf_tokenizer::GgufTokenizer;

    let path = std::env::var("BOOSTR_GGUF_DUMP").expect("set BOOSTR_GGUF_DUMP");
    let gguf = Gguf::open(&path).expect("open gguf");
    let md = gguf.metadata();

    eprintln!(
        "  tokenizer.ggml.model = {:?}",
        md.get_string("tokenizer.ggml.model")
    );
    let toks = md.get_array("tokenizer.ggml.tokens").expect("tokens array");
    eprintln!("  vocab len = {}", toks.len());
    // bert-base-uncased reference ids.
    for (id, expect) in [
        (1996usize, "the"),
        (7592, "hello"),
        (101, "[CLS]"),
        (102, "[SEP]"),
    ] {
        let got = toks.get(id).and_then(|v| v.as_string()).unwrap_or("<oob>");
        eprintln!("  vocab[{id}] = {got:?}   (bert-base-uncased expects {expect:?})");
    }
    eprintln!(
        "  first 10 vocab entries: {:?}",
        toks.iter()
            .take(10)
            .map(|v| v.as_string().unwrap_or("?"))
            .collect::<Vec<_>>()
    );

    let tok = GgufTokenizer::from_gguf(&gguf).expect("tokenizer");
    let text = "hello the quick brown fox";
    let ids = tok.encode(text);
    eprintln!("  encode({text:?}) = {ids:?}");
    eprintln!("  decode(...)      = {:?}", tok.decode(&ids));
}

/// Determine the exact vocab convention: how many tokens carry the
/// SentencePiece `▁` word-boundary marker vs the WordPiece `##` continuation
/// marker.
#[test]
#[ignore]
fn vocab_marker_census() {
    let path = std::env::var("BOOSTR_GGUF_DUMP").expect("set BOOSTR_GGUF_DUMP");
    let gguf = Gguf::open(&path).expect("open gguf");
    let toks = gguf
        .metadata()
        .get_array("tokenizer.ggml.tokens")
        .expect("tokens");

    let strs: Vec<&str> = toks.iter().map(|v| v.as_string().unwrap_or("")).collect();
    let n_underscore = strs.iter().filter(|t| t.starts_with('\u{2581}')).count();
    let n_hash = strs.iter().filter(|t| t.starts_with("##")).count();
    let n_special = strs
        .iter()
        .filter(|t| t.starts_with('[') && t.ends_with(']'))
        .count();
    eprintln!(
        "  total={}  with_U2581={n_underscore}  with_##={n_hash}  special=[{n_special}]",
        strs.len()
    );

    // bert-base-uncased: 2003="is", 2015="##s", 1997="of", 2064="can"
    for id in [2003usize, 2015, 1997, 2064, 3335] {
        eprintln!("  vocab[{id}] = {:?}", strs.get(id));
    }
    eprintln!(
        "  sample continuation-looking entries: {:?}",
        strs.iter()
            .filter(|t| !t.starts_with('\u{2581}') && !t.starts_with('[') && t.len() <= 3)
            .take(15)
            .collect::<Vec<_>>()
    );
}

/// Probe punctuation / digit ids, which in bert-base-uncased are word-INITIAL
/// tokens with no `##`. If they are stored here without `▁`, a naive
/// "no marker => continuation" rule would corrupt them.
#[test]
#[ignore]
fn vocab_punctuation_probe() {
    let path = std::env::var("BOOSTR_GGUF_DUMP").expect("set BOOSTR_GGUF_DUMP");
    let gguf = Gguf::open(&path).expect("open gguf");
    let toks = gguf
        .metadata()
        .get_array("tokenizer.ggml.tokens")
        .expect("tokens");
    let strs: Vec<&str> = toks.iter().map(|v| v.as_string().unwrap_or("")).collect();
    // bert-base-uncased: 999="!", 1010=",", 1012=".", 1029="?", 1015="1", 1037="a"
    for (id, expect) in [
        (999usize, "!"),
        (1010, ","),
        (1012, "."),
        (1029, "?"),
        (1015, "1"),
        (1037, "a"),
    ] {
        eprintln!(
            "  vocab[{id}] = {:?}  (bert expects {expect:?})",
            strs.get(id)
        );
    }
}

/// Are distinct vocab rows of `token_embd.weight` actually distinct once
/// loaded? GGUF stores dims fastest-first, so `shape=[hidden, vocab]` means
/// memory is `vocab_id * hidden + h`. If the loader builds a tensor whose row
/// stride is wrong, every "row" is a blend of the table and the model becomes
/// input-independent.
#[test]
#[ignore]
fn token_embedding_rows_are_distinct() {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    let path = std::env::var("BOOSTR_GGUF_DUMP").expect("set BOOSTR_GGUF_DUMP");
    let mut gguf = Gguf::open(&path).expect("open gguf");
    let info = gguf.tensor_info("token_embd.weight").expect("info").clone();
    eprintln!("  token_embd.weight gguf shape = {:?}", info.shape);

    let device = CpuDevice::new();
    let t = gguf
        .load_tensor_f32_streaming::<CpuRuntime>("token_embd.weight", &device)
        .expect("load token_embd");
    eprintln!("  loaded tensor shape = {:?}", t.shape());

    let data: Vec<f32> = t.to_vec();
    let hidden = 768usize;
    let row = |id: usize| -> &[f32] { &data[id * hidden..(id + 1) * hidden] };

    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    };

    // 7592="hello", 4419="fox", 1996="the", 100="[UNK]"
    for (a, b, label) in [
        (7592usize, 4419usize, "hello vs fox"),
        (7592, 1996, "hello vs the"),
        (4419, 1996, "fox vs the"),
    ] {
        eprintln!(
            "  cos(row[{a}], row[{b}]) = {:.6}   ({label})",
            cos(row(a), row(b))
        );
    }
    let n0: f32 = row(7592).iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("  ||row[7592]|| = {n0:.6}");
}

/// Split the fused `attn_qkv.weight` the way `from_weights_nomic` does and
/// report each third's magnitude. A near-zero third means the split is reading
/// the wrong region: Q≈0 makes every attention score ~equal, so softmax is
/// uniform and each token becomes the mean of all values — the encoder then
/// averages the sequence away layer by layer.
#[test]
#[ignore]
fn qkv_split_thirds_have_signal() {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    let path = std::env::var("BOOSTR_GGUF_DUMP").expect("set BOOSTR_GGUF_DUMP");
    let mut gguf = Gguf::open(&path).expect("open gguf");
    let info = gguf
        .tensor_info("blk.0.attn_qkv.weight")
        .expect("info")
        .clone();
    eprintln!(
        "  gguf shape = {:?}  type = {:?}",
        info.shape, info.ggml_type
    );

    let device = CpuDevice::new();
    let t = gguf
        .load_tensor_f32_streaming::<CpuRuntime>("blk.0.attn_qkv.weight", &device)
        .expect("load qkv");
    eprintln!("  loaded shape = {:?}", t.shape());

    let data: Vec<f32> = t.to_vec();
    let hidden = 768usize;
    let proj = hidden * hidden;
    eprintln!("  total elems = {} (expect {})", data.len(), 3 * proj);

    for (name, slice) in [
        ("Q", &data[0..proj]),
        ("K", &data[proj..2 * proj]),
        ("V", &data[2 * proj..3 * proj]),
    ] {
        let rms = (slice.iter().map(|x| x * x).sum::<f32>() / slice.len() as f32).sqrt();
        let maxabs = slice.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let zeros = slice.iter().filter(|x| **x == 0.0).count();
        eprintln!(
            "  {name}: rms={rms:.6}  max|x|={maxabs:.6}  zeros={zeros}/{}",
            slice.len()
        );
    }
}

/// Ground-truth check on K-quant dequantization: load the SAME tensor from an
/// f16 GGUF and a K-quantized GGUF of the same model and compare.
///
/// Correct dequantization of Q4_K should track the f16 reference closely
/// (cosine ≳0.99 per row, small relative error). A large deviation means the
/// dequant kernel is misreading the block layout, which silently corrupts every
/// weight in the model.
///
/// ```bash
/// BOOSTR_GGUF_REF=/path/model.f16.gguf BOOSTR_GGUF_QUANT=/path/model.Q4_K_M.gguf \
///   cargo nextest run -p boostr --test gguf_dump_diag dequant_matches --no-capture --run-ignored all
/// ```
#[test]
#[ignore]
fn dequant_matches_f16_reference() {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    let ref_path = std::env::var("BOOSTR_GGUF_REF").expect("set BOOSTR_GGUF_REF");
    let quant_path = std::env::var("BOOSTR_GGUF_QUANT").expect("set BOOSTR_GGUF_QUANT");
    let device = CpuDevice::new();

    let mut g_ref = Gguf::open(&ref_path).expect("open ref");
    let mut g_q = Gguf::open(&quant_path).expect("open quant");

    for name in [
        "token_embd.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_output.weight",
    ] {
        let (ti_r, ti_q) = (
            g_ref.tensor_info(name).expect("ref info").clone(),
            g_q.tensor_info(name).expect("quant info").clone(),
        );
        let a: Vec<f32> = g_ref
            .load_tensor_f32_streaming::<CpuRuntime>(name, &device)
            .expect("load ref")
            .to_vec();
        let b: Vec<f32> = g_q
            .load_tensor_f32_streaming::<CpuRuntime>(name, &device)
            .expect("load quant")
            .to_vec();
        assert_eq!(a.len(), b.len(), "{name}: element count differs");

        let dot: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum();
        let na: f64 = a
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = b
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * nb);

        let num: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| ((*x - *y) as f64) * ((*x - *y) as f64))
            .sum::<f64>()
            .sqrt();
        let rel_err = num / na;

        eprintln!(
            "  {name}\n    ref={:?} quant={:?}\n    cosine={cos:.6}  rel_err={rel_err:.6}  \
             rms_ref={:.6} rms_quant={:.6}",
            ti_r.ggml_type,
            ti_q.ggml_type,
            (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt(),
            (b.iter().map(|x| x * x).sum::<f32>() / b.len() as f32).sqrt(),
        );
    }
}
