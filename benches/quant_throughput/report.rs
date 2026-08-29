//! The comparison table, and the caveats a fair reading of it needs.
//!
//! Rows are ordered so a matched pair sits together: same backend, same size
//! class, same shape, same operation, TCF above GGUF. The `tcf/gguf` column on
//! the TCF row is the ratio of instructions per work unit — below 1.00 means
//! TCF costs less.

/// One measured case.
pub struct Row {
    pub id: String,
    pub pair: usize,
    pub codec: &'static str,
    pub encoding: String,
    pub bpw: f64,
    pub backend: &'static str,
    pub op: &'static str,
    pub shape: String,
    pub shape_label: &'static str,
    pub m: usize,
    pub iters: u64,
    pub unit: &'static str,
    pub units: u64,
    /// Retired user-space instructions per iteration, setup subtracted.
    pub instructions: Option<f64>,
    /// The same figure divided by the case's work units.
    pub per_unit: Option<f64>,
    /// Minimum reference cycles per iteration. Load-sensitive.
    pub cycles: Option<f64>,
    /// Minimum elapsed nanoseconds per iteration. Load-sensitive.
    pub ns: Option<f64>,
    pub alloc_count: Option<f64>,
    pub alloc_bytes: Option<f64>,
    pub error: Option<String>,
}

/// Run-wide facts the table cannot carry per row.
pub struct Context {
    pub perf: bool,
    pub load_before: Option<f64>,
    pub load_after: Option<f64>,
    pub threads: usize,
    pub reps: usize,
    pub csv: bool,
}

/// The sort key that puts a matched pair on adjacent lines.
fn key(row: &Row) -> (&'static str, usize, &'static str, usize, &'static str, u8) {
    (
        row.backend,
        row.pair,
        row.shape_label,
        row.m,
        row.op,
        u8::from(row.codec == "gguf"),
    )
}

pub fn print(rows: &[Row], context: &Context) {
    let mut ordered: Vec<&Row> = rows.iter().collect();
    ordered.sort_by(|a, b| key(a).cmp(&key(b)));
    if context.csv {
        print_csv(&ordered);
    } else {
        print_table(&ordered, context);
        print_caveats(context);
    }
}

fn print_csv(rows: &[&Row]) {
    println!(
        "id,codec,encoding,bpw,backend,op,shape,m,iters,unit,units,instructions_per_iter,\
instructions_per_unit,ratio_tcf_over_gguf,cycles_min,ns_min,alloc_count_per_iter,\
alloc_bytes_per_iter,error"
    );
    for (index, row) in rows.iter().enumerate() {
        println!(
            "{},{},{},{:.4},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            row.id,
            row.codec,
            row.encoding,
            row.bpw,
            row.backend,
            row.op,
            row.shape,
            row.m,
            row.iters,
            row.unit,
            row.units,
            opt(row.instructions),
            opt(row.per_unit),
            opt(ratio(rows, index)),
            opt(row.cycles),
            opt(row.ns),
            opt(row.alloc_count),
            opt(row.alloc_bytes),
            row.error.as_deref().unwrap_or(""),
        );
    }
}

fn print_table(rows: &[&Row], context: &Context) {
    println!();
    println!(
        "TCF vs GGUF quantized throughput — {} threads, perf {}, {} reps per phase",
        context.threads,
        if context.perf { "on" } else { "OFF" },
        context.reps,
    );
    println!(
        "load average 1m: {} before, {} after",
        opt(context.load_before),
        opt(context.load_after),
    );
    println!();
    let header = format!(
        "{:<5} {:<12} {:>5} {:<5} {:<8} {:<10} {:>10} {:>4} {:>6} {:>12} {:>12} {:>6} {:>9} {:>12} {:>12} {:>6} {:>10}",
        "codec",
        "encoding",
        "bpw",
        "back",
        "op",
        "shape",
        "NxK",
        "M",
        "iters",
        "instr/iter",
        "instr/unit",
        "unit",
        "tcf/gguf",
        "cycles*",
        "ns*",
        "alloc",
        "bytes",
    );
    println!("{header}");
    println!("{}", "-".repeat(header.chars().count()));

    let mut previous: Option<(&str, usize, &str)> = None;
    for (index, row) in rows.iter().enumerate() {
        let group = (row.backend, row.pair, row.shape_label);
        if previous.is_some_and(|p| p != group) {
            println!();
        }
        previous = Some(group);

        if let Some(error) = row.error.as_deref() {
            println!(
                "{:<5} {:<12} {:>5.2} {:<5} {:<8} {:<10} {:>10} {:>4} {:>6}  {error}",
                row.codec,
                row.encoding,
                row.bpw,
                row.backend,
                row.op,
                row.shape_label,
                row.shape,
                row.m,
                row.iters,
            );
            continue;
        }
        println!(
            "{:<5} {:<12} {:>5.2} {:<5} {:<8} {:<10} {:>10} {:>4} {:>6} {:>12} {:>12} {:>6} {:>9} {:>12} {:>12} {:>6} {:>10}",
            row.codec,
            row.encoding,
            row.bpw,
            row.backend,
            row.op,
            row.shape_label,
            row.shape,
            row.m,
            row.iters,
            si(row.instructions),
            fine(row.per_unit),
            row.unit,
            fine(ratio(rows, index)),
            si(row.cycles),
            si(row.ns),
            si(row.alloc_count),
            si(row.alloc_bytes),
        );
    }
}

/// TCF instructions per unit over the matched GGUF row's, on the TCF row only.
fn ratio(rows: &[&Row], index: usize) -> Option<f64> {
    let row = rows.get(index)?;
    if row.codec != "tcf" {
        return None;
    }
    let partner = rows.get(index + 1)?;
    if partner.codec != "gguf"
        || partner.pair != row.pair
        || partner.backend != row.backend
        || partner.shape_label != row.shape_label
        || partner.m != row.m
        || partner.op != row.op
    {
        return None;
    }
    let (mine, theirs) = (row.per_unit?, partner.per_unit?);
    if theirs <= 0.0 {
        return None;
    }
    Some(mine / theirs)
}

fn print_caveats(context: &Context) {
    println!();
    println!("How to read this");
    println!("  instr/iter  retired user-space instructions per iteration, with a zero-iteration");
    println!("              run of the same case subtracted, so setup and warm-up are removed.");
    println!("              Deterministic. This is the metric the comparison rests on.");
    println!("  instr/unit  the same figure per element (dequant) or per multiply-accumulate");
    println!("              (matmul), so shapes and batch sizes share one scale.");
    println!("  tcf/gguf    instr/unit ratio against the GGUF row below. Under 1.00 favours TCF.");
    println!("  alloc/bytes heap allocations per iteration. Deterministic.");
    println!("  cycles*/ns* MINIMUM over iterations. WALL-CLOCK FAMILY, load-sensitive.");
    println!("              Read them only when the load average above is near zero.");
    if !context.perf {
        println!();
        println!("  perf is OFF, so every instruction column is blank. Install perf, or lower");
        println!("  kernel.perf_event_paranoid, and rerun.");
    }
    println!();
    println!("What this does NOT measure");
    println!("  - Quality. Section 8.4's gate is cost AND quality; this is the cost half.");
    println!("  - A layout comparison on CPU. boostr's TCF fused matmul is SCALAR");
    println!("    (quant/cpu/kernels/tcf/matmul.rs), while Q4_K and Q6_K run an AVX2 integer");
    println!("    dp4a-style path over Q8_K-quantized activations and Q8_0 runs a dequantize");
    println!("    row plus AVX2/FMA dot. A CPU fused-vs-fused row is mostly SIMD versus scalar.");
    println!("  - Kernel time on CUDA or WebGPU. Instructions there count host-side LAUNCH");
    println!("    work. Judge those rows by ns*, and only on a verified quiet machine.");
    println!("  - End-to-end model throughput, memory bandwidth, or load time.");
    println!("  - An allocator-free baseline. The counting global allocator adds two atomic");
    println!("    increments per allocation, to every row equally.");
    println!("  - Any encoding outside the three matched size classes.");
}

fn opt(value: Option<f64>) -> String {
    value.map_or_else(|| "".to_string(), |v| format!("{v:.6}"))
}

/// Compact magnitude, for a column that spans nanoseconds to gigainstructions.
fn si(value: Option<f64>) -> String {
    let Some(v) = value else {
        return "-".to_string();
    };
    let abs = v.abs();
    if abs >= 1e9 {
        format!("{:.2}G", v / 1e9)
    } else if abs >= 1e6 {
        format!("{:.2}M", v / 1e6)
    } else if abs >= 1e3 {
        format!("{:.2}k", v / 1e3)
    } else {
        format!("{v:.2}")
    }
}

/// Small values keep their digits: an instr/unit figure is often under 10.
fn fine(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |v| format!("{v:.3}"))
}
