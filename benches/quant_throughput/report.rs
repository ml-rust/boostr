//! The results table, and the caveats a fair reading of it needs.
//!
//! Rows are grouped by backend, size class, and shape, so one kernel's rows
//! across batch sizes sit together and two runs of the table diff line by
//! line.

/// One measured case.
pub struct Row {
    pub id: String,
    pub class: usize,
    pub encoding: &'static str,
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

/// The sort key that groups one kernel's rows.
fn key(row: &Row) -> (&'static str, usize, &'static str, usize, &'static str) {
    (row.backend, row.class, row.shape_label, row.m, row.op)
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
        "id,encoding,bpw,backend,op,shape,m,iters,unit,units,instructions_per_iter,\
instructions_per_unit,cycles_min,ns_min,alloc_count_per_iter,alloc_bytes_per_iter,error"
    );
    for row in rows {
        println!(
            "{},{},{:.4},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            row.id,
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
        "GGUF block quantized throughput — {} threads, perf {}, {} reps per phase",
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
        "{:<12} {:>5} {:<5} {:<8} {:<10} {:>10} {:>4} {:>6} {:>12} {:>12} {:>6} {:>12} {:>12} {:>6} {:>10}",
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
        "cycles*",
        "ns*",
        "alloc",
        "bytes",
    );
    println!("{header}");
    println!("{}", "-".repeat(header.chars().count()));

    let mut previous: Option<(&str, usize, &str)> = None;
    for row in rows {
        let group = (row.backend, row.class, row.shape_label);
        if previous.is_some_and(|p| p != group) {
            println!();
        }
        previous = Some(group);

        if let Some(error) = row.error.as_deref() {
            println!(
                "{:<12} {:>5.2} {:<5} {:<8} {:<10} {:>10} {:>4} {:>6}  {error}",
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
            "{:<12} {:>5.2} {:<5} {:<8} {:<10} {:>10} {:>4} {:>6} {:>12} {:>12} {:>6} {:>12} {:>12} {:>6} {:>10}",
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
            si(row.cycles),
            si(row.ns),
            si(row.alloc_count),
            si(row.alloc_bytes),
        );
    }
}

fn print_caveats(context: &Context) {
    println!();
    println!("How to read this");
    println!("  instr/iter  retired user-space instructions per iteration, with a zero-iteration");
    println!("              run of the same case subtracted, so setup and warm-up are removed.");
    println!("              Deterministic. This is the metric a kernel change is judged on.");
    println!("  instr/unit  the same figure per element (dequant) or per multiply-accumulate");
    println!("              (matmul), so shapes and batch sizes share one scale.");
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
    println!("  - Quality. A format is judged on cost AND quality; this is the cost half.");
    println!("  - A layout comparison on CPU. The Q8_0 row dequantizes each row to f32 and");
    println!("    dots it with AVX2/FMA, while Q4_K and Q6_K run an AVX2 INTEGER path over");
    println!("    Q8_K-quantized activations. Those rows compare float arithmetic against");
    println!("    integer arithmetic, not a layout against a layout.");
    println!("  - Kernel time on CUDA or WebGPU. Instructions there count host-side LAUNCH");
    println!("    work. Judge those rows by ns*, and only on a verified quiet machine.");
    println!("  - End-to-end model throughput, memory bandwidth, or load time.");
    println!("  - An allocator-free baseline. The counting global allocator adds two atomic");
    println!("    increments per allocation, to every row equally.");
    println!("  - Any format outside the three size classes.");
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
