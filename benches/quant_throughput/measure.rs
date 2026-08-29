//! Command-line handling, the `perf` driver, and the two-run subtraction.
//!
//! The parent process never runs a kernel. It spawns one child per phase, so a
//! backend that aborts, hangs, or exhausts device memory takes down one case
//! rather than the suite — the process isolation `fluxbench` would otherwise
//! have provided.

use std::process::{Command, Stdio};

use crate::cases::{self, Backend, Case};
use crate::payload;
use crate::report::{self, Row};
use crate::worker;

/// Work units a case aims to cover per measurement, before clamping.
const DEFAULT_BUDGET: u64 = 2_000_000_000;

/// Upper bound on iterations, so a cheap case does not spend minutes.
const DEFAULT_MAX_ITERS: u64 = 200;

/// Repetitions of each phase. Retired instructions are near-deterministic, so
/// two runs and a minimum are enough to shed a stray interrupt.
const DEFAULT_REPS: usize = 2;

struct Options {
    backends: Vec<Backend>,
    filter: Option<String>,
    iters: Option<u64>,
    budget: u64,
    max_iters: u64,
    reps: usize,
    use_perf: bool,
    csv: bool,
    list: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            backends: Vec::new(),
            filter: None,
            iters: None,
            budget: DEFAULT_BUDGET,
            max_iters: DEFAULT_MAX_ITERS,
            reps: DEFAULT_REPS,
            use_perf: true,
            csv: false,
            list: false,
        }
    }
}

const HELP: &str = "\
TCF vs GGUF quantized throughput, at matched bits per weight.

  --list                 print every case id and exit
  --backend <name>       restrict to cpu | cuda | wgpu (repeatable)
  --filter <substring>   restrict to case ids containing this
  --iters <n>            fixed iterations per case (default: derived from work)
  --budget <n>           work units per case when deriving iterations
  --max-iters <n>        cap on derived iterations
  --reps <n>             repetitions of each phase, minimum taken
  --no-perf              skip instruction counting, report the other columns
  --csv                  machine-readable output
  --help                 this text
";

/// Route to the worker path or the parent path.
pub fn dispatch(args: &[String]) -> Result<(), String> {
    if let Some(position) = args.iter().position(|a| a == "--worker") {
        let index: usize = value_at(args, position)?;
        let iters: u64 = match args.iter().position(|a| a == "--iters") {
            Some(p) => value_at(args, p)?,
            None => 0,
        };
        return run_worker(index, iters as usize);
    }
    let options = parse(args)?;
    if options.list {
        for (index, case) in select(&options).into_iter() {
            println!("{index}\t{}", case.id());
        }
        return Ok(());
    }
    run_parent(&options)
}

fn value_at<T: std::str::FromStr>(args: &[String], position: usize) -> Result<T, String> {
    args.get(position + 1)
        .ok_or_else(|| format!("{} needs a value", args[position]))?
        .parse()
        .map_err(|_| format!("{} got an unparseable value", args[position]))
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "--list" => options.list = true,
            "--no-perf" => options.use_perf = false,
            "--csv" => options.csv = true,
            "--backend" => {
                let name: String = value_at(args, index)?;
                options.backends.push(match name.as_str() {
                    "cpu" => Backend::Cpu,
                    "cuda" => Backend::Cuda,
                    "wgpu" => Backend::Wgpu,
                    other => return Err(format!("unknown backend {other}")),
                });
                index += 1;
            }
            "--filter" => {
                options.filter = Some(value_at(args, index)?);
                index += 1;
            }
            "--iters" => {
                options.iters = Some(value_at(args, index)?);
                index += 1;
            }
            "--budget" => {
                options.budget = value_at(args, index)?;
                index += 1;
            }
            "--max-iters" => {
                options.max_iters = value_at(args, index)?;
                index += 1;
            }
            "--reps" => {
                options.reps = value_at::<usize>(args, index)?.max(1);
                index += 1;
            }
            // `cargo bench` forwards its own flags; ignoring them keeps the
            // documented invocation working unchanged.
            "--bench" | "--nocapture" => {}
            other => return Err(format!("unknown argument {other}")),
        }
        index += 1;
    }
    Ok(options)
}

/// The cases this invocation will measure, paired with their matrix index.
fn select(options: &Options) -> Vec<(usize, Case)> {
    cases::enumerate()
        .into_iter()
        .enumerate()
        .filter(|(_, case)| options.backends.is_empty() || options.backends.contains(&case.backend))
        .filter(|(_, case)| match options.filter.as_deref() {
            Some(needle) => case.id().contains(needle),
            None => true,
        })
        .collect()
}

/// Child mode: one case, one phase, one line of output.
fn run_worker(index: usize, iters: usize) -> Result<(), String> {
    let all = cases::enumerate();
    let case = all
        .get(index)
        .ok_or_else(|| format!("case index {index} is outside the matrix"))?;
    let outcome = payload::packed(case.scheme(), case.n(), case.k())
        .and_then(|bytes| worker::run_on_backend(case, &bytes, iters));
    match outcome {
        Ok(sample) => println!(
            "QTP ok=1 alloc_count={} alloc_bytes={} min_ns={} min_cycles={}",
            sample.alloc_count, sample.alloc_bytes, sample.min_ns, sample.min_cycles,
        ),
        // A failed case is a reported row, not a failed suite: a WebGPU buffer
        // limit must not hide the CPU and CUDA numbers beside it.
        Err(message) => println!("QTP ok=0 err={}", message.replace('\n', " ")),
    }
    Ok(())
}

/// What one child invocation yielded.
#[derive(Default)]
struct Phase {
    instructions: Option<u64>,
    alloc_count: Option<u64>,
    alloc_bytes: Option<u64>,
    min_ns: Option<u64>,
    min_cycles: Option<u64>,
    error: Option<String>,
}

fn run_parent(options: &Options) -> Result<(), String> {
    let selected = select(options);
    if selected.is_empty() {
        return Err("no case matched the filters".into());
    }
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate this binary: {e}"))?;
    let exe = exe.to_string_lossy().into_owned();
    let perf = options.use_perf && perf_usable(&exe);

    let load_before = load_average();
    let mut rows = Vec::with_capacity(selected.len());
    for (index, case) in &selected {
        rows.push(measure_case(&exe, *index, case, options, perf));
    }
    let load_after = load_average();

    report::print(
        &rows,
        &report::Context {
            perf,
            load_before,
            load_after,
            threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(0),
            reps: options.reps,
            csv: options.csv,
        },
    );
    Ok(())
}

fn measure_case(exe: &str, index: usize, case: &Case, options: &Options, perf: bool) -> Row {
    let iters = iterations(case, options);
    let (units, unit) = case.work_units();

    let mut hi = Phase::default();
    let mut lo = Phase::default();
    for _ in 0..options.reps {
        hi = merge(hi, child(exe, index, iters, perf));
        if perf {
            lo = merge(lo, child(exe, index, 0, true));
        }
    }

    // Subtraction isolates the measured loop: both phases pay the same setup,
    // the same warm-up, and the same process start.
    let per_iter = match (hi.instructions, lo.instructions) {
        (Some(high), Some(low)) if high > low && iters > 0 => {
            Some((high - low) as f64 / iters as f64)
        }
        _ => None,
    };

    Row {
        id: case.id(),
        pair: case.pair,
        codec: case.codec.label(),
        encoding: case.encoding_name(),
        bpw: case.bpw(),
        backend: case.backend.label(),
        op: case.op.label(),
        shape: format!("{}x{}", case.n(), case.k()),
        shape_label: case.shape_label(),
        m: case.op.m(),
        iters,
        unit,
        units,
        instructions: per_iter,
        per_unit: per_iter.map(|i| i / units.max(1) as f64),
        cycles: hi.min_cycles.filter(|c| *c > 0).map(|c| c as f64),
        ns: hi.min_ns.map(|n| n as f64),
        alloc_count: hi.alloc_count.map(|c| c as f64 / iters as f64),
        alloc_bytes: hi.alloc_bytes.map(|b| b as f64 / iters as f64),
        error: hi.error,
    }
}

/// Keep the lower instruction count and the lower elapsed time of two runs of
/// the same phase. A minimum discards interference; it never invents speed.
fn merge(current: Phase, next: Phase) -> Phase {
    Phase {
        instructions: min_option(current.instructions, next.instructions),
        min_ns: min_option(current.min_ns, next.min_ns),
        min_cycles: min_option(current.min_cycles, next.min_cycles),
        alloc_count: next.alloc_count.or(current.alloc_count),
        alloc_bytes: next.alloc_bytes.or(current.alloc_bytes),
        error: next.error.or(current.error),
    }
}

fn min_option(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

/// One child run, optionally wrapped in `perf stat`.
fn child(exe: &str, index: usize, iters: u64, perf: bool) -> Phase {
    let mut command = if perf {
        let mut c = Command::new("perf");
        c.args(["stat", "-x,", "-e", "instructions:u", "--", exe]);
        c
    } else {
        Command::new(exe)
    };
    command
        .arg("--worker")
        .arg(index.to_string())
        .arg("--iters")
        .arg(iters.to_string())
        .stdin(Stdio::null());

    let output = match command.output() {
        Ok(output) => output,
        Err(e) => {
            return Phase {
                error: Some(format!("spawn failed: {e}")),
                ..Phase::default()
            };
        }
    };
    let mut phase = parse_worker_line(&String::from_utf8_lossy(&output.stdout));
    if perf {
        phase.instructions = parse_perf(&String::from_utf8_lossy(&output.stderr));
    }
    if !output.status.success() && phase.error.is_none() {
        phase.error = Some(format!("child exited {}", output.status));
    }
    phase
}

/// Read the worker's single `QTP` line.
fn parse_worker_line(stdout: &str) -> Phase {
    let mut phase = Phase::default();
    let Some(line) = stdout.lines().find(|l| l.starts_with("QTP ")) else {
        phase.error = Some("worker produced no result line".into());
        return phase;
    };
    if let Some(rest) = line.split_once(" err=") {
        phase.error = Some(rest.1.trim().to_string());
        return phase;
    }
    for field in line.split_whitespace() {
        let Some((key, value)) = field.split_once('=') else {
            continue;
        };
        let parsed = value.parse::<u64>().ok();
        match key {
            "alloc_count" => phase.alloc_count = parsed,
            "alloc_bytes" => phase.alloc_bytes = parsed,
            "min_ns" => phase.min_ns = parsed,
            "min_cycles" => phase.min_cycles = parsed,
            _ => {}
        }
    }
    phase
}

/// Read `perf stat -x,` output. A counter that could not be read appears as
/// `<not counted>` and yields `None` rather than a zero that would look fast.
fn parse_perf(stderr: &str) -> Option<u64> {
    stderr.lines().find_map(|line| {
        let fields: Vec<&str> = line.split(',').collect();
        let event = fields.get(2)?;
        if !event.starts_with("instructions") {
            return None;
        }
        fields.first()?.trim().replace('.', "").parse::<u64>().ok()
    })
}

/// Whether `perf` exists AND can actually count this process. A machine with
/// `perf_event_paranoid` locked down reports the tool as present and every
/// counter as unreadable, which must be detected once rather than per case.
fn perf_usable(exe: &str) -> bool {
    let probe = Command::new("perf")
        .args([
            "stat",
            "-x,",
            "-e",
            "instructions:u",
            "--",
            exe,
            "--worker",
            "0",
            "--iters",
            "0",
        ])
        .stdin(Stdio::null())
        .output();
    match probe {
        Ok(output) => parse_perf(&String::from_utf8_lossy(&output.stderr)).is_some(),
        Err(_) => false,
    }
}

fn iterations(case: &Case, options: &Options) -> u64 {
    if let Some(fixed) = options.iters {
        return fixed.max(1);
    }
    let (units, _) = case.work_units();
    (options.budget / units.max(1)).clamp(1, options.max_iters.max(1))
}

/// The 1-minute load average, or `None` where `/proc/loadavg` is absent.
fn load_average() -> Option<f64> {
    std::fs::read_to_string("/proc/loadavg")
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}
