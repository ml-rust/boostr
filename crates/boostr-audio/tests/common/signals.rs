//! Deterministic test signals for the narration glitch oracle, built bit for
//! bit as `tests/tools/narration_oracle/signals.mjs` builds them.
//!
//! No transcendental function runs: a tone comes from the recurrence
//! `y[n] = c*y[n-1] - y[n-2]` with `c = 2cos(w)` and `y[1] = sin(w)` written
//! as literals, noise from an integer xorshift32 generator, envelopes from
//! linear ramps. Each op adds into the buffer in `f64` and stores `f32`.

/// `2cos(w)` and `sin(w)` for a tone of the named frequency at 48 kHz, or at
/// 44.1 kHz for the `_44` names.
const TONES: [(&str, f64, f64); 13] = [
    ("60", 1.9999383152895793, 0.007853900888711332),
    ("120", 1.9997532649633212, 0.015707317311820672),
    ("125", 1.9997322758191236, 0.01636173162648678),
    ("130", 1.9997104300531479, 0.01701613893230685),
    ("135", 1.9996877276747522, 0.017670538948953582),
    ("150", 1.9996144809641296, 0.0196336924606283),
    ("180", 1.9994448604360011, 0.023559764833610154),
    ("200", 1.9993146499511145, 0.026176948307873153),
    ("220", 1.9991707363874793, 0.02879395236826347),
    ("260", 1.9988418040633458, 0.034027350502167444),
    ("300", 1.9984580724814458, 0.03925981575906861),
    ("70_44", 1.999900533911886, 0.009973144676492593),
    ("200_44", 1.9991880801438362, 0.028491315390844964),
];

/// One step of building a signal.
#[derive(Debug, Clone, Copy)]
pub enum Op {
    /// A tone of `len` samples from `start`: amplitude `amp`, rising
    /// linearly over `attack` samples and falling over `release`.
    Tone {
        hz: &'static str,
        start: usize,
        len: usize,
        amp: f64,
        attack: usize,
        release: usize,
    },
    /// Uniform noise in `[-amp, amp)` from xorshift32 seeded with `seed`.
    Noise {
        start: usize,
        len: usize,
        amp: f64,
        seed: u32,
    },
    /// One sample of `value` added at `at`.
    Spike { at: usize, value: f64 },
}

fn tone(hz: &'static str, start: usize, len: usize, amp: f64, attack: usize, release: usize) -> Op {
    Op::Tone {
        hz,
        start,
        len,
        amp,
        attack,
        release,
    }
}

fn noise(start: usize, len: usize, amp: f64, seed: u32) -> Op {
    Op::Noise {
        start,
        len,
        amp,
        seed,
    }
}

fn spike(at: usize, value: f64) -> Op {
    Op::Spike { at, value }
}

/// Adds each op into `n` zero samples, in order.
pub fn render(n: usize, ops: &[Op]) -> Vec<f32> {
    let mut x = vec![0.0f32; n];
    for op in ops {
        match *op {
            Op::Tone {
                hz,
                start,
                len,
                amp,
                attack,
                release,
            } => {
                let &(_, c, s1) = TONES
                    .iter()
                    .find(|t| t.0 == hz)
                    .unwrap_or_else(|| panic!("signals: no tone {hz}"));
                let (mut a, mut b) = (0.0f64, 0.0f64);
                for i in 0..len {
                    let y = match i {
                        0 => 0.0,
                        1 => s1,
                        _ => c * b - a,
                    };
                    a = b;
                    b = y;
                    let mut g = i as f64 / attack as f64;
                    if g > 1.0 {
                        g = 1.0;
                    }
                    let r = (len - 1 - i) as f64 / release as f64;
                    if r < g {
                        g = r;
                    }
                    let s = &mut x[start + i];
                    *s = (f64::from(*s) + amp * g * y) as f32;
                }
            }
            Op::Noise {
                start,
                len,
                amp,
                seed,
            } => {
                let mut s = seed;
                for i in 0..len {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    let v = &mut x[start + i];
                    *v = (f64::from(*v) + amp * (f64::from(s) / 2147483648.0 - 1.0)) as f32;
                }
            }
            Op::Spike { at, value } => x[at] = (f64::from(x[at]) + value) as f32,
        }
    }
    x
}

/// FNV-1a (32-bit) over each sample's `f32` bit pattern, low byte first, as
/// 8 hex digits.
pub fn fnv1a(x: &[f32]) -> String {
    let mut h: u32 = 0x811c9dc5;
    for s in x {
        for byte in s.to_bits().to_le_bytes() {
            h ^= u32::from(byte);
            h = h.wrapping_mul(0x01000193);
        }
    }
    format!("{h:08x}")
}

/// The voice median a heat reading is taken against.
#[derive(Debug, Clone, Copy)]
pub enum VoiceMedian {
    /// No reference pitch.
    Absent,
    /// A fixed pitch in Hz.
    Hz(f64),
    /// The case's own median pitch.
    Own,
}

/// The checks read on a case.
#[derive(Debug, Clone)]
pub enum Run {
    /// Tidy, the click check and the click repair.
    Edges,
    /// The pitch track, its median, and heat against each voice median.
    Pitch(Vec<VoiceMedian>),
}

/// A named signal and the checks read on it.
#[derive(Debug, Clone)]
pub struct Case {
    pub name: &'static str,
    pub rate: u32,
    pub n: usize,
    pub run: Run,
    pub ops: Vec<Op>,
}

fn edges(name: &'static str, rate: u32, n: usize, ops: Vec<Op>) -> Case {
    Case {
        name,
        rate,
        n,
        run: Run::Edges,
        ops,
    }
}

fn pitch(name: &'static str, n: usize, medians: Vec<VoiceMedian>, ops: Vec<Op>) -> Case {
    Case {
        name,
        rate: 48000,
        n,
        run: Run::Pitch(medians),
        ops,
    }
}

/// Every case, in the oracle's order.
pub fn cases() -> Vec<Case> {
    use VoiceMedian::{Absent, Hz, Own};
    // Glitch cases: 1 s at 48 kHz, a noise floor and one sustained tone.
    let floor = noise(0, 48000, 0.00001, 1);
    let thump_front = tone("60", 960, 1440, 0.3, 240, 240);
    let thump_back = tone("60", 40800, 1920, 0.3, 240, 240);
    let speech = tone("200", 9600, 33600, 0.4, 2400, 4800);
    // Click cases: 2 s at 48 kHz, speech, a pause from 0.8 to 1.2 s, speech.
    let paused = |extra: &[Op]| {
        let mut ops = vec![
            noise(0, 96000, 0.00002, 7),
            tone("200", 0, 38400, 0.3, 480, 2400),
            tone("200", 57600, 38400, 0.3, 2400, 480),
        ];
        ops.extend_from_slice(extra);
        ops
    };
    let steps = |hz: [&'static str; 4]| {
        hz.iter()
            .enumerate()
            .map(|(k, h)| tone(h, k * 24000, 24000, 0.3, 480, 480))
            .collect::<Vec<_>>()
    };
    vec![
        edges(
            "speech ending in silence",
            48000,
            48000,
            vec![floor, tone("200", 4800, 38400, 0.4, 2400, 4800)],
        ),
        edges(
            "thump before the speech",
            48000,
            48000,
            vec![floor, thump_front, speech],
        ),
        edges(
            "thump after the speech",
            48000,
            48000,
            vec![floor, tone("200", 2400, 33600, 0.4, 2400, 4800), thump_back],
        ),
        edges(
            "thumps at both edges",
            48000,
            48000,
            vec![
                floor,
                thump_front,
                tone("200", 9600, 28800, 0.4, 2400, 4800),
                thump_back,
            ],
        ),
        edges(
            "speech cut off at the end",
            48000,
            48000,
            vec![floor, tone("200", 4800, 43200, 0.4, 2400, 1)],
        ),
        edges(
            "speech decaying at the end",
            48000,
            48000,
            vec![floor, tone("200", 4800, 43200, 0.4, 2400, 960)],
        ),
        edges(
            "dark burst too close to the speech",
            48000,
            48000,
            vec![
                floor,
                thump_front,
                tone("200", 3360, 38400, 0.4, 2400, 4800),
            ],
        ),
        edges(
            "bright burst before the speech",
            48000,
            48000,
            vec![floor, noise(960, 1440, 0.3, 99), speech],
        ),
        edges(
            "dark sound too long for a thump",
            48000,
            48000,
            vec![floor, tone("60", 960, 3840, 0.3, 240, 240), speech],
        ),
        edges(
            "buffer shorter than a frame",
            48000,
            100,
            vec![noise(0, 100, 0.1, 5)],
        ),
        edges("empty buffer", 48000, 0, vec![]),
        edges(
            "44.1 kHz thump before the speech, click after it",
            44100,
            44100,
            vec![
                noise(0, 44100, 0.00001, 3),
                tone("70_44", 882, 1323, 0.3, 220, 220),
                tone("200_44", 8820, 30870, 0.4, 2205, 4410),
                spike(42000, 0.2),
            ],
        ),
        edges(
            "one click in a pause",
            48000,
            96000,
            paused(&[spike(48010, 0.2)]),
        ),
        edges(
            "clicks in a pause",
            48000,
            96000,
            paused(&[spike(42010, 0.15), spike(46810, -0.2), spike(51650, 0.25)]),
        ),
        edges(
            "clicks under 0.05 s apart",
            48000,
            96000,
            paused(&[spike(46000, 0.2), spike(47300, 0.2)]),
        ),
        edges(
            "click inside speech",
            48000,
            96000,
            paused(&[spike(20010, 0.3)]),
        ),
        edges(
            "wide burst in a pause",
            48000,
            96000,
            paused(&[noise(48000, 480, 0.2, 11)]),
        ),
        edges(
            "click removed at 2 ms",
            48000,
            96000,
            paused(&[spike(47949, 0.2)]),
        ),
        edges(
            "click removed at 4 ms",
            48000,
            96000,
            paused(&[spike(47900, 0.2)]),
        ),
        edges(
            "click removed at 8 ms",
            48000,
            96000,
            paused(&[spike(48250, 0.2)]),
        ),
        edges(
            "click removed at 12 ms",
            48000,
            96000,
            paused(&[spike(48250, 0.2), spike(49250, 0.2)]),
        ),
        edges(
            "click left after every repair",
            48000,
            96000,
            paused(&[spike(48250, 0.2), spike(49250, 0.2), spike(50150, 0.2)]),
        ),
        pitch(
            "steady voice",
            96000,
            vec![Absent, Hz(0.0), Hz(150.0), Own],
            vec![tone("150", 0, 96000, 0.3, 480, 480)],
        ),
        pitch(
            "heated climb",
            96000,
            vec![Hz(120.0), Own],
            steps(["180", "220", "260", "300"]),
        ),
        pitch(
            "calm read",
            96000,
            vec![Hz(125.0), Own],
            steps(["120", "125", "130", "135"]),
        ),
        pitch(
            "part too short to judge",
            24000,
            vec![Hz(150.0)],
            vec![tone("150", 0, 24000, 0.3, 480, 480)],
        ),
        pitch(
            "voiceless part",
            96000,
            vec![Hz(150.0), Own],
            vec![noise(0, 96000, 0.005, 13)],
        ),
    ]
}
