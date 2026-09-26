#!/usr/bin/env python3
"""Oracle values for the VITS rational-quadratic spline unit test.

Calls HF `transformers.models.vits.modeling_vits._unconstrained_rational_quadratic_spline`
with num_bins=10 and tail_bound=5.0, forward and reverse. Spline parameters
come from closed-form f64 expressions cast to f32, so the Rust test rebuilds
them without a pasted table. The printed constants are the ones pasted into
the `tests` module of `src/model/audio/vits/spline/rq.rs`.
It also prints the per-element round-trip log_abs_det bound, derived from
the reverse log_abs_det slope at each forward output.

Requires: numpy, torch, transformers.

Usage: python3 tests/tools/vits_rq_spline_oracle/rq_spline_oracle.py
"""
import math
import numpy as np
import torch
from transformers.models.vits.modeling_vits import _unconstrained_rational_quadratic_spline as spline

NB = 10
TB = 5.0

def params(n):
    w = np.array([[math.sin(0.7 * i + 1.3 * j) for j in range(NB)] for i in range(n)])
    h = np.array([[1.5 * math.cos(0.5 * i - 0.9 * j) for j in range(NB)] for i in range(n)])
    d = np.array([[2.0 * math.sin(0.3 * i * j + 0.2) for j in range(NB - 1)] for i in range(n)])
    f = lambda a: torch.tensor(a.astype(np.float32))
    return f(w), f(h), f(d)

base = [-6.0, -5.0, -4.2, -2.5, -0.3, 0.0, 1.7, 3.3, 4.99, 5.0, 5.5]
n = len(base) + 2
w, h, d = params(n)

def knots(u, row, k):
    x = torch.softmax(u[row], -1)
    x = 1e-3 + (1 - 1e-3 * NB) * x
    c = torch.nn.functional.pad(torch.cumsum(x, -1), (1, 0))
    c = 2 * TB * c - TB
    return c[k].item()

# Row 11 sits on interior width knot 4, row 12 on interior height knot 6.
xs = base + [knots(w, 11, 4), knots(h, 12, 6)]
x = torch.tensor(xs, dtype=torch.float32)

def emit(name, t):
    print(f"const {name}: [f32; {t.numel()}] = [")
    print("    " + ", ".join(str(np.float32(v)) for v in t.tolist()) + ",")
    print("];")

emit("INPUTS", x)
for rev in (False, True):
    y, ld = spline(x.clone(), w.clone(), h.clone(), d.clone(), reverse=rev, tail_bound=TB)
    tag = "REV" if rev else "FWD"
    emit(f"{tag}_OUT", y)
    emit(f"{tag}_LOGDET", ld)


def reverse_logdet_slope(row, y):
    """Largest one-sided |d log_abs_det / dy| of the reverse map at y, in f64."""
    if abs(y) >= TB:
        return 0.0
    args = (w[row:row + 1].double(), h[row:row + 1].double(), d[row:row + 1].double())
    ld = lambda t: spline(torch.tensor([t], dtype=torch.float64), *args, reverse=True, tail_bound=TB)[1].item()
    e = 1e-7
    return max(abs(ld(min(y + e, TB)) - ld(y)), abs(ld(y) - ld(max(y - e, -TB)))) / e


# Round-trip log_abs_det bound per element: 1e-5 plus the reverse slope times one
# f32 ulp of the forward output, since a 1-ulp change in y moves the reverse
# log_abs_det by that product.
y_fwd, _ = spline(x.clone(), w.clone(), h.clone(), d.clone(), reverse=False, tail_bound=TB)
bound = [
    1e-5 + reverse_logdet_slope(i, v) * abs(float(np.spacing(np.float32(v))))
    for i, v in enumerate(y_fwd.tolist())
]
# Rounded up to whole 1e-6 steps so the printed bound never shrinks.
steps = [math.ceil(b / 1e-6 - 1e-9) for b in bound]
print(f"const ROUND_TRIP_LOGDET_TOL: [f32; {len(steps)}] = [")
print("    " + ", ".join(f"{k}e-6" for k in steps) + ",")
print("];")
