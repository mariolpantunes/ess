"""Dispersion benchmark for the toroidal (torann-backed) ESS.

Measures, for d in {2, 4, 8, 16, 32, 64} and both search modes, the
quality of the dispersion, the wall time and the epochs the early stop
actually used, over several independent seeds driving the initial
candidates (LHS init, smart-init pools, tie-breaking noise and the
torann hash functions).

The headline metric is **toroidal separation** -- the shortest distance
from any point of the design to any other, wrap included. It is defined once,
in ``torann.metrics``, and shared with torann, because two copies of a metric
is exactly how this file ended up asking for a key that no longer existed.
Higher is better, and it keeps discriminating at every dimension here:
against an LHS batch it reads 13.7x at d=2 and still 1.3x at d=64, where the
alternatives have collapsed to a fraction of a percent.

``torann.metrics`` records the four metrics that were measured and rejected,
with numbers. The short version: toroidal Clark-Evans and coverage radius
both stop resolving above d~16 (coverage radius reads 13.183 against 13.180
at d=64, with the better design fractionally behind); wrap-around discrepancy
manages a 0.9% gap at d=64.

  projection discrepancy wrap-around L2 discrepancy averaged over 1-D and
                         2-D projections, normalized so 1.0 = random. Held
                         here as a *reported* column and deliberately never
                         selected on. It has a fixed scale at every
                         dimension, which is why it looks like the
                         high-dimensional criterion, but it scores marginal
                         coverage -- which is what a Latin hypercube is
                         constructed to optimize, so an unrelaxed LHS wins
                         it at every d by construction. It is a real
                         criterion for the DoE effect-sparsity question;
                         it is not a ranking of how well the voids got
                         filled, and selecting on it inverts the conclusion.

  wrap_disc              full-dimensional wrap-around discrepancy, likewise
                         reported and not selected on: a discrepancy falls
                         monotonically with more relaxation, so any tuning
                         phase that maximises it just asks for more epochs.

**Do not read the epoch counts as a comparison between two builds.** ESS
stops on a plateau detector -- a 1% relative-improvement threshold with
``patience`` -- so a single near-threshold comparison flipping resets the
streak and costs a full ``patience`` worth of epochs. An ulp-level difference
in a distance is enough to move the total by tens: measured across a torann
kernel change, the seed-to-seed spread within one build at d=8 was 95-185
epochs, wider than any difference between builds. For attributing a speed
change, use a fixed-epoch harness (torann ships ``bench_ess_loop.py``); for
reporting epochs here, read the spread over seeds and not one number.

The number of points scales with the dimension -- 256 points in 64D is
far too sparse for any packing question to be meaningful.

Three phases plus a plot, each writing JSON to ``examples/out/``::

    python examples/benchmark_dispersion.py --phase all
    python examples/benchmark_dispersion.py --phase main --seeds 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from torann.metrics import toroidal_separation

import ess
from ess.samplers import LHCSampler
from ess.utils import (
    expected_discrepancy,
    projection_discrepancy,
    wrap_around_discrepancy,
)

OUT = os.path.join(os.path.dirname(__file__), "out")

DIMS = [2, 4, 8, 16, 32, 64]
# The force and tuning phases sweep a cheaper subset; both follow
# `--dims` so the whole run can be shrunk to a smoke test.
TUNE_DIMS = [2, 8, 32]
MODES = ["k_nn", "radius"]
FORCES = ["gaussian", "softened_inverse", "linear", "cauchy"]

# n grows with d: a fixed 256 would leave 64D so sparse that the ideal
# packing spacing exceeds the domain and there is nothing to optimize.
N_BY_DIM = {2: 256, 4: 256, 8: 512, 16: 512, 32: 1024, 64: 1024}


def score(pts, dim, anchors=None):
    """The metric panel for one design (see the module docstring)."""
    n = len(pts)
    return {
        "separation": toroidal_separation(pts, anchors),
        "wrap_disc": wrap_around_discrepancy(pts) / expected_discrepancy(len(pts), pts.shape[1]),
        "proj1": projection_discrepancy(pts, 1) / expected_discrepancy(n, 1),
        "proj2": (
            projection_discrepancy(pts, 2, max_projections=60)
            / expected_discrepancy(n, 2)
        ),
    }


def run_once(dim, mode, seed, metric, n=None, **params):
    """One ESS run from scratch; returns the measured record."""
    n = n or N_BY_DIM.get(dim, 512)
    bounds = np.array([[0.0, 1.0]] * dim)
    stats = {}
    t0 = time.perf_counter()
    pts = ess.esa(
        np.empty((0, dim)), bounds, n=n, seed=seed, search_mode=mode,
        metric=metric, stats=stats, **params,
    )
    wall = time.perf_counter() - t0

    return {
        "dim": dim, "mode": mode, "seed": seed, "metric": metric, "n": n,
        **{k: v for k, v in params.items() if not isinstance(v, (list, dict))},
        **score(pts, dim),
        "time_s": wall,
        "epochs": stats["epochs_total"],
        "radius": stats["radius"],
    }


def baseline(dim, seed, n=None):
    """LHS initialization, i.e. the quality of not relaxing at all."""
    n = n or N_BY_DIM.get(dim, 512)
    pts = LHCSampler(random_state=seed).sample(n, dim)
    return {"dim": dim, "mode": "LHS (no relaxation)", "seed": seed,
            "n": n, **score(pts, dim), "time_s": 0.0, "epochs": 0}


# The headline is `separation` -- the shared metric, defined once in
# `torann.metrics` so this project and torann cannot drift apart again. It is
# HIGHER-better, like the `torus_ce` it replaces, so the selections below read
# as maxima. `wrap_disc` and the projection discrepancies stay as reported
# columns and are deliberately NOT selected on: a discrepancy falls
# monotonically with more relaxation, and projection discrepancy scores
# marginal coverage, which is what an LHS is built to optimize. See
# `torann.metrics` for the four metrics that were measured and rejected.
COLS = ("separation", "wrap_disc", "proj1", "proj2", "epochs", "time_s")


def summarize(rows, keys):
    """Group rows by `keys`, aggregate the measured columns (mean/std)."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for gk, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rec = dict(zip(keys, gk))
        for col in COLS:
            vals = np.array([r[col] for r in rs], dtype=float)
            rec[col] = float(vals.mean())
            rec[col + "_std"] = float(vals.std())
        rec["runs"] = len(rs)
        out.append(rec)
    return out


def table(rows, keys):
    head = list(keys) + ["separation", "wrap disc", "proj1", "proj2",
                         "epochs", "time[s]"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [str(r[k]) for k in keys]
        cells += [f"{r['separation']:.4f}", f"{r['wrap_disc']:.3f}",
                  f"{r['proj1']:.3f}", f"{r['proj2']:.3f}",
                  f"{r['epochs']:.0f}", f"{r['time_s']:.2f}"]
        print("| " + " | ".join(cells) + " |")


def save(name, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"[saved {path}: {len(rows)} runs]")


def phase_force(seeds):
    """Which force law disperses best? Subset of dims, both modes."""
    rows = [
        run_once(dim, mode, seed, metric)
        for dim in TUNE_DIMS
        for mode in MODES
        for metric in FORCES
        for seed in range(seeds)
    ]
    save("bench_force.json", rows)
    table(summarize(rows, ("metric", "mode", "dim")), ("metric", "mode", "dim"))
    best = max(summarize(rows, ("metric",)), key=lambda r: r["separation"])
    print(f"\n>>> best force law: {best['metric']} "
          f"(separation {best['separation']:.4f}, higher is better)")
    return best["metric"]


def phase_tune(metric, seeds):
    """The two knobs that dominate quality (batch_size, patience) and
    the one that dominates high-dimensional behaviour (k)."""
    dims = TUNE_DIMS

    rows = [
        run_once(dim, "k_nn", seed, metric, batch_size=bs, patience=pat)
        for bs in (50, None)
        for pat in (5, 25, 50)
        for dim in dims
        for seed in range(seeds)
    ]
    save("bench_tune_stop.json", rows)
    agg = summarize(rows, ("batch_size", "patience"))
    table(agg, ("batch_size", "patience"))
    top = max(r["separation"] for r in agg)
    ok = [r for r in agg if r["separation"] >= top * 0.99]
    best = min(ok, key=lambda r: r["epochs"])
    print(f"\n>>> batch_size={best['batch_size']}, patience={best['patience']}")

    rows_k = [
        run_once(dim, "k_nn", seed, metric, batch_size=best["batch_size"],
                 patience=best["patience"], k=k)
        for k in (5, 9, 17, 33)
        for dim in dims
        for seed in range(seeds)
    ]
    save("bench_tune_k.json", rows_k)
    agg_k = summarize(rows_k, ("k",))
    table(agg_k, ("k",))
    # keep every projection better than random, then maximise regularity
    safe = [r for r in agg_k if r["proj1"] < 1.0] or agg_k
    # Same guard `patience` gets, and for the same reason: quality saturates
    # in k while cost does not, so an unguarded `max` just picks the largest
    # k on the grid and calls it tuning. Cheapest k within 1% of the best.
    top_k = max(r["separation"] for r in safe)
    best_k = min((r for r in safe if r["separation"] >= top_k * 0.99),
                 key=lambda r: r["time_s"])
    print(f"\n>>> k={best_k['k']} (separation {best_k['separation']:.4f}, "
          f"within 1% of the best at the lowest cost; "
          f"proj1 {best_k['proj1']:.3f} < 1 = better than random)")
    return {"batch_size": best["batch_size"], "patience": best["patience"],
            "k": best_k["k"]}


def phase_main(metric, params, seeds):
    """The headline benchmark: dims x modes x seeds with tuned settings."""
    rows = [
        run_once(dim, mode, seed, metric, **params)
        for dim in DIMS
        for mode in MODES
        for seed in range(seeds)
    ] + [baseline(dim, seed) for dim in DIMS for seed in range(seeds)]
    save("bench_main.json", rows)
    table(summarize(rows, ("dim", "mode")), ("dim", "mode"))
    return rows


def phase_plot():
    """Four-panel summary figure from bench_main.json."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(OUT, "bench_main.json")) as fh:
        rows = json.load(fh)
    agg = summarize(rows, ("dim", "mode"))

    slate, blue, teal, amber = "#64748b", "#3b82f6", "#14b8a6", "#f59e0b"
    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 9, "text.color": slate,
        "axes.edgecolor": slate, "axes.labelcolor": slate,
        "xtick.color": slate, "ytick.color": slate,
        "axes.titlecolor": slate, "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.5))
    panels = (
        ("separation", "toroidal separation\n(higher better)", False),
        ("proj1", "1-D projection discrepancy\n(lower better, 1 = random)", True),
        ("epochs", "epochs used (early stop)", False),
        ("time_s", "wall time per run [s]", True),
    )
    series = (("k_nn", blue), ("radius", teal), ("LHS (no relaxation)", amber))
    for ax, (col, title, logy) in zip(axes, panels):
        for mode, color in series:
            sel = sorted((r for r in agg if r["mode"] == mode), key=lambda r: r["dim"])
            if not sel or (col in ("epochs", "time_s") and mode.startswith("LHS")):
                continue
            dims = [r["dim"] for r in sel]
            mean = np.array([r[col] for r in sel])
            std = np.array([r[col + "_std"] for r in sel])
            ax.plot(dims, mean, "o-", color=color, label=mode, ms=4)
            ax.fill_between(dims, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DIMS, [str(d) for d in DIMS])
        ax.set_xlabel("dimensions")
        ax.set_title(title, fontsize=9)
        if logy:
            ax.set_yscale("log")

        if col == "proj1":
            ax.axhline(1.0, color=slate, lw=0.8, ls="--")
        ax.legend(fontsize=8)
    fig.suptitle("toroidal ESS: n scaled with d (256 to 1024), 10 seeds, tuned defaults",
                 color=slate)
    fig.tight_layout()
    path = os.path.join(OUT, "bench_dispersion.png")
    fig.savefig(path)
    print(f"[saved {path}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="override the dimension sweep (default: 2..64); "
                         "a short list makes this runnable as a smoke test")
    ap.add_argument("--phase", default="all",
                    choices=["force", "tune", "main", "plot", "all"])
    ap.add_argument("--seeds", type=int, default=10,
                    help="independent seeds per configuration (main phase)")
    ap.add_argument("--tune-seeds", type=int, default=3,
                    help="seeds per cell in the force/tune phases")
    ap.add_argument("--metric", default=None,
                    help="skip the force phase and use this law")
    args = ap.parse_args()
    if args.dims:
        DIMS[:] = args.dims
        TUNE_DIMS[:] = args.dims

    metric = args.metric
    params = {}
    if args.phase in ("force", "all") and metric is None:
        print("\n=== phase: force selection ===")
        metric = phase_force(args.tune_seeds)
    metric = metric or "gaussian"

    if args.phase in ("tune", "all"):
        print(f"\n=== phase: tuning (metric={metric}) ===")
        params = phase_tune(metric, args.tune_seeds)

    if args.phase in ("main", "all"):
        print(f"\n=== phase: main benchmark (metric={metric}, {params}) ===")
        phase_main(metric, params, args.seeds)

    if args.phase in ("plot", "all"):
        phase_plot()
