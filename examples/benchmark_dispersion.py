"""Dispersion benchmark for the toroidal (torann-backed) ESS.

Measures, for d in {2, 4, 8, 16, 32, 64} and both search modes, the
quality of the dispersion, the wall time and the epochs the early stop
actually used, over several independent seeds driving the initial
candidates (LHS init, smart-init pools, tie-breaking noise and the
torann hash functions).

Metrics are chosen per dimension, because none is valid everywhere:

  toroidal Clark-Evans   nearest-neighbour regularity in the geometry
                         that is actually optimized. Calibrated (random
                         = 1) and bounded by 2/Gamma(1+1/d). Reported
                         for every d but only *trusted* up to ~16: above
                         that, concentration of measure flattens the
                         distance distribution and every design looks
                         alike. The Euclidean box version is not used at
                         all -- it is biased upward (random scores 1.42
                         at d=64) and rewards piling points on the walls.

  projection discrepancy wrap-around L2 discrepancy averaged over 1-D
                         and 2-D projections, normalized so 1.0 = random.
                         Fixed scale regardless of ambient dimension, no
                         distance contrasts involved: this is the
                         high-dimensional criterion, and it also encodes
                         the DoE effect-sparsity principle (what matters
                         is that every factor and every pair be covered).

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

import ess  # noqa: E402
from ess.samplers import LHCSampler  # noqa: E402
from ess.utils import (  # noqa: E402
    expected_discrepancy,
    projection_discrepancy,
    wrap_around_discrepancy,
    expected_discrepancy,
)

OUT = os.path.join(os.path.dirname(__file__), "out")

DIMS = [2, 4, 8, 16, 32, 64]
MODES = ["k_nn", "radius"]
FORCES = ["gaussian", "softened_inverse", "linear", "cauchy"]

# n grows with d: a fixed 256 would leave 64D so sparse that the ideal
# packing spacing exceeds the domain and there is nothing to optimize.
N_BY_DIM = {2: 256, 4: 256, 8: 512, 16: 512, 32: 1024, 64: 1024}


def score(pts, dim):
    """The metric panel for one design (see the module docstring)."""
    n = len(pts)
    return {
        "wrap_disc": wrap_around_discrepancy(pts) / expected_discrepancy(len(pts), pts.shape[1]),
        "proj1": projection_discrepancy(pts, 1) / expected_discrepancy(n, 1),
        "proj2": (
            projection_discrepancy(pts, 2, max_projections=60)
            / expected_discrepancy(n, 2)
        ),
    }


def run_once(dim, mode, seed, metric, n=None, **params):
    """One ESS run from scratch; returns the measured record."""
    n = n or N_BY_DIM[dim]
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
    n = n or N_BY_DIM[dim]
    pts = LHCSampler(random_state=seed).sample(n, dim)
    return {"dim": dim, "mode": "LHS (no relaxation)", "seed": seed,
            "n": n, **score(pts, dim), "time_s": 0.0, "epochs": 0}


# `torus_ce` was dropped from the metric panel (Clark-Evans stopped being
# trustworthy above d~16). `wrap_disc` replaces it as the headline scalar --
# and it runs the other way: CE was higher-better, a normalised discrepancy
# is LOWER-better, with 1.0 = random. Every selection below is flipped to
# match; getting that wrong picks the worst design rather than the best.
COLS = ("wrap_disc", "proj1", "proj2", "epochs", "time_s")


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
    head = list(keys) + ["wrap disc", "proj1", "proj2", "epochs", "time[s]"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [str(r[k]) for k in keys]
        cells += [f"{r['wrap_disc']:.3f}", f"{r['proj1']:.3f}", f"{r['proj2']:.3f}",
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
        for dim in (2, 8, 32)
        for mode in MODES
        for metric in FORCES
        for seed in range(seeds)
    ]
    save("bench_force.json", rows)
    table(summarize(rows, ("metric", "mode", "dim")), ("metric", "mode", "dim"))
    best = min(summarize(rows, ("metric",)), key=lambda r: r["wrap_disc"])
    print(f"\n>>> best force law: {best['metric']} "
          f"(wrap disc {best['wrap_disc']:.3f}, lower is better)")
    return best["metric"]


def phase_tune(metric, seeds):
    """The two knobs that dominate quality (batch_size, patience) and
    the one that dominates high-dimensional behaviour (k)."""
    dims = (2, 8, 32)

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
    top = min(r["wrap_disc"] for r in agg)
    ok = [r for r in agg if r["wrap_disc"] <= top * 1.01]
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
    best_k = min(safe, key=lambda r: r["wrap_disc"])
    print(f"\n>>> k={best_k['k']} (proj1 {best_k['proj1']:.3f} < 1 = better than random)")
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
        ("wrap_disc", "wrap-around discrepancy\n(lower better, 1 = random)", True),
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
        if col == "wrap_disc":
            ax.axhline(1.0, color=slate, lw=0.8, ls="--")
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
    ap.add_argument("--phase", default="all",
                    choices=["force", "tune", "main", "plot", "all"])
    ap.add_argument("--seeds", type=int, default=10,
                    help="independent seeds per configuration (main phase)")
    ap.add_argument("--tune-seeds", type=int, default=3,
                    help="seeds per cell in the force/tune phases")
    ap.add_argument("--metric", default=None,
                    help="skip the force phase and use this law")
    args = ap.parse_args()

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
