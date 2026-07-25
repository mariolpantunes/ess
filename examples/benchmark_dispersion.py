"""Dispersion benchmark for the toroidal (torann-backed) ESS.

Measures, for d in {2, 4, 8, 16, 32, 64} and both search modes, the
quality of the dispersion (Clark-Evans index and the maximin gain over
a same-seed random-uniform baseline), the wall time and the epochs the
early stop actually used — every run repeated over several independent
seeds that drive the initial candidates (LHS init, smart-init pools and
the hash functions).

Three phases, each writing JSON to ``examples/out/``:

  force   pick the best-performing force law (subset of dims/seeds)
  tune    grid-search lr x decay, then patience, with the winning law
  main    the full dims x modes x seeds sweep with the tuned settings

Run from the repository root::

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
from ess import utils  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")

DIMS = [2, 4, 8, 16, 32, 64]
MODES = ["k_nn", "radius"]
FORCES = ["softened_inverse", "gaussian", "linear", "cauchy"]
N_POINTS = 256


def run_once(dim, mode, seed, metric, n=N_POINTS, **params):
    """One ESS run from scratch; returns the measured record."""
    bounds = np.array([[0.0, 1.0]] * dim)
    stats = {}
    t0 = time.perf_counter()
    pts = ess.esa(
        np.empty((0, dim)),
        bounds,
        n=n,
        seed=seed,
        search_mode=mode,
        metric=metric,
        stats=stats,
        **params,
    )
    wall = time.perf_counter() - t0

    rng = np.random.default_rng(seed)
    random_pts = rng.uniform(0.0, 1.0, (n, dim))
    maximin = utils.calculate_min_pairwise_distance(pts)
    maximin_rnd = utils.calculate_min_pairwise_distance(random_pts)

    return {
        "dim": dim,
        "mode": mode,
        "seed": seed,
        "metric": metric,
        **params,
        "ce": utils.calculate_clark_evans_index(pts, bounds),
        "maximin": maximin,
        "maximin_gain": maximin / max(maximin_rnd, 1e-12),
        "time_s": wall,
        "epochs": stats["epochs_total"],
        "epochs_per_batch": float(np.mean(stats["batch_epochs"])),
        "radius": stats["radius"],
    }


def summarize(rows, keys):
    """Group rows by `keys`, aggregate the measured columns (mean/std)."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for gk, rs in sorted(groups.items()):
        rec = dict(zip(keys, gk))
        for col in ("ce", "maximin_gain", "time_s", "epochs_per_batch"):
            vals = np.array([r[col] for r in rs], dtype=float)
            rec[col] = float(vals.mean())
            rec[col + "_std"] = float(vals.std())
        rec["runs"] = len(rs)
        out.append(rec)
    return out


def table(rows, keys, cols=("ce", "maximin_gain", "epochs_per_batch", "time_s")):
    head = list(keys) + [c for col in cols for c in (col, "±")]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [str(r[k]) for k in keys]
        for col in cols:
            cells += [f"{r[col]:.3f}", f"{r[col + '_std']:.3f}"]
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
    agg = summarize(rows, ("metric", "mode", "dim"))
    table(agg, ("metric", "mode", "dim"))

    # winner: best mean CE across all cells, tie-broken by maximin gain
    by_metric = summarize(rows, ("metric",))
    best = max(by_metric, key=lambda r: (round(r["ce"], 2), r["maximin_gain"]))
    print(f"\n>>> best force law: {best['metric']} "
          f"(CE {best['ce']:.3f}, maximin x{best['maximin_gain']:.2f})")
    return best["metric"]


def phase_tune(metric, seeds):
    """Grid lr x decay (patience fixed), then patience, on 3 dims."""
    dims = (4, 16, 64)
    rows = [
        run_once(dim, mode, seed, metric, lr=lr, decay=decay)
        for lr in (0.005, 0.01, 0.02)
        for decay in (0.90, 0.95, 0.99)
        for dim in dims
        for mode in MODES
        for seed in range(seeds)
    ]
    save("bench_tune_lr.json", rows)
    agg = summarize(rows, ("lr", "decay"))
    table(agg, ("lr", "decay"))
    # quality first (CE within 1% of the best), then fewest epochs
    top_ce = max(r["ce"] for r in agg)
    ok = [r for r in agg if r["ce"] >= top_ce * 0.99]
    best = min(ok, key=lambda r: r["epochs_per_batch"])
    lr, decay = best["lr"], best["decay"]
    print(f"\n>>> lr={lr}, decay={decay} "
          f"(CE {best['ce']:.3f}, {best['epochs_per_batch']:.0f} ep/batch)")

    rows_p = [
        run_once(dim, mode, seed, metric, lr=lr, decay=decay, patience=patience)
        for patience in (5, 10, 20)
        for dim in dims
        for mode in MODES
        for seed in range(seeds)
    ]
    save("bench_tune_patience.json", rows_p)
    agg_p = summarize(rows_p, ("patience",))
    table(agg_p, ("patience",))
    top_ce = max(r["ce"] for r in agg_p)
    ok = [r for r in agg_p if r["ce"] >= top_ce * 0.99]
    best_p = min(ok, key=lambda r: r["epochs_per_batch"])
    print(f"\n>>> patience={best_p['patience']} "
          f"(CE {best_p['ce']:.3f}, {best_p['epochs_per_batch']:.0f} ep/batch)")
    return {"lr": lr, "decay": decay, "patience": best_p["patience"]}


def phase_main(metric, params, seeds):
    """The headline benchmark: dims x modes x seeds with tuned settings."""
    rows = [
        run_once(dim, mode, seed, metric, **params)
        for dim in DIMS
        for mode in MODES
        for seed in range(seeds)
    ]
    save("bench_main.json", rows)
    agg = summarize(rows, ("dim", "mode"))
    table(agg, ("dim", "mode"))
    return rows


def phase_plot():
    """Three-panel summary figure from bench_main.json."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(OUT, "bench_main.json")) as fh:
        rows = json.load(fh)
    agg = summarize(rows, ("dim", "mode"))

    slate, blue, teal = "#64748b", "#3b82f6", "#14b8a6"
    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 10, "text.color": slate,
        "axes.edgecolor": slate, "axes.labelcolor": slate,
        "xtick.color": slate, "ytick.color": slate,
        "axes.titlecolor": slate, "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    panels = (
        ("ce", "Clark-Evans index (higher = more dispersed)"),
        ("epochs_per_batch", "epochs per batch (early stop)"),
        ("time_s", "wall time per run [s]"),
    )
    for ax, (col, title) in zip(axes, panels):
        for mode, color in (("k_nn", blue), ("radius", teal)):
            sel = sorted((r for r in agg if r["mode"] == mode),
                         key=lambda r: r["dim"])
            dims = [r["dim"] for r in sel]
            mean = np.array([r[col] for r in sel])
            std = np.array([r[col + "_std"] for r in sel])
            ax.plot(dims, mean, "o-", color=color, label=mode, ms=4)
            ax.fill_between(dims, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DIMS, [str(d) for d in DIMS])
        ax.set_xlabel("dimensions")
        ax.set_title(title, fontsize=10)
        if col == "ce":
            ax.axhline(1.0, color=slate, lw=0.8, ls="--")
            ax.annotate("random", (DIMS[-1], 1.0), textcoords="offset points",
                        xytext=(0, 4), ha="right", color=slate, fontsize=8)
        ax.legend()
    fig.suptitle(f"toroidal ESS: {N_POINTS} points from scratch, "
                 "10 seeds, tuned defaults", color=slate)
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
    metric = metric or "softened_inverse"

    if args.phase in ("tune", "all"):
        print(f"\n=== phase: tuning (metric={metric}) ===")
        params = phase_tune(metric, args.tune_seeds)

    if args.phase in ("main", "all"):
        print(f"\n=== phase: main benchmark (metric={metric}, {params}) ===")
        phase_main(metric, params, args.seeds)

    if args.phase in ("plot", "all"):
        phase_plot()
