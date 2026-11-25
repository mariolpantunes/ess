r"""Clearance and dispersion benchmark for toroidal (torann-backed) ESS.

Measures, for d in {2, 4, 8, 16, 32, 64} and both search modes, the
per-new-point nearest-neighbour clearance normalised by the fair-share radius
$R_1 = \text{radius\_for\_count}(d, N_{\text{total}}, 1)$, wall time, and
epochs used over independent seeds.

Designs are ranked on **normalised clearance** (`gap_p05`, `gap_min`, `gap_med`,
all higher-better) per the project evaluation spec, comparing relaxed ESS
against placement-only (`epochs=0, w=0`) and uniform random (scale reference
only).

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

from torann import ToroidalNN

import ess
from ess import geometry

OUT = os.path.join(os.path.dirname(__file__), "out")

DIMS = [2, 4, 8, 16, 32, 64]
TUNE_DIMS = [2, 8, 32]
MODES = ["k_nn", "radius"]
FORCES = ["gaussian", "softened_inverse", "linear", "cauchy"]

N_BY_DIM = {2: 256, 4: 256, 8: 512, 16: 512, 32: 1024, 64: 1024}


def score(pts, dim, anchors=None, p: float = 1.0):
    """Per-new-point nearest-neighbour clearance normalised by R1."""
    static = np.empty((0, dim)) if anchors is None else anchors
    n_total = static.shape[0] + pts.shape[0]
    r1 = geometry.radius_for_count(dim, n_total, target=1, p=p)
    idx = ToroidalNN(seed=0, p=p, brute_threshold=n_total + 1).fit(static, pts, k=1)
    _, dists = idx.query(k=1)
    gaps = np.asarray(dists, dtype=np.float64).ravel() / r1
    return {
        "gap_min": float(gaps.min()),
        "gap_p05": float(np.quantile(gaps, 0.05)),
        "gap_med": float(np.median(gaps)),
    }


def run_once(dim, mode, seed, metric, n=None, **params):
    """One incremental ESS extension run; returns the measured record."""
    n = n or N_BY_DIM.get(dim, 512)
    n_anchors = max(16, n // 4)
    rng = np.random.default_rng(seed)
    anchors = rng.uniform(0.0, 1.0, (n_anchors, dim))
    bounds = np.array([[0.0, 1.0]] * dim)
    stats: dict = {}
    t0 = time.perf_counter()
    pts = ess.esa(
        anchors, bounds, n=n, seed=seed, search_mode=mode,
        metric=metric, stats=stats, **params,
    )
    wall = time.perf_counter() - t0

    return {
        "dim": dim, "mode": mode, "seed": seed, "metric": metric, "n": n,
        **{k: v for k, v in params.items() if not isinstance(v, (list, dict))},
        **score(pts, dim, anchors=anchors),
        "time_s": wall,
        "epochs": stats["epochs_total"],
        "radius": stats["radius"],
    }


def baseline(dim, seed, n=None):
    """Placement-only (`epochs=0, w=0`) reference on the same anchors."""
    return {
        **run_once(dim, "auto", seed, "gaussian", n=n, epochs=0, w=0.0),
        "mode": "placement-only",
    }


COLS = ("gap_p05", "gap_min", "gap_med", "epochs", "time_s")


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
    head = list(keys) + ["gap_p05/R1", "gap_min/R1", "gap_med/R1",
                         "epochs", "time[s]"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [str(r[k]) for k in keys]
        cells += [f"{r['gap_p05']:.4f}", f"{r['gap_min']:.4f}",
                  f"{r['gap_med']:.4f}", f"{r['epochs']:.0f}",
                  f"{r['time_s']:.2f}"]
        print("| " + " | ".join(cells) + " |")


def save(name, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"[saved {path}: {len(rows)} runs]")


def phase_force(seeds):
    """Which force law maximises normalised clearance? Subset of dims, both modes."""
    rows = [
        run_once(dim, mode, seed, metric)
        for dim in TUNE_DIMS
        for mode in MODES
        for metric in FORCES
        for seed in range(seeds)
    ]
    save("bench_force.json", rows)
    table(summarize(rows, ("metric", "mode", "dim")), ("metric", "mode", "dim"))
    best = max(summarize(rows, ("metric",)), key=lambda r: r["gap_p05"])
    print(f"\n>>> best force law: {best['metric']} "
          f"(gap_p05/R1 {best['gap_p05']:.4f}, higher is better)")
    return best["metric"]


def phase_tune(metric, seeds):
    """Sweep batch_size, patience, and k on normalised clearance vs cost."""
    dims = TUNE_DIMS

    rows = [
        run_once(dim, "k_nn", seed, metric, batch_size=bs, patience=pat)
        for bs in (50, None)
        for pat in (5, 12, 25)
        for dim in dims
        for seed in range(seeds)
    ]
    save("bench_tune_stop.json", rows)
    agg = summarize(rows, ("batch_size", "patience"))
    table(agg, ("batch_size", "patience"))
    top = max(r["gap_p05"] for r in agg)
    ok = [r for r in agg if r["gap_p05"] >= top * 0.99]
    best = min(ok, key=lambda r: r["time_s"])
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
    top_k = max(r["gap_p05"] for r in agg_k)
    best_k = min((r for r in agg_k if r["gap_p05"] >= top_k * 0.99),
                 key=lambda r: r["time_s"])
    print(f"\n>>> k={best_k['k']} (gap_p05/R1 {best_k['gap_p05']:.4f}, "
          f"within 1% of the best at the lowest cost)")
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
        ("gap_p05", "5% gap / R1\n(higher better)", False),
        ("gap_min", "min gap / R1\n(higher better)", False),
        ("epochs", "epochs used (early stop)", False),
        ("time_s", "wall time per run [s]", True),
    )
    series = (("k_nn", blue), ("radius", teal), ("placement-only", amber))
    for ax, (col, title, logy) in zip(axes, panels):
        for mode, color in series:
            sel = sorted((r for r in agg if r["mode"] == mode), key=lambda r: r["dim"])
            if not sel or (col == "epochs" and mode == "placement-only"):
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
        ax.legend(fontsize=8)
    fig.suptitle("toroidal ESS: normalised clearance vs dimension", color=slate)
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
