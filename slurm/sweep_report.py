"""Reduce a fetched fixes sweep to one CSV and print the paired comparisons.

Every comparison is paired per seed inside a cell (scenario, d, n0, ratio):
the median over seeds of B / A - 1 on a metric, and how many seeds B wins.
Dispersion (`hole_max`, `hole_mean`) is lower-better.

Run: python slurm/sweep_report.py results/fixes1 [--csv .agents/notes/fixes1.csv]
"""

import argparse
import csv
import json
import pathlib

import numpy as np

# label, A (reference), B (the change)
PAIRS = [
    ("placement: sequential vs batch, cap 2%", "batch_c02", "head_c02"),
    ("placement: sequential vs batch, cap 1%", "batch_c01", "head_c01"),
    ("placement alone (no relaxation)", "batch_place", "head_place"),
    ("relaxation gain, sequential", "head_place", "head_c01"),
    ("relaxation gain, batch", "batch_place", "batch_c01"),
    ("cap 1% vs 2%", "head_c02", "head_c01"),
    ("cap 0.5% vs 2%", "head_c02", "head_c005"),
    ("cap 4% vs 2%", "head_c02", "head_c04"),
    ("init_pool 1 vs 64", "head_c01", "head_pool1"),
    ("init_pool 16 vs 64", "head_c01", "head_pool16"),
    ("init_pool 256 vs 64", "head_c01", "head_pool256"),
    ("ess vs random", "random", "head_c01"),
    ("p=0.5: one cap vs extra L1 cap", "l1cap_p05", "head_p05_c02"),
    ("p=0.25: one cap vs extra L1 cap", "l1cap_p025", "head_p025_c02"),
    ("p=0.5: cap 1% vs 2%", "head_p05_c02", "head_p05_c01"),
    ("p=0.5 vs p=1 (L1 dispersion)", "head_c01", "head_p05_c01"),
]
METRICS = ("hole_max", "hole_mean", "sep_min", "wrap_disc", "time_s")


def load(root: pathlib.Path) -> dict:
    """Map (cell, variant) to its per-seed rows."""
    runs = {}
    for path in sorted(root.glob("*/result.json")):
        if not (path.parent / "DONE").exists():
            continue
        data = json.loads(path.read_text())
        meta = data["meta"]
        cell = (meta["scenario"], meta["d"], meta["n0"], meta["ratio"])
        prefix = f"{meta['scenario']}_d{meta['d']}_n{meta['n0']}_r{meta['ratio']:g}_"
        runs[cell, path.parent.name.removeprefix(prefix)] = data["rows"]
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--csv", type=pathlib.Path)
    args = ap.parse_args()
    runs = load(args.root)
    print(f"{len(runs)} finished runs in {args.root}")
    if args.csv:
        keys = sorted({k for rows in runs.values() for row in rows for k in row})
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            out = csv.writer(handle)
            out.writerow(["scenario", "d", "n0", "ratio", "variant", *keys])
            for (cell, variant), rows in sorted(runs.items()):
                for row in rows:
                    out.writerow([*cell, variant, *(row.get(k, "") for k in keys)])
    cells = sorted({cell for cell, _ in runs})
    for label, a, b in PAIRS:
        print(f"\n== {label}  ({b} against {a}; median paired change, seeds won by {b})")
        print("scenario   d    n0 ratio | " + " | ".join(f"{m:>16}" for m in METRICS))
        for cell in cells:
            if (cell, a) not in runs or (cell, b) not in runs:
                continue
            ra = {r["seed"]: r for r in runs[cell, a]}
            rb = {r["seed"]: r for r in runs[cell, b]}
            seeds = sorted(ra.keys() & rb.keys())
            cols = []
            for metric in METRICS:
                va = np.array([ra[s].get(metric, np.nan) for s in seeds])
                vb = np.array([rb[s].get(metric, np.nan) for s in seeds])
                if np.isnan(va).any() or np.isnan(vb).any() or not np.all(va > 0):
                    cols.append(f"{'-':>16}")
                    continue
                better = vb > va if metric == "sep_min" else vb < va
                cols.append(f"{np.median(vb / va - 1) * 100:+7.1f}% {int(better.sum()):2d}/{len(seeds):<2d}  ")
            print(f"{cell[0]:9s} {cell[1]:3d} {cell[2]:5d} {cell[3]:5g} | " + " | ".join(cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
