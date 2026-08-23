"""Microbenchmark the torann index in isolation from the ESS loop.

Phase 0.2 of ``plans/plan_optimize.md``. Measuring the index *through* a
simulation confounds three different things — query cost, re-index cost,
and how many epochs convergence needs — so this harness drives
`torann.ToroidalNN` directly and reports each separately, together with
the recall the LSH parameters actually deliver.

Recall is measured against exact brute force on a subsample of queries,
because it is the quantity that decides whether cheaper LSH settings are
usable at all: if re-exploration quality tolerates reduced recall, a
large speed envelope opens; if not, the LSH knobs cannot help.

The ESS access pattern is reproduced faithfully: points barely move
between epochs (the step cap is a small fraction of the local spacing),
so `update` is called with coordinates perturbed by that much rather
than with fresh random ones.

Run from the repository root::

    python examples/profile_torann.py --sizes 10000 50000 --dims 32
    python examples/profile_torann.py --grid          # (B, K, L, probes)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from torann import ToroidalNN
from torann.brute import exact_knn

OUT = os.path.join(os.path.dirname(__file__), "out")

# Fraction of a spacing a point travels per epoch (mirrors ess.STEP_CAP).
STEP_FRACTION = 0.02


def recall_at_k(index, arena, queries, k, n_probe=256, seed=0):
    """Mean fraction of the true k nearest neighbours the index returns.

    Args:
        index (ToroidalNN): Fitted index.
        arena (np.ndarray): All indexed points, for the exact reference.
        queries (np.ndarray): Candidate-tier coordinates.
        k (int): Neighbours per query.
        n_probe (int): Queries sampled for the exact comparison, which is
            $O(n)$ per query and therefore the expensive part.
        seed (int): Subsample seed.

    Returns:
        float: Recall in [0, 1].
    """
    rng = np.random.default_rng(seed)
    take = min(n_probe, queries.shape[0])
    rows = rng.choice(queries.shape[0], take, replace=False)
    q = np.ascontiguousarray(queries[rows])

    approx, _ = index.query(k=k, queries=q)
    exact, _ = exact_knn(arena, q, k + 1)  # +1: the query may be indexed

    hits = 0
    for i in range(take):
        hits += len(set(approx[i].tolist()) & set(exact[i].tolist()))
    return hits / float(take * k)


def probe(n_static, n_cand, dim, k, epochs=10, seed=0, measure_recall=True,
          **index_kwargs):
    """Time fit / query / update for one configuration."""
    rng = np.random.default_rng(seed)
    static = rng.random((n_static, dim))
    cand = rng.random((n_cand, dim))

    index = ToroidalNN(seed=seed, **index_kwargs)
    t0 = time.perf_counter()
    index.fit(static, cand, k=k)
    fit_s = time.perf_counter() - t0

    # A spacing-sized perturbation, as the ESS step would produce.
    spacing = dim / 4.0 / max(n_static + n_cand, 2) ** (1.0 / dim)
    query_s = update_s = 0.0
    for _ in range(epochs):
        t0 = time.perf_counter()
        index.query(k=k)
        query_s += time.perf_counter() - t0

        moved = np.mod(
            index.candidates
            + rng.normal(0.0, STEP_FRACTION * spacing / dim, cand.shape),
            1.0,
        )
        t0 = time.perf_counter()
        index.update(moved)
        update_s += time.perf_counter() - t0

    rec = (
        recall_at_k(index, np.mod(np.vstack([static, cand]), 1.0),
                    index.candidates, k)
        if measure_recall else float("nan")
    )
    return {
        "n_static": n_static, "n_cand": n_cand, "dim": dim, "k": k,
        "backend": index.backend_name or "brute",
        "approx": bool(index.is_approximate), "tables": int(index.n_tables),
        **{key: val for key, val in index_kwargs.items()},
        "fit_s": fit_s,
        "query_ms": 1000.0 * query_s / epochs,
        "update_ms": 1000.0 * update_s / epochs,
        "recall": rec,
    }


def table(rows, keys):
    cols = ("fit_s", "query_ms", "update_ms", "recall")
    print("| " + " | ".join(list(keys) + list(cols)) + " |")
    print("|" + "---|" * (len(keys) + len(cols)))
    for r in rows:
        cells = [str(r.get(key)) for key in keys]
        cells += [f"{r['fit_s']:.2f}", f"{r['query_ms']:.1f}",
                  f"{r['update_ms']:.1f}", f"{r['recall']:.3f}"]
        print("| " + " | ".join(cells) + " |")


def save(name, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"[saved {os.path.join(OUT, name)}: {len(rows)} rows]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 50000])
    ap.add_argument("--dims", type=int, nargs="+", default=[32])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--grid", action="store_true",
                    help="sweep (resolution, dims_per_table, tables, probes)")
    args = ap.parse_args()

    rows = []
    if not args.grid:
        print("baseline: auto-tuned LSH parameters\n")
        for dim in args.dims:
            for n in args.sizes:
                rows.append(probe(n, n, dim, args.k, epochs=args.epochs))
                print(f"  d={dim} n={n}+{n} done", flush=True)
        save("profile_torann_baseline.json", rows)
        table(rows, ("dim", "n_static", "backend", "tables"))
    else:
        print("grid: LSH parameters vs query cost and recall\n")
        dim, n = args.dims[0], args.sizes[0]
        for resolution in (2, 3, 4):
            for tables in (4, 8, 16, 24):
                for probes in (0, 2, 4, 8):
                    rows.append(probe(
                        n, n, dim, args.k, epochs=args.epochs,
                        resolution=resolution, num_tables=tables,
                        probes=probes,
                    ))
                    print(f"  B={resolution} L={tables} probes={probes} "
                          f"query={rows[-1]['query_ms']:.1f}ms "
                          f"recall={rows[-1]['recall']:.3f}", flush=True)
        save("profile_torann_grid.json", rows)
        table(rows, ("resolution", "num_tables", "probes", "tables"))
