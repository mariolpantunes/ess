"""Re-exploration benchmark: find the voids left by a fixed anchor set.

This is what ESS is *for*. Given points already evaluated — a design of
experiments so far, the population of a blind optimiser, the history
before a restart — where should the next batch go? A space-filling
sampler cannot answer that: it ignores the anchors and samples the whole
domain, so some of its points land where you have already looked. ESS
places them in the empty regions.

Benchmarking it from an empty domain (as `benchmark_dispersion.py` does)
measures the wrong thing: with no anchors the task degenerates into plain
space filling, which is exactly what LHS and Sobol are built for and
where they win on cost. The comparison that matters is this one.

Methods compared (all asked to add `n` points to `a` fixed anchors):

    ess-<version>   the library, this checkout
    init-lhs        Latin hypercube sample of n points, anchors ignored
    init-random     uniform sample of n points, anchors ignored
    init-sobol      Sobol sequence, anchors ignored (if SciPy is present)

The "init-" methods are the initialisation strategies ESS itself starts
from, so they are the honest floor: any gain ESS reports has to be a gain
over its own starting point.

Metrics (see `report` for the printed names):

    void_mean / void_min  toroidal L1 distance from each NEW point to the
                          nearest ANCHOR, mean and worst. This is the
                          objective: larger means the batch really did go
                          where nothing had been evaluated.
    combined_ce           toroidal Clark-Evans of anchors + new points:
                          is the union a well-spread design?
    marginal_disc         1-D wrap-around discrepancy of the **union**,
                          normalised so 1.0 = a random sample. Guards
                          against buying void coverage by wrecking the
                          per-factor coverage.

                          It must be scored on the union, not on the new
                          batch alone. The voids left by an anchor set
                          are not spread uniformly along each axis, so a
                          method that correctly fills them cannot have
                          uniform marginals *within the batch* — scoring
                          the batch alone rewards ignoring the anchors,
                          which is why init-lhs reaches 0.010 there while
                          ESS reads 3.97. On the union the ordering is
                          sane: anchors alone 0.640, ESS 0.616.

Run from the repository root::

    python examples/benchmark_reexploration.py
    python examples/benchmark_reexploration.py --dims 2 8 --seeds 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ess  # noqa: E402
from ess.samplers import LHCSampler, UniformSampler  # noqa: E402
from ess.utils import (  # noqa: E402
    expected_discrepancy,
    toroidal_clark_evans,
    wrap_around_discrepancy,
)

OUT = os.path.join(os.path.dirname(__file__), "out")
VERSION = getattr(ess, "__version__", "dev")


def _toroidal_l1(a, b):
    """Pairwise toroidal L1 distances between rows of `a` and rows of `b`."""
    delta = np.abs(a[:, None, :] - b[None, :, :])
    return np.minimum(delta, 1.0 - delta).sum(axis=2)


def score(anchors, new, marginal_dims=8):
    """The metric panel for one batch of new points."""
    n = len(new)
    void = _toroidal_l1(new, anchors).min(axis=1)
    union = np.vstack([anchors, new])
    dims = range(min(marginal_dims, new.shape[1]))
    disc = np.mean([wrap_around_discrepancy(union[:, [j]]) for j in dims])
    batch_disc = np.mean([wrap_around_discrepancy(new[:, [j]]) for j in dims])
    return {
        "void_mean": float(void.mean()),
        "void_min": float(void.min()),
        "combined_ce": toroidal_clark_evans(union),
        "marginal_disc": float(disc / expected_discrepancy(len(union), 1)),
        # Diagnostic only -- see the module docstring on why the batch
        # alone is the wrong set to score.
        "marginal_disc_batch": float(batch_disc / expected_discrepancy(n, 1)),
    }


def methods():
    """name -> callable(anchors, bounds, n, seed) returning the new points."""
    out = {
        f"ess-{VERSION}": lambda A, b, n, s: ess.esa(A, b, n=n, seed=s),
        "init-lhs": lambda A, b, n, s: LHCSampler(random_state=s).sample(
            n, A.shape[1]
        ),
        "init-random": lambda A, b, n, s: UniformSampler(random_state=s).sample(
            n, A.shape[1]
        ),
    }
    try:
        from scipy.stats import qmc

        out["init-sobol"] = lambda A, b, n, s: qmc.Sobol(
            A.shape[1], scramble=True, seed=s
        ).random(n)
    except ImportError:
        pass
    return out


def run(dims, n_anchors, n_new, seeds):
    rows = []
    for dim in dims:
        bounds = np.array([[0.0, 1.0]] * dim)
        for seed in range(seeds):
            anchors = np.random.default_rng(1000 + seed).random((n_anchors, dim))
            for name, fn in methods().items():
                t0 = time.perf_counter()
                new = np.asarray(fn(anchors, bounds, n_new, seed), dtype=np.float64)
                wall = time.perf_counter() - t0
                rows.append({
                    "method": name, "dim": dim, "seed": seed,
                    "n_anchors": n_anchors, "n_new": n_new,
                    "time_s": wall, **score(anchors, new),
                })
                print(f"  {name:16s} d={dim:<3} seed={seed} "
                      f"void_mean={rows[-1]['void_mean']:.3f} "
                      f"t={wall:.2f}s", flush=True)
    return rows


COLS = ("void_mean", "void_min", "combined_ce", "marginal_disc", "time_s")
LABEL = {
    "void_mean": "void dist (mean)", "void_min": "void dist (worst)",
    "combined_ce": "combined CE", "marginal_disc": "marginal disc",
    "time_s": "time[s]",
}


def report(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["dim"], r["method"]), []).append(r)
    print("\n| dim | method | " + " | ".join(LABEL[c] for c in COLS) + " |")
    print("|" + "---|" * (len(COLS) + 2))
    for (dim, method) in sorted(groups, key=lambda k: (k[0], k[1])):
        rs = groups[(dim, method)]
        cells = [f"{np.mean([r[c] for r in rs]):.3f}" for c in COLS]
        print(f"| {dim} | {method} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 8, 32])
    ap.add_argument("--anchors", type=int, default=500)
    ap.add_argument("--new", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(f"re-exploration: {args.anchors} anchors + {args.new} new points")
    rows = run(args.dims, args.anchors, args.new, args.seeds)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bench_reexploration.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    report(rows)
