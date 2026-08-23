"""A fast smoke test of the ESS entry point — not a benchmark.

Renamed from ``benchmark.py``, because it never was one. It runs 10 static
points plus 50 new ones, and torann's brute/LSH crossover is 512, so every
dimension here takes the exact brute-force path: the whole file finishes in
under half a second and never touches the index this project exists to drive.
What it is good for is catching a crash or an obvious quality collapse in
``ess.esa`` across a spread of dimensions, quickly, which is worth having.

``benchmark_dispersion.py`` is the benchmark. ``benchmark_reexploration.py``
measures the task ESS is actually for.

Quality is reported as toroidal separation of the new batch against the
static points — the shared metric, see ``torann.metrics``. It used to be
``euclidean_separation``, which ignores the wrap and so scored the one thing
this library is about as if it did not happen.

Run from the repository root::

    python examples/smoke_esa.py
    python examples/smoke_esa.py --dimensions 2 8 --out /tmp/smoke.json
"""

import argparse
import json
import time

import numpy as np

import ess
from ess import utils


def run_benchmark(dimensions, n_samples=10, n_new=50, seed=42):
    results = {}
    rng = np.random.default_rng(seed)

    for dim in dimensions:
        print(f"Checking dimension {dim}...")

        # 1. Initialize bounds and static points
        bounds = np.array([[0, 1]] * dim, dtype=np.float32)
        samples = rng.uniform(0, 1, (n_samples, dim)).astype(np.float32)

        # 2. Run Generation (using defaults: softening force, clip boundary)
        start_time = time.perf_counter()
        # Use a fixed seed for reproducible results
        new_points = ess.esa(samples, bounds, n=n_new, epochs=200, seed=seed)
        elapsed = time.perf_counter() - start_time

        # 3. Calculate spatial metrics on generated points
        min_dist = utils.toroidal_separation(new_points, samples)

        # Grid coverage (limit grid resolution in higher dimensions to prevent memory/time explosion)
        grid_res = 3 if dim > 5 else 10
        coverage = utils.calculate_grid_coverage(new_points, bounds, grid=grid_res)

        results[str(dim)] = {
            "execution_time_seconds": elapsed,
            "toroidal_separation": min_dist,
            "wrap_discrepancy": (
                utils.wrap_around_discrepancy(new_points)
                / utils.expected_discrepancy(len(new_points), new_points.shape[1])
            ),
            "grid_coverage": coverage,
            "grid_resolution": grid_res,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test Empty Space Search (ESS) execution."
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[2, 4, 8, 10, 20],
        help="List of dimensions to check.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="baseline.json",
        help="Filename to write baseline JSON results.",
    )
    args = parser.parse_args()

    results = run_benchmark(args.dimensions)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Benchmark results written to {args.out}")


if __name__ == "__main__":
    main()
