import argparse
import json
import time
import numpy as np
import ess
import ess.utils as utils


def run_benchmark(dimensions, n_samples=10, n_new=50, seed=42):
    results = {}
    rng = np.random.default_rng(seed)

    for dim in dimensions:
        print(f"Benchmarking dimension {dim}...")

        # 1. Initialize bounds and static points
        bounds = np.array([[0, 1]] * dim, dtype=np.float32)
        samples = rng.uniform(0, 1, (n_samples, dim)).astype(np.float32)

        # 2. Run Generation (using defaults: softening force, clip boundary)
        start_time = time.perf_counter()
        # Use a fixed seed for reproducible results
        new_points = ess.esa(samples, bounds, n=n_new, epochs=200, seed=seed)
        elapsed = time.perf_counter() - start_time

        # 3. Calculate spatial metrics on generated points
        min_dist = utils.euclidean_separation(new_points)

        # Grid coverage (limit grid resolution in higher dimensions to prevent memory/time explosion)
        grid_res = 3 if dim > 5 else 10
        coverage = utils.calculate_grid_coverage(new_points, bounds, grid=grid_res)

        results[str(dim)] = {
            "execution_time_seconds": elapsed,
            "min_pairwise_distance": min_dist,
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
        description="Benchmark Empty Space Search (ESS) execution."
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[2, 4, 8, 10, 20],
        help="List of dimensions to benchmark.",
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
