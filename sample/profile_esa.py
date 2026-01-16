import argparse
import cProfile
import io
import logging
import pstats
import time

import numpy as np

import ess.ess as ess

# Disable logging to keep profile output clean
logging.basicConfig(level=logging.CRITICAL)


def profile_esa_split(
    n=2000,
    search_mode="k_nn",
    border_strategy="clip",
    k=None,
    radius=None,
    metric="gaussian",
    batch_size=50,
):
    """
    Runs a profile for a specific N value and configuration.
    Allows passing parameters dynamically via CLI.
    """
    dimensions = [2, 4, 8, 16, 32]
    initial_pop_sizes = [128, 256, 512]

    # Use the n provided via args (Single N per run)
    n_values = [n]

    bound_min, bound_max = -10, 10

    print("Starting Profile Run...")
    print(f"Configuration: N={n}, Search={search_mode}, Border={border_strategy}")
    print(f"K={k}, Radius={radius}, Metric={metric}")
    print(f"Dims={dimensions}")
    print(f"Populations: {initial_pop_sizes}")
    print("-" * 65)

    # LOOP 1: Iterate over N values (Single value list)
    for n_gen in n_values:
        print(f"\n{'=' * 60}")
        print(f" PROFILING RUN: N = {n_gen}")
        print(f"{'=' * 60}")

        # Initialize Profiler
        pr = cProfile.Profile()
        pr.enable()

        run_start = time.perf_counter()

        # LOOP 2 & 3: Workload
        for dim in dimensions:
            bounds = np.array([[bound_min, bound_max]] * dim)

            for pop_size in initial_pop_sizes:
                # Setup
                rng = np.random.default_rng(42)
                points0 = rng.uniform(bound_min, bound_max, (pop_size, dim))

                # Timer per line
                step_start = time.perf_counter()

                # Run ESA
                ess.esa(
                    points0,
                    bounds,
                    n=n_gen,
                    batch_size=batch_size,
                    epochs=100,
                    search_mode=search_mode,
                    border_strategy=border_strategy,
                    k=k,
                    radius=radius,
                    metric=metric,
                )

                step_end = time.perf_counter()
                elapsed = step_end - step_start

                print(
                    f"Done: Dim={dim:<2}, Pop={pop_size:<3}, N={n_gen:<4} ({elapsed:.4f}s)"
                )

        # Stop Profiler
        run_end = time.perf_counter()
        pr.disable()

        print("-" * 60)
        print(f"Total Time for N={n_gen}: {run_end - run_start:.4f}s")
        print("-" * 60)

        # Print Stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats(20)
        print(s.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile ESA Algorithm")
    parser.add_argument(
        "--n", type=int, default=2000, help="Number of points to generate (default: 2000)"
    )
    parser.add_argument(
        "--search_mode",
        type=str,
        default="k_nn",
        choices=["k_nn", "radius"],
        help="Search mode (default: k_nn)",
    )
    parser.add_argument(
        "--border_strategy",
        type=str,
        default="clip",
        choices=["clip", "repulsive"],
        help="Border strategy (default: clip)",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Number of neighbors (default: None)"
    )
    parser.add_argument(
        "--radius", type=float, default=None, help="Search radius (default: None)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="gaussian",
        help="Distance metric function name (default: gaussian)",
    )

    args = parser.parse_args()

    profile_esa_split(
        n=args.n,
        search_mode=args.search_mode,
        border_strategy=args.border_strategy,
        k=args.k,
        radius=args.radius,
        metric=args.metric,
    )