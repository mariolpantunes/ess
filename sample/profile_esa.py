import cProfile
import io
import logging
import pstats
import time

import numpy as np

import ess.ess as ess

# Disable logging to keep profile output clean
logging.basicConfig(level=logging.CRITICAL)


def profile_esa_split(batch_size=50):
    """
    Runs separate profiles for different N values to isolate
    NumpyNN (Small N) vs FaissHNSW (Large N) performance.
    """
    dimensions = [2, 4, 8, 16, 32]
    initial_pop_sizes = [128, 256, 512]

    # 2000 -> NumpyNN
    # 5000 -> FaissHNSWFlatNN
    n_values = [2000, 5000]

    bound_min, bound_max = -10, 10

    print("Starting Split Profiling...")
    print(f"Configurations: Dims={dimensions}")
    print(f"Populations: {initial_pop_sizes}")
    print("-" * 65)

    # LOOP 1: Iterate over N values (The Profile Splitter)
    for n_gen in n_values:
        print(f"\n{'=' * 60}")
        print(f" PROFILING RUN: N = {n_gen}")
        print(f"{'=' * 60}")

        # Initialize Profiler for THIS specific N value
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
                ess.esa(points0, bounds, n=n_gen, batch_size=batch_size, epochs=100)

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

        # Print Stats for this N
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats(20)
        print(s.getvalue())


if __name__ == "__main__":
    profile_esa_split()
