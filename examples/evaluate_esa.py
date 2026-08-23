"""Current `esa` against the legacy `_esa_01`, on speed and on quality.

Quality is toroidal separation of the produced batch against the seed points
-- the shared metric, defined once in `torann.metrics`. It used to be a
normalised wrap-around discrepancy *labelled* "Clark-Evans", which inverted
the reading: a discrepancy is lower-better and a Clark-Evans index is
higher-better, so the summary reported 25.39 for legacy against 2.39 for
current and looked like a 10x regression when it was a 10x improvement. The
script ran clean the whole time, which is why nobody noticed.

Run from the repository root::

    python examples/evaluate_esa.py
"""

import time

import numpy as np

from ess import esa as current_esa
from ess.legacy import _esa_01 as legacy_esa
from ess.utils import toroidal_separation


def run_benchmark():
    dimensions = [2, 5, 10, 20, 40]
    populations = [64, 128, 256]
    results = []

    print(
        f"{'Dim':>4} | {'N':>4} | {'Legacy(s)':>10} | {'Current(s)':>10} | "
        f"{'Speedup':>8} | {'L-Sep':>10} | {'C-Sep':>10}"
    )
    print("-" * 75)

    for d in dimensions:
        for n in populations:
            bounds = np.array([[0, 1]] * d)
            # Use a small seed set (10% of N)
            n_initial = max(1, n // 10)
            samples = np.random.uniform(0, 1, (n_initial, d)).astype(np.float32)

            # --- Benchmark Legacy Version ---
            # _esa_01 optimizes each point independently against 'samples'
            start_legacy = time.perf_counter()
            points_legacy = legacy_esa(samples, bounds, n=n, seed=42)
            time_legacy = time.perf_counter() - start_legacy
            quality_legacy = toroidal_separation(points_legacy, samples)

            # --- Benchmark Current Version ---
            # esa optimizes points as a batch, repelling each other and 'samples'
            start_current = time.perf_counter()
            points_current = current_esa(samples, bounds, n=n, seed=42)
            time_current = time.perf_counter() - start_current
            quality_current = toroidal_separation(points_current, samples)

            speedup = time_legacy / time_current

            row = {
                "Dim": d,
                "N": n,
                "Legacy_T": time_legacy,
                "Current_T": time_current,
                "Speedup": speedup,
                "Legacy_Q": quality_legacy,
                "Current_Q": quality_current,
            }
            results.append(row)

            print(
                f"{d:4d} | {n:4d} | {time_legacy:10.4f} | {time_current:10.4f} | "
                f"{speedup:7.2f}x | {quality_legacy:10.4f} | {quality_current:10.4f}"
            )

    return results


def print_markdown(data):
    # 1. Detailed Table
    print("\n### Detailed Comparison Table")
    headers = [
        "Dim",
        "N",
        "Legacy Time (s)",
        "Current Time (s)",
        "Speedup",
        "Legacy separation",
        "Current separation",
    ]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    print(header_line)
    print(sep_line)

    for r in data:
        line = (
            f"| {r['Dim']} | {r['N']} | {r['Legacy_T']:.4f} | {r['Current_T']:.4f} | "
            f"{r['Speedup']:.2f}x | {r['Legacy_Q']:.4f} | {r['Current_Q']:.4f} |"
        )
        print(line)

    # 2. Aggregated Metrics
    speedups = [r["Speedup"] for r in data]
    l_qualities = [r["Legacy_Q"] for r in data]
    c_qualities = [r["Current_Q"] for r in data]

    avg_speedup = sum(speedups) / len(speedups)
    med_speedup = sorted(speedups)[len(speedups) // 2]
    avg_l_q = sum(l_qualities) / len(l_qualities)
    avg_c_q = sum(c_qualities) / len(c_qualities)

    print("\n### Aggregated Performance Summary")
    print("| Metric | Value |")
    print("| --- | --- |")
    print(f"| Average Speedup | {avg_speedup:.2f}x |")
    print(f"| Median Speedup | {med_speedup:.2f}x |")
    print(f"| Avg Legacy separation (higher better) | {avg_l_q:.4f} |")
    print(f"| Avg Current separation (higher better) | {avg_c_q:.4f} |")


if __name__ == "__main__":
    results_data = run_benchmark()
    print_markdown(results_data)
