r"""Current `esa` against the legacy `_esa_01`, on speed and clearance.

Designs are ranked on per-new-point toroidal L1 nearest-neighbour clearance
normalised by the fair-share radius $R_1 = \text{radius\_for\_count}(d, N_{\text{total}}, 1)$:
`gap_min / R1` and `gap_p05 / R1` (**higher is better**), reported beside wall
time and epochs used.

Run from the repository root::

    python examples/evaluate_esa.py
"""

import time

import numpy as np
from torann import ToroidalNN

from ess import esa as current_esa
from ess import geometry
from ess.legacy import _esa_01 as legacy_esa


def clearance_stats(anchors: np.ndarray, new_pts: np.ndarray, p: float = 1.0) -> dict[str, float]:
    """Per-new-point nearest-neighbour gap normalised by fair-share radius R1."""
    n_total = anchors.shape[0] + new_pts.shape[0]
    dim = new_pts.shape[1]
    r1 = geometry.radius_for_count(dim, n_total, target=1, p=p)
    idx = ToroidalNN(seed=0, p=p, brute_threshold=n_total + 1).fit(anchors, new_pts, k=1)
    _, dists = idx.query(k=1)
    gaps = np.asarray(dists, dtype=np.float64).ravel() / r1
    return {
        "gap_min": float(gaps.min()),
        "gap_p05": float(np.quantile(gaps, 0.05)),
        "gap_med": float(np.median(gaps)),
    }


def run_benchmark():
    dimensions = [2, 5, 10, 20, 40]
    populations = [64, 128, 256]
    results = []

    print(
        f"{'Dim':>4} | {'N':>4} | {'Legacy(s)':>10} | {'Current(s)':>10} | "
        f"{'Speedup':>8} | {'L-gap05':>8} | {'C-gap05':>8} | {'C-min':>8} | {'Ep':>4}"
    )
    print("-" * 88)

    for d in dimensions:
        for n in populations:
            bounds = np.array([[0.0, 1.0]] * d)
            n_initial = max(1, n // 10)
            samples = np.random.default_rng(42).uniform(0.0, 1.0, (n_initial, d))

            start_legacy = time.perf_counter()
            points_legacy = legacy_esa(samples, bounds, n=n, seed=42)
            time_legacy = time.perf_counter() - start_legacy
            q_legacy = clearance_stats(samples, points_legacy)

            stats: dict = {}
            start_current = time.perf_counter()
            points_current = current_esa(samples, bounds, n=n, seed=42, stats=stats)
            time_current = time.perf_counter() - start_current
            q_current = clearance_stats(samples, points_current)

            speedup = time_legacy / time_current

            row = {
                "Dim": d,
                "N": n,
                "Legacy_T": time_legacy,
                "Current_T": time_current,
                "Speedup": speedup,
                "Legacy_Q": q_legacy["gap_p05"],
                "Current_Q": q_current["gap_p05"],
                "Current_Min": q_current["gap_min"],
                "Epochs": int(stats.get("epochs_total", 0)),
            }
            results.append(row)

            print(
                f"{d:4d} | {n:4d} | {time_legacy:10.4f} | {time_current:10.4f} | "
                f"{speedup:7.2f}x | {row['Legacy_Q']:8.4f} | {row['Current_Q']:8.4f} | "
                f"{row['Current_Min']:8.4f} | {row['Epochs']:4d}"
            )

    return results


def print_markdown(data):
    print("\n### Detailed Comparison Table")
    headers = [
        "Dim",
        "N",
        "Legacy Time (s)",
        "Current Time (s)",
        "Speedup",
        "Legacy gap_p05/R1",
        "Current gap_p05/R1",
        "Current gap_min/R1",
        "Epochs",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in data:
        print(
            f"| {r['Dim']} | {r['N']} | {r['Legacy_T']:.4f} | {r['Current_T']:.4f} | "
            f"{r['Speedup']:.2f}x | {r['Legacy_Q']:.4f} | {r['Current_Q']:.4f} | "
            f"{r['Current_Min']:.4f} | {r['Epochs']} |"
        )

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
    print(f"| Avg Legacy gap_p05/R1 (HIGHER better) | {avg_l_q:.4f} |")
    print(f"| Avg Current gap_p05/R1 (HIGHER better) | {avg_c_q:.4f} |")


if __name__ == "__main__":
    results_data = run_benchmark()
    print_markdown(results_data)
