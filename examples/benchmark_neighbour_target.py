r"""Is `NEIGHBOUR_TARGET` the right force scale for k-NN mode?

In k-NN mode the interaction radius selects nothing -- `k` does that. It is
the unit of length the run is expressed in: $\hat{d} = d_{L1}/R$ feeds every
force law, the step is $\eta F R$, and `STEP_CAP` bounds travel to a
fraction of $R$. `ess.NEIGHBOUR_TARGET` is how that length is specified, as
the neighbour count $R$ is derived to hold.

The value is a compromise between two readings that move in opposite
directions, and this script measures both:

1. **Dispersion** — how uniform the design is, on the discrepancies
   `ess.utils` ranks with (never on a toroidal point metric, which would be
   grading the relaxation with its own loss). Reported from scratch and
   normalised so 1.0 is a random design, plus a paired per-seed check,
   because the differences at low dimension are a few percent and a median
   over independent runs cannot tell those from noise.
2. **The attraction response** — how far a guided placement is dragged
   toward the good region, against the same run with the attraction off. A
   larger $R$ shrinks every $\hat{d}$; the repulsion is capped near zero by
   its softening and the attraction's longer tail is not, so the balance
   between them moves with this constant.

The tables it prints are the evidence quoted in `ess.geometry`'s docstring
for the constant. Re-run it when the force laws, the metrics or the
defaults change, and update that docstring with what comes out.

Usage:
    python examples/benchmark_neighbour_target.py
    python examples/benchmark_neighbour_target.py --dims 2 3 --seeds 4
"""

import argparse
import statistics
import sys

import numpy as np

import ess
from ess.utils import (
    expected_discrepancy,
    projection_discrepancy,
    wrap_around_discrepancy,
)

#: Design sizes the dispersion benchmark uses, so the two are comparable.
N_BY_DIM = {2: 256, 3: 256, 5: 256, 8: 512, 10: 512, 16: 512, 20: 512,
            32: 1024, 64: 1024}

#: The guided-placement setup: "good" means near the origin, so the
#: attraction's effect is measurable as a radius.
PULL_DIM, PULL_N, PULL_STATIC = 8, 30, 40


def design(dim, n, target, seed, **kwargs):
    """One from-scratch k-NN run at `target`, as `esa` would do it."""
    bounds = np.tile([0.0, 1.0], (dim, 1))
    return ess.esa(np.empty((0, dim)), bounds, n=n, seed=seed,
                   search_mode="k_nn", radius_target=target, **kwargs)


def dispersion(pts, dim):
    """The metric panel, normalised so 1.0 is a random design."""
    n = len(pts)
    return {
        "wrap": wrap_around_discrepancy(pts) / expected_discrepancy(n, dim),
        "proj1": projection_discrepancy(pts, 1) / expected_discrepancy(n, 1),
        "proj2": (projection_discrepancy(pts, 2, max_projections=60)
                  / expected_discrepancy(n, 2)),
    }


def pull(target, seed, weight):
    """How far the attraction drags the placement toward the good region.

    Returns the repulsion-only mean radius minus the composite's, so a
    larger number means the attraction won more of the balance.
    """
    bounds = np.tile([-5.0, 5.0], (PULL_DIM, 1))
    rng = np.random.default_rng(seed)
    static = rng.uniform(-5, 5, (PULL_STATIC, PULL_DIM))
    attract = -np.linalg.norm(static, axis=1)
    shared = {"n": PULL_N, "seed": 11 + seed, "search_mode": "k_nn",
              "radius_target": target}
    repulsive = ess.esa(static, bounds, **shared)
    composite = ess.esa(static, bounds, attractiveness=attract,
                        attraction_weight=weight, attraction_metric="cauchy",
                        attraction_kwargs={"power": 1.0}, **shared)
    return float(np.linalg.norm(repulsive, axis=1).mean()
                 - np.linalg.norm(composite, axis=1).mean())


def report_dispersion(dims, targets, seeds):
    print("=== dispersion, from scratch (1.0 = random, lower better) ===\n")
    for dim in dims:
        n = N_BY_DIM.get(dim, 512)
        print(f"  d={dim}, n={n}, median of {seeds} seeds")
        print(f"{'target':>8}{'wrap':>9}{'proj1':>9}{'proj2':>9}")
        for target in targets:
            rows = [dispersion(design(dim, n, target, s), dim)
                    for s in range(seeds)]
            cells = "".join(f"{statistics.median(r[k] for r in rows):>9.4f}"
                            for k in ("wrap", "proj1", "proj2"))
            print(f"{target:>8}{cells}")
        print()
        sys.stdout.flush()


def report_paired(dims, targets, seeds, reference):
    """The same comparison per seed, which is what resolves a few percent.

    Independent medians cannot separate a 5% effect from run-to-run spread;
    the same seed under two targets can, because everything else about the
    run is held fixed.
    """
    print(f"=== paired against target {reference}, {seeds} seeds "
          f"(wrap-around discrepancy) ===\n")
    print(f"{'d':>5}{'target':>8}{'wins':>9}{'median':>10}")
    for dim in dims:
        n = N_BY_DIM.get(dim, 512)
        base = [dispersion(design(dim, n, reference, s), dim)["wrap"]
                for s in range(seeds)]
        for target in targets:
            if target == reference:
                continue
            alt = [dispersion(design(dim, n, target, s), dim)["wrap"]
                   for s in range(seeds)]
            wins = sum(a < b for a, b in zip(alt, base))
            change = statistics.median((a - b) / b for a, b in zip(alt, base))
            print(f"{dim:>5}{target:>8}{f'{wins}/{seeds}':>9}"
                  f"{change * 100:>+9.1f}%")
        sys.stdout.flush()
    print()


def report_pull(targets, seeds, weights):
    print(f"=== attraction response at d={PULL_DIM} "
          f"(larger = the attraction wins more) ===\n")
    print(f"{'target':>8}" + "".join(f"{f'w={w}':>10}" for w in weights))
    for target in targets:
        cells = "".join(
            f"{statistics.median(pull(target, s, w) for s in range(seeds)):>10.3f}"
            for w in weights)
        print(f"{target:>8}{cells}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[2, 3, 5, 8, 20])
    parser.add_argument("--targets", type=int, nargs="+",
                        default=[1, 2, 3, 5, 8, 16])
    parser.add_argument("--seeds", type=int, default=12,
                        help="seeds per cell for the medians")
    parser.add_argument("--paired-seeds", type=int, default=30,
                        help="seeds for the paired check; 0 skips it")
    parser.add_argument("--reference", type=int, default=ess.NEIGHBOUR_TARGET,
                        help="the target the paired check compares against")
    parser.add_argument("--weights", type=float, nargs="+",
                        default=[0.2, 0.5, 1.0],
                        help="attraction weights for the response table")
    args = parser.parse_args()

    report_dispersion(args.dims, args.targets, args.seeds)
    if args.paired_seeds:
        report_paired(args.dims, args.targets, args.paired_seeds,
                      args.reference)
    report_pull(args.targets, args.seeds, args.weights)


if __name__ == "__main__":
    main()
