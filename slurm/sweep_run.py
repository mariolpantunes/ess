"""One cell of the fixes sweep: one code arm, one scenario, all seeds.

The arm is a source tree (`--src`), so the same runner scores every commit
on the same inputs. Designs are scored on **dispersion** of the union of
the existing and the new points: the largest and the mean distance from a
probe to its nearest point (toroidal L1, lower is better). Separation and
wrap-around discrepancy are recorded as diagnostics.

Run: python slurm/sweep_run.py --src ARM/src --out DIR --scenario uniform
     --d 8 --n0 256 --ratio 1 --seeds 20 [--cap 0.01] [--pool 64] [--p 0.5]
"""

import argparse
import inspect
import json
import logging
import pathlib
import platform
import resource
import sys
import time

import numpy as np

PROBES = 8192
CHUNK = 3e7  # floats per distance block


def nearest(a: np.ndarray, b: np.ndarray, p: float = 1.0, same: bool = False):
    """Toroidal L^p distance from each row of `a` to its nearest row of `b`.

    Args:
        a (np.ndarray): Query rows, shape (M, D).
        b (np.ndarray): Reference rows, shape (N, D).
        p (float): Exponent, 0 < p <= 1.
        same (bool): `a` is `b`; a row is not its own neighbour.

    Returns:
        np.ndarray: Shape (M,).
    """
    out = np.empty(a.shape[0])
    rows = max(1, int(CHUNK / max(b.size, 1)))
    for lo in range(0, a.shape[0], rows):
        gap = np.abs(a[lo:lo + rows, None, :] - b[None, :, :])
        np.minimum(gap, 1.0 - gap, out=gap)
        if p != 1.0:
            gap **= p
        dist = gap.sum(axis=2)
        if p != 1.0:
            dist **= 1.0 / p
        if same:
            idx = np.arange(lo, min(lo + rows, a.shape[0]))
            dist[idx - lo, idx] = np.inf
        out[lo:lo + rows] = dist.min(axis=1)
    return out


def make_static(scenario: str, n0: int, dim: int, rng: np.random.Generator):
    """Existing points on the unit torus for one scenario."""
    if scenario == "scratch" or n0 == 0:
        return np.empty((0, dim))
    if scenario == "uniform":
        return rng.random((n0, dim))
    if scenario == "clustered":
        centres = rng.uniform(0.2, 0.8, size=(6, dim))
        pts = centres[rng.integers(6, size=n0)] + rng.normal(0.0, 0.06, size=(n0, dim))
        return np.clip(pts, 0.0, 1.0 - 1e-9)
    raise ValueError(scenario)


def score(static: np.ndarray, new: np.ndarray, probes: np.ndarray, p: float, utils) -> dict:
    """Dispersion of the union, plus the diagnostics."""
    union = np.concatenate((static, new)) if static.size else new
    hole = nearest(probes, union)
    sep = nearest(union, union, same=True)
    row = {
        "hole_max": float(hole.max()),
        "hole_p95": float(np.quantile(hole, 0.95)),
        "hole_mean": float(hole.mean()),
        "sep_min": float(sep.min()),
        "sep_med": float(np.median(sep)),
        "new_sep_min": float(nearest(new, new, same=True).min()),
    }
    if static.size:
        row["new_to_static_min"] = float(nearest(new, static).min())
    if p != 1.0:
        row["hole_max_lp"] = float(nearest(probes, union, p).max())
        row["sep_min_lp"] = float(nearest(union, union, p, same=True).min())
    if utils is not None:
        row["wrap_disc"] = float(
            utils.wrap_around_discrepancy(union)
            / utils.expected_discrepancy(union.shape[0], union.shape[1]))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src", help="source tree of the arm; omit for the random baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", required=True, choices=("scratch", "uniform", "clustered"))
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--n0", type=int, required=True)
    ap.add_argument("--ratio", type=float, default=1.0, help="points added / n0")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cap", type=float, help="override STEP_CAP")
    ap.add_argument("--pool", type=int, help="init_pool")
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, help="0 scores the placement alone")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    esa = core = utils = None
    if args.src:
        sys.path.insert(0, str(pathlib.Path(args.src).resolve()))
        import ess  # noqa: PLC0415  (the arm under test, chosen at run time)
        from ess import utils  # noqa: PLC0415

        if not pathlib.Path(ess.__file__).resolve().is_relative_to(pathlib.Path(args.src).resolve()):
            raise SystemExit(f"imported {ess.__file__}, not the arm in {args.src}")
        core = sys.modules["ess.ess"]
        esa = core.esa
        if args.cap is not None:
            core.STEP_CAP = args.cap
    kwargs = {}
    if esa is not None:
        accepted = inspect.signature(esa).parameters
        if args.pool is not None:
            kwargs["init_pool"] = args.pool
        if args.epochs is not None:
            kwargs["epochs"] = args.epochs
        if args.p != 1.0:
            kwargs["p"] = args.p
        missing = [k for k in kwargs if k not in accepted]
        if missing:
            raise SystemExit(f"arm does not accept {missing}")

    n_new = int(round(args.n0 * args.ratio))
    bounds = np.array([[0.0, 1.0]] * args.d)
    rows = []
    for seed in range(args.seeds):
        static = make_static(args.scenario, args.n0, args.d, np.random.default_rng([seed, 1]))
        probes = np.random.default_rng([seed, 2]).random((PROBES, args.d))
        started = time.perf_counter()
        if esa is None:
            new = np.random.default_rng([seed, 3]).random((n_new, args.d))
        else:
            new = np.mod(np.asarray(esa(static, bounds, n=n_new, seed=seed, **kwargs)), 1.0)
        elapsed = time.perf_counter() - started
        if new.shape != (n_new, args.d) or not np.all(np.isfinite(new)):
            raise SystemExit(f"seed {seed}: bad output {new.shape}")
        row = {"seed": seed, "time_s": elapsed}
        row.update(score(static, new, probes, args.p, utils))
        rows.append(row)
        print(f"seed {seed} {elapsed:.1f}s hole_max={row['hole_max']:.4f}", flush=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = vars(args) | {
        "n_new": n_new,
        "step_cap": None if core is None else core.STEP_CAP,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "host": platform.node(),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (out / "result.json").write_text(json.dumps({"meta": meta, "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
