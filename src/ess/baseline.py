r"""Reference space-filling constructions, for testing ESS rather than for use.

Nothing here is a production engine. These exist to answer "is the relaxation
doing anything?" and "did that refactor change the answer?", which need
constructions simple enough that they cannot themselves be wrong:

* `dart` — Mitchell's best-candidate. Draw `k_cand` uniform candidates, keep
  the one furthest from everything placed so far, repeat. No forces, no step
  size, no convergence criterion, so nothing in it can be tuned into or out
  of a result. It is the ablation of ESS's relaxation: `esa` starts from the
  same idea (`_smart_init`) and then relaxes, so `esa` minus `dart` is what
  the force kernel contributes.
* `random_fill` — the null. Same point count, no search at all. Any margin
  `dart` holds over this is attributable to looking for empty space, and any
  margin `esa` holds over `dart` to relaxing afterwards.
* `grid_oracle` — the emptiest point on a lattice, by exhaustion. Exact up to
  the lattice, and usable only at very low dimension; it exists to prove
  `dart` approximates the quantity it claims to.

Everything measures **toroidal L1**, the metric ESS optimises, via
`torann.brute.pairwise_l1` — one implementation of the wrap rather than a
private copy that can drift from it.
"""

from __future__ import annotations

import numpy as np
from torann.brute import pairwise_l1

__all__ = ["dart", "grid_oracle", "random_fill"]


def _unit(samples: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map `samples` onto the unit torus; return it with the frame to undo it."""
    lower, upper = bounds[:, 0], bounds[:, 1]
    span = np.where(upper - lower == 0, 1.0, upper - lower)
    pts = np.asarray(samples, dtype=np.float64).reshape(-1, bounds.shape[0])
    return np.mod((pts - lower) / span, 1.0), lower, span


def dart(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    n: int,
    seed: int | np.random.Generator | None = None,
    k_cand: int = 64,
    sequential: bool = True,
) -> np.ndarray:
    r"""Mitchell's best-candidate: place `n` points, greedily, farthest-first.

    For each new point, draw `k_cand` uniform candidates and keep

    $$ c^* = \arg\max_{c} \; \min_{p \in P} d^{tor}_{L_1}(c, p) $$

    over everything already present, $P$ — the existing samples plus, when
    `sequential`, the points placed so far in this call.

    **`sequential` is the difference between this and `esa`'s
    `_smart_init`**, and the two are not ranked the way the names suggest.
    Sequential adds each point to $P$ before choosing the next — Mitchell's
    algorithm as defined. Batch picks every slot at once against a fixed
    reference, so two slots cannot see each other's choice.

    Batch is cheaper ($O(1)$ index queries against $O(n)$) *and* more uniform.
    Measured on a $1N$ sample + $1N$ quasi-opposite + $1N$ probe pool, 15
    seeds, normalised 2-D projection discrepancy, sequential against batch:
    **-14.4% at $d = 8$ (0/15 seeds), -12.2% at $d = 16$ (1/15), -7.5% at
    $d = 32$ (0/15)** — sequential is worse everywhere.

    The reason is that greedy farthest-first is a *traversal*: each pick is
    maximally far from everything chosen so far, which chains outward to the
    extremes and clumps. Batch spreads across the original set's voids
    without chaining.

    Sequential does win on one thing, which is why it stays reachable:
    **minimum separation among the placed points**, since batch can put two
    slots on top of each other. If a design is being judged on its worst
    pair rather than on uniformity, that is the one to use.

    Args:
        samples (np.ndarray): Points already present, shape $(M, D)$. May be
            empty.
        bounds (np.ndarray): Domain, shape $(D, 2)$.
        n (int): How many points to place.
        seed (int | Generator | None): Seed or Generator.
        k_cand (int): Candidates drawn per placement. Higher is closer to the
            true largest-empty-sphere centre and linearly more expensive.
        sequential (bool): Update the reference set after each placement.

    Returns:
        np.ndarray: The `n` placed points, shape $(n, D)$, in the original
        coordinate system.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    dim = bounds.shape[0]
    n = int(n)
    if n <= 0:
        return np.empty((0, dim))

    static, lower, span = _unit(samples, bounds)
    placed = np.empty((n, dim))

    if not sequential:
        cand = rng.random((n * k_cand, dim))
        if static.shape[0]:
            d = pairwise_l1(cand, static).min(axis=1)
        else:
            d = rng.random(n * k_cand)  # nothing to be far from
        best = d.reshape(n, k_cand).argmax(axis=1)
        placed = cand.reshape(n, k_cand, dim)[np.arange(n), best]
        return lower + placed * span

    ref = static
    for i in range(n):
        cand = rng.random((k_cand, dim))
        if ref.shape[0]:
            placed[i] = cand[int(np.argmax(pairwise_l1(cand, ref).min(axis=1)))]
        else:
            placed[i] = cand[0]
        ref = np.vstack((ref, placed[i][None, :]))

    return lower + placed * span


def random_fill(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    n: int,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """`n` uniform points: the null that isolates "searched for empty space".

    Same signature and same point count as `dart` and `esa`, with no search,
    so a margin either of them holds over this is attributable to the search
    and not to having produced more points.

    Args:
        samples (np.ndarray): Ignored; accepted so this is drop-in.
        bounds (np.ndarray): Domain, shape $(D, 2)$.
        n (int): How many points to place.
        seed (int | Generator | None): Seed or Generator.

    Returns:
        np.ndarray: Shape $(n, D)$, in the original coordinate system.
    """
    del samples
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    dim = bounds.shape[0]
    if int(n) <= 0:
        return np.empty((0, dim))
    lower, upper = bounds[:, 0], bounds[:, 1]
    return rng.uniform(lower, upper, size=(int(n), dim))


def grid_oracle(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    resolution: int = 64,
    max_nodes: int = 40_000_000,
) -> tuple[np.ndarray, float]:
    r"""The emptiest point in the domain, found by exhausting a lattice.

    Evaluates $\min_{p} d^{tor}_{L_1}(x, p)$ at every node of a regular
    `resolution`-per-axis lattice and returns the argmax. Exact up to the
    lattice, in the metric ESS actually uses — which is what `dart` is
    checked against.

    **Only usable at very low dimension**, and resolution binds before node
    count does. Cost is $G^D$ nodes; more importantly the answer is the best
    *node*, and the true optimum can sit half a cell away, so the reported
    distance carries a worst-case error of $D/(2G)$ in L1. If that is not far
    below the gap being arbitrated, this cannot arbitrate it — refine `G`
    until the answer stops moving, and if it never does, the oracle is out of
    range. At $D = 4$, $G = 64$ that is 16.8M nodes and an error of 0.031;
    by $D = 6$ the lattice alone is a billion nodes.

    Args:
        samples (np.ndarray): Points to be far from, shape $(M, D)$.
        bounds (np.ndarray): Domain, shape $(D, 2)$.
        resolution (int): Nodes per axis, $G$.
        max_nodes (int): Refuse rather than exhaust memory.

    Returns:
        tuple: `(point, distance)` — the emptiest lattice node in the original
        coordinate system, and its toroidal-L1 distance to the nearest sample.

    Raises:
        ValueError: If the lattice would exceed `max_nodes`.
    """
    dim = bounds.shape[0]
    nodes = resolution**dim
    if nodes > max_nodes:
        raise ValueError(
            f"a {resolution}-per-axis lattice in {dim}D is {nodes:.3g} nodes, "
            f"over the {max_nodes:.3g} budget. This oracle is a low-dimension "
            f"instrument: lower `resolution`, or use the discrepancy panel, "
            f"which needs no lattice."
        )

    static, lower, span = _unit(samples, bounds)
    axis = (np.arange(resolution) + 0.5) / resolution
    grid = np.stack(np.meshgrid(*([axis] * dim), indexing="ij"), axis=-1)
    grid = grid.reshape(-1, dim)

    best_d, best_i = -np.inf, 0
    block = max(1, 4_000_000 // max(1, static.shape[0]))
    for s in range(0, grid.shape[0], block):
        chunk = np.ascontiguousarray(grid[s : s + block])
        d = pairwise_l1(chunk, static).min(axis=1)
        j = int(np.argmax(d))
        if d[j] > best_d:
            best_d, best_i = float(d[j]), s + j

    return lower + grid[best_i] * span, best_d
