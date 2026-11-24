"""Inverse-distance fitness terrain for never-measured positions on the unit torus.

Fitness is only ever known where the objective was actually paid for.
Everything else is an estimate, and this module is where estimates come from.

**The space guides, not the points.** In ESS, only particles (the points being
relaxed and the static anchors) exert forces -- repulsion, to maximise space
exploration. The fitness field $f(x)$ is the terrain those particles traverse:
a non-convex, spatially bounded function built from the measured anchors at
zero objective evaluation cost, where **lower is closer to the ground
(better)** and **higher is uphill (worse)**. Its slope $\\nabla f(x)$ modulates
the repulsive displacement along the terrain (braking uphill movement towards
higher cost), and its value scores candidate positions during placement.

Three properties follow from that design:

1. **Being bounded by the measured range is a guarantee, not a handicap.**
   `InverseDistance` returns a convex combination of the anchors and can never
   leave their range, so it can never steer a point toward a value nobody
   observed.
2. **Degrading to the mean is the right failure.** Where the field has no
   information its slope vanishes and the run reduces to pure repulsion.
3. **The field is static for a whole call.** The anchors do not move during
   relaxation, so the field and its leave-one-out neighbourhood size (`choose_k`)
   are fitted once at the start of `esa` and only read as the moving points
   traverse it.

**Why only InverseDistance (IDW) survived high dimensionality.**
In a 468-arm factorial sweep inside OBLESA across $d = 8..64$, parametric and
basis-fitted models (`detrended`, `projection`, `fourier`, `auto`) all fitted
$2d$ coefficients from a fixed population size and collapsed below the $33\\%$
random parity line as $d$ grew (`detrended`: $70.4 \\to 41.9 \\to 18.0 \\to 13.8\\%$;
`projection`: $71.4 \\to 56.8 \\to 41.1 \\to 27.9\\%$), steering points in the wrong
direction once under-determined. `InverseDistance` has zero fitted coefficients
and held flat at $74.9 / 73.1 / 72.1 / 71.0\\%$ across $d = 8 / 16 / 32 / 64$.

At low dimension a local average (`k = 8`) preserves sharp regional detail;
at high dimension ($d \\ge 16$) pairwise distances concentrate and only a wide
average (`k = 128` to all anchors) preserves rank correlation and slope
direction. `choose_k` selects `k` per call via buffered leave-one-out
cross-validation at the fair-share radius $R_1$, and returns the field's
leave-one-out rank correlation as its `trust` in $[0, 1]$.
"""

import numpy as np
from torann import ToroidalNN

from . import geometry


class InverseDistance:
    """Shepard inverse-distance interpolation over the `k` nearest measured points.

    No coefficients, so nothing to under-determine: the estimate is a convex
    combination of measured values and can never leave their range.

    **The neighbour search is `torann`, not a second implementation.** The
    library already delegates every toroidal query to `ToroidalNN`, and
    this model's estimate is a toroidal k-NN like any other, so it uses an
    index rather than scanning. That index is separate from the relaxation's
    on purpose: the relaxation's holds the moving candidates as well, and an
    estimate must average over *sources* -- points whose objective was paid
    for -- never over the points being placed.

    The index is fitted once, in `fit`, and queried on every refresh. Above
    `torann`'s crossover the search uses LSH (unless `k` covers every source,
    where exact brute force is forced via `brute_threshold`).

    Args:
        k (int): Neighbours averaged over.
        power (float): Inverse-distance exponent.
        backend (str): Forwarded to `ToroidalNN`; `'auto'` picks the compiled
            kernel when it is installed.
        p (float): Exponent of the toroidal $L^p$ metric the neighbours are
            ranked and weighted by, in $(0, 1]$.
        seed (int | None): Seed of the index's hash functions. Fixed by
            default so that a seeded run is reproducible whatever the
            number of sources; ``None`` draws fresh ones on every fit.
    """

    def __init__(self, k: int = 8, power: float = 2.0, backend: str = "auto",
                 p: float = 1.0, seed: int | None = 0):
        self.k = int(k)
        self.power = float(power)
        self.backend = str(backend)
        self.p = geometry.check_p(p)
        self.seed = seed
        self._val = np.empty(0)
        self._pos = np.empty((0, 0))
        self._index: ToroidalNN | None = None

    def fit(self, positions: np.ndarray, values: np.ndarray) -> "InverseDistance":
        """Fit the terrain to the measured anchor positions and fitness values.

        Args:
            positions (np.ndarray): Measured points, shape $(M, D)$, in $[0, 1)$.
            values (np.ndarray): Their fitness (elevation), shape $(M,)$,
                lower is closer to the ground (better).

        Returns:
            InverseDistance: self.
        """
        pos = np.asarray(positions, dtype=np.float64)
        self._val = np.asarray(values, dtype=np.float64)
        self._pos = np.mod(pos, 1.0)
        self._index = None
        if pos.shape[0]:
            n = pos.shape[0]
            self._index = ToroidalNN(
                seed=self.seed, backend=self.backend, p=self.p,
                brute_threshold=n + 1 if self.k >= n - 1 else None).fit(
                self._pos, k=min(self.k, n))
        return self

    def at(self, positions: np.ndarray) -> np.ndarray:
        """Expected fitness (elevation) at `positions`, shape $(Q,)$.

        A position sitting exactly on a source takes that source's value
        rather than dividing by zero.
        """
        queries = np.asarray(positions, dtype=np.float64)
        if self._index is None or queries.shape[0] == 0:
            return np.zeros(queries.shape[0], dtype=np.float64)

        q = np.mod(queries, 1.0)
        kk = min(self.k, self._val.shape[0])
        ids, dist = self._index.query(k=kk, queries=q)

        missing = ids < 0
        safe = np.where(missing, 0, ids)
        f_near = self._val[safe]

        exact = (dist <= 0.0) & ~missing
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(exact | missing, 0.0, dist ** (-self.power))

        total = w.sum(axis=1)
        out = np.full(queries.shape[0], float(self._val.mean()))
        ok = total > 0
        if ok.any():
            out[ok] = np.einsum("qk,qk->q", w[ok], f_near[ok]) / total[ok]
        hit = exact.any(axis=1)
        if hit.any():
            out[hit] = f_near[hit, np.argmax(exact[hit], axis=1)]
        return out

    def guide(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""The terrain elevation and its uphill slope at `positions`.

        The field is $f(x) = \sum_j w_j f_j / \sum_j w_j$ with
        $w_j = d_j^{-q}$ over the neighbours `at` uses. Its gradient is

        $$ \nabla f(x) = -q \sum_j \tilde w_j \,\frac{f_j - f(x)}{d_j}\,
           \nabla d_j, \qquad \tilde w_j = \frac{w_j}{\sum_l w_l} $$

        (the neighbour set is held fixed). Because lower $f$ is closer to the
        ground (better) and higher $f$ is uphill (worse), $\nabla f(x)$ points
        **uphill** towards higher cost, and $-\nabla f(x)$ points **downhill**
        towards the ground. It is a property of the position: a mean, not a
        sum, so a region probed by ten sources of one value guides like a
        region probed by one; a constant field has no slope.

        The direction of $\nabla d_j$ is the library's: $\operatorname{sign}$
        of the toroidal displacement, normalised to unit $L^1$ length (see
        `ess._compute_forces`). The returned vector is therefore
        $\nabla f$ divided by the number of coordinates that differ, in units
        of fitness per unit distance; a caller multiplies by its interaction
        radius to compare it with a push.

        Args:
            positions (np.ndarray): Query positions, shape $(Q, D)$.

        Returns:
            tuple[np.ndarray, np.ndarray]: The estimated elevation, shape $(Q,)$,
            and the uphill slope vector, shape $(Q, D)$, pointing towards higher
            fitness (uphill). A query on a source, or with no usable neighbour,
            has a zero vector.
        """
        queries = np.asarray(positions, dtype=np.float64)
        dim = queries.shape[1] if queries.ndim == 2 else 0
        out = np.zeros((queries.shape[0], dim), dtype=np.float64)
        if self._index is None or queries.shape[0] == 0:
            return np.zeros(queries.shape[0], dtype=np.float64), out

        q = np.mod(queries, 1.0)
        kk = min(self.k, self._val.shape[0])
        ids, dist = self._index.query(k=kk, queries=q)
        missing = ids < 0
        safe = np.where(missing, 0, ids)
        f_near = self._val[safe]
        exact = (dist <= 0.0) & ~missing
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(exact | missing, 0.0, dist ** (-self.power))
        total = w.sum(axis=1)
        est = np.full(queries.shape[0], float(self._val.mean()))
        ok = total > 0
        if ok.any():
            est[ok] = np.einsum("qk,qk->q", w[ok], f_near[ok]) / total[ok]
        hit = exact.any(axis=1)
        if hit.any():
            est[hit] = f_near[hit, np.argmax(exact[hit], axis=1)]

        rows = np.flatnonzero(ok & ~hit)
        step = max(1, 4_000_000 // max(1, ids.shape[1] * dim))
        for lo in range(0, rows.size, step):
            r = rows[lo : lo + step]
            with np.errstate(divide="ignore", invalid="ignore"):
                coef = np.where(
                    w[r] > 0.0,
                    w[r] / total[r, None] * (f_near[r] - est[r, None]) / dist[r],
                    0.0)
            disp = q[r, None, :] - self._pos[safe[r]]
            disp -= np.round(disp)
            grad = np.sign(disp)
            norms = np.maximum(np.abs(grad).sum(axis=2, keepdims=True), 1e-9)
            out[r] = -self.power * np.einsum("qk,qkd->qd", coef, grad / norms)
        return est, out


K_CANDIDATES = (8, 32, 128)
"""tuple: Neighbour counts `choose_k` compares, beside "every source"."""

K_TOLERANCE = 0.02
"""float: Rank correlation a smaller neighbourhood may lose and still be chosen."""


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; 0 when one side is constant."""
    if a.size < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return 0.0
    ra = np.argsort(np.argsort(a, kind="stable")).astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable")).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def leave_one_out(positions, values, ks=None, power: float = 2.0,
                  p: float = 1.0, max_queries: int = 512,
                  seed: int | None = 0, buffer: float = 0.0) -> dict[int, float]:
    r"""How well the distance-weighted average predicts the sources it is made of.

    Each source is estimated from the others (itself left out) with the
    $k$ nearest of them, for every $k$ in `ks`; the score of a $k$ is the
    rank correlation of those estimates with the measured values. It costs
    no evaluation of the objective: the sources are the only measured
    points, and they are all the evidence there is of what the field is
    worth at the dimension, the count and the landscape of this call.

    A score near 1 means the field orders positions as the objective does;
    near 0 (or below) means it holds no information, whatever $k$.

    **The buffer.** The field is read where the points are placed: in empty
    regions, about one fair-share radius from every source. A source inside
    a cluster is predicted by its own cluster, which says nothing of that
    reading. With `buffer` > 0 the sources nearer than it are left out too
    (a buffered, or spatial, leave-one-out), so each source is predicted
    from as far away as an empty region is.

    Args:
        positions (np.ndarray): Sources, shape $(M, D)$, in $[0, 1)$.
        values (np.ndarray): Their fitness, shape $(M,)$.
        ks (Iterable[int] | None): Neighbour counts; ``None`` is
            `K_CANDIDATES` and every source. Capped at $M - 1$.
        power (float): Inverse-distance exponent.
        p (float): Exponent of the toroidal $L^p$ metric.
        max_queries (int): Sources that are left out, at most (a seeded
            sample above it); bounds the cost at `max_queries` $\times M$.
        seed (int | None): Seed of that sample and of the index.
        buffer (float): Distance under which a neighbour is left out too.

    Returns:
        dict[int, float]: Score per neighbour count, ascending.
    """
    pos = np.mod(np.asarray(positions, dtype=np.float64), 1.0)
    val = np.asarray(values, dtype=np.float64)
    n = val.shape[0]
    if n < 3:
        return {}
    wanted = (*K_CANDIDATES, n - 1) if ks is None else tuple(ks)
    cand = sorted({min(max(int(k), 1), n - 1) for k in wanted})
    if buffer > 0.0:
        cand = sorted({*cand, n - 1})  # the search must reach past the buffer
    rows = np.arange(n)
    if n > max_queries:
        rows = np.sort(np.random.default_rng(seed).choice(n, max_queries, replace=False))
    k_max = min(cand[-1] + 1, n)
    index = ToroidalNN(
        seed=seed, p=geometry.check_p(p),
        brute_threshold=n + 1 if k_max >= n else None).fit(pos, k=k_max)
    ids, dist = index.query(k=k_max, queries=pos[rows])
    ids = np.asarray(ids)
    dist = np.asarray(dist, dtype=np.float64)
    dist = np.where((ids == rows[:, None]) | (ids < 0) | (dist < buffer), np.inf, dist)
    order = np.argsort(dist, axis=1, kind="stable")
    ids = np.take_along_axis(ids, order, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)
    usable = np.isfinite(dist)
    with np.errstate(divide="ignore"):
        w = np.where(usable, np.maximum(dist, 1e-12) ** (-float(power)), 0.0)
    f_near = np.where(usable, val[np.where(ids < 0, 0, ids)], 0.0)
    num = np.cumsum(w * f_near, axis=1)
    den = np.cumsum(w, axis=1)
    out = {}
    for k in (cand if ks is None else sorted({min(max(int(k), 1), n - 1) for k in wanted})):
        col = min(k, dist.shape[1]) - 1
        est = np.where(den[:, col] > 0.0, num[:, col] / np.maximum(den[:, col], 1e-300),
                       float(val.mean()))
        out[k] = _rank_corr(est, val[rows])
    return out


def choose_k(positions, values, power: float = 2.0, p: float = 1.0,
             seed: int | None = 0) -> tuple[int, float]:
    r"""Neighbour count of the field for these sources, and how far to trust it.

    The count is the smallest candidate whose leave-one-out score
    (`leave_one_out`) is within `K_TOLERANCE` of the best: a local average
    where it is as good as a wide one (low dimension), every source where
    only the wide one keeps an order (high dimension, where the nearest
    sources are barely nearer than the rest). "Every source" is returned as
    the number of sources.

    The trust is the best score, clipped to $[0, 1]$: 0 for a field that
    predicts nothing (a deceptive or flat landscape), so a caller can scale
    its guide by it.

    The leave-one-out is buffered by the fair-share radius of the sources
    (`geometry.radius_for_count` with a target of one), the distance at
    which a point placed in an empty region reads the field.

    Returns:
        tuple[int, float]: `(k, trust)`.
    """
    val = np.asarray(values, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    buffer = 0.0
    if val.shape[0] >= 3 and pos.ndim == 2:
        buffer = geometry.radius_for_count(pos.shape[1], val.shape[0], target=1, p=p)
    scores = leave_one_out(pos, val, power=power, p=p, seed=seed, buffer=buffer)
    if not scores:
        return max(1, val.shape[0]), 0.0
    best = max(scores.values())
    k = min(k for k, s in scores.items() if s >= best - K_TOLERANCE)
    if k >= val.shape[0] - 1:
        k = val.shape[0]
    return int(k), float(min(max(best, 0.0), 1.0))
