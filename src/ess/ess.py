r"""Core ESA/ESS logic on the flat torus, powered by torann.

The simulation runs on the unit torus $[0, 1)^d$ under the toroidal L1
metric: opposite faces of the domain are identified, so there is no
boundary. This removes the wall-repulsion machinery entirely — the two
historic edge artifacts (pile-up against hard clipping, tuning of soft
walls) cannot occur in a space that has no walls. The position update is
simply

$$ x_{t+1} = (x_t + \eta_t \, F(x_t)) \bmod 1 $$

Neighbour search is delegated to `torann.ToroidalNN`, which speaks this
geometry natively (exact brute force at small $n$, LSH above its
threshold) and whose two-tier lifecycle (static anchors + moving
candidates) matches the ESA batch loop one to one.

Note:
    Because the domain is periodic, the scaled minimum and maximum of
    each dimension meet: a point at $0$ and a point at $1-\epsilon$ are
    close. For space-filling designs this is the intended behaviour —
    it is what makes the relaxation seamless — but it is the one
    semantic difference from the old bounded-box implementation.
"""

import collections.abc
import logging
import math
import statistics
import time

import numpy as np
from torann import ToroidalNN

from . import samplers

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
"""logging.Logger: Module-level logger for debugging ESA optimization steps."""

STEP_CAP = 0.02
r"""float: Per-epoch travel budget, as a fraction of the interaction radius.

Displacement is capped in the **L1** metric the radius is expressed in,
so a point never crosses more than 2% of its local spacing per epoch.
Early on the cap binds, which turns the update into a constant-speed
flow along the force direction — only the *direction* matters, which is
why `lr` barely affects the outcome; as the learning rate decays the
step falls below the cap and the run anneals into place.

Two failure modes this avoids, both measured at $d = 32$: capping the
L2 norm instead lets each coordinate move by $R/\sqrt{d}$ rather than
$R/d$ (a quarter of the domain per epoch, toroidal Clark-Evans 1.28 ->
1.06), and a loose cap of 0.25 leaves every step maximal, which never
settles (1.14 versus 1.30 here).
"""

NEIGHBOUR_TARGET = 2
r"""int: Neighbours the default interaction radius should contain.

The smallest radius that costs nothing. k-NN mode is insensitive to it
above 2 (it only normalises the force there — measured 1.469 vs 1.477
at $d=8$, 1.299 vs 1.296 at $d=32$ for targets 2 and 3), while 1 is too
tight in low dimension (2D toroidal Clark-Evans 1.86 versus 2.08).
Radius mode, where it is the actual search cutoff, keeps improving with
larger values at proportionally higher cost, so callers who want that
should pass ``radius=`` (or ``k=``) explicitly.
"""

K_LOCAL = 5
r"""int: Cap on the number of interacting neighbours per point.

Repulsion is a *local* effect: the nearest shell sets the packing, and
further neighbours only add an isotropic mean-field pressure that moves
points without improving separation. The historic default $k = 2d+1$
grows with dimension and crosses into that regime badly — at $d = 64$
with $n = 512$ it makes every point interact with a quarter of the whole
design, which collapses the one-dimensional marginals (projection
discrepancy $105\times$ worse than random) *and* leaves toroidal
packing below random.

Measured over $d \in \{2, \dots, 64\}$, a cap of 5 is best or tied
everywhere: it maximises toroidal Clark-Evans at every dimension above 8
while keeping every projection better than random. See
``examples/benchmark_dispersion.py``.
"""


# --- Force Functions -------------------------------------------------------
#
# Every force law is expressed in log-space over the *normalised* distance
# $\hat{d} = d_{L1} / R$, where $R$ is the interaction radius (heuristic or
# user-provided). Normalisation makes the laws dimension-free: $\hat{d} = 1$
# always means "at the interaction radius", in 2 dimensions or in 200, so
# the old per-law, per-dimension parameter scaling is gone.


def gaussian_force(
    d: np.ndarray, sigma: float = 0.5, alpha: float = 5.0, **kwargs
) -> np.ndarray:
    r"""Gaussian repulsion in log-space over the normalised distance.

    $$ \log F(\hat{d}) = \log \alpha - \frac{\hat{d}^2}{2\sigma^2} $$

    The defaults are calibrated so the force is $O(1)$ at the interaction
    radius: with $\sigma = 1/2$, $F(1) = \alpha e^{-2} \approx 0.14\alpha
    \approx 0.7$ — strong enough that a step $\eta F$ moves a point a few
    percent of $R$ per epoch, matching the other laws.

    Args:
        d (np.ndarray): Normalised distances $\hat{d} = d_{L1}/R$.
        sigma (float): Spread $\sigma$ in units of the radius.
        alpha (float): Maximum force magnitude $\alpha$ (at $\hat{d}=0$).

    Returns:
        np.ndarray: Log-force magnitudes.
    """
    return np.log(alpha) - (d * d) / (2.0 * sigma * sigma)


def softened_inverse_force(
    d: np.ndarray, epsilon: float = 0.5, alpha: float = 1.0, power: float = 2.0,
    **kwargs,
) -> np.ndarray:
    r"""Softened inverse-power repulsion in log-space (the default law).

    $$ \log F(\hat{d}) = \log \alpha
       - \frac{p}{2} \log(\hat{d}^2 + \epsilon^2) $$

    The magnitude decays as $\hat{d}^{-p}$; the softening $\epsilon$
    bounds the force at $\hat{d} = 0$ to $\alpha\,\epsilon^{-p}$. The old
    dimension-dependent exponent $\max(2, D-1)$ is gone: normalising by
    the interaction radius already absorbs the dimensional scale, so a
    fixed $p = 2$ behaves consistently across dimensions. With
    $\alpha = 1$ the force is $\approx 0.8$ at the interaction radius,
    so the default step $\eta F$ is a meaningful fraction of the local
    spacing.

    The softening also sets the *ceiling*: $\epsilon = 1/2$ caps the
    force at $\alpha\epsilon^{-p} = 4$, in line with the other laws. A
    much smaller $\epsilon$ (0.1 caps at 100) makes close pairs fire
    steps that saturate the per-epoch displacement cap, so points move
    randomly instead of settling — in 2D that alone drops the toroidal
    Clark-Evans index from 1.94 to 1.01.

    Args:
        d (np.ndarray): Normalised distances $\hat{d} = d_{L1}/R$.
        epsilon (float): Softening $\epsilon$ (prevents infinities).
        alpha (float): Magnitude scale $\alpha$.
        power (float): Decay exponent $p$.

    Returns:
        np.ndarray: Log-force magnitudes.
    """
    return np.log(alpha) - 0.5 * power * np.log((d * d) + (epsilon * epsilon))


def linear_force(
    d: np.ndarray, alpha: float = 4.0, eps: float = 1e-9, **kwargs
) -> np.ndarray:
    r"""Linear (triangular) repulsion in log-space with a hard cutoff.

    $$ \log F(\hat{d}) = \log \alpha + \log \max(\epsilon,\; 1 - \hat{d}) $$

    The force falls to zero exactly at the interaction radius
    ($\hat{d} = 1$); beyond it only $\epsilon$ remains, so far neighbours
    contribute nothing. $\alpha$ lifts the ramp so the typical force
    (around $\hat{d} \approx 3/4$) is $O(1)$, in line with the other laws.

    Args:
        d (np.ndarray): Normalised distances $\hat{d} = d_{L1}/R$.
        alpha (float): Magnitude scale $\alpha$ (force at $\hat{d} = 0$).
        eps (float): Floor $\epsilon$ that keeps the logarithm finite.

    Returns:
        np.ndarray: Log-force magnitudes.
    """
    return np.log(alpha) + np.log(np.maximum(eps, 1.0 - d))


def cauchy_force(
    d: np.ndarray, alpha: float = 2.0, power: float = 2.0, **kwargs
) -> np.ndarray:
    r"""Long-tailed Cauchy repulsion in log-space.

    $$ \log F(\hat{d}) = \log \alpha - \frac{p}{2} \log(1 + \hat{d}^2) $$

    Finite at zero ($F(0) = \alpha$), heavy-tailed at range ($F(1) =
    \alpha/2$) — useful when far neighbours should keep contributing
    (global untangling), at the price of slower local convergence.

    Args:
        d (np.ndarray): Normalised distances $\hat{d} = d_{L1}/R$.
        alpha (float): Magnitude scale $\alpha$ (force at $\hat{d} = 0$).
        power (float): Decay exponent $p$.

    Returns:
        np.ndarray: Log-force magnitudes.
    """
    return np.log(alpha) - 0.5 * power * np.log(1.0 + (d * d))


METRIC_REGISTRY = {
    "gaussian": gaussian_force,
    "softened_inverse": softened_inverse_force,
    "linear": linear_force,
    "cauchy": cauchy_force,
}


# --- Helpers ----------------------------------------------------------------
def _scale(
    arr: np.ndarray,
    min_val: np.ndarray | np.number | float | int | None = None,
    max_val: np.ndarray | np.number | float | int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | np.number | float | int,
    np.ndarray | np.number | float | int,
]:
    r"""Normalizes the input array to the unit hypercube $[0, 1]^D$.

    Min-max scaling is performed column-wise (per dimension). If explicit
    bounds are not provided, they are inferred from the data:

    $$ x' = \frac{x - x_{min}}{x_{max} - x_{min}} $$

    Constant dimensions ($x_{max} = x_{min}$) use a denominator of 1.0 to
    avoid division by zero.

    Args:
        arr (np.ndarray): Input data array of shape $(N, D)$.
        min_val (np.ndarray | np.number | None): Optional pre-computed
            minimum values. If None, computed from `arr`.
        max_val (np.ndarray | np.number | None): Optional pre-computed
            maximum values. If None, computed from `arr`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - **scaled_arr**: The normalized data in range $[0, 1]$.
            - **min_val**: The minimum values used for scaling.
            - **max_val**: The maximum values used for scaling.
    """
    used_min_val = np.min(arr, axis=0) if min_val is None else min_val
    used_max_val = np.max(arr, axis=0) if max_val is None else max_val

    denom = used_max_val - used_min_val
    denom = np.where(denom == 0, 1.0, denom)

    return (arr - used_min_val) / denom, used_min_val, used_max_val


def _inv_scale(
    scl_arr: np.ndarray,
    min_val: np.ndarray | np.number | float | int,
    max_val: np.ndarray | np.number | float | int,
) -> np.ndarray:
    r"""Restores scaled data from $[0, 1)^D$ back to its original domain.

    $$ x = x' \cdot (x_{max} - x_{min}) + x_{min} $$

    Args:
        scl_arr (np.ndarray): Scaled input array in $[0, 1)$.
        min_val (np.ndarray | np.number): Minimum values of the original domain.
        max_val (np.ndarray | np.number): Maximum values of the original domain.

    Returns:
        np.ndarray: The array projected back into the original bounds.
    """
    return scl_arr * (max_val - min_val) + min_val


def _l1_radius_heuristic(
    dim: int, n_points: int, target: int = NEIGHBOUR_TARGET
) -> float:
    r"""Radius expected to contain `target` neighbours, from the exact
    toroidal L1 distance law.

    Per dimension the toroidal distance between two uniform points is
    $u = \min(\delta,\, 1-\delta)$ with $\delta$ uniform, so
    $u \sim \mathrm{U}(0, 1/2)$ *exactly*; the full distance is a sum of
    $d$ such terms, with mean $d/4$ and variance $d/48$. Inverting

    $$ N \cdot P(\mathrm{dist} \le R) = \text{target} $$

    gives the radius in either regime:

    * **Dense** — while $R \le 1/2$ the sum is still in its first
      Irwin-Hall piece, where $P(\mathrm{dist} \le R) = (2R)^d/d!$ is
      exactly the L1 ball volume, so
      $R = \tfrac{1}{2}\,(\text{target} \cdot d!/N)^{1/d}$.
    * **Sparse** — beyond that the ball volume exceeds the torus and the
      formula is meaningless, but the central limit theorem applies:
      $R = d/4 + z_{\text{target}/N}\,\sqrt{d/48}$.

    Note:
        The previous version multiplied the packing radius by a fixed
        1.25 "safety margin". A margin on the *radius* is a margin of
        $1.25^d$ on the *count* — 1.6 neighbours at $d=2$, but 1262 at
        $d=32$ and $1.6\times10^{6}$ at $d=64$ — so the neighbourhood
        grew without bound and had to be clamped at the mean pairwise
        distance, which is why radius mode went global (and blew up
        memory) in high dimension. Targeting the count directly is what
        keeps it local at every $d$.

    Args:
        dim (int): Dimensionality $d$.
        n_points (int): Total number of points $N$ (static + generated).
        target (int): Desired neighbours inside the ball. Defaults to
            `NEIGHBOUR_TARGET` — the smallest radius that does not cost
            quality; callers wanting a wider interaction pass ``radius``
            to `esa` directly.

    Returns:
        float: The interaction radius in toroidal L1 units.
    """
    n = max(n_points, 2)
    count = max(min(target, n - 1), 1)

    log_r = (math.lgamma(dim + 1) + math.log(count) - math.log(n)) / dim
    dense = 0.5 * math.exp(log_r)
    if dense <= 0.5:
        return dense  # exact: still inside the first Irwin-Hall piece

    z = statistics.NormalDist().inv_cdf(count / n)
    return min(max(dim / 4.0 + z * math.sqrt(dim / 48.0), 1e-6), dim / 2.0)


def _smart_init(
    index: ToroidalNN,
    n_new: int,
    dim: int,
    rng: np.random.Generator,
    init_sampler: samplers.Sampler,
    pool: int = 15,
) -> np.ndarray:
    r"""Initializes new points by Best Candidate Sampling against the index.

    For each of the $n$ slots, a pool of candidate positions is drawn with
    the space-filling sampler and the one farthest (toroidal L1) from every
    already-indexed point wins:

    $$ c^* = \arg\max_{c \in \text{pool}} \;
       \min_{p \in \text{index}} d_{L1}^{tor}(c, p) $$

    A small jitter $\xi \sim U(-10^{-3}, 10^{-3})$ breaks exact overlaps;
    the result is reduced mod 1 (no clipping — the torus has no edge to
    clip against).

    Args:
        index (ToroidalNN): Fitted index holding all existing points.
        n_new (int): Number of points to initialize.
        dim (int): Dimensionality of the space.
        rng (np.random.Generator): Random number generator.
        init_sampler (samplers.Sampler): Candidate-pool sampler (e.g. LHS).
        pool (int): Candidates drawn per slot.

    Returns:
        np.ndarray: Initial positions, shape $(n_{new}, D)$, in $[0, 1)$.
    """
    candidates = init_sampler.sample(n_new * pool, dim, rng).astype(np.float64)
    _, dists = index.query(k=1, queries=candidates)
    best = dists.reshape(n_new, pool).argmax(axis=1)
    picked = candidates.reshape(n_new, pool, dim)[np.arange(n_new), best]
    jitter = rng.uniform(-1e-3, 1e-3, size=picked.shape)
    return np.mod(picked + jitter, 1.0)


def _pad_ragged(
    results: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Packs per-query variable-length (ids, dists) lists into dense arrays.

    Rows are padded with ``-1`` / ``inf`` — the same missing-neighbour
    convention `ToroidalNN.query` uses — so radius-mode results feed the
    exact same force kernel as k-NN results.

    Args:
        results (list[tuple[np.ndarray, np.ndarray]]): One (ids, distances)
            pair per query, as returned by `ToroidalNN.query_radius`.

    Returns:
        tuple[np.ndarray, np.ndarray]: (ids, distances) of shape
            $(M, m_{max})$; ``m_max`` is the largest neighbourhood found
            (at least 1, so downstream shapes stay valid).
    """
    n = len(results)
    width = max((ids.shape[0] for ids, _ in results), default=0)
    width = max(width, 1)
    ids = np.full((n, width), -1, dtype=np.int64)
    dists = np.full((n, width), np.inf)
    for i, (row_ids, row_dists) in enumerate(results):
        ids[i, : row_ids.shape[0]] = row_ids
        dists[i, : row_dists.shape[0]] = row_dists
    return ids, dists


#: Elements per intermediate array in the force kernel (~64 MB in
#: float64). The kernel materialises several $(\text{rows}, m, D)$
#: tensors at once, so this is what keeps radius mode — whose neighbour
#: lists become the whole design in sparse high-dimensional regimes —
#: from allocating gigabytes.
_FORCE_BLOCK_ELEMENTS = 8_000_000

#: Floor on a coordinate gap before it is raised to $p - 1 < 0$ in the force
#: direction. Only reached by near-coincident coordinates, which are rare
#: enough that four orders of magnitude barely move the result — a guard
#: against division by zero, not a knob.
_GRAD_EPS = 1e-9


def _accumulate(timers: dict | None, key: str, started: float) -> None:
    """Adds the seconds elapsed since `started` to ``timers[key]``.

    A no-op when `timers` is None, so the instrumented call sites cost a
    single `time.perf_counter()` when statistics are not requested.

    Args:
        timers (dict | None): Accumulator, or None to disable.
        key (str): Bucket name.
        started (float): `time.perf_counter()` value from before the work.
    """
    if timers is not None:
        timers[key] = timers.get(key, 0.0) + (time.perf_counter() - started)


def _row_blocks(rows: int, width: int, dim: int):
    """Yields ``(start, stop)`` row slices with a bounded working set.

    Args:
        rows (int): Number of active points $M$.
        width (int): Neighbour-list width $m$.
        dim (int): Dimensionality $D$.

    Yields:
        tuple[int, int]: Half-open row ranges covering ``range(rows)``,
        each sized so that ``block * width * dim`` stays within
        `_FORCE_BLOCK_ELEMENTS`.
    """
    per_row = max(width * dim, 1)
    step = max(1, min(rows, _FORCE_BLOCK_ELEMENTS // per_row))
    for start in range(0, rows, step):
        yield start, min(start + step, rows)


def _compute_forces(
    active: np.ndarray,
    all_data: np.ndarray,
    ids: np.ndarray,
    dists: np.ndarray,
    radius: float,
    metric_fn: collections.abc.Callable,
    rng: np.random.Generator,
    force_p: float = 1.0,
    **metric_kwargs,
) -> np.ndarray:
    r"""Net repulsive force on each active point from its neighbour list.

    This is the single force kernel for both search modes: k-NN passes the
    dense `ToroidalNN.query` result, radius mode passes the padded output
    of `_pad_ragged`. For active point $x_i$ with neighbours $y_j$:

    $$ \vec{F}_i = \sum_{j} \frac{\vec{u}_{ij}}{\lVert\vec{u}_{ij}\rVert_2}
       \, f\!\left(\frac{d_{L1}(x_i, y_j)}{R}\right), \qquad
       u_{ij,l} \;=\; \lvert r_{ij,l}\rvert^{\,p-1}\operatorname{sign}(r_{ij,l}) $$

    where $\vec{r}_{ij}$ is the **toroidal** displacement — each component
    wrapped into $[-1/2, 1/2)$ by $r \leftarrow r - \operatorname{round}(r)$
    — so near the seam the push points the short way around, never across
    the whole domain.

    **The direction is the gradient of the metric supplying the
    magnitudes**, which is what `force_p` selects. $\vec{u}$ is
    $\nabla d_p$ up to the positive scalar $d_p^{1-p}$, which the
    normalisation removes:

    | `force_p` | direction | pushes hardest along |
    |---|---|---|
    | 2 | $r_l$ | the coordinates that already differ most |
    | 1 | $\operatorname{sign}(r_l)$ | every coordinate equally |
    | <1 | $\lvert r_l\rvert^{p-1}\operatorname{sign}(r_l)$ | the coordinates that nearly **coincide** |

    Until this was parameterised the kernel always used $r_l/\lVert r
    \rVert_2$ — the $p = 2$ gradient — while `dists` carried toroidal
    **L1** magnitudes, so the applied force was the gradient of no
    potential ESS defines. The default is now $1$, matching the metric.
    Passing ``force_p=2.0`` restores the historic behaviour exactly and is
    a regression anchor, not a recommended mode: every ESS quality figure
    recorded before 2026-07-28 was measured with it.

    Two conventions at $r_l = 0$, where $\lvert r_l\rvert^{p-1}
    \operatorname{sign}(r_l)$ jumps from $-\infty$ to $+\infty$. An
    *exactly* shared coordinate contributes nothing, since
    $\operatorname{sign}(0) = 0$ — the symmetric subgradient, and the only
    choice that does not invent a direction. A *nearly* shared one gets the
    largest push, bounded by `_GRAD_EPS`. Exact ties have measure zero
    here, so the second case is the one that occurs.

    Numerics: $f$ is evaluated in log-space, the per-point maximum
    $M_i$ is subtracted before exponentiation (log-sum-exp trick), and the
    restored scale $e^{M_i}$ is capped at $10^3$. Exactly coincident
    neighbours ($\lVert\vec{r}\rVert < 10^{-9}$) get a random unit
    direction to break the tie.

    Args:
        active (np.ndarray): Active positions, shape $(M, D)$, in $[0, 1)$.
        all_data (np.ndarray): All positions (static + active), indexed by
            the global ids in `ids`.
        ids (np.ndarray): Neighbour ids, shape $(M, m)$; ``-1`` = missing.
        dists (np.ndarray): Toroidal L1 distances, shape $(M, m)$;
            ``inf`` = missing.
        radius (float): Interaction radius $R$ used to normalise distances.
        metric_fn (Callable): Log-space force law $\log f(\hat{d})$.
        rng (np.random.Generator): Generator for tie-breaking noise.
        force_p (float): Exponent whose gradient sets the push direction.
            Must match the metric that produced `dists` for the force to be
            a gradient; ``1.0`` is toroidal L1 and is the default, ``2.0``
            reproduces the historic direction.
        **metric_kwargs: Extra arguments for `metric_fn`.

    Returns:
        np.ndarray: Net force vectors, shape $(M, D)$.
    """
    if not np.any(ids >= 0):
        return np.zeros_like(active)

    forces = np.empty_like(active)
    for lo, hi in _row_blocks(active.shape[0], ids.shape[1], active.shape[1]):
        block_ids, block_dists = ids[lo:hi], dists[lo:hi]
        valid = block_ids >= 0

        safe_ids = np.where(valid, block_ids, 0)
        disp = active[lo:hi, None, :] - all_data[safe_ids]
        disp -= np.round(disp)  # toroidal wrap: shortest displacement per axis
        norms = np.linalg.norm(disp, axis=2, keepdims=True)

        stacked = (norms[..., 0] < 1e-9) & valid
        if np.any(stacked):
            noise = rng.standard_normal(size=disp.shape)
            noise /= np.linalg.norm(noise, axis=2, keepdims=True) + 1e-9
            disp = np.where(stacked[..., None], noise, disp)
            norms = np.where(stacked[..., None], 1.0, norms)

        # grad(d_p) up to a positive scalar. p=2 is the displacement itself,
        # so that branch runs the arithmetic the kernel has always run and
        # stays bit-identical; the rng draw above is unconditional, so seeds
        # hit the same code path whichever direction is selected.
        if force_p == 2.0:
            grad = disp
        elif force_p == 1.0:
            grad = np.sign(disp)
            norms = np.linalg.norm(grad, axis=2, keepdims=True)
        else:
            grad = np.abs(disp)
            np.maximum(grad, _GRAD_EPS, out=grad)
            np.power(grad, force_p - 1.0, out=grad)
            grad *= np.sign(disp)
            norms = np.linalg.norm(grad, axis=2, keepdims=True)

        d_hat = np.where(valid, block_dists, 1.0) / radius
        log_mag = metric_fn(d_hat, **metric_kwargs)
        log_mag = np.where(valid, log_mag, -np.inf)

        m_i = np.max(log_mag, axis=1, keepdims=True)
        m_i = np.where(np.isneginf(m_i), 0.0, m_i)
        weights = np.exp(log_mag - m_i)
        weights[~valid] = 0.0

        directions = grad / np.maximum(norms, 1e-9)
        net = np.sum(directions * weights[..., None], axis=1)

        force_cap = 1000.0
        forces[lo:hi] = np.exp(np.minimum(m_i, np.log(force_cap))) * net

    return forces


# --- Core Logic --------------------------------------------------------------
def _esa(
    samples01: np.ndarray,
    index: ToroidalNN,
    *,
    n: int,
    dim: int,
    epochs: int,
    lr: float,
    decay: float,
    batch_size: int,
    k: int,
    radius: float,
    search_mode: str,
    tol: float,
    patience: int,
    metric_fn: collections.abc.Callable,
    rng: np.random.Generator,
    init_sampler: samplers.Sampler,
    force_p: float = 1.0,
    stats: dict | None = None,
    **metric_kwargs,
) -> np.ndarray:
    r"""Executes the ESA optimization loop on the unit torus.

    **Per batch:**

    1. Initialize positions (`_smart_init` once the index has points,
       the raw sampler for the very first from-scratch batch).
    2. For each epoch $t$: query neighbours, compute forces, step

       $$ x_{t+1} = (x_t + \eta_t \vec{F}_t) \bmod 1, \qquad
          \eta_{t+1} = \gamma \eta_t $$

       (per-point steps are norm-capped at $1/4$ so a force spike can
       never wrap a point across the torus), then `ToroidalNN.update`.
    3. On convergence, `ToroidalNN.promote` freezes the batch into the
       static tier and installs the next one.

    **Early stopping** is learning-rate-decoupled, so the decay schedule
    cannot fake convergence. The monitored signal is the largest force
    magnitude $\max_i \lVert \vec{F}_i \rVert$, smoothed by an EMA
    ($\beta = 1/2$). The loop stops when the signal *plateaus*: no
    relative improvement of at least 1% over its best value for
    `patience` consecutive epochs — i.e. when the physics has stopped
    settling, at whatever force level the packing frustration allows.
    Two additional guards: the absolute floor `tol` (forces genuinely
    vanished — isolated points), and the annealing floor
    $\eta_t \cdot \text{EMA} < 10^{-9}$ (steps too small to matter).
    Benchmarked over $d \in \{2, \dots, 64\}$
    (``examples/benchmark_dispersion.py``), the plateau fires after
    8-38 epochs per batch where pure annealing would grind on for 300+,
    at equal Clark-Evans quality.

    Args:
        samples01 (np.ndarray): Static points already scaled to $[0, 1)$.
        index (ToroidalNN): The (unfitted) neighbour index to drive.
        n (int): Number of points to generate.
        dim (int): Dimensionality of the space.
        epochs (int): Maximum update steps per batch.
        lr (float): Initial step size $\eta_0$.
        decay (float): Learning-rate decay $\gamma$ per epoch.
        batch_size (int): Points optimized together per batch.
        k (int): Neighbours per query (k-NN mode).
        radius (float): Interaction radius $R$ (search cutoff in radius
            mode; force normalisation scale in both modes).
        search_mode (str): ``"k_nn"`` or ``"radius"``.
        tol (float): Absolute convergence floor on the force EMA.
        patience (int): Consecutive non-improving epochs (< 1% relative)
            before the plateau stop fires.
        metric_fn (Callable): Log-space force law.
        rng (np.random.Generator): Random number generator.
        init_sampler (samplers.Sampler): Sampler for initial positions.
        force_p (float): Exponent whose gradient sets the push direction;
            see `_compute_forces`.
        stats (dict | None): Optional run-statistics sink; see `esa`.
        **metric_kwargs: Extra arguments for `metric_fn`.

    Returns:
        np.ndarray: Generated points in $[0, 1)$, shape $(n, D)$.
    """
    n_static = samples01.shape[0]
    all_data = np.empty((n_static + n, dim))
    all_data[:n_static] = np.mod(samples01, 1.0)
    cursor = n_static

    radius_hint = radius if search_mode == "radius" else None
    fitted = n_static > 0
    if fitted:
        index.fit(all_data[:n_static], k=k, radius=radius_hint)

    num_batches = math.ceil(n / batch_size)
    logger.debug(
        "Starting ESA: %d points, %d batches, mode=%s, R=%.4f",
        n, num_batches, search_mode, radius,
    )

    timers: dict | None = {} if stats is not None else None

    for _ in range(num_batches):
        current_n = min(batch_size, n_static + n - cursor)
        if current_n <= 0:
            break

        started = time.perf_counter()
        if fitted:
            init = _smart_init(index, current_n, dim, rng, init_sampler)
            index.promote(init)
        else:
            # From scratch: nothing to anchor against — the first batch
            # starts straight from the space-filling sampler.
            init = np.mod(
                init_sampler.sample(current_n, dim, rng).astype(np.float64), 1.0
            )
            index.fit(np.empty((0, dim)), init, k=k, radius=radius_hint)
            fitted = True
        _accumulate(timers, "setup_s", started)

        all_data[cursor : cursor + current_n] = init
        active = all_data[cursor : cursor + current_n]  # view into the buffer

        current_lr = lr
        ema = None
        best_ema = np.inf
        rel_improve = 0.01
        calm_streak = 0
        epochs_used = 0
        for epochs_used in range(1, epochs + 1):
            started = time.perf_counter()
            if search_mode == "radius":
                ids, dists = _pad_ragged(index.query_radius(radius))
            else:
                ids, dists = index.query(k=k)
            _accumulate(timers, "query_s", started)

            started = time.perf_counter()
            forces = _compute_forces(
                active, all_data, ids, dists, radius, metric_fn, rng,
                force_p=force_p, **metric_kwargs,
            )
            _accumulate(timers, "force_s", started)

            started = time.perf_counter()

            # Steps are measured in units of the interaction radius, so
            # stability does not depend on n or d: the forces are already
            # dimensionless (they see d/R), and a step of lr*R is the same
            # fraction of the local spacing at every density.
            step = forces * (current_lr * radius)
            # Cap travel in the metric the radius is expressed in: an L1
            # budget of a quarter of the local spacing. Capping the L2
            # norm instead lets each coordinate move by R/sqrt(d) rather
            # than R/d, which in 32 dimensions is a quarter of the domain
            # per epoch and destroys the packing.
            step_norm = np.abs(step).sum(axis=1, keepdims=True)
            np.multiply(
                step,
                np.minimum(1.0, STEP_CAP * radius / np.maximum(step_norm, 1e-12)),
                out=step,
            )
            active += step
            np.mod(active, 1.0, out=active)
            _accumulate(timers, "step_s", started)

            started = time.perf_counter()
            index.update(active)
            _accumulate(timers, "update_s", started)

            f_max = float(np.max(np.linalg.norm(forces, axis=1)))
            ema = f_max if ema is None else 0.5 * ema + 0.5 * f_max
            if ema < best_ema * (1.0 - rel_improve):
                best_ema = ema
                calm_streak = 0
            else:
                calm_streak += 1
            if ema < tol or calm_streak >= patience:
                break  # converged: forces vanished, or stopped improving
            if current_lr * ema < 1e-9:
                break  # annealing floor: steps too small to matter
            current_lr *= decay

        logger.debug(
            "Batch [%d:%d] stopped after %d/%d epochs (force EMA %.4g)",
            cursor, cursor + current_n, epochs_used, epochs, ema or 0.0,
        )
        if stats is not None:
            stats.setdefault("batch_epochs", []).append(epochs_used)
            stats.setdefault("batch_force_ema", []).append(
                float(ema) if ema is not None else 0.0
            )
        cursor += current_n

    if stats is not None:
        stats["radius"] = radius
        stats["epochs_total"] = int(np.sum(stats.get("batch_epochs", [0])))
        stats.update(timers or {})

    return all_data[n_static:cursor]


def esa(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    n: int,
    index: ToroidalNN | None = None,
    epochs: int = 1024,
    lr: float = 0.5,
    search_mode: str = "k_nn",
    decay: float = 0.99,
    batch_size: int | None = None,
    k: int | None = None,
    radius: float | None = None,
    tol: float = 1e-2,
    patience: int = 25,
    metric: str | collections.abc.Callable = "gaussian",
    seed: int | np.random.Generator | None = None,
    init_sampler: samplers.Sampler | int | None = None,
    force_p: float = 1.0,
    stats: dict | None = None,
    **metric_kwargs,
) -> np.ndarray:
    r"""Empty Space Algorithm (ESA): returns only the generated points.

    Public API for the toroidal relaxation. It scales the domain to the
    unit torus $[0, 1)^d$, derives the interaction radius, runs `_esa`,
    and maps the result back:

    $$ R = \min\!\left(\tfrac{5}{8}\,(d!/N)^{1/d},\; d/4\right)
       \quad \text{(when not given; see `_l1_radius_heuristic`)} $$

    The same $R$ is the range cutoff in radius mode and the distance
    normalisation of every force law in both modes, which is what keeps
    force parameters dimension-free.

    The defaults are calibrated by ``examples/benchmark_dispersion.py``
    over $d \in \{2, \dots, 64\}$. Two of them dominate the achieved
    quality and interact, so they must be read together:

    * ``batch_size=None`` relaxes every point at once. Freezing points
      in batches is greedy — an early batch settles into an arrangement
      optimal for *itself*, then never moves while the rest squeeze into
      its gaps — and costs about a third of the achievable dispersion.
    * ``patience=25`` gives the plateau detector enough evidence.
      Stopping sooner is what makes single-batch relaxation look bad:
      with one batch there is no per-batch restart of the annealing, so
      a hair-trigger stop ends the run before it has done anything.

    Together, in 2D at $n = 256$, they take the toroidal Clark-Evans
    index from 1.71 to 2.08 against a theoretical ceiling of
    $2/\Gamma(1+1/d) = 2.257$ — 92% of optimal — and improve the worst
    pair separation by $6.7\times$. ``lr`` and ``decay`` are
    comparatively inert once the stop criterion is right.

    Args:
        samples (np.ndarray): Existing points to avoid, shape $(N_0, D)$.
            May be empty.
        bounds (np.ndarray): Domain boundaries, shape $(D, 2)$.
        n (int): Number of new points to create.
        index (ToroidalNN | None): Optional pre-configured index (e.g. a
            specific backend or LSH parameters). It is re-fitted; when
            None a default `ToroidalNN` is created. Exact vs LSH search
            is the index's own size-based decision — there are no engine
            thresholds left in ESS.
        epochs (int): Maximum iterations per batch.
        lr (float): Initial step size $\eta_0$, **in units of the
            interaction radius**. Largely inert: `STEP_CAP` binds for
            most of the run, so it mainly sets when annealing takes
            over (0.2 and 0.5 differ by <2% on every benchmark cell).
        search_mode (str): ``"k_nn"`` (rank-based neighbourhood) or
            ``"radius"`` (metric ball).
        decay (float): Learning-rate decay $\gamma$ per epoch.
        batch_size (int | None): Points optimized together. ``None``
            (default) relaxes **all** $n$ points simultaneously, which
            is what the toroidal index makes affordable. A smaller value
            processes the points in sequential batches, each frozen once
            converged — a greedy approximation.

            Batching is *not* the way to make large runs fast. Measured
            at $d = 8$ from scratch, wall time and toroidal Clark-Evans:

            | $n$ | all at once | batch 5000 | batch 50 |
            | --- | --- | --- | --- |
            | 10 000 | 11.5 s / 1.487 | — | 5.3 s / 1.352 |
            | 40 000 | 113 s / 1.491 | 94 s / 1.396 | > 750 s |

            Batching saves at most ~20% at $n = 40\,000$ while giving up
            most of the quality, and small batches become *pathological*
            — 800 sequential batches, each paying full query cost
            against an index that keeps growing. Cost is superlinear
            (~$n^{1.6}$) either way, dominated by index and query work
            rather than by the batch structure. Set this only to bound
            the per-epoch working set ($O(\text{batch} \cdot k \cdot D)$)
            when memory, not time, is the constraint.
        k (int | None): Neighbours in k-NN mode; default
            $\min(2D + 1, \text{`K_LOCAL`})$.
        radius (float | None): Interaction radius; default heuristic.
        tol (float): Absolute early-stop floor on the EMA of the largest
            force magnitude (fires only when forces genuinely vanish;
            the working criterion is the plateau — see `_esa`).
        patience (int): Consecutive epochs without a 1% relative
            improvement of the force EMA before the batch is declared
            converged. Quality saturates at 25-50; below ~10 the
            detector fires on a transient and the run stops early.
        metric (str | Callable): Force-law name in `METRIC_REGISTRY`, or
            a callable $\log f(\hat{d})$.
        seed (int | np.random.Generator | None): Seed or Generator.
        init_sampler (samplers.Sampler | int | None): Initial-position
            sampler; None = LHS.
        force_p (float): Exponent whose gradient sets the push direction.
            The default ``1.0`` is the gradient of toroidal L1, the metric
            the distances are measured in, so the force is the gradient of
            the potential ESS actually defines. ``2.0`` restores the
            pre-2026-07-28 behaviour — the kernel pushed along
            $\delta/\lVert\delta\rVert_2$ whatever metric supplied the
            magnitudes — and exists as a regression anchor for reproducing
            figures recorded under it. See `_compute_forces`.
        stats (dict | None): Optional dictionary filled in place with run
            statistics — ``batch_epochs`` (epochs used per batch),
            ``batch_force_ema`` (final force EMA per batch),
            ``epochs_total``, ``radius``, and a wall-clock decomposition
            in seconds: ``query_s`` (neighbour search), ``force_s`` (the
            force kernel), ``step_s`` (position update), ``update_s``
            (re-indexing the moved points) and ``setup_s``
            (initialisation plus promote/fit). The buckets are what turn
            "the large runs are slow" into an attributable cost, and
            they cover the whole inner loop, so they should sum to just
            under the total. Passing None disables the timers entirely.
        **metric_kwargs: Extra arguments for the force law.

    Returns:
        np.ndarray: The generated points, shape $(n, D)$, in the original
        coordinate system.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        samples = samples.reshape(-1, bounds.shape[0])

    if isinstance(metric, str):
        metric_fn = METRIC_REGISTRY.get(metric.lower())
        if metric_fn is None:
            raise ValueError(f"Unknown metric '{metric}'")
    else:
        metric_fn = metric

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    dim = bounds.shape[0]
    min_val = bounds[:, 0]
    max_val = bounds[:, 1]
    scaled_samples, _, _ = _scale(samples, min_val, max_val)

    k_value = k if k is not None else min(2 * dim + 1, K_LOCAL)
    batch = batch_size if batch_size is not None else max(n, 1)
    final_radius = (
        radius if radius is not None
        else _l1_radius_heuristic(dim, samples.shape[0] + n)
    )
    logger.debug("Interaction radius (toroidal L1): %.4f", final_radius)

    if index is None:
        index = ToroidalNN(seed=int(rng.integers(2**31)))

    generated = _esa(
        scaled_samples,
        index,
        n=n,
        dim=dim,
        epochs=epochs,
        lr=lr,
        decay=decay,
        batch_size=batch,
        k=k_value,
        radius=final_radius,
        search_mode=search_mode,
        tol=tol,
        patience=patience,
        metric_fn=metric_fn,
        rng=rng,
        init_sampler=samplers.check_sampler(init_sampler, default_random_state=rng),
        force_p=float(force_p),
        stats=stats,
        **metric_kwargs,
    )
    return _inv_scale(generated, min_val, max_val)


def ess(
    samples: np.ndarray | list,
    bounds: np.ndarray,
    *,
    n: int,
    **kwargs,
) -> np.ndarray:
    r"""Empty Space Strategy (ESS): returns the combined data set.

    Convenience wrapper that runs `esa` and concatenates the result with
    the original samples:

    $$ \text{Result} = \text{samples} \cup \text{ESA}(\text{samples}, \dots) $$

    Args:
        samples (np.ndarray | list): Existing points, shape $(N_0, D)$.
        bounds (np.ndarray): Domain boundaries, shape $(D, 2)$.
        n (int): Number of new points to generate.
        **kwargs: Forwarded verbatim to `esa` (epochs, lr, search_mode,
            metric, tol, patience, index, seed, ...).

    Returns:
        np.ndarray: Array of shape $(N_0 + n, D)$ with original and new
        points.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        samples = samples.reshape(-1, bounds.shape[0])

    new_points = esa(samples, bounds, n=n, **kwargs)
    return np.concatenate((samples, new_points), axis=0)
