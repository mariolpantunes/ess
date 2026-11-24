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
import inspect
import logging
import math
import time

import numpy as np
from torann import ToroidalNN
from torann import rust as _torann_rust

from . import attraction, geometry, samplers

logger = logging.getLogger(__name__)
"""logging.Logger: Module-level logger for debugging ESA optimization steps."""

PLACEMENT_ROUND = 32
r"""int: Slots placed per round of the best-candidate initialisation.

A slot must be far from the slots placed before it, not only from what
was indexed when the batch began (see `_smart_init`). Inside a round that
is an exact running minimum, quadratic in the round; between rounds it is
the index, which costs a `promote`. Measured at 16 to 256 on clustered
static data ($n = 20000$ at $d = 2, 5$; $n = 5000$ at $d = 16$) the
placement is the same quality at every size and cheapest here: at 256 the
running minimum costs more than the neighbour queries it sits beside, and
below 32 the per-round overhead starts to show.
"""

GAP_QUANTILE = 0.05
r"""float: Quantile of the nearest-neighbour distance the stop rule watches.

Each epoch the neighbour query already returns, for every point being
relaxed, the distance to its nearest other point. The relaxation exists to
raise the low end of that distribution, so the stop rule watches its 5%
quantile: the minimum is one point and too noisy to wait on, the median
saturates while the closest pairs are still separating.
"""

GAP_IMPROVE = 0.01
r"""float: Relative gain of the watched gap that counts as progress.

A batch is settled when the `GAP_QUANTILE` gap has not grown by 1% over
its best value (and, when guided, the mean field value has not dropped by
its spread-scaled `ATT_IMPROVE`) for `patience` consecutive epochs.
"""

ATT_IMPROVE = 0.005
"""float: Drop of the mean estimated fitness (in units of the batch spread) that counts as progress."""

_RANK_STD = 12.0 ** -0.5
"""float: Standard deviation of a rank spread evenly over 0 to 1."""

ATT_EVERY = 5
"""int: Epochs between evaluations of the fitness terrain guide during relaxation."""

PLACEMENT_WEIGHT_CAP = 0.5
"""float: Maximum fitness weight applied during best-candidate placement so placement never over-clusters."""

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
# $\hat{d} = d / R$ (toroidal $L^p$, $L^1$ by default), where $R$ is the interaction radius (heuristic or
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
        d (np.ndarray): Normalised distances $\hat{d} = d/R$.
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
        d (np.ndarray): Normalised distances $\hat{d} = d/R$.
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
        d (np.ndarray): Normalised distances $\hat{d} = d/R$.
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
        d (np.ndarray): Normalised distances $\hat{d} = d/R$.
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


def _rank_normalise(values: np.ndarray) -> np.ndarray:
    r"""Fitness (elevation) to $[0, 1]$ by rank, lowest = 0 (ground), highest = 1 (uphill).

    **The terrain must never see the caller's raw units.** Objective values
    span orders of magnitude between problems and between regions of one
    problem, so a `fitness_weight` calibrated on a landscape scoring
    $10^{-3}$ means something else entirely on one scoring $10^{6}$ — the
    balance point would be set by the objective rather than by the knob.

    That exact defect has been paid for once already, in an optimizer that
    blended fitness and diversity through a Boltzmann softmax over raw
    scores: the blend point moved with the objective's scale, so the weight
    was inert on one pool and saturated on another, and two axes of a
    399-arm factorial measured nothing.

    Ranks are used rather than z-scores because they are bounded and
    outlier-proof: one catastrophic sample cannot compress every other point
    into a corner of the range. Ties share the mean rank, so equally good
    points sit at equal elevation. A constant vector maps to all zeros
    (flat ground).

    Args:
        values (np.ndarray): Raw fitness, lower = closer to the ground (better).

    Returns:
        np.ndarray: Ranks scaled to $[0, 1]$, same shape (0 = best, 1 = worst).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return v
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    # average the ranks of tied values so equals share the same elevation
    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    if uniq.size < v.size:
        sums = np.zeros(uniq.size)
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    span = ranks.max() - ranks.min()
    return np.zeros_like(ranks) if span == 0 else (ranks - ranks.min()) / span


def _check_metric_kwargs(
    metric_fn: collections.abc.Callable, metric_kwargs: dict, named: bool
) -> None:
    """Reject keywords the force law does not actually have a parameter for.

    Every law in `METRIC_REGISTRY` ends in ``**kwargs`` — that is what lets a
    sweep pass one law's parameters to another without crashing, and it is
    also what made `esa` silently accept anything at all. A misspelled
    ``sigma`` used to run to completion with the default and report a number,
    which is the worst possible failure: not an error, just a quietly wrong
    measurement. This turns that back into a `TypeError`.

    Validation is against the *named* parameters, since ``**kwargs`` is
    precisely the thing that cannot discriminate. A user-supplied callable
    that declares ``**kwargs`` is taken at its word and left alone — that is
    an explicit choice in their code, where the registry's is an
    implementation detail of this module.

    Args:
        metric_fn (Callable): The resolved force law.
        metric_kwargs (dict): Extra keywords bound for it.
        named (bool): True when `metric` was a registry name, so the law is
            ours and its ``**kwargs`` carries no promise.

    Raises:
        TypeError: Naming the offending keyword and what is accepted.
    """
    if not metric_kwargs:
        return
    try:
        params = inspect.signature(metric_fn).parameters
    except (TypeError, ValueError):  # builtins, C callables: nothing to check
        return
    if not named and any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        return
    accepted = {
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    } - {"d"}
    unknown = sorted(set(metric_kwargs) - accepted)
    if unknown:
        label = getattr(metric_fn, "__name__", repr(metric_fn))
        raise TypeError(
            f"{label}() got unexpected keyword argument(s) "
            f"{', '.join(map(repr, unknown))}; it accepts "
            f"{', '.join(sorted(accepted)) or '(none)'}. Extra keywords are "
            f"forwarded to the force law, so a misspelled one would "
            f"otherwise be ignored and the run would report a number "
            f"measured with the default."
        )


# --- Helpers ----------------------------------------------------------------
def _scale(
    arr: np.ndarray,
    min_val: np.ndarray | np.number | float | None = None,
    max_val: np.ndarray | np.number | float | None = None,
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
    min_val: np.ndarray | np.number | float,
    max_val: np.ndarray | np.number | float,
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




class _AttractionField:
    r"""Fitness (elevation) as a terrain field over the torus, and the only
    source of estimates for it.

    Two things this exists to make unmissable.

    **Where estimation happens.** Every inferred fitness in the library
    comes out of `at`. Nothing else writes into the estimated region, so the
    question "is this number measured or inferred?" is answered by reading one
    class rather than tracing an array through a loop.

    **That fitness belongs to a position, not to a point.** The estimate is
    a property of the space itself (`attraction.InverseDistance`), fitted once
    from the measured anchors at the start of `esa` and queried as the moving
    points traverse the terrain.

    Args:
        positions (np.ndarray): Measured points, shape $(M, D)$, in $[0, 1)$.
        values (np.ndarray): Their fitness (elevation), shape $(M,)$, normalised
            so lower = closer to the ground (better).
        k (int): Neighbours an inverse-distance estimate averages over.
        power (float): Its inverse-distance exponent.
        p (float): Exponent of the toroidal $L^p$ metric.
    """

    __slots__ = ("_pos", "_val", "model", "n_measured")

    def __init__(self, positions, values, k=8, power=2.0, p: float = 1.0):
        self._pos = np.asarray(positions, dtype=np.float64)
        self._val = np.asarray(values, dtype=np.float64)
        self.n_measured = int(self._val.shape[0])
        self.model = attraction.InverseDistance(k=k, power=power, p=p).fit(
            self._pos, self._val)

    def at(self, positions: np.ndarray) -> np.ndarray:
        """Estimated fitness (elevation) at `positions`, shape $(Q,)$."""
        return self.model.at(np.asarray(positions, dtype=np.float64))

    def guide(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate and uphill slope of the terrain at `positions`."""
        return self.model.guide(np.asarray(positions, dtype=np.float64))


def _zscore(values: np.ndarray) -> np.ndarray:
    """Standardise, returning zeros when every value is the same.

    Used to put novelty and estimated fitness on one scale so the
    weight between them means the same thing on any objective.
    """
    sd = values.std()
    if sd <= 0.0:
        return np.zeros_like(values)
    return (values - values.mean()) / sd


def _smart_init(
    index: ToroidalNN,
    n_new: int,
    dim: int,
    rng: np.random.Generator,
    init_sampler: samplers.Sampler,
    pool: int = 15,
    field: "_AttractionField | None" = None,
    fitness_weight: float = 0.0,
    p: float = 1.0,
) -> np.ndarray:
    r"""Initializes new points by Best Candidate Sampling against the index.

    For each of the $n$ slots, a pool of candidate positions is drawn with
    the space-filling sampler and the one farthest (toroidal, in the index's
    metric) from every already-indexed point **and from every slot picked
    before it in this call** wins:

    $$ c_i^* = \arg\max_{c \in \text{pool}_i} \;
       \min_{p \,\in\, \text{index} \,\cup\, \{c_1^*, \dots, c_{i-1}^*\}}
       d^{tor}(c, p) $$

    **The second term is what makes this Mitchell's algorithm rather than a
    race.** Scored against the index alone, every slot of a call sees the
    same voids and none of them sees the others, so they all pick the same
    few emptiest pockets -- and the larger the pool, the more reliably they
    find them. With clustered static data in 2-D and $n = 5000$ that put the
    whole batch in the corners: 20% grid coverage where the raw sampler
    (``pool=1``) reaches 100%, and a largest hole that did not shrink with
    $n$. The relaxation cannot repair it, because the own-gap step schedule
    $\text{near}_i / (t + C)$ bounds travel to the *local* spacing and the
    imbalance is global. The cost of seeing the earlier slots is quadratic
    in the slots of one call, which is why `_esa` calls this in rounds of
    `PLACEMENT_ROUND`.

    A small jitter $\xi \sim U(-10^{-3}, 10^{-3})$ breaks exact overlaps;
    the result is reduced mod 1 (no clipping — the torus has no edge to
    clip against).

    **The placement mirrors the mode.** Given a fitness `field`, the winner is
    chosen on a standardised blend of novelty (large gap) and the fitness
    (low elevation $\hat f$, closer to the ground) the candidate is *expected*
    to have,

    $$ s(c) = z\big(\text{novelty}(c)\big)
              - w \, z\big(\hat f(c)\big) $$

    so a repulsive ESS places repulsively and a guided ESS places towards
    empty regions with lower expected fitness. Measured on the OBLESA sweep,
    placing on novelty alone is worth nothing: unguided dart tied uniform
    noise 233-243 over 480 paired cells, while the same search with a fitness
    term in the *placement* beat it 309-169. Where you probe matters more than
    how evenly you spread.

    Args:
        index (ToroidalNN): Fitted index holding all existing points.
        n_new (int): Number of points to initialize.
        dim (int): Dimensionality of the space.
        rng (np.random.Generator): Random number generator.
        init_sampler (samplers.Sampler): Candidate-pool sampler (e.g. LHS).
        pool (int): Candidates drawn per slot.
        field (_AttractionField | None): Source of the fitness
            estimate. Omit for repulsive placement.
        fitness_weight (float): Balance against novelty in the composite
            score. Zero reproduces the repulsive placement exactly.
        p (float): Exponent of the metric the index ranks by, so that the
            distance to an earlier slot is the same distance as the one to
            an indexed point.

    Returns:
        np.ndarray: Initial positions, shape $(n_{new}, D)$, in $[0, 1)$.
    """
    candidates = init_sampler.sample(n_new * pool, dim, rng).astype(np.float64)
    _, dists = index.query(k=1, queries=candidates)
    # A copy: it is lowered in place as slots are picked, and the index owns
    # what it returned.
    novelty = np.array(dists, dtype=np.float64).reshape(n_new, pool)
    slots = candidates.reshape(n_new, pool, dim)

    # Repulsive ESS places on novelty alone; guided ESS places on a blend
    # preferring larger gap and lower expected fitness (closer to the ground).
    f_hat = None
    if field is not None and fitness_weight != 0.0:
        f_hat = field.at(candidates).reshape(n_new, pool)

    best = np.empty(n_new, dtype=np.intp)
    for i in range(n_new):
        if f_hat is None:
            best[i] = novelty[i].argmax()
        else:
            # Standardised per slot, so `fitness_weight` is scale-free:
            # novelty shrinks as the space fills, and without this the balance
            # would drift over the batch.
            best[i] = (_zscore(novelty[i])
                       - fitness_weight * _zscore(f_hat[i])).argmax()
        if pool == 1 or i + 1 == n_new:
            continue  # nothing to choose between, or nobody to tell
        # The slots still to be picked now have one more point to be far
        # from. Toroidal gap in the metric the index ranks by, so the two
        # halves of the min are the same distance.
        gap = np.abs(slots[i + 1:] - slots[i, best[i]])
        np.minimum(gap, 1.0 - gap, out=gap)
        taken = geometry.lp_norm(gap, p, axis=2)
        np.minimum(novelty[i + 1:], taken, out=novelty[i + 1:])

    picked = slots[np.arange(n_new), best]
    jitter = rng.uniform(-1e-3, 1e-3, size=picked.shape)
    return np.mod(picked + jitter, 1.0)


#: Below this many direction elements per block, the compiled contraction
#: loses to NumPy and is not used.
#:
#: The kernel is row-parallel, so it pays a rayon dispatch per call and ESS
#: calls it once per block per epoch -- hundreds of times. Measured end to
#: end, that overhead swamps the win on a small neighbourhood: at the k-NN
#: default of `K_LOCAL = 5` neighbours the compiled path ran 0.74x at d=40,
#: a real regression on the *default* configuration. It reaches 1.6-3.6x at
#: 64 neighbours, which is the regime radius mode needs and where the
#: `(rows, k, D)` tensor is tens of megabytes.
#:
#: Counted as `block_ids.size * D` -- the elements the tensor would have had
#: -- because that, not the row count, is what the kernel is avoiding.
_KERNEL_MIN_WORK = 200_000

#: Elements per intermediate array in the force kernel (~64 MB in
#: float64). The kernel materialises several $(\text{rows}, m, D)$
#: tensors at once, so this is what keeps radius mode — whose neighbour
#: lists become the whole design in sparse high-dimensional regimes —
#: from allocating gigabytes.
_FORCE_BLOCK_ELEMENTS = 8_000_000



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
    **metric_kwargs,
) -> np.ndarray:
    r"""Net repulsive force on each active point from its neighbour list.

    This is the single force kernel for both search modes: k-NN passes the
    dense `ToroidalNN.query` result, radius mode passes
    `ToroidalNN.query_radius(pad=True)`, which is the same shape under the
    same missing-neighbour convention. For active point $x_i$ with
    neighbours $y_j$:

    $$ \vec{F}_i = \sum_{j} \frac{\vec{u}_{ij}}{\lVert\vec{u}_{ij}\rVert_1}
       \, f\!\left(\frac{d_{L^p}(x_i, y_j)}{R}\right), \qquad
       u_{ij,l} \;=\; \operatorname{sign}(r_{ij,l}) $$

    where $\vec{r}_{ij}$ is the **toroidal** displacement — each component
    wrapped into $[-1/2, 1/2)$ by $r \leftarrow r - \operatorname{round}(r)$
    — so near the seam the push points the short way around, never across
    the whole domain.

    **The direction is the gradient of the metric supplying the
    magnitudes.** For toroidal L1 that is $\operatorname{sign}(r_l)$: every
    coordinate is pushed equally, rather than hardest along the one that
    already differs most. Until 2026-07-28 the kernel used
    $r_l/\lVert r\rVert_2$ — the $L_2$ gradient — while `dists` carried
    L1 magnitudes, so the applied force descended no potential ESS
    defines. An exactly shared coordinate contributes nothing, since
    $\operatorname{sign}(0) = 0$, which is the symmetric subgradient and
    the only choice that does not invent a direction.

    **Below $p = 1$ the direction stays $\operatorname{sign}(r_l)$, and
    that is a measurement, not an omission.** The gradient of the $L^p$
    distance is $\lvert r_l\rvert^{p-1}\operatorname{sign}(r_l)$: singular
    at a shared coordinate, and concentrated on whichever gap is smallest.
    A quasi-norm is concave along each axis, so that is exactly where a
    step stops paying soonest. $\operatorname{sign}(r)$ widens every gap at
    once and therefore increases the distance under *every* exponent; it is
    the one ascent direction they share. Relaxed from scratch and scored on
    nearest-neighbour separation in the run's own $L^p$, relative to an
    $L^1$ run (median of 8 seeds, median / minimum separation):

    | $p$ | $d$ | $L^p$ gradient | $\operatorname{sign}$ |
    | --- | --- | --- | --- |
    | 0.75 | 2 | 0.94 / 0.94 | 0.98 / 0.99 |
    | 0.75 | 4 | 0.95 / 0.92 | 1.04 / 1.07 |
    | 0.75 | 16 | 0.97 / 0.94 | 1.00 / 1.00 |
    | 0.5 | 2 | 0.92 / 1.00 | 0.95 / 1.11 |
    | 0.5 | 4 | 0.98 / 1.08 | 1.13 / 1.34 |
    | 0.5 | 16 | 0.95 / 0.90 | 1.04 / 1.04 |

    The gradient loses in every cell. What the exponent changes is
    therefore which points interact and how strongly -- the neighbour list
    and $\hat d$ are $L^p$ -- and how far a step may go, not where the push
    points. It also means the compiled kernel serves every exponent.

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
        dists (np.ndarray): Toroidal $L^p$ distances, shape $(M, m)$;
            ``inf`` = missing.
        radius (float): Interaction radius $R$ used to normalise distances.
        metric_fn (Callable): Log-space force law $\log f(\hat{d})$.
        rng (np.random.Generator): Generator for tie-breaking noise.
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

        # Weights first, geometry second. Nothing above depends on a
        # displacement -- the force law reads `dists`, which torann already
        # returned -- so the whole $(rows, k, D)$ tensor can be deferred, and
        # then not built at all on the compiled path.
        d_hat = np.where(valid, block_dists, 1.0) / radius
        log_mag = metric_fn(d_hat, **metric_kwargs)
        log_mag = np.where(valid, log_mag, -np.inf)

        m_i = np.max(log_mag, axis=1, keepdims=True)
        m_i = np.where(np.isneginf(m_i), 0.0, m_i)
        weights = np.exp(log_mag - m_i)
        weights[~valid] = 0.0

        # Coincident points get a random direction rather than none, and the
        # randomness is this function's `rng` -- a NumPy Generator, whose
        # stream cannot move into a compiled kernel without changing it. So
        # a block containing one takes the NumPy path entire. Exact
        # coincidence is close to measure-zero here, which is what makes
        # "fall back for the whole block" cheaper than splitting the pairs.
        stacked = valid & (np.where(valid, block_dists, np.inf) < 1e-9)
        wd = _torann_rust.weighted_directions
        kernel = (
            wd
            if (wd is not None
                and not np.any(stacked)
                and block_ids.size * active.shape[1] >= _KERNEL_MIN_WORK)
            else None
        )
        if kernel is None:
            disp = active[lo:hi, None, :] - all_data[safe_ids]
            disp -= np.round(disp)  # toroidal wrap: shortest per axis
            if np.any(stacked):
                noise = rng.standard_normal(size=disp.shape)
                noise /= np.abs(noise).sum(axis=2, keepdims=True) + 1e-9
                disp = np.where(stacked[..., None], noise, disp)

            grad = np.sign(disp)
            norms = np.abs(grad).sum(axis=2, keepdims=True)
            directions = grad / np.maximum(norms, 1e-9)
            net = np.sum(directions * weights[..., None], axis=1)
        else:
            (net,) = kernel(all_data, active[lo:hi], block_ids, weights[None])

        force_cap = 1000.0
        forces[lo:hi] = np.exp(np.minimum(m_i, np.log(force_cap))) * net

    return forces


def _apply_guide(forces: np.ndarray, guide: np.ndarray, weight: float) -> None:
    r"""Damp the uphill component of the repulsive force along the terrain slope, in place.

    The fitness field $f(x)$ is the terrain that particles traverse (lower =
    closer to the ground = better; higher = uphill = worse). Repulsion moving
    downhill ($\vec F_{\text{rep}} \cdot \hat u < 0$, towards the ground) is
    untouched; the share $\min(w, 1)$ of the component of $\vec F_{\text{rep}}$
    that points uphill ($\vec F_{\text{rep}} \cdot \hat u > 0$, climbing against
    gravity towards higher fitness/cost) is removed:

    $$ \vec F' = \vec F_{\text{rep}} - \min(w, 1)\,\max(\vec F_{\text{rep}}\cdot\hat u,\, 0)\,\hat u,
       \qquad \hat u = \frac{\nabla f}{\lVert\nabla f\rVert_2} $$

    No external force is injected: a point at rest under repulsion stays at
    rest, the push is never reversed and never grows, and a row whose guide is
    zero (flat field, or a point on a source) is untouched.

    Args:
        forces (np.ndarray): Repulsive push on each active point, shape $(M, D)$.
        guide (np.ndarray): $R\,\nabla f$ (uphill slope) at each point, shape $(M, D)$.
        weight (float): $w \ge 0$, effective space weight.
    """
    g_len = np.sqrt(np.einsum("md,md->m", guide, guide))
    rows = np.flatnonzero(g_len > 0.0)
    if rows.size == 0:
        return
    unit = guide[rows] / g_len[rows, None]
    along = np.einsum("md,md->m", forces[rows], unit)
    up = np.maximum(along, 0.0)
    forces[rows] -= min(float(weight), 1.0) * up[:, None] * unit


# --- Core Logic --------------------------------------------------------------
def _esa(
    samples01: np.ndarray,
    index: ToroidalNN,
    *,
    n: int,
    dim: int,
    epochs: int,
    C: float,
    w: float,
    batch_size: int,
    k: int,
    radius: float,
    search_mode: geometry.ResolvedMode,
    patience: int,
    metric_fn: collections.abc.Callable,
    rng: np.random.Generator,
    init_sampler: samplers.Sampler,
    init_pool: int = 64,
    p: float = 1.0,
    fit_norm: np.ndarray | None = None,
    fitness_weight: float = 1.0,
    stats: dict | None = None,
    **metric_kwargs,
) -> np.ndarray:
    r"""Executes the ESA optimization loop on the unit torus.

    **Per batch:**

    1. Initialize positions (`_smart_init` once the index has points,
       the raw sampler for the very first from-scratch batch).
    2. For each epoch $t$: query neighbours, compute repulsive forces,
       damp uphill motion along the fitness terrain when `fit_norm` is
       given, and step in units of each point's own nearest-neighbour
       clearance (Vlasiuk et al. 2018):

       $$ x_{i,t+1} = \left(x_{i,t} + \frac{\text{near}_{i,t}}{t + C}
          \,\frac{\vec F_{i,t}}{\lVert\vec F_{i,t}\rVert_p}\right) \bmod 1 $$

       then `ToroidalNN.update`.
    3. On convergence, offer the lowest-scoring share `w` of the batch a
       best-candidate position (`_smart_init`), accepting only moves that
       improve the score, and freeze the batch into the static tier.

    **Early stopping** watches the `GAP_QUANTILE` (5%) quantile of the
    nearest-neighbour distance of the points being relaxed (and, when guided,
    the mean field elevation scaled by the batch's initial field spread). The
    loop stops when neither has improved by `GAP_IMPROVE` (1%) or the spread-
    scaled `ATT_IMPROVE` (0.5% drop towards the ground) for `patience`
    consecutive epochs.

    Args:
        samples01 (np.ndarray): Static points already scaled to $[0, 1)$.
        index (ToroidalNN): The (unfitted) neighbour index to drive.
        n (int): Number of points to generate.
        dim (int): Dimensionality of the space.
        epochs (int): Maximum update steps per batch.
        C (float): Offset of the gap step schedule $\text{near}_i / (t + C)$.
        w (float): Share of worst points offered re-placement after relaxation.
        batch_size (int): Points optimized together per batch.
        k (int): Neighbours per query (k-NN mode).
        radius (float): Interaction radius $R$ (search cutoff in radius
            mode; force normalisation scale in both modes).
        search_mode (geometry.ResolvedMode): ``"k_nn"`` or ``"radius"``.
            Already resolved -- ``"auto"`` is answered in `esa`.
        patience (int): Consecutive epochs without progress before the
            plateau stop fires.
        metric_fn (Callable): Log-space force law.
        rng (np.random.Generator): Random number generator.
        init_sampler (samplers.Sampler): Sampler for initial positions.
        init_pool (int): Candidates per slot for `_smart_init`; see `esa`.
        p (float): Exponent of the toroidal $L^p$ metric the index ranks
            by; `radius` is in its units.
        fit_norm (np.ndarray | None): Normalised fitness of `samples01` in $[0, 1]$
            (lower = closer to the ground = better).
        fitness_weight (float): Balance against repulsion.
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

    # One field, built once from the measured points: the anchors do not move,
    # so the terrain they define is static for the whole call.
    field = None
    trust = 1.0
    if fit_norm is not None and n_static > 0:
        k_att, trust = attraction.choose_k(
            all_data[:n_static], fit_norm[:n_static], power=2.0, p=p)
        if stats is not None:
            stats["k_att"] = int(k_att)
            stats["field_trust"] = float(trust)
        field = _AttractionField(all_data[:n_static], fit_norm[:n_static],
                                 k=k_att, power=2.0, p=p)
    if fitted:
        index.fit(all_data[:n_static], k=k, radius=radius_hint)

    num_batches = math.ceil(n / batch_size)
    logger.debug(
        "Starting ESA: %d points, %d batches, mode=%s, R=%.4f",
        n, num_batches, search_mode, radius,
    )

    timers: dict | None = {} if stats is not None else None
    place_w = min(float(fitness_weight), PLACEMENT_WEIGHT_CAP) if fitness_weight > 0.0 else 0.0

    for _ in range(num_batches):
        current_n = min(batch_size, n_static + n - cursor)
        if current_n <= 0:
            break

        started = time.perf_counter()
        if fitted:
            # Placed in rounds, each one indexed before the next is drawn, so
            # that a slot is far from the whole batch placed so far and not
            # only from the batches before it. Within a round `_smart_init`
            # does the same thing exactly.
            rounds = PLACEMENT_ROUND
            init = np.empty((current_n, dim))
            for lo in range(0, current_n, rounds):
                hi = min(lo + rounds, current_n)
                init[lo:hi] = _smart_init(
                    index, hi - lo, dim, rng, init_sampler, pool=init_pool,
                    field=field, fitness_weight=place_w, p=p,
                )
                if hi < current_n:
                    index.promote(init[lo:hi])
            if current_n > rounds:
                # The earlier rounds went into the static tier to be seen;
                # the relaxation needs the whole batch back in the candidate
                # tier, and a refit is the only way to say that.
                index.fit(all_data[:cursor], init, k=k, radius=radius_hint)
            else:
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

        space_weight = float(fitness_weight) * trust
        use_guide = (field is not None and fit_norm is not None
                     and space_weight > 0.0)
        att_step = ATT_IMPROVE
        if field is not None and fit_norm is not None:
            est = field.at(active)
            best_fit = float(est.mean())
            if current_n > 1:
                att_step = ATT_IMPROVE * min(1.0, max(0.01, float(est.std()) / _RANK_STD))
        else:
            best_fit = 0.0
        fit_start = best_fit
        guide_vec = None
        both_streak = 0
        ema = None
        best_gap = 0.0
        gap_streak = 0
        epochs_used = 0

        for epochs_used in range(1, epochs + 1):
            started = time.perf_counter()
            if search_mode == "radius":
                ids, dists = index.query_radius(radius, pad=True)
            else:
                ids, dists = index.query(k=k)
            _accumulate(timers, "query_s", started)

            # The distance from each point to its nearest other point, read
            # off the query. Radius mode reports nothing beyond the radius,
            # so a point alone inside it counts as one radius away.
            near = dists.min(axis=1)
            isolated = ~np.isfinite(near)
            near[isolated] = radius
            gap = float(np.quantile(near, GAP_QUANTILE))
            progress = (not np.all(isolated)) and (gap > best_gap * (1.0 + GAP_IMPROVE))
            if progress:
                best_gap = gap
                gap_streak = 0
            else:
                gap_streak += 1

            if use_guide and field is not None and (
                    guide_vec is None or epochs_used % ATT_EVERY == 0):
                # The field is fixed for the call; what is refreshed is
                # where the moving points read it. Lower elevation is better.
                est, guide_vec = field.guide(active)
                guide_vec *= radius
                fit_now = float(est.mean())
                if fit_now < best_fit - att_step:
                    best_fit = fit_now
                    progress = True
            both_streak = 0 if progress else both_streak + 1

            started = time.perf_counter()
            forces = _compute_forces(
                active, all_data, ids, dists, radius, metric_fn, rng,
                **metric_kwargs,
            )
            if use_guide and guide_vec is not None:
                _apply_guide(forces, guide_vec, space_weight)
            _accumulate(timers, "force_s", started)

            started = time.perf_counter()
            # The point's own clearance is the unit, and it shrinks as
            # 1/(t + C) along the unit force direction in the run's L^p.
            f_norm = geometry.lp_norm(forces, p, axis=1)[:, None]
            step = forces / np.maximum(f_norm, 1e-12)
            step *= (near / (epochs_used + C))[:, None]
            active += step
            np.mod(active, 1.0, out=active)
            _accumulate(timers, "step_s", started)

            started = time.perf_counter()
            index.update(active)
            _accumulate(timers, "update_s", started)

            f_max = float(np.max(np.abs(forces).sum(axis=1)))
            ema = f_max if ema is None else 0.5 * ema + 0.5 * f_max
            settled = both_streak if use_guide else gap_streak
            if f_max <= 1e-12 or settled >= patience:
                break

        replaced = 0
        if w > 0.0 and current_n > 1:
            # Worst first, in rounds, each one indexed before the next is
            # drawn, as the placement does. Higher score = larger gap and
            # lower fitness (closer to the ground).
            _, d_near = index.query(k=2, queries=active)
            d_near = np.asarray(d_near, dtype=np.float64).reshape(current_n, -1)
            own = np.where(d_near > 1e-12, d_near, np.inf).min(axis=1)
            own[~np.isfinite(own)] = 0.0
            guided = field is not None and fit_norm is not None and place_w > 0.0
            own_fit = field.at(active) if guided and field is not None else None
            rank_by = (own if own_fit is None
                       else _zscore(own) - place_w * _zscore(own_fit))
            order = np.argsort(rank_by)[: math.ceil(w * current_n)]
            for lo in range(0, order.size, PLACEMENT_ROUND):
                rows = order[lo : lo + PLACEMENT_ROUND]
                offer = _smart_init(index, rows.size, dim, rng, init_sampler,
                                    pool=init_pool, p=p,
                                    field=field if guided else None,
                                    fitness_weight=place_w if guided else 0.0)
                _, d_offer = index.query(k=1, queries=offer)
                gain = np.asarray(d_offer, dtype=np.float64).reshape(rows.size, -1)[:, 0]
                offer_fit = None
                mu_g = sd_g = 0.0
                fit_term = np.zeros(2 * rows.size)
                if own_fit is not None and field is not None:
                    offer_fit = field.at(offer)
                    gaps = np.concatenate((own[rows], gain))
                    mu_g, sd_g = float(gaps.mean()), float(gaps.std())
                    fit_term = -place_w * _zscore(
                        np.concatenate((own_fit[rows], offer_fit)))
                best_first = np.argsort(
                    -gain if offer_fit is None
                    else -(_zscore(gain) + fit_term[rows.size:]))
                moved = False
                taken = np.empty((0, dim))
                for at, (row, j) in enumerate(zip(rows, best_first)):
                    clear = gain[j]
                    if taken.shape[0]:
                        sep = np.abs(taken - offer[j])
                        np.minimum(sep, 1.0 - sep, out=sep)
                        clear = min(clear, float(geometry.lp_norm(sep, p, axis=1).min()))
                    if offer_fit is None:
                        better = clear > own[row]
                    else:
                        scale = sd_g if sd_g > 0.0 else 1.0
                        better = ((clear - mu_g) / scale + fit_term[rows.size + j]
                                  > (own[row] - mu_g) / scale + fit_term[at])
                    if better:
                        active[row] = offer[j]
                        taken = np.concatenate((taken, offer[j][None]))
                        replaced += 1
                        moved = True
                if moved:
                    index.update(active)

        logger.debug(
            "Batch [%d:%d] stopped after %d/%d epochs (force EMA %.4g)",
            cursor, cursor + current_n, epochs_used, epochs, ema or 0.0,
        )
        if stats is not None:
            stats.setdefault("batch_epochs", []).append(epochs_used)
            stats.setdefault("batch_replaced", []).append(replaced)
            if field is not None and fit_norm is not None:
                stats.setdefault("batch_att_gain", []).append(
                    fit_start - float(field.at(active).mean()))
            stats.setdefault("batch_force_ema", []).append(
                float(ema) if ema is not None else 0.0
            )
        cursor += current_n

    if stats is not None:
        stats["radius"] = radius
        stats["search_mode"] = search_mode
        stats["epochs_total"] = int(np.sum(stats.get("batch_epochs", [0])))
        stats.update(timers or {})

    return all_data[n_static:cursor]


def esa(
    samples: np.ndarray | list,
    bounds: np.ndarray | list,
    *,
    fitness: np.ndarray | list | None = None,
    n: int | None = None,
    p: float | np.floating = 1.0,
    C: float = 12.0,
    w: float = 0.05,
    fitness_weight: float = 1.0,
    index: ToroidalNN | None = None,
    epochs: int = 1024,
    patience: int = 12,
    search_mode: geometry.SearchMode = "auto",
    batch_size: int | None = None,
    k: int | None = None,
    radius: float | None = None,
    radius_target: int | None = None,
    metric: str | collections.abc.Callable = "gaussian",
    seed: int | np.random.Generator | None = None,
    init_sampler: samplers.Sampler | int | None = None,
    init_pool: int = 64,
    stats: dict | None = None,
    **metric_kwargs,
) -> np.ndarray:
    r"""Empty Space Algorithm (ESA): returns only the generated points.

    Public API for the toroidal relaxation. At its simplest it takes only the
    existing `samples`, the domain `bounds`, and optionally `fitness`
    (with every parameter from `fitness` onwards passed by keyword):

    * When `n` is omitted (`None`), it defaults to $\max(1, N_0)$, doubling the
      existing sample set (or generating 1 point when starting from scratch).
    * Each epoch steps along the unit repulsive force in the run's toroidal
      $L^p$ metric by the point's own nearest-neighbour clearance divided by
      $t + C$ (Vlasiuk et al. 2018), eliminating separate learning-rate, decay
      and travel-cap parameters.
    * When `fitness` is provided (**lower = closer to the ground = better**),
      the space itself acts as a terrain (`attraction.InverseDistance`) built
      once from the measured anchors via `attraction.choose_k` (automatic
      neighbourhood size $k$ and leave-one-out `trust` in $[0, 1]$). During
      relaxation, uphill motion along the terrain slope (climbing towards
      higher fitness/cost) is damped by $\min(\text{fitness\_weight} \cdot \text{trust}, 1)$,
      while downhill motion towards the ground is unbraked, without injecting
      point-source attraction forces that could collapse the design.
    * After convergence, the lowest-scoring fraction `w` (default 5%) of the
      batch is offered a best-candidate re-placement (`_smart_init`), taken
      only when it improves the point's clearance (or composite score).

    Args:
        samples (np.ndarray | list): Existing points to avoid, shape $(N_0, D)$.
            May be empty.
        bounds (np.ndarray | list): Domain boundaries, shape $(D, 2)$.
        fitness (np.ndarray | list | None): One value per row of
            `samples`, **lower = closer to the ground = better**. ``None``
            (default) runs pure repulsion. Values are rank-normalised to
            $[0, 1]$ internally ($0$ = best/ground, $1$ = worst/uphill),
            and a constant vector is treated as pure repulsion.
        n (int | None): Number of new points to create. ``None`` (default)
            generates $\max(1, N_0)$ points.
        p (float): Exponent of the toroidal $L^p$ metric,
            `geometry.P_MIN` $\le p \le 1$. Default ``1.0`` (or taken from
            `index` when supplied and `p` is left at ``1.0``).
        C (float): Offset of the single-parameter gap step schedule
            $\Delta x_i = \frac{\text{near}_i}{t + C}\frac{\vec F_i}{\lVert\vec F_i\rVert_p}$.
            Default ``12.0``.
        w (float): Share in $[0, 1]$ of the lowest-scoring points in each batch
            offered a post-relaxation best-candidate re-placement. Default
            ``0.05`` (5%); ``0.0`` disables it.
        fitness_weight (float): Terrain uphill damping and placement weight when
            `fitness` is given. Default ``1.0`` (scaled by the field's
            leave-one-out `trust` during relaxation, and capped at
            `PLACEMENT_WEIGHT_CAP` = 0.5 during placement). ``0.0`` disables
            guidance and reproduces pure repulsion bit-for-bit.
        index (ToroidalNN | None): Optional pre-configured index. It is
            re-fitted; when ``None`` a default `ToroidalNN` is created.
        epochs (int): Maximum iterations per batch. Default ``1024``.
        patience (int): Consecutive epochs without a `GAP_IMPROVE` (1%) gain
            in the 5% nearest-neighbour gap quantile (or spread-scaled
            `ATT_IMPROVE` field drop towards the ground when guided) before
            stopping. Default ``12``.
        search_mode (geometry.SearchMode): ``"auto"`` (default), ``"k_nn"`` or
            ``"radius"``, resolved via `geometry.neighbourhood_for`.
        batch_size (int | None): Points optimized together. ``None`` (default)
            relaxes all $n$ points simultaneously.
        k (int | None): Neighbours in k-NN mode; default
            $\min(2D + 1, \text{`K_LOCAL`})$.
        radius (float | None): Interaction radius as a fraction of the torus
            diameter, in $(0, 1]$. ``None`` or ``0`` derives it from
            `geometry.radius_for_count`.
        radius_target (int | None): Neighbours the derived radius should
            contain when `radius` is not given.
        metric (str | Callable): Force-law name in `METRIC_REGISTRY`, or
            a callable $\log f(\hat{d})$. Default ``"gaussian"``.
        seed (int | np.random.Generator | None): Seed or Generator.
        init_sampler (samplers.Sampler | int | None): Candidate-pool sampler;
            ``None`` = LHS.
        init_pool (int): Candidates per slot for `_smart_init`. Default ``64``.
        stats (dict | None): Optional dictionary filled in place with run
            statistics (`batch_epochs`, `batch_replaced`, `batch_force_ema`,
            `epochs_total`, `radius`, `search_mode`, `k_att`, `field_trust`,
            and timing buckets `query_s`, `force_s`, `step_s`, `update_s`,
            `setup_s`).
        **metric_kwargs: Extra arguments for the force law.

    Returns:
        np.ndarray: The generated points, shape $(n, D)$, in the original
        coordinate system.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(
            f"`bounds` must have shape (D, 2), got {bounds.shape}")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError(
            "`bounds` must be finite with lower <= upper on every axis")
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        samples = samples.reshape(-1, bounds.shape[0])

    n_gen = max(1, samples.shape[0]) if n is None else int(n)
    if C <= 0.0:
        raise ValueError(f"`C` must be positive, got {C!r}")
    if not 0.0 <= float(w) <= 1.0:
        raise ValueError(f"`w` must be in [0, 1], got {w!r}")

    if isinstance(metric, str):
        metric_fn = METRIC_REGISTRY.get(metric.lower())
        if metric_fn is None:
            raise ValueError(f"Unknown metric '{metric}'")
    else:
        metric_fn = metric
    _check_metric_kwargs(metric_fn, metric_kwargs, isinstance(metric, str))

    fit_norm = None
    if fitness is not None:
        raw_fit = np.asarray(fitness, dtype=np.float64).ravel()
        if raw_fit.size != samples.shape[0]:
            raise ValueError(
                f"fitness has {raw_fit.size} values for {samples.shape[0]} "
                f"samples. One per existing point: the points being placed "
                f"have no evaluated quality yet, so they carry none."
            )
        if raw_fit.size > 0 and np.ptp(raw_fit) > 0.0 and float(fitness_weight) != 0.0:
            fit_norm = _rank_normalise(raw_fit)

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    dim = bounds.shape[0]
    min_val = bounds[:, 0]
    max_val = bounds[:, 1]
    scaled_samples, _, _ = _scale(samples, min_val, max_val)
    if scaled_samples.size:
        outside = int(np.any(
            (scaled_samples < 0.0) | (scaled_samples > 1.0), axis=1).sum())
        if outside:
            logger.warning(
                "%d of %d samples lie outside `bounds`. On the torus they "
                "wrap to the opposite side and act as obstacles there. "
                "Widen `bounds` to contain the samples, or drop them.",
                outside, scaled_samples.shape[0])

    k_value = k if k is not None else min(2 * dim + 1, K_LOCAL)
    batch = batch_size if batch_size is not None else max(n_gen, 1)
    index_p = None if index is None else float(getattr(index, "p", 1.0))
    if p is None or (index_p is not None and p == 1.0):
        exponent = geometry.check_p(1.0 if index_p is None else index_p)
    else:
        exponent = geometry.check_p(p)
        if index_p is not None and not math.isclose(
                index_p, exponent, rel_tol=1e-6):
            raise ValueError(
                f"p={exponent} but the supplied index was built with "
                f"p={index_p}. Pass one of them, or make them agree.")
        if index_p is not None:
            exponent = index_p

    n_total = samples.shape[0] + n_gen
    mode, target = geometry.neighbourhood_for(
        dim, n_total, search_mode, radius_target)
    final_radius = (
        geometry.radius_for_count(dim, n_total, target, exponent)
        if not radius
        else geometry.radius_from_normalized(float(radius), dim, exponent)
    )
    logger.debug("Interaction radius (toroidal L^%g): %.4f",
                 exponent, final_radius)

    if index is None:
        index = ToroidalNN(seed=int(rng.integers(2**31)), p=exponent)

    generated = _esa(
        scaled_samples,
        index,
        n=n_gen,
        dim=dim,
        epochs=epochs,
        C=float(C),
        w=float(w),
        batch_size=batch,
        k=k_value,
        radius=final_radius,
        search_mode=mode,
        patience=patience,
        metric_fn=metric_fn,
        rng=rng,
        init_sampler=samplers.check_sampler(init_sampler, default_random_state=rng),
        init_pool=int(init_pool),
        p=exponent,
        fit_norm=fit_norm,
        fitness_weight=float(fitness_weight),
        stats=stats,
        **metric_kwargs,
    )
    return _inv_scale(generated, min_val, max_val)


def ess(
    samples: np.ndarray | list,
    bounds: np.ndarray | list,
    *,
    fitness: np.ndarray | list | None = None,
    n: int | None = None,
    p: float | np.floating = 1.0,
    **kwargs,
) -> np.ndarray:
    r"""Empty Space Strategy (ESS): returns the combined data set.

    Convenience wrapper that runs `esa` and concatenates the result with
    the original samples:

    $$ \text{Result} = \text{samples} \cup \text{ESA}(\text{samples}, \dots) $$

    Args:
        samples (np.ndarray | list): Existing points, shape $(N_0, D)$.
        bounds (np.ndarray | list): Domain boundaries, shape $(D, 2)$.
        fitness (np.ndarray | list | None): Optional per-sample
            fitness values (lower = closer to the ground = better).
        n (int | None): Number of new points to generate; ``None`` defaults
            to $\max(1, N_0)$.
        p (float): Exponent of the toroidal $L^p$ metric,
            `geometry.P_MIN` $\le p \le 1$. Default ``1.0``.
        **kwargs: Forwarded verbatim to `esa` (`C`, `w`, `fitness_weight`,
            `epochs`, `patience`, `search_mode`, `metric`, `index`, `seed`, ...).

    Returns:
        np.ndarray: Array of shape $(N_0 + n, D)$ with original and new
        points.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        samples = samples.reshape(-1, np.asarray(bounds).shape[0])

    new_points = esa(samples, bounds, fitness=fitness, n=n, p=p, **kwargs)
    return np.concatenate((samples, new_points), axis=0)
