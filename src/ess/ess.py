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


def _rank_normalise(values: np.ndarray) -> np.ndarray:
    r"""Attractiveness to $[0, 1]$ by rank, highest = 1.

    **The force must never see the caller's raw units.** Objective values
    span orders of magnitude between problems and between regions of one
    problem, so an `attraction_weight` calibrated on a landscape scoring
    $10^{-3}$ means something else entirely on one scoring $10^{6}$ — the
    balance point would be set by the objective rather than by the knob.

    That exact defect has been paid for once already, in an optimizer that
    blended fitness and diversity through a Boltzmann softmax over raw
    scores: the blend point moved with the objective's scale, so the weight
    was inert on one pool and saturated on another, and two axes of a
    399-arm factorial measured nothing.

    Ranks are used rather than z-scores because they are bounded and
    outlier-proof: one catastrophic sample cannot compress every other point
    into a corner of the range. Ties share the mean rank, so equally
    attractive points pull equally. A constant vector maps to all ones —
    every point equally attractive, which is the sensible reading.

    Args:
        values (np.ndarray): Raw attractiveness, higher = more attractive.

    Returns:
        np.ndarray: Ranks scaled to $[0, 1]$, same shape.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return v
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    # average the ranks of tied values so equals pull equally
    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    if uniq.size < v.size:
        sums = np.zeros(uniq.size)
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    span = ranks.max() - ranks.min()
    return np.ones_like(ranks) if span == 0 else (ranks - ranks.min()) / span


def _check_attraction_balance(
    metric_fn: collections.abc.Callable,
    metric_kwargs: dict,
    attraction_fn: collections.abc.Callable,
    attraction_kwargs: dict,
    weight: float,
) -> None:
    r"""Refuse an attraction that can out-pull repulsion at contact.

    Dart cannot collapse: it picks from discrete candidates and re-measures
    novelty against everything placed. A continuous relaxation can — an
    attraction that dominates at short range piles every active point onto
    the most attractive static one, and the plateau detector would report
    that as convergence, because the forces really have stopped changing.

    Every law in `METRIC_REGISTRY` is finite at $\hat d = 0$, so the guard
    is one inequality and can be *checked* rather than tested for:

    $$ w \cdot a_{\max} \cdot F_{\text{att}}(0) \;<\; F_{\text{rep}}(0) $$

    with $a_{\max} = 1$ after `_rank_normalise`, which is the worst case.

    A second condition decides whether the attraction does anything at all.
    It only overcomes repulsion somewhere if it decays *more slowly*, so the
    net force must change sign at some separation; if it never does, the
    term merely scales repulsion down near attractive neighbours. That is a
    legitimate configuration — points still settle closer to good regions —
    but it is not what "attraction" suggests, so it warns. The common way to
    land there is using one law for both sides: two `linear` terms are
    proportional and can never cross.

    Args:
        metric_fn (Callable): Repulsion law, log-space.
        metric_kwargs (dict): Its parameters.
        attraction_fn (Callable): Attraction law, log-space.
        attraction_kwargs (dict): Its parameters.
        weight (float): $w$.

    Raises:
        ValueError: If the guard fails, naming both magnitudes.
    """
    zero = np.zeros(1)
    f_rep = float(np.exp(metric_fn(zero, **metric_kwargs))[0])
    f_att = float(np.exp(attraction_fn(zero, **attraction_kwargs))[0])
    if weight * f_att >= f_rep:
        raise ValueError(
            f"attraction would out-pull repulsion at contact: "
            f"weight * F_att(0) = {weight * f_att:.4g} >= F_rep(0) = "
            f"{f_rep:.4g}. Every active point would collapse onto its most "
            f"attractive neighbour, and the plateau detector would call that "
            f"convergence. Lower attraction_weight below {f_rep / f_att:.4g}, "
            f"or give the attraction a shallower law."
        )

    d_hat = np.linspace(1e-3, 4.0, 512)
    net = (np.exp(metric_fn(d_hat, **metric_kwargs))
           - weight * np.exp(attraction_fn(d_hat, **attraction_kwargs)))
    if not (net < 0).any():
        logger.warning(
            "attraction never overcomes repulsion at any separation, so it "
            "only weakens the push near attractive neighbours rather than "
            "pulling toward them. This is what happens when both sides use "
            "the same law with the same shape (two `linear` terms are "
            "proportional and can never cross). Give the attraction a "
            "slower-decaying law via attraction_metric, or raise "
            "attraction_weight."
        )


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


def _estimate_attractiveness(
    queries: np.ndarray,
    static: np.ndarray,
    attract_static: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    r"""Attractiveness a candidate position is *expected* to have.

    Attractiveness is only ever known for the static points -- they are the
    ones whose objective has been paid for. A candidate has none, and treating
    that as zero is not neutral: `_rank_normalise` puts the scale on $[0, 1]$,
    so zero is the *bottom* of it, and every candidate is modelled as the least
    attractive thing in the space. Candidates that should pull on each other
    repel instead, and the only attraction in the system is toward the static
    points.

    The estimate is Shepard's method -- inverse-distance weighting over the `k`
    nearest static points, under the same toroidal $L_1$ metric everything else
    here uses:

    $$ \hat a(c) = \frac{\sum_i w_i\, a_i}{\sum_i w_i},
       \qquad w_i = d_{L1}^{tor}(c, x_i)^{-p} $$

    A candidate sitting exactly on a static point takes that point's value
    rather than dividing by zero.

    The known limitation is the one every distance-weighted surrogate carries:
    in high dimension distances concentrate, the weights flatten and $\hat a$
    tends to the mean of `attract_static`. It degrades to "no information"
    rather than to something wrong, which is the right failure mode, but the
    guidance does fade as `dim` grows.

    Args:
        queries (np.ndarray): Positions to estimate, shape $(Q, D)$, in
            $[0, 1)$.
        static (np.ndarray): Points with known attractiveness, shape $(M, D)$,
            in $[0, 1)$.
        attract_static (np.ndarray): Their attractiveness, shape $(M,)$,
            already normalised to $[0, 1]$.
        k (int): Neighbours averaged over. Clamped to $M$.
        power (float): Inverse-distance exponent $p$.
        confidence (np.ndarray | None): Per-source weight in $(0, 1]$, folded
            into $w_i$. Measured points sit at 1; a value inferred earlier
            enters at less than 1 so a chain of inference fades toward the
            mean instead of asserting itself. ``None`` treats every source as
            measured.

    Returns:
        np.ndarray: Estimated attractiveness, shape $(Q,)$, within the range of
        `attract_static`.
    """
    m = static.shape[0]
    q = queries.shape[0]
    if m == 0 or q == 0:
        return np.zeros(q, dtype=np.float64)

    kk = min(int(k), m)
    out = np.empty(q, dtype=np.float64)
    fallback = float(attract_static.mean())

    # Chunked over queries. The obvious `queries[:, None, :] - static[None]`
    # builds a (Q, M, D) temporary, which at the sizes this is actually called
    # with -- 3840 candidates against 90 static points in 100 dimensions -- is
    # 276 MB per call, and the run spent 71% of its initialization time
    # allocating and touching it. Chunking caps that at a few MB and leaves
    # the arithmetic identical.
    step = max(1, int(2_000_000 // max(m * max(queries.shape[1], 1), 1)))
    for lo in range(0, q, step):
        hi = min(lo + step, q)
        # Toroidal L1: per axis the wrap-around distance is min(|d|, 1-|d|),
        # because the space is the unit torus -- the metric the index uses.
        delta = np.abs(queries[lo:hi, None, :] - static[None, :, :])
        dist = np.minimum(delta, 1.0 - delta).sum(axis=2)      # (chunk, M)

        nearest = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        rows = np.arange(hi - lo)[:, None]
        d_near = dist[rows, nearest]
        a_near = attract_static[nearest]

        exact = d_near <= 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(exact, 0.0, d_near ** (-float(power)))
        if confidence is not None:
            w = w * confidence[nearest]
        total = w.sum(axis=1)
        chunk = np.full(hi - lo, fallback)
        ok = total > 0
        if ok.any():
            chunk[ok] = np.einsum("qk,qk->q", w[ok], a_near[ok]) / total[ok]
        hit = exact.any(axis=1)
        if hit.any():
            # On a known point: take its value rather than dividing by zero.
            chunk[hit] = a_near[hit, np.argmax(exact[hit], axis=1)]
        out[lo:hi] = chunk
    return out


class _AttractionField:
    r"""Attractiveness as a field over the torus, and the only source of
    estimates for it.

    Two things this exists to make unmissable.

    **Where estimation happens.** Every inferred attractiveness in the library
    comes out of `at`. Nothing else writes into the estimated region, so the
    question "is this number measured or inferred?" is answered by reading one
    class rather than tracing an array through a loop.

    **That attractiveness belongs to a position, not to a point.** The estimate
    used to be taken once at placement and carried by the point for the whole
    relaxation, which made one side of the force balance live and the other
    stale. A point that relaxed *into* a good region kept its old low value and
    never became an attractor; worse, a point that drifted *out* of one kept
    its old high value and went on pulling everything else toward where it no
    longer was. Repulsion has always depended on current positions; now
    attraction does too.

    Sources carry a confidence. Measured points -- the ones whose objective was
    actually paid for -- sit at 1 and never decay. A value that was itself
    inferred enters at `decay` times the confidence of what it was inferred
    from, so a chain of inference fades toward the mean instead of laundering
    itself into ground truth.

    Args:
        positions (np.ndarray): Measured points, shape $(M, D)$, in $[0, 1)$.
        values (np.ndarray): Their attractiveness, shape $(M,)$, normalised.
        k (int): Neighbours an estimate averages over.
        power (float): Inverse-distance exponent.
        decay (float): Confidence multiplier applied to inferred sources.
    """

    __slots__ = ("_conf", "_pos", "_val", "decay", "k", "n_measured", "power")

    def __init__(self, positions, values, k=8, power=2.0, decay=0.5):
        self._pos = np.asarray(positions, dtype=np.float64)
        self._val = np.asarray(values, dtype=np.float64)
        self._conf = np.ones(self._val.shape[0], dtype=np.float64)
        self.n_measured = int(self._val.shape[0])
        self.k = int(k)
        self.power = float(power)
        self.decay = float(decay)

    def at(self, positions: np.ndarray) -> np.ndarray:
        """Estimated attractiveness at `positions`, shape $(Q,)$."""
        return _estimate_attractiveness(
            np.asarray(positions, dtype=np.float64), self._pos, self._val,
            k=self.k, power=self.power, confidence=self._conf,
        )

    def add_inferred(self, positions: np.ndarray, values: np.ndarray) -> None:
        """Fold settled-but-unmeasured points in as reduced-confidence sources.

        For a batched run, where an earlier batch has finished relaxing and its
        points now anchor the next one. Their values were inferred, so they
        enter at `decay` times the mean confidence of the sources that produced
        them rather than as ground truth.
        """
        if len(values) == 0:
            return
        conf = self.decay * float(self._conf.mean())
        self._pos = np.vstack((self._pos, np.asarray(positions, np.float64)))
        self._val = np.concatenate((self._val, np.asarray(values, np.float64)))
        self._conf = np.concatenate(
            (self._conf, np.full(len(values), conf, dtype=np.float64)))


def _zscore(values: np.ndarray) -> np.ndarray:
    """Standardise, returning zeros when every value is the same.

    Used to put novelty and estimated attractiveness on one scale so the
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
    attraction_weight: float = 0.0,
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

    **The placement mirrors the mode.** Given `attract_static`, the winner is
    chosen on a standardised blend of novelty and the attractiveness the
    candidate is *expected* to have,

    $$ s(c) = z\big(\text{novelty}(c)\big)
              + w \, z\big(\hat a(c)\big) $$

    so a repulsive ESS places repulsively and a composite ESS places
    compositely. Measured on the OBLESA sweep, placing on novelty alone is
    worth nothing: unguided dart tied uniform noise 233-243 over 480 paired
    cells, while the same search with a fitness term in the *placement* beat
    it 309-169. Where you probe matters more than how evenly you spread.

    Args:
        index (ToroidalNN): Fitted index holding all existing points.
        n_new (int): Number of points to initialize.
        dim (int): Dimensionality of the space.
        rng (np.random.Generator): Random number generator.
        init_sampler (samplers.Sampler): Candidate-pool sampler (e.g. LHS).
        pool (int): Candidates drawn per slot.
        field (_AttractionField | None): Source of the attractiveness
            estimate. Omit for repulsive placement.
        attraction_weight (float): Balance against novelty in the composite
            score. Zero reproduces the repulsive placement exactly.

    Returns:
        np.ndarray: Initial positions, shape $(n_{new}, D)$, in $[0, 1)$.
    """
    candidates = init_sampler.sample(n_new * pool, dim, rng).astype(np.float64)
    _, dists = index.query(k=1, queries=candidates)
    novelty = dists.reshape(n_new, pool)

    if field is None or attraction_weight == 0.0:
        # Repulsive ESS: placement is repulsive too. Bit-identical to the
        # behaviour before the composite mode existed, so a run without
        # `attractiveness` is unchanged and the composite path is an ablation
        # rather than a different algorithm.
        best = novelty.argmax(axis=1)
    else:
        # Composite ESS: placement is composite too. Scoring on novelty alone
        # puts every candidate as far from everything as possible -- including
        # as far from the good regions -- and then asks the relaxation's
        # attraction to drag it back. That is the attraction spending its
        # budget undoing the placement instead of refining it.
        a_hat = field.at(candidates).reshape(n_new, pool)
        score = np.empty_like(novelty)
        for i in range(n_new):
            # Standardised per slot, so `attraction_weight` is scale-free:
            # novelty shrinks as the space fills, and without this the balance
            # would drift over the batch.
            score[i] = _zscore(novelty[i]) + attraction_weight * _zscore(a_hat[i])
        best = score.argmax(axis=1)

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
    attract: np.ndarray | None = None,
    attraction_weight: float = 1.0,
    attraction_fn: collections.abc.Callable | None = None,
    attraction_kwargs: dict | None = None,
    **metric_kwargs,
) -> np.ndarray:
    r"""Net force on each active point from its neighbour list.

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

    **With `attract` given the sum has a second, signed term:**

    $$ \vec{F}_i = \sum_j \hat u_{ij} f_{\text{rep}}(\hat d_{ij})
       \;-\; w \sum_j \hat u_{ij}\, a_j\, f_{\text{att}}(\hat d_{ij}) $$

    where $a_j \in [0, 1]$ is the neighbour's normalised attractiveness. Both
    terms share the neighbour list and the direction $\hat u$; only the
    magnitude law and the sign differ. Static points carry $a_j$; points this
    call is placing carry zero, because they have no evaluated quality and
    inventing one would be fabrication.

    Note this is a *pairwise* pull, not the gradient of an interpolated
    field, so a weight tuned against a Shepard-style surrogate does not
    transfer to it.

    **The direction is the gradient of the metric supplying the
    magnitudes.** For toroidal L1 that is $\operatorname{sign}(r_l)$: every
    coordinate is pushed equally, rather than hardest along the one that
    already differs most. Until 2026-07-28 the kernel used
    $r_l/\lVert r\rVert_2$ — the $L_2$ gradient — while `dists` carried
    L1 magnitudes, so the applied force descended no potential ESS
    defines. An exactly shared coordinate contributes nothing, since
    $\operatorname{sign}(0) = 0$, which is the symmetric subgradient and
    the only choice that does not invent a direction.

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
        attract (np.ndarray | None): Per-row attractiveness over `all_data`,
            already rank-normalised to $[0, 1]$, zero for rows that are not
            static points. ``None`` evaluates no second term at all — not a
            second term weighted zero — so the default path keeps exactly the
            instruction stream it had before attraction existed.
        attraction_weight (float): $w$, the balance against repulsion.
        attraction_fn (Callable | None): Attraction law, log-space.
        attraction_kwargs (dict | None): Its parameters.
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
        # Every norm here is L1, because every distance here is L1: torann
        # returns toroidal L1, the force law is evaluated on it, and the step
        # is capped in it. A single geometry end to end.
        norms = np.abs(disp).sum(axis=2, keepdims=True)

        stacked = (norms[..., 0] < 1e-9) & valid
        if np.any(stacked):
            noise = rng.standard_normal(size=disp.shape)
            noise /= np.abs(noise).sum(axis=2, keepdims=True) + 1e-9
            disp = np.where(stacked[..., None], noise, disp)
            norms = np.where(stacked[..., None], 1.0, norms)

        # grad(d_L1) = sign(delta): every coordinate is pushed equally, which
        # is what descending L1 means. An exactly shared coordinate
        # contributes nothing, since sign(0) = 0 — the symmetric subgradient,
        # and the only choice that does not invent a direction.
        grad = np.sign(disp)
        norms = np.abs(grad).sum(axis=2, keepdims=True)

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

        # Both are set together by `esa`; testing the callable is what makes
        # that pairing explicit, and it is the thing actually invoked.
        if attract is not None and attraction_fn is not None:
            # The log-sum-exp trick is per *term*, not across terms: a signed
            # sum cannot share one max-subtract, because the shift that keeps
            # one term in range can push the other under it. So the attraction
            # gets its own maximum, its own exponentiation, and the two are
            # combined in linear space. It looks fusable and is not.
            a = attract[safe_ids] * valid
            log_att = attraction_fn(d_hat, **(attraction_kwargs or {}))
            log_att = np.where(valid, log_att, -np.inf)
            m_a = np.max(log_att, axis=1, keepdims=True)
            m_a = np.where(np.isneginf(m_a), 0.0, m_a)
            w_att = np.exp(log_att - m_a) * a
            w_att[~valid] = 0.0
            pull = np.sum(directions * w_att[..., None], axis=1)
            # `directions` points from the neighbour towards the active point,
            # so repulsion adds along it and attraction subtracts.
            forces[lo:hi] -= (
                attraction_weight
                * np.exp(np.minimum(m_a, np.log(force_cap)))
                * pull
            )

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
    init_pool: int = 64,
    attract: np.ndarray | None = None,
    attraction_weight: float = 1.0,
    attraction_fn: collections.abc.Callable | None = None,
    attraction_kwargs: dict | None = None,
    k_att: int = 8,
    att_power: float = 2.0,
    placement_weight: float | None = None,
    att_every: int = 5,
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
        init_pool (int): Candidates per slot for `_smart_init`; see `esa`.
        attract (np.ndarray | None): Normalised attractiveness over
            `all_data`; see `_compute_forces`.
        attraction_weight (float): Balance against repulsion.
        attraction_fn (Callable | None): Attraction law, log-space.
        attraction_kwargs (dict | None): Its parameters.
        k_att (int): Neighbours the attractiveness estimate averages over.
        att_power (float): Inverse-distance exponent of that estimate.
        placement_weight (float | None): Attraction weight for the placement
            only; ``None`` uses `attraction_weight`.
        att_every (int): Refresh the attractiveness field every this many
            epochs. 1 refreshes every epoch.
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

    # One field, built from the measured points, used for both the placement
    # score and the per-epoch refresh below. Every estimate in the run comes
    # out of this object.
    field = None
    if attract is not None and n_static > 0:
        field = _AttractionField(all_data[:n_static], attract[:n_static],
                                 k=k_att, power=att_power)
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
            init = _smart_init(
                index, current_n, dim, rng, init_sampler, pool=init_pool,
                field=field,
                attraction_weight=(attraction_weight
                                   if placement_weight is None
                                   else placement_weight),
            )
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

        if field is not None and attract is not None:
            # Seed the active rows. Left at the zero this buffer was created
            # with they would sit at the *bottom* of the normalised scale --
            # modelled as the least attractive things in the space -- so they
            # would repel each other with no attraction between them. Refreshed
            # every epoch below, because the value belongs to the position.
            attract[cursor : cursor + current_n] = field.at(active)

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

            if (field is not None and attract is not None
                    and epochs_used % att_every == 0):
                # Attractiveness is a field, so it is read at the positions
                # the points currently occupy rather than carried from where
                # they started.
                #
                # On a stride because `STEP_CAP` bounds a point to 2% of the
                # interaction radius per epoch, while the field varies on the
                # scale of the radius itself -- so between refreshes the value
                # can drift by only a few percent, and refreshing every epoch
                # buys accuracy the physics cannot use. It is the difference
                # between the composite path costing ~1.5x the repulsive one
                # and ~1.1x.
                attract[cursor : cursor + current_n] = field.at(active)

            started = time.perf_counter()
            forces = _compute_forces(
                active, all_data, ids, dists, radius, metric_fn, rng,
                attract=attract,
                attraction_weight=attraction_weight,
                attraction_fn=attraction_fn,
                attraction_kwargs=attraction_kwargs,
                **metric_kwargs,
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

            f_max = float(np.max(np.abs(forces).sum(axis=1)))
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
    init_pool: int = 64,
    attractiveness: np.ndarray | None = None,
    attraction_weight: float = 0.5,
    attraction_metric: str | collections.abc.Callable | None = None,
    attraction_kwargs: dict | None = None,
    k_att: int = 8,
    att_power: float = 2.0,
    placement_weight: float | None = None,
    att_every: int = 5,
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
        init_sampler (samplers.Sampler | int | None): Candidate-pool
            sampler; None = LHS. It does not place the initial positions
            directly — it proposes `init_pool` candidates per slot and
            `_smart_init` keeps the farthest. The exception is a run with no
            static points, where there is nothing to be far from and the
            sampler's output is used as drawn.
        init_pool (int): Candidates per slot for that selection, i.e. the
            $k$ of Mitchell's best-candidate. ``1`` disables the selection
            and uses the sampler's raw output.

**The selection earns its place; the pool size is a small,
            dimension-fading tuning.** Measured on a $1N$ sample + $1N$
            quasi-opposite + $1N$ probe pool, 20 seeds, paired per seed
            against the old hardcoded 15, on normalised 2-D projection
            discrepancy (positive = better, with wins out of 20):

            | $d$ | `1` | `64` | best | at |
            | --- | --- | --- | --- | --- |
            | 8 | -5.7% | **+4.4%** (16) | +6.2% (17) | 1024 |
            | 16 | -3.5% | **+1.5%** (15) | +3.1% (17) | 512 |
            | 32 | -0.5% | +0.6% (12) | +2.9% (20) | 2048 |
            | 64 | -0.6% | -0.4% (8) | +0.8% (14) | 1024 |

            ``1`` — no selection at all — is worse at every dimension, which
            is the result that matters: the best-candidate step is doing
            real work. Beyond that the gain rises and then plateaus inside
            its own noise, so there is no interior optimum to find; an
            earlier 6-seed reading suggested one and did not survive more
            seeds.

            The knob fades with dimension: 4-6% at $d = 8$, 1.5-3% through
            $d = 32$, and essentially inert by $d = 64$, where `wrap` does
            not move at all across the whole range. Concentration of measure
            — once every candidate is far from everything, "farthest" stops
            discriminating. 64 is the default because it takes most of the
            low-$d$ gain at 1.7x the cost of 15 and does no harm above;
            1024 buys another 2% at $d = 8$ for roughly 15x the cost.
        attractiveness (np.ndarray | None): One value per row of `samples`,
            **higher = more attractive**. ``None`` (default) is pure
            repulsion and evaluates no second term at all -- not a term
            weighted zero -- so the default path is unchanged.

            The polarity is in the name on purpose: ESS is not told whether
            the caller minimises or maximises, and cannot guess. A caller
            minimising an objective passes the negated values.

            Only existing points carry it. The points being placed have no
            evaluated quality, and inventing one would be fabrication; they
            exert repulsion alone.

            Values are rank-normalised to $[0, 1]$ internally, so the units
            never reach the force and `attraction_weight` means the same
            thing on every objective. See `_rank_normalise`.
        attraction_weight (float): $w$, the balance against repulsion.
            Refused at setup if $w \cdot F_{\text{att}}(0) \ge
            F_{\text{rep}}(0)$, which would let attraction win at contact and
            collapse every active point onto its most attractive neighbour --
            a failure the plateau detector would report as convergence.
        attraction_metric (str | Callable | None): Force law for the pull.
            ``None`` means the same law as `metric`.

            **The same law on both sides cannot produce a well.** Identical
            shapes are proportional, so the net force never changes sign and
            attraction only weakens the push near attractive neighbours
            rather than pulling toward them. That is a legitimate mode -- and
            it is what `linear` on both sides can *only* do, having no shape
            parameter -- but it is rarely what is wanted, so it warns. For a
            genuine equilibrium give the attraction a slower-decaying law,
            e.g. ``metric="softened_inverse", attraction_metric="cauchy"``.
        placement_weight (float | None): Attraction weight used when
            *placing* the points, separately from the weight used in the
            relaxation. ``None`` (default) uses `attraction_weight` for both,
            which is the sensible pairing.

            The two are different mechanisms and separating them is what makes
            their contributions measurable: ``placement_weight=0`` is guided
            relaxation on a repulsive placement (ESS before 0.5.0), while
            ``attraction_weight=0`` with a non-zero `placement_weight` is a
            guided placement that then relaxes on repulsion alone. The second
            is not recommended -- with nothing holding them, the placed points
            drift off the good regions they were put on and push the rest of
            the design around -- but it is the arm that shows why the
            relaxation term is needed rather than assuming it.
        att_every (int): Epochs between refreshes of the attractiveness
            field. The estimate belongs to a position, so it is re-read as the
            points move; a stride of 5 is enough because a point travels at
            most 2% of the interaction radius per epoch.
        attraction_kwargs (dict | None): Parameters for that law, kept
            separate from `metric_kwargs` because both laws take parameters
            of the same names and one ``**kwargs`` cannot serve two.
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
    _check_metric_kwargs(metric_fn, metric_kwargs, isinstance(metric, str))

    attract = None
    attraction_fn = None
    attraction_kw = dict(attraction_kwargs or {})
    if attractiveness is not None:
        # Shape first: it is the most basic thing that can be wrong, and its
        # message should not be pre-empted by a balance complaint.
        a = _rank_normalise(np.asarray(attractiveness, dtype=np.float64))
        if a.size != samples.shape[0]:
            raise ValueError(
                f"attractiveness has {a.size} values for {samples.shape[0]} "
                f"samples. One per existing point: the points being placed "
                f"have no evaluated quality yet, so they carry none."
            )
        # Laid out over `all_data`, which is [static | placed]; the placed
        # rows stay zero and so exert no pull.
        attract = np.zeros(samples.shape[0] + n, dtype=np.float64)
        attract[: samples.shape[0]] = a

        chosen = metric if attraction_metric is None else attraction_metric
        if isinstance(chosen, str):
            attraction_fn = METRIC_REGISTRY.get(chosen.lower())
            if attraction_fn is None:
                raise ValueError(f"Unknown attraction_metric '{chosen}'")
        else:
            attraction_fn = chosen
        if attraction_metric is None and not attraction_kw:
            # same law, same parameters: proportional, so it can never cross
            attraction_kw = dict(metric_kwargs)
        _check_metric_kwargs(attraction_fn, attraction_kw, isinstance(chosen, str))
        if attraction_weight != 0.0:
            # At weight zero there is no attraction to balance, and the
            # "never overcomes repulsion" warning is both true and useless --
            # that is the point of the setting. It is the ablation arm.
            _check_attraction_balance(
                metric_fn, metric_kwargs, attraction_fn, attraction_kw,
                float(attraction_weight),
            )

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
        init_pool=int(init_pool),
        attract=attract,
        attraction_weight=float(attraction_weight),
        attraction_fn=attraction_fn,
        attraction_kwargs=attraction_kw,
        k_att=int(k_att),
        att_power=float(att_power),
        placement_weight=(None if placement_weight is None
                          else float(placement_weight)),
        att_every=max(1, int(att_every)),
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
