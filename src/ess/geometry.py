r"""Toroidal-L1 geometry: how wide a neighbourhood is, in a given dimension.

Its own module for two reasons. `attraction` needs the radius law and
`ess` imports `attraction`, so the law cannot live in `ess` without a
cycle. And it is the layer where the ``(0, 1]`` normalization is
interpreted: a radius is normalized precisely so it can cross a boundary
where the geometry is not known, which makes "who converts it back" a
question worth answering in one file rather than at each caller.

The library ranks and relaxes under torann's toroidal $L_1$ metric, so
everything here is derived from that distance law rather than assumed.
"""

from __future__ import annotations

import math
import statistics
from typing import Literal, NamedTuple, get_args

#: Dimension from which radius mode is worth choosing at all.
#:
#: Not a knob inside the target law -- a statement about *which mode* to run.
#: Below it k-NN wins on optimizer outcome at every dimension measured
#: (`cs`: -0.167 against radius's -0.160 at D=2, -0.182 against -0.159 at
#: D=3, -0.124 against -0.096 at D=5) and is cheaper besides. From D=10
#: radius wins on quality, and from D=40 on cost as well.
#:
#: Passed to `radius_target_for` as `low_dim` so it can be moved, and
#: measured rather than assumed -- stage g put the crossover here.
LOW_DIM = 10

__all__ = [
    "LOW_DIM",
    "NEIGHBOUR_TARGET",
    "RADIUS_TARGET",
    "SEARCH_MODES",
    "Neighbourhood",
    "ResolvedMode",
    "SearchMode",
    "l1_radius_for_count",
    "neighbourhood_for",
    "radius_for_target",
    "radius_target_for",
]

NEIGHBOUR_TARGET = 2
r"""int: Neighbours the interaction radius should contain in **k-NN mode**.

Here the radius never selects anything -- `k` does that -- and this only
sets the scale the force law is evaluated on, $\hat{d} = d_{L1}/R$. The
target is inert above 2 for dispersion (1.469 vs 1.477 at $d=8$, 1.299 vs
1.296 at $d=32$ for targets 2 and 3), so the cheapest safe value wins; 1 is
too tight in low dimension (2D toroidal Clark-Evans 1.86 versus 2.08).

It is *not* inert for the attraction balance, which is why this stayed at 2
when `RADIUS_TARGET` moved: raising it enlarges $R$, shrinks every
$\hat{d}$, and shifts where on the force curve a given
`attraction_weight` lands. At 5 a weight of 0.2 stops clearing the margin
a test pins it to. Two jobs, two constants.
"""

RADIUS_TARGET = 5
r"""int: Floor on the neighbour count in **radius mode**.

Was the whole answer; is now the lower bound of `radius_target_for`, which
scales the target with the dimension. It still sets the floor for the same
reason it was chosen: below about five neighbours the estimate is starved,
and at two roughly one point in eight at $D=100$ has an empty neighbourhood,
feels no force at all and never moves.

Matched to `ess.K_LOCAL`, so the two modes start from the same
neighbourhood and differ only in whether the *count* or the *volume* is
held fixed. That is the comparison worth making; anything else compares two
differently sized neighbourhoods and calls it a comparison of modes.

Radius mode inherited `NEIGHBOUR_TARGET`, and 2 was measured for the other
mode. Here the number *is* the search cutoff, which is a different job: at a
target of 2, **10.7% of points at $d=100$ and 14.5% at $d=200$ have an empty
neighbourhood**, feel no force at all and never move. By 4 that is 1.3%; by
8 it is zero.

Five is an interim value, not a measured optimum. Design quality keeps
improving with larger targets at every dimension tested -- radius mode
overtakes k-NN at 32, 64 and 128 for $d = 40, 100, 200$ -- so the right
default is likely higher and probably dimension-dependent. Five is where the
dead-point problem is largely gone and the two modes are comparable; the
sweep that settles the rest is running.
"""


type SearchMode = Literal["auto", "k_nn", "radius"]
"""What a caller may ask for: a mode, or ``"auto"`` to be told one."""

type ResolvedMode = Literal["k_nn", "radius"]
""""What a run actually does: ``"auto"`` is answered here, never forwarded."""

#: The accepted spellings, read off `SearchMode` rather than repeated, so the
#: validation and the annotation cannot drift apart.
SEARCH_MODES: tuple[str, ...] = get_args(SearchMode.__value__)


class Neighbourhood(NamedTuple):
    """A resolved neighbourhood: which query to run, and how wide.

    One value rather than two, because it is one decision. The count means
    different things in the two modes -- in ``"radius"`` it *is* the search
    cutoff, in ``"k_nn"`` it only sets the scale the force law is evaluated
    on -- so a mode and a target that were not resolved together are a bug
    waiting to happen. It unpacks like the pair it replaces.
    """

    mode: ResolvedMode
    target: int


#: Largest possible toroidal-L1 distance, per dimension. Each axis
#: contributes at most 1/2 (past that the wrap is the shorter way round), so
#: the diameter of the unit torus under this metric is exactly ``dim / 2``.
#: That is the scale a normalized radius is expressed in.
_AXIS_MAX = 0.5


def l1_radius_for_count(
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


def radius_target_for(
    dim: int,
    n_points: int,
    *,
    low_dim: int = LOW_DIM,
    low_target: int = RADIUS_TARGET,
) -> int:
    r"""Neighbours radius mode should aim for, from the dimension and the
    design size. This is what makes radius mode parameter-free.

    $$ t = \min\big(\max(2D,\ \texttt{RADIUS\_TARGET}),\ \lfloor N/2 \rfloor\big) $$

    **Why $2D$.** It is the spanning requirement, and it long predates this
    function -- it was the original k-NN default, on the argument that in one
    dimension a point needs a neighbour on each side to settle, so $D$
    dimensions need $2D$. Measured against the sweep optima at
    ``force_weight=1``, it is also simply the best-fitting law: mean factor
    error 1.22x, against 1.45x for $c\sqrt{D}$ and 1.58x for $c\log_2 D$.
    Those two fail on *curvature*, not calibration -- fitted optimally they
    still over-provision 2x at $D=10$ (34 against a measured 16) and
    under-provision at $D=40$ (67 against 96), because neither can bend
    enough. Observed optima were 16 at $D=10$, 32-48 at $D=20$ and 96 at
    $D=40$.

    **Why the cap, and why it is not optional.** $2D$ is unbounded and the
    design is not. A population-sized pool holds $3N \approx 30\sqrt{D}$
    points, so demand grows like $D$ while supply grows like $\sqrt{D}$:
    $2D$ is 42% of the pool at $D=40$, 95% at $D=200$, and **exceeds it
    entirely at $D=225$**. Past that the rule would silently ask for more
    neighbours than exist. The cap is where $\sqrt{D}$ belongs -- $N/2$ is a
    $\sqrt{D}$ ceiling because $N$ is -- and at $D=1000$ it yields 474
    rather than 2000.

    Half the pool is also where the neighbourhood stops being local: the
    measured optimum at $D=40$ is 96 of 189 points, 51%. Beyond that the
    radius grows enough that every $\hat{d} = d_{L1}/R$ compresses and the
    force law flattens toward uniform, which is the mechanism behind the
    optima being *interior* rather than "more is better". Too few neighbours
    leaves points with none at all; too many erase the local structure the
    field exists to express.

    When the cap binds, the answer is a larger pool -- `n_ess`, or more
    rounds -- not a different formula. The design has run out of points.

    Note:
        Fitted at ``force_weight=1``. The stronger-attraction operating point
        has its optima pinned at the pool ceiling throughout, so whether it
        wants a term of its own is open until that sweep lands.

        Radius mode itself is worth choosing from about $D=10$: below that
        k-NN is both better and cheaper, above it radius wins on quality from
        $D=10$ and on cost from $D=40$.

    **Below `low_dim`.** Radius mode is not the mode to be running there --
    k-NN wins on optimizer outcome at every dimension measured below the
    crossover -- so this returns a flat `low_target` rather than pretending
    the law extends down. It is flat rather than tuned because tuning it buys
    nothing: measured on projection discrepancy, a fixed 5 is 9% worse than
    $2D$ at $D=3$ and 33% worse at $D=5$, and **not faster** at either (15.7
    ms against 12.8 at $D=3$), because a pool of 51 points has no work for a
    smaller neighbourhood to save. The saving only appears at $D=10$, which
    is the far side of the crossover.

    So `low_dim` marks where the *mode* stops being the right choice, and
    both it and `low_target` are arguments rather than constants in the body
    because both are measurements and measurements move.

    Args:
        dim (int): Dimensionality $D$.
        n_points (int): Points the neighbourhood is drawn from (static +
            generated) -- the pool, not the population.
        low_dim (int): Dimension below which the flat `low_target` is used
            instead of the law. Defaults to `LOW_DIM` (10), the measured
            crossover.
        low_target (int): The flat count used below `low_dim`, and the floor
            everywhere else. Defaults to `RADIUS_TARGET` (5).

    Returns:
        int: Neighbours to target, at least `RADIUS_TARGET` unless the design
            is too small to hold that many, and never more than half of it.

    Example:
        >>> radius_target_for(40, 189)      # 2D = 80, well under the cap
        80
        >>> radius_target_for(1000, 948)    # 2D = 2000; the cap binds
        474
    """
    want = low_target if dim < low_dim else max(2 * dim, low_target)
    cap = max(1, n_points // 2)
    return int(min(want, cap))



def neighbourhood_for(
    dim: int,
    n_points: int,
    mode: SearchMode = "auto",
    target: int | None = None,
    *,
    low_dim: int = LOW_DIM,
    low_target: int = RADIUS_TARGET,
) -> Neighbourhood:
    r"""Which neighbour query to run here, and how wide -- the whole of the
    ``"auto"`` policy, in one place.

    `radius_target_for` answers *how many* neighbours radius mode should
    aim for. This answers the question before it: whether radius mode is
    the mode to be running at all. Both are needed for the default path to
    be parameter-free, and both are measurements.

    **The rule.** Below `low_dim`, k-NN; from `low_dim` up, radius with the
    target the law gives. Radius mode loses on optimizer outcome at every
    dimension measured below the crossover -- `cs` reaches $-0.167$ against
    radius's $-0.160$ at $D=2$, $-0.182$ against $-0.159$ at $D=3$,
    $-0.124$ against $-0.096$ at $D=5$ -- and is not cheaper there either,
    because a 51-point pool has no work for a smaller neighbourhood to
    save. From $D=10$ radius wins on quality, and from $D=40$ on cost as
    well (0.90x the k-NN time at $D=40$, 0.73x at $D=100$), which is the
    regime that matters: it is the *high*-dimensional runs that are
    expensive.

    So ``"auto"`` is not a compromise between the two modes. It runs each
    one where that one is both better and cheaper, and the crossover
    between those regimes is a single dimension.

    **Passing a mode still means it.** An explicit ``"radius"`` at $D=3$
    gets radius mode; the flat `low_target` in `radius_target_for` is what
    keeps that sensible. ``"auto"`` is a default, not a policy imposed on a
    caller who has decided.

    Note:
        Measured on the repulsion, which is what `search_mode` governs. The
        attraction estimate is a separate use of the index with its own
        argument (`ess.esa`'s ``att_search_mode``) and its own neighbour
        count, and this crossover is not evidence about it.

    Args:
        dim (int): Dimensionality $D$.
        n_points (int): Points the neighbourhood is drawn from (static +
            generated) -- the pool, not the population.
        mode (SearchMode): ``"auto"``, ``"k_nn"`` or ``"radius"``.
        target (int | None): Neighbours to aim for. ``None`` takes the
            default for whichever mode is chosen: `NEIGHBOUR_TARGET` for
            k-NN, where the radius only scales the force law, and
            `radius_target_for` for radius, where it is the search cutoff.
        low_dim (int): Dimension from which ``"auto"`` chooses radius mode.
            Defaults to `LOW_DIM` (10), the measured crossover.
        low_target (int): Floor on the radius target, and the flat count
            `radius_target_for` uses below `low_dim`. Defaults to
            `RADIUS_TARGET` (5).

    Returns:
        Neighbourhood: The mode to run and the neighbour count to aim for.

    Raises:
        ValueError: If `mode` is not one of `SEARCH_MODES`.

    Example:
        >>> neighbourhood_for(3, 51)        # below the crossover
        Neighbourhood(mode='k_nn', target=2)
        >>> neighbourhood_for(40, 189)      # above it, 2D = 80
        Neighbourhood(mode='radius', target=80)
        >>> neighbourhood_for(40, 189, "k_nn")
        Neighbourhood(mode='k_nn', target=2)
    """
    if mode not in SEARCH_MODES:
        raise ValueError(
            f"search_mode must be one of {SEARCH_MODES}, got {mode!r}")

    chosen: ResolvedMode = (
        ("k_nn" if dim < low_dim else "radius") if mode == "auto" else mode
    )
    if target is not None:
        return Neighbourhood(chosen, int(target))
    if chosen == "k_nn":
        return Neighbourhood(chosen, NEIGHBOUR_TARGET)
    return Neighbourhood(chosen, radius_target_for(
        dim, n_points, low_dim=low_dim, low_target=low_target))


def radius_from_normalized(value: float, dim: int) -> float:
    """A normalized radius as a toroidal-L1 distance.

    The single place the ``(0, 1]`` convention is interpreted. It lives here
    rather than at any caller because the conversion needs ``dim`` and the
    metric, and this is the layer that owns both -- a caller like OBLESA
    holds points that came out of `esa` and has no way to know the geometry
    they were relaxed under.

    **What the number means in the caller's own units.** `esa` min-maxes
    each axis onto ``[0, 1]`` *independently*, and the metric is an L1 sum
    over all of them, so the internal radius is a sum of `dim` dimensionless
    per-axis fractions -- there is no single length in domain units it
    corresponds to unless every axis happens to share a unit and a width.
    The scale is still readable per axis, and that is the useful reading:
    ``value / 2`` is the mean fraction of *each axis's own range* the ball
    reaches. On bounds of ``[-5, 5]``, ``value=0.2`` reaches 10% of the
    range, so 1.0 in the caller's units, on a typical axis.

    Args:
        value (float): Fraction of the torus diameter, in ``(0, 1]``.
        dim (int): Dimensionality.

    Returns:
        float: The radius in toroidal L1 units.

    Raises:
        ValueError: If `value` is outside ``[0, 1]``.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"radius must be in [0, 1] (0 = auto), got {value}")
    return float(value) * dim * _AXIS_MAX


def radius_for_target(
    dim: int, n_points: int, target: int = NEIGHBOUR_TARGET
) -> float:
    r"""Normalized radius expected to contain `target` neighbours.

    The companion to passing ``radius=`` explicitly. A radius is normalized
    against the torus diameter so it can be handed across a layer that does
    not know the geometry, but that makes the *useful* band narrow and
    dimension-dependent: it sits near $2R/d$, and its width shrinks like
    $1/\sqrt{d}$ because the distance distribution concentrates. At $d=1000$
    a normalized $0.48$ contains around 1% of the points and $0.52$ around
    99%. Guessing a value inside that band is not something a caller should
    be asked to do.

    So this answers the question a caller actually has -- "how wide is about
    eight neighbours, here?" -- with the number to pass. It is
    `l1_radius_for_count`, which is what ``radius=0`` uses internally,
    divided by the diameter.

    Args:
        dim (int): Dimensionality $d$.
        n_points (int): Total points $N$ the radius will search among
            (static + generated).
        target (int): Desired neighbours inside the ball.

    Returns:
        float: A value in $(0, 1]$ for ``esa(radius=...)`` or
            ``oblesa(radius=...)``.

    Example:
        >>> round(radius_for_target(100, 400), 3)
        0.426
    """
    return l1_radius_for_count(dim, n_points, target) / (dim * _AXIS_MAX)
