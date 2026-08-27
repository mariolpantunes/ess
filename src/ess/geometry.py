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

__all__ = [
    "NEIGHBOUR_TARGET",
    "l1_radius_for_count",
    "radius_for_target",
]

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
