"""Models for the attractiveness of a position that was never measured.

Attractiveness is only ever known where the objective was actually paid for.
Everything else is an estimate, and this module is where estimates come from.

**What these models are for, and what they are not.** Every force in ESS acts
between *points*: repulsion pushes a node away from a point, attraction pulls
it toward one. Both therefore need the attractiveness of *both* endpoints, and
the two endpoints are not alike:

* **Anchors are static and measured.** Their positions do not move and their
  attractiveness is the objective value that was paid for. Nothing here is
  used for them.
* **Candidates move, and are never measured.** Their repulsion is still exact
  -- it depends only on distance -- but their attractiveness cannot be
  computed, and leaving it out is not neutral. `_rank_normalise` puts the
  scale on `[0, 1]`, so an unset value reads as the *bottom* of it: every
  candidate would be modelled as the least attractive object in the space,
  candidates that should pull on one another would repel instead, and the only
  attraction in the system would be toward the anchors.

So a model here exists to **keep the pairwise force balance well posed for
points that have no measurement**. It is not a search for the optimum, and it
is not asked to name a position better than anything measured. Do not reason
about it as a surrogate that has to *extrapolate*: what it has to do is supply
a plausible, correctly-ranked value at a moving position.

Three consequences that are easy to get backwards:

1. **Being bounded by the measured range is not a handicap.**
   `InverseDistance` returns a convex combination and so can never leave that
   range. For this job that is a *guarantee*, not a limitation -- the value
   only has to be plausible and correctly ordered, and a bounded estimate
   cannot inject a force from a number that was never observed.
2. **Degrading to the mean is the correct failure.** When a model has nothing
   to say it should say nothing, so the force balance falls back to
   repulsion. A model that instead fits confidently to noise is *worse than
   useless*: it pulls the block somewhere unjustified. This is why `Auto`
   scores abstention fairly and why `HarmonicRidge` is dangerous below
   `M > 2 * d`, which is why no such model is left here.
3. **The field is fitted once, from the anchors, before the relaxation
   loop.** The anchors are static, so nothing about the fit changes as the
   run proceeds; `_AttractionField` builds it in its constructor. What repeats
   every `att_every` epochs is *evaluation* at the candidates' new positions,
   which cannot be hoisted out: attractiveness belongs to a position, not to a
   point, and a point that has relaxed into a good region must become an
   attractor rather than carry its placement-time value. `InverseDistance`
   has no closed form to tabulate, so each refresh pays `O(M d)`; see
   `ess._toroidal_l1` for what that costs and what was done about it.

**One model, and what replaced the choice.** Every parametric model this
module carried fitted `2d` coefficients against `M` measured points, so each
one needed `M > 2d` to be identifiable -- a condition set by the caller's
population, not by anything this module controls. `InverseDistance` fits
nothing, has no such threshold, and is a convex combination of measured
values, so it provably cannot leave their range. It is the only built-in left,
and `k` and `power` are the whole of its configuration.

Its two costs, both measured rather than argued: the query is `O(M d)` with no
closed form to tabulate, and in high dimension distances concentrate so extra
sources stop sharpening the estimate -- leave-one-out error at `d = 100` moves
only 0.624 to 0.614 as `M` goes 60 to 300. Neither bites at the anchor counts
a population-sized caller supplies. Degrading to a *local* mean is also the
right failure: it still moves with position, where a shrunk parametric fit
degrades to a constant and says nothing at all.

**Custom models.** Subclass :class:`AttractionModel`, implement `fit` and `at`,
and pass the instance as `att_model`::

    class MyModel(ess.AttractionModel):
        def fit(self, positions, values, confidence):
            ...
            return self

        def at(self, positions):
            return ...

    ess.esa(samples, bounds, n=60, attractiveness=-scores, att_model=MyModel())
"""

import abc
import inspect

import numpy as np
from torann import ToroidalNN

from . import geometry


class AttractionModel(abc.ABC):
    """Estimates attractiveness at positions on the unit torus.

    `fit` is called whenever the measured set changes and must tolerate being
    called repeatedly. `at` must accept any number of query positions and
    return one value per row, on the same scale as the values it was fitted
    on.
    """

    @abc.abstractmethod
    def fit(self, positions: np.ndarray, values: np.ndarray,
            confidence: np.ndarray) -> "AttractionModel":
        """Take the measured sources.

        Args:
            positions (np.ndarray): Measured points, shape $(M, D)$, in
                $[0, 1)$.
            values (np.ndarray): Their attractiveness, shape $(M,)$, higher
                is more attractive.
            confidence (np.ndarray): Per-source weight in $(0, 1]$. Measured
                points sit at 1; a value inferred earlier enters lower, so a
                chain of inference fades toward the mean rather than
                laundering itself into ground truth.

        Returns:
            AttractionModel: self.
        """

    @abc.abstractmethod
    def at(self, positions: np.ndarray) -> np.ndarray:
        """Estimated attractiveness at `positions`, shape $(Q,)$."""


class InverseDistance(AttractionModel):
    """Shepard interpolation over the `k` nearest measured points.

    No coefficients, so nothing to under-determine: the estimate is a convex
    combination of measured values and can never leave their range.

    **That bound is a guarantee, not a handicap** -- see the module docstring.
    The estimate is not asked to name a position better than anything measured;
    it is asked to give a moving candidate a plausible, correctly-ordered value
    so the force balance is well posed. Being unable to leave the measured
    range means it can never inject a force from a number nobody observed.

    The textbook objection is that distances concentrate in high dimension, so
    the weights flatten and the estimate tends to the mean. That is true of the
    *values* and is the right failure mode -- "no information", not something
    wrong. It is **not** true that the guidance stops working: measured inside
    OBLESA at `attraction_weight=2`, the share of the selected population that
    the relaxed block wins is 74.9 / 73.1 / 72.1 / 71.0 percent at
    `d = 8 / 16 / 32 / 64`, essentially flat, while `Detrended` over the same
    range falls 70.4 -> 41.9 -> 18.0 -> 13.8 as its fit stops being
    identifiable. Do not drop this model on the concentration argument alone.

    **The neighbour search is `torann`, not a second implementation.** The
    library already delegates every toroidal-L1 query to `ToroidalNN`, and
    this model's estimate is a toroidal-L1 k-NN like any other, so it uses an
    index rather than scanning. That index is separate from the relaxation's
    on purpose: the relaxation's holds the moving candidates as well, and an
    estimate must average over *sources* -- points whose objective was paid
    for -- never over the points being placed.

    The index is fitted once, in `fit`, and queried on every refresh. It also
    means the backend decision is `torann`'s: below its crossover the search
    is exact brute force, and above it the same call is LSH, so a caller who
    supplies far more anchors than a population gets the faster path without
    changing anything here.

    **Let it take the LSH path.** The approximation looks alarming measured
    against exact k-NN -- rank correlation between the two estimates falls to
    0.638 at `M = 15360, d = 100` -- but that is the wrong reference. Both are
    noisy readings of one field, and disagreeing with each other says nothing
    about which is closer to it. Scored against the truth they are recovering:

    ==========  =====  ==========  ========  =========
    M           d      exact rho   LSH rho   speed-up
    ==========  =====  ==========  ========  =========
    960         100    0.4649      0.4648    25x
    3840        100    0.5208      0.5281    45x
    15360       100    0.5147      0.5429    111x
    15360       8      0.9613      0.9613    73x
    ==========  =====  ==========  ========  =========

    LSH is as good or better everywhere, and at the largest cell it is
    *better* than exact -- in high dimension the formally-nearest `k` are
    barely nearer than any other `k`, so paying to identify them exactly buys
    precision in a quantity that has stopped carrying information. Forcing
    `brute_threshold` there spends two orders of magnitude for nothing.

    **k-NN or radius.** The two differ in what they hold fixed. ``'k_nn'``
    averages over a fixed *count*, so the neighbourhood grows and shrinks with
    the local density and every candidate is estimated from the same number of
    sources -- a candidate in a void reaches far away for its `k` and is
    smoothed by distant, weakly-related values. ``'radius'`` fixes the
    *volume* instead: a candidate in a dense region averages many sources, one
    in a void averages few or none and falls back to the mean, which is the
    honest answer for a position nothing nearby has measured. Which is better
    is an empirical question and the reason this is a switch.

    Args:
        k (int): Neighbours averaged over in ``'k_nn'`` mode.
        power (float): Inverse-distance exponent.
        backend (str): Forwarded to `ToroidalNN`; `'auto'` picks the compiled
            kernel when it is installed.
        search_mode (str): ``'k_nn'`` or ``'radius'``.
        radius (float | None): Normalized radius in $(0, 1]$ for
            ``'radius'`` mode -- a fraction of the torus diameter, see
            `geometry.radius_from_normalized`. ``None`` or ``0`` derives one
            from the fitted anchors that targets `k` neighbours, so the two
            modes are calibrated to the same neighbourhood by default and
            differ only in which of count and volume is held fixed.
    """

    def __init__(self, k: int = 8, power: float = 2.0, backend: str = "auto",
                 search_mode: str = "k_nn", radius: float | None = None):
        if search_mode not in ("k_nn", "radius"):
            raise ValueError(
                f"search_mode must be 'k_nn' or 'radius', got {search_mode!r}")
        self.k = int(k)
        self.power = float(power)
        self.backend = str(backend)
        self.search_mode = str(search_mode)
        self.radius = radius
        self._val = np.empty(0)
        self._conf = np.empty(0)
        self._index: ToroidalNN | None = None
        self._r: float = 0.0

    def fit(self, positions, values, confidence):
        pos = np.asarray(positions, dtype=np.float64)
        self._val = np.asarray(values, dtype=np.float64)
        self._conf = np.asarray(confidence, dtype=np.float64)
        self._index = None
        if pos.shape[0]:
            n, dim = pos.shape
            # Resolved here, not in `at`: the radius is a property of the
            # source set, which is fixed at fit time, and `at` runs on every
            # refresh.
            self._r = (
                geometry.l1_radius_for_count(dim, n, min(self.k, n))
                if not self.radius
                else geometry.radius_from_normalized(float(self.radius), dim)
            )
            self._index = ToroidalNN(backend=self.backend).fit(
                np.mod(pos, 1.0), k=min(self.k, n))
        return self

    def at(self, positions):
        """Attractiveness a candidate position is *expected* to have.

        Attractiveness is only ever known for the sources -- they are the ones
        whose objective has been paid for. A candidate has none, and treating
        that as zero is not neutral: `_rank_normalise` puts the scale on
        `[0, 1]`, so zero is the *bottom* of it, every candidate is modelled
        as the least attractive thing in the space, candidates that should
        pull on one another repel instead, and the only attraction left in the
        system is toward the sources.

        A candidate sitting exactly on a source takes that source's value
        rather than dividing by zero.
        """
        queries = np.asarray(positions, dtype=np.float64)
        if self._index is None or queries.shape[0] == 0:
            return np.zeros(queries.shape[0], dtype=np.float64)

        q = np.mod(queries, 1.0)
        if self.search_mode == "radius":
            # `pad=True` returns the same dense (m, width) shape `query` does,
            # under the same -1 / inf convention -- which is the whole reason
            # radius mode costs two lines here instead of a second weighting
            # path. A query that found nothing is an all-padded row, and the
            # fallback to the mean below is already what handles it.
            ids, dist = self._index.query_radius(self._r, queries=q, pad=True)
        else:
            kk = min(self.k, self._val.shape[0])
            ids, dist = self._index.query(k=kk, queries=q)

        # LSH mode guarantees `k` results, but a source set smaller than `k`
        # pads with -1 / inf, and a radius that found nothing pads the whole
        # row. All three are handled by zeroing the weight.
        missing = ids < 0
        safe = np.where(missing, 0, ids)
        a_near = self._val[safe]

        exact = (dist <= 0.0) & ~missing
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(exact | missing, 0.0, dist ** (-self.power))
        w = w * self._conf[safe]

        total = w.sum(axis=1)
        out = np.full(queries.shape[0], float(self._val.mean()))
        ok = total > 0
        if ok.any():
            out[ok] = np.einsum("qk,qk->q", w[ok], a_near[ok]) / total[ok]
        hit = exact.any(axis=1)
        if hit.any():
            out[hit] = a_near[hit, np.argmax(exact[hit], axis=1)]
        return out


#: Built-in models, by the name `att_model` accepts. One, now.
#:
#: `'fourier'`, `'detrended'`, `'auto'` and `'projection'` were removed after a
#: 468-arm factorial measured them against `'idw'` inside OBLESA. Every one of
#: them fits `2d` coefficients, and a caller supplies anchors by population
#: size rather than by dimension, so identifiability needs a population growing
#: linearly in `d` -- where every anchor is a paid objective call.
#:
#: Measured as the share of the selected population the relaxed block won,
#: against 33% parity:
#:
#: ============  =====  =====  =====  =====
#: model         d=8    d=16   d=32   d=64
#: ============  =====  =====  =====  =====
#: detrended     70.4   41.9   18.0   13.8
#: projection    71.4   56.8   41.1   27.9
#: idw           74.9   73.1   72.1   71.0
#: ============  =====  =====  =====  =====
#:
#: `'detrended'` and `'projection'` both end below parity, which means their
#: field steers wrong rather than merely saying nothing. `'auto'` additionally
#: cross-validated four candidates on every fit: 22.8 ms at 60 sources, 86
#: seconds at 15360.
#:
#: The one thing lost with the harmonic basis is a query cost flat in `M` --
#: tabulating the torus as `d` one-dimensional curves made a query integer
#: indexing and a sum, where inverse-distance weighting pays `O(M d)`. That
#: matters above roughly four thousand anchors and not below; a
#: population-sized caller supplies 60 to 400.
MODELS = {"idw": InverseDistance}


def get_model(spec, **kwargs) -> AttractionModel:
    """Resolve `att_model` to an instance.

    Accepts a registered name, a subclass, or an already-built instance, so a
    caller can pass `'fourier'`, `HarmonicRidge(harmonics=3)`, or a model of
    their own.

    Args:
        spec (str | type | AttractionModel): What to resolve.
        **kwargs: Forwarded to the constructor when `spec` names a class.
            Ignored for an instance, which is already configured.

    Returns:
        AttractionModel: A model ready to `fit`.

    Raises:
        ValueError: If `spec` is an unknown name.
        TypeError: If `spec` is not a model at all.
    """
    if isinstance(spec, AttractionModel):
        return spec
    if isinstance(spec, str):
        if spec not in MODELS:
            raise ValueError(
                f"unknown att_model {spec!r}; expected one of "
                f"{sorted(MODELS)}, an AttractionModel subclass, or an "
                f"instance of one")
        spec = MODELS[spec]
    if isinstance(spec, type) and issubclass(spec, AttractionModel):
        params = inspect.signature(spec).parameters
        return spec(**{k: v for k, v in kwargs.items() if k in params})
    raise TypeError(
        f"att_model must be a name, an AttractionModel subclass or an "
        f"instance, got {type(spec).__name__}")
