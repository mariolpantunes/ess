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
   `M > 2 * d` -- see the table below.
3. **The field is fitted once, from the anchors, before the relaxation
   loop.** The anchors are static, so nothing about the fit changes as the
   run proceeds; `_AttractionField` builds it in its constructor. What repeats
   every `att_every` epochs is *evaluation* at the candidates' new positions,
   which cannot be hoisted out: attractiveness belongs to a position, not to a
   point, and a point that has relaxed into a good region must become an
   attractor rather than carry its placement-time value. The harmonic models
   make that cheap by tabulating the whole torus as `d` one-dimensional curves
   at fit time (`_build_table`), after which a query is integer indexing and a
   sum; `InverseDistance` has no closed form to tabulate and pays `O(M d)` per
   query instead.

**What fits a torus.** The space is $[0, 1)^d$ with opposite faces identified,
so a model of it must be periodic: a polynomial is discontinuous at the seam
and would assert a gradient across a boundary the space does not have. The
periodic analogue of a quadratic well is the von Mises density,
$p(\\theta) \\propto \\exp(\\kappa \\cos(\\theta - \\mu))$, whose logarithm is

$$ \\kappa\\cos(\\theta - \\mu)
   = (\\kappa\\cos\\mu)\\cos\\theta + (\\kappa\\sin\\mu)\\sin\\theta $$

-- one first-harmonic Fourier term per axis. So a first-harmonic model *is* an
additive log-von-Mises field, and that is the right lowest-order shape here
rather than a convenient one. Higher harmonics buy narrower wells and
multimodality, at `2 * harmonics * d` coefficients.

**Which model, and when.** The deciding ratio is coefficients against measured
points, not the choice of basis:

===========================  ============================  =================
model                        coefficients                  identifiable when
===========================  ============================  =================
:class:`HarmonicRidge`       ``2 * harmonics * d + 1``      ``M > 2 * d``
:class:`HarmonicProjection`  ``2 * d`` (each an average)    always
:class:`InverseDistance`     none                          always
===========================  ============================  =================

`HarmonicRidge` *solves* for its coefficients, so it needs more points than
unknowns; below that the ridge penalty shrinks it toward the global mean,
which carries no spatial information at all. `HarmonicProjection` estimates the
same first-harmonic coefficients by **correlating** the attractiveness signal
against the basis instead of inverting a normal-equation matrix. Each
coefficient is then a weighted average, with the variance of a mean rather
than of a regression, and it stays defined however few points there are. The
two agree when the design is orthogonal; they part company exactly where the
solve stops being well-posed.

:class:`InverseDistance` fits nothing and interpolates locally. It degrades to
a *local* mean as distances concentrate, which still moves with position --
where a shrunk parametric fit degrades to a constant.

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


def _wmean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(values.mean()) if total <= 0 else float(
        np.dot(values, weights) / total)


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

    Args:
        k (int): Neighbours averaged over.
        power (float): Inverse-distance exponent.
    """

    def __init__(self, k: int = 8, power: float = 2.0):
        self.k = int(k)
        self.power = float(power)
        self._pos = np.empty((0, 0))
        self._val = np.empty(0)
        self._conf = np.empty(0)

    def fit(self, positions, values, confidence):
        self._pos = np.asarray(positions, dtype=np.float64)
        self._val = np.asarray(values, dtype=np.float64)
        self._conf = np.asarray(confidence, dtype=np.float64)
        return self

    def at(self, positions):
        # Imported here: `ess.ess` imports this module, so a module-level
        # import back into it would close a cycle at import time.
        from ess.ess import _estimate_attractiveness

        return _estimate_attractiveness(
            np.asarray(positions, dtype=np.float64), self._pos, self._val,
            k=self.k, power=self.power, confidence=self._conf,
        )


class _Harmonic(AttractionModel):
    """Shared machinery for the additive trigonometric models.

    Both subclasses fit the same shape,

    $$ \\hat a(x) = b + \\sum_{j}\\sum_{h=1}^{H}
       \\alpha_{jh}\\sin 2\\pi h x_j + \\beta_{jh}\\cos 2\\pi h x_j $$

    and differ only in how the coefficients are obtained. Being additive over
    the axes is what makes the model tabulable as `d` one-dimensional curves
    rather than one `d`-dimensional surface -- a grid would be $g^d$ cells,
    which at $d = 100$ is beyond counting.
    """

    def __init__(self, harmonics: int = 1, bins: int = 4096):
        self.harmonics = max(1, int(harmonics))
        self.bins = int(bins)
        self._w = None
        self._table = None
        self._bias = 0.0

    def _features(self, positions: np.ndarray) -> np.ndarray:
        """`[sin 2*pi*h*x, cos 2*pi*h*x]` for `h = 1..H`, shape $(M, 2Hd)$."""
        ang = 2.0 * np.pi * positions
        blocks = []
        for h in range(1, self.harmonics + 1):
            blocks.append(np.sin(h * ang))
            blocks.append(np.cos(h * ang))
        return np.concatenate(blocks, axis=1)

    def _build_table(self, dim: int) -> None:
        """Tabulate as `d` one-dimensional curves, one per axis.

        `d * bins` floats -- 3.3 MB at `d=100` with 4096 bins -- and
        prediction becomes integer indexing and a sum, with no transcendental
        call at all. The curves are periodic, so the table wraps rather than
        clamps, which is the same reason the basis is trigonometric.
        """
        if self._w is None:
            self._table = None
            return
        grid = (np.arange(self.bins) + 0.5) / self.bins
        ang = 2.0 * np.pi * grid
        table = np.zeros((dim, self.bins))
        for h in range(1, self.harmonics + 1):
            off = 2 * dim * (h - 1)
            table += (np.sin(h * ang)[None, :] * self._w[off:off + dim, None]
                      + np.cos(h * ang)[None, :]
                      * self._w[off + dim:off + 2 * dim, None])
        self._table = table

    def at(self, positions):
        pos = np.asarray(positions, dtype=np.float64)
        if self._w is None:
            return np.full(pos.shape[0], self._bias)
        if self._table is None:
            return self._features(pos) @ self._w + self._bias
        # Linear interpolation between adjacent bins, wrapping at the end.
        # -0.5 because the curves are stored at bin *centres*, (i + 0.5) / B;
        # indexing as though they sat at the left edges offsets every lookup
        # by half a bin, which costs more than the interpolation recovers.
        t = (pos % 1.0) * self.bins - 0.5
        lo = np.floor(t)
        frac = t - lo
        i0 = lo.astype(np.intp, copy=False) % self.bins
        i1 = (i0 + 1) % self.bins
        axes = np.arange(pos.shape[1])
        v0 = self._table[axes, i0]
        v1 = self._table[axes, i1]
        return (v0 + (v1 - v0) * frac).sum(axis=1) + self._bias


class HarmonicProjection(_Harmonic):
    """Coefficients by correlation instead of by solving. **Any `M`.**

    The least-squares estimate needs more points than unknowns. The Fourier
    coefficient of a signal is also just its projection onto the basis,

    $$ \\alpha_{jh} = \\frac{2}{\\sum_i w_i}\\sum_i w_i
       \\left(a_i - \\bar a\\right)\\sin 2\\pi h x_{ij} $$

    which is a weighted average -- variance $\\sigma^2/M$, no matrix to invert,
    defined for any $M \\ge 1$. It coincides with the ridge solution when the
    design is orthogonal, and stays usable exactly where the solve stops being
    well-posed. Centring on $\\bar a$ is what makes it a *contrast*: an axis
    on which attractive and unattractive points sit at the same phase
    contributes nothing rather than contributing noise.

    The price is bias, and it has to be paid for. Correlation ignores the
    correlation *between* basis columns, so overlapping structure is double
    counted where least squares would apportion it, and the per-coefficient
    noise accumulates over all `2d` of them. Left raw, the estimate is worse
    than predicting the mean once `M` falls far enough below `2d`. A
    James-Stein factor keeps only the fitted energy that exceeds what noise
    alone would produce, which is what makes the trade a real one.

    Args:
        harmonics (int): Terms per axis.
        bins (int): Resolution of the tabulated curves.
    """

    def fit(self, positions, values, confidence):
        pos = np.asarray(positions, dtype=np.float64)
        val = np.asarray(values, dtype=np.float64)
        wts = np.asarray(confidence, dtype=np.float64)
        self._bias = _wmean(val, wts) if val.size else 0.0
        total = float(wts.sum())
        if val.size == 0 or total <= 0:
            self._w, self._table = None, None
            return self
        phi = self._features(pos)
        y = (val - self._bias) * wts
        raw = 2.0 * (phi.T @ y) / total

        # Shrink, or the estimate is worse than useless. Each coefficient is a
        # mean of M terms, so it carries noise of variance ~2 Var(y) / M, and
        # the p of them accumulate: unshrunk, the d=100, M=30 field scored a
        # held-out error of 1.93 against 1.0 for simply predicting the mean.
        # The James-Stein factor asks how much of the fitted energy exceeds
        # what noise alone would produce, and keeps only that -- so a model
        # with nothing to say returns the mean instead of shouting.
        energy = float(raw @ raw)
        var = float(np.average((val - self._bias) ** 2, weights=wts))
        noise = raw.size * 2.0 * var / max(1.0, float(val.size))
        self._w = raw * max(0.0, 1.0 - noise / energy) if energy > 0 else None
        self._build_table(pos.shape[1])
        return self


#: Built-in models, by the name `att_model` accepts.
#:
#: `'fourier'`, `'detrended'` and `'auto'` were removed after a 468-arm
#: factorial measured them against `'idw'` inside OBLESA. All three fit `2d`
#: coefficients (`'detrended'` through its trend, `'auto'` through whichever
#: candidate it picks), and a caller supplies anchors by population rather
#: than by dimension, so identifiability needed a population growing linearly
#: in `d` -- where every anchor is a paid objective call. Measured as the
#: share of the selected population the relaxed block won, against 33% parity:
#: `'detrended'` fell 70.4 -> 13.8 across d = 8..64 while `'idw'` held
#: 74.9 -> 71.0. `'auto'` additionally cross-validated four candidates on
#: every fit, which cost 22.8 ms at 60 sources and 86 seconds at 15360.
MODELS = {
    "idw": InverseDistance,
    "projection": HarmonicProjection,
}


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
