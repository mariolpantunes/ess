r"""
ESS: Empty Space Strategy
=========================

A library for generating spatially diverse point distributions in
high-dimensional spaces using physics-based repulsion simulations.

The Empty Space Strategy (ESS) fills the "voids" in a design space by
introducing active particles that repel each other and existing static
points. This is particularly useful for:

1. **Optimization Initialization**: Diverse starting points for
   population-based algorithms.
2. **Design of Experiments (DoE)**: Space-filling designs.
3. **Sampling**: High-entropy distributions in bounded domains.

Key Algorithms
--------------
The library provides two main entry points:

* `esa`: **Empty Space Algorithm** — returns *only* the new generated points.
* `ess`: **Empty Space Strategy** — returns the combined set (original + new).

Architecture
------------
The simulation runs on the unit torus $[0, 1)^d$ under the toroidal L1
metric — opposite faces are identified, so the relaxation has no walls
and no edge artifacts. Neighbour search is a single engine,
`torann.ToroidalNN`: exact brute force at small $n$, toroidal LSH above
its threshold, chosen internally by size.

Modules
-------
* `ess`: Core generation logic and force-field definitions.
* `samplers`: Space-filling initial-position samplers (LHS, uniform).
* `utils`: Metrics for spatial distribution. **Rank designs with
  `projection_discrepancy` and `wrap_around_discrepancy`** — they measure
  deviation from uniformity and reference no point metric at all, so no
  choice of geometry can flatter an arm that optimised it.
  `toroidal_separation` is a raw $L_1$ gap, useful as a diagnostic within
  one fixed geometry; `euclidean_separation` is its non-wrapping
  counterpart, kept for provenance. Clark-Evans is gone: it only has
  meaning divided by an expected nearest-neighbour distance, which makes it
  a statistic *about a metric* rather than about uniformity, and it was
  measurably blunt besides.
* `baseline`: Reference constructions for *testing* ESS, not for using it
  -- `dart` (Mitchell's best-candidate: the ablation of the relaxation),
  `random_fill` (the null), and `grid_oracle` (the emptiest point by
  exhaustion, at low `d`).
* `legacy`: Reference implementations of earlier sequential strategies.
"""

# 1. Internal Module Imports
# 3. Neighbour-search engine (re-exported for custom configuration)
from torann import ToroidalNN

from . import baseline, legacy, samplers, utils

# 2. Main API Exports
from .ess import esa, ess

# 4. Sampler Exports
from .samplers import LHCSampler, Sampler, UniformSampler, check_sampler

# Define __all__ to control what 'from ess import *' exports
__all__ = [
    "LHCSampler",
    "Sampler",
    "ToroidalNN",
    "UniformSampler",
    "baseline",
    "check_sampler",
    "esa",
    "ess",
    "legacy",
    "samplers",
    "utils",
]
