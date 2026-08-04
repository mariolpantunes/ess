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
* `utils`: Metrics for spatial distribution. Three families, and the
  distinction matters: `projection_discrepancy` / `wrap_around_discrepancy`
  reference no point metric and are the ones that rank designs;
  `toroidal_separation` / `toroidal_clark_evans` are $L_1$ statistics valid
  within one fixed geometry; `euclidean_separation` /
  `euclidean_clark_evans` are their non-wrapping counterparts, kept for
  provenance. Never rank arms that optimised different geometries with the
  latter two families.
* `legacy`: Reference implementations of earlier sequential strategies.
"""

# 1. Internal Module Imports
from . import legacy, samplers, utils

# 2. Main API Exports
from .ess import esa, ess

# 3. Neighbour-search engine (re-exported for custom configuration)
from torann import ToroidalNN

# 4. Sampler Exports
from .samplers import LHCSampler, Sampler, UniformSampler, check_sampler

# Define __all__ to control what 'from ess import *' exports
__all__ = [
    "esa",
    "ess",
    "ToroidalNN",
    "Sampler",
    "LHCSampler",
    "UniformSampler",
    "check_sampler",
    "utils",
    "legacy",
    "samplers",
]
