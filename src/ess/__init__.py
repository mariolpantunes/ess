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
and no edge artifacts. `esa(p=...)` selects a toroidal $L^p$ quasi-norm,
$0 < p \le 1$, in its place; the exponent governs the search, the radius,
the step schedule, the placement and the fitness terrain estimate together. Neighbour search is a single engine,
`torann.ToroidalNN`: exact brute force at small $n$, toroidal LSH above
its threshold, chosen internally by size.

Modules
-------
* `ess`: Core generation logic and force-field definitions.
* `attraction`: Static inverse-distance fitness terrain (`InverseDistance`)
  and buffered leave-one-out neighbourhood selection (`choose_k`).
* `geometry`: Toroidal $L^p$ radius laws, diameter, and automatic
  neighbourhood selection (`neighbourhood_for`).
* `samplers`: Space-filling initial-position samplers (LHS, uniform).
* `utils`: Metrics for spatial distribution:
  - `toroidal_separation` (shared with `torann.metrics`): minimum toroidal
    $L_1$ distance from each new point to its nearest other point (existing
    anchor or new point), which directly measures void-filling clearance in
    iterative re-exploration.
  - `wrap_around_discrepancy`, `projection_discrepancy`, and
    `expected_discrepancy`: periodic $L_2$ uniformity diagnostics over the
    full torus or low-dimensional coordinate projections ($1.0$ = random).
  - `euclidean_separation` and `calculate_grid_coverage`: non-wrapping
    separation and low-dimensional grid occupancy, retained for provenance.
* `baseline`: Reference constructions for *testing* ESS, not for using it
  -- `dart` (Mitchell's best-candidate: the ablation of the relaxation),
  `random_fill` (the null), and `grid_oracle` (the emptiest point by
  exhaustion, at low `d`).
* `legacy`: Reference implementations of earlier sequential strategies.

Division of labour with torann
------------------------------
`torann` answers *geometric* questions — which points are near which, and how
far apart they are. It owns the neighbour index and the **definitions** of
toroidal point metrics, because each is one exact k-NN scan.

ESS owns everything about *purpose*: how new points are placed and relaxed
into empty regions (with optional terrain guidance), and how iterative
space exploration is evaluated.

Keep that line. It was drawn after the alternative failed: with a metric
defined in one project and chosen in another, a rename in one silently
outlived the other. Two benchmark scripts kept asking for a removed key and
died in their reporting *after completing every run*, and a third printed a
lower-is-better discrepancy under a higher-is-better heading for weeks —
reporting a 10x improvement as a 10x regression without ever failing.
`test/test_examples.py` runs every example end to end for that reason;
importing them is not enough to catch a bad key in a report function.
"""

import importlib.metadata

__author__ = "Mário Antunes"
__license__ = "MIT"
__email__ = "mario.antunes@ua.pt"
__url__ = "https://github.com/mariolpantunes/ess"
__status__ = "Development"

# Read from the installed distribution rather than a literal here: a hand-kept
# copy drifts from pyproject.toml without failing anything, which is how
# pyBlindOpt shipped 0.3.0 reporting 0.2.0. Source checkouts that were never
# installed have no metadata, hence the fallback.
try:
    __version__ = importlib.metadata.version("EmptySpaceSearch")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

# 1. Internal Module Imports
# 3. Neighbour-search engine (re-exported for custom configuration)
from torann import ToroidalNN

from . import attraction, baseline, geometry, legacy, samplers, utils

# 2. Main API Exports
from .attraction import InverseDistance
from .ess import esa, ess
from .geometry import (
    LOW_DIM,
    NEIGHBOUR_TARGET,
    RADIUS_TARGET,
    Neighbourhood,
    diameter,
    l1_radius_for_count,
    neighbourhood_for,
    radius_for_count,
    radius_for_target,
    radius_target_for,
)

# 4. Sampler Exports
from .samplers import LHCSampler, Sampler, UniformSampler, check_sampler

# Define __all__ to control what 'from ess import *' exports
__all__ = [
    "LOW_DIM",
    "NEIGHBOUR_TARGET",
    "RADIUS_TARGET",
    "InverseDistance",
    "LHCSampler",
    "Neighbourhood",
    "Sampler",
    "ToroidalNN",
    "UniformSampler",
    "attraction",
    "baseline",
    "check_sampler",
    "diameter",
    "esa",
    "ess",
    "geometry",
    "l1_radius_for_count",
    "legacy",
    "neighbourhood_for",
    "radius_for_count",
    "radius_for_target",
    "radius_target_for",
    "samplers",
    "utils",
]
