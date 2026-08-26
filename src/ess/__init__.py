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
  choice of geometry can flatter an arm that optimised it. ESS relaxes
  under a toroidal-$L_1$ repulsion, so ranking it on a toroidal-$L_1$ point
  metric would be grading the optimiser with its own loss, and would report
  a large margin for exactly that reason and no other.
  `toroidal_separation` is that point metric — a raw $L_1$ gap, useful as a
  diagnostic within one fixed geometry, and **never** a ranking here. Its
  definition comes from `torann.metrics` (see *Division of labour* below);
  `euclidean_separation` is its non-wrapping counterpart, kept for
  provenance. Clark-Evans is gone: it only has meaning divided by an
  expected nearest-neighbour distance, which makes it a statistic *about a
  metric* rather than about uniformity, and it was measurably blunt
  besides.
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

ESS owns everything about *purpose*: how designs are generated, and how they
are **ranked**. Which metric decides that a design is better is a question
about what the points are for, and the reasoning above — rank on
discrepancies, never on a point metric ESS itself optimises — is this
project's to make and lives here.

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

from . import attraction, baseline, legacy, samplers, utils

# 2. Main API Exports
from .attraction import AttractionModel, InverseDistance
from .ess import esa, ess

# 4. Sampler Exports
from .samplers import LHCSampler, Sampler, UniformSampler, check_sampler

# Define __all__ to control what 'from ess import *' exports
__all__ = [
    "AttractionModel",
    "InverseDistance",
    "LHCSampler",
    "Sampler",
    "ToroidalNN",
    "UniformSampler",
    "attraction",
    "baseline",
    "check_sampler",
    "esa",
    "ess",
    "legacy",
    "samplers",
    "utils",
]
