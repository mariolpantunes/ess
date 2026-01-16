"""
ESS: Empty Space Strategy
=========================

A library for generating spatially diverse point distributions in high-dimensional
spaces using physics-based repulsion simulations.

The Empty Space Strategy (ESS) fills the "voids" in a design space by introducing
active particles that repel each other and existing static points. This is particularly
useful for:
1. **Optimization Initialization**: Finding diverse starting points for population-based algorithms.
2. **Design of Experiments (DoE)**: Creating space-filling designs.
3. **Sampling**: Generating high-entropy distributions in bounded domains.

Key Algorithms
--------------
The library provides two main entry points:
* `esa`: **Empty Space Algorithm** - Returns *only* the new generated points.
* `ess`: **Empty Space Strategy** - Returns the combined set (Original + New).

Architecture
------------
The simulation is powered by swappable Nearest Neighbor (NN) engines to handle
forces efficiently across different scales:
* **Small N**: Uses `NumpyNN` for vectorized brute-force exact calculation.
* **Large N**: Uses `FaissHNSWFlatNN` for approximate, graph-based queries.

Modules
-------
* `ess`: Core generation logic and force field definitions.
* `nn`: Abstract and concrete NN implementations (Numpy, Faiss).
* `utils`: Metrics for spatial distribution (Coverage, Clark-Evans Index, Maximin).
* `legacy`: Reference implementations of earlier sequential strategies.
"""

# 1. Internal Module Imports
from . import legacy, nn, utils

# 2. Main API Exports
from .ess import esa, ess

# 3. NN Engine Exports
from .nn import FaissFlatL2NN, FaissHNSWFlatNN, NearestNeighbors, NumpyNN

# Define __all__ to control what 'from ess import *' exports
__all__ = [
    "esa",
    "ess",
    "NearestNeighbors",
    "NumpyNN",
    "FaissFlatL2NN",
    "FaissHNSWFlatNN",
    "nn",
    "utils",
    "legacy",
]
