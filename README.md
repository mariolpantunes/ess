# ![ESS Logo](assets/ess_logo.svg) Empty Space Search (ESS)

![PyPI - Version](https://img.shields.io/pypi/v/EmptySpaceSearch)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/EmptySpaceSearch)
![GitHub License](https://img.shields.io/github/license/mariolpantunes/ess)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mariolpantunes/ess/main.yml)
![GitHub last commit](https://img.shields.io/github/last-commit/mariolpantunes/ess)

**ESS** is a high-performance Python library that implements the Electrostatic Search Algorithm (ESA), a novel method for generating spatially diverse point distributions. It simulates electrostatic repulsive forces to "relax" new points into the empty spaces of a high-dimensional domain, making it ideal for sampling, coverage optimization, and exploratory data analysis.

## Features

* **Electrostatic Search Algorithm (ESA)**: Uses physics-inspired repulsive forces (Gaussian, Softened Inverse, etc.) to maximize the separation between points.
* **Toroidal Geometry (New in v0.4.0)**: The relaxation runs on the unit torus $[0, 1)^d$ under the toroidal L1 metric — opposite faces are identified, so there are no walls, no clipping, and no edge-clumping artifacts by construction.
* **Radius-Based Interactions**: Supports physical range searches (interacting with *all* neighbors within a radius) alongside standard k-NN, with automatic L1-ball radius estimation for high-dimensional spaces.
* **Single Scalable Engine**: Neighbour search is [torann](https://github.com/mariolpantunes/torann) — exact brute force at small N, toroidal LSH above its threshold, chosen internally; per-epoch coordinate updates are native (no index rebuilds).
* **Robust Early Stopping**: Convergence is detected on the force field itself (plateau of the largest net force, learning-rate-decoupled), typically stopping in tens of epochs instead of hundreds.
* **High-Dimensional Metrics**: Includes robust coverage metrics (Maximin, Clark-Evans Index, Sparse Grid Coverage) optimized for dimensions > 32D.
* **Smart Initialization**: Uses a vectorized "Best Candidate" sampling strategy to seed new batches in the most promising void regions.

> **Note:** The library is designed to be compliant with modern Python 3.12+ standards.

## Installation

The library can be installed directly from [PyPI](https://pypi.org/project/EmptySpaceSearch/):

```bash
pip install EmptySpaceSearch
```

Alternatively, you can install the latest development version directly from GitHub:

```bash
pip install git+[https://github.com/mariolpantunes/ess.git](https://github.com/mariolpantunes/ess.git)
```

**Requirements:**

* Python >= 3.10
* numpy
* torann

## Usage

### Basic Example

Generate 100 new points in a 2D space using the default settings (LHS Initialization + Auto-Radius + Repulsive Walls):

```python
import numpy as np
import ess

# Define existing points (e.g., obstacles)
obstacles = np.array([[0.5, 0.5]]) 
bounds = np.array([[0, 1], [0, 1]])

# Generate 100 new points
# 'ess' returns the combined set (obstacles + new points)
result = ess.ess(obstacles, bounds, n=100, seed=42)

print(f"Total points: {len(result)}")

```

### Advanced Usage with a Custom Index & LHS Sampler

A pre-configured `torann.ToroidalNN` (specific backend, LSH parameters)
can be passed in, together with space-filling samplers and the
physics-based radius mode:

```python
import numpy as np
from ess import esa, ToroidalNN, LHCSampler

# 1000 existing points in 50 dimensions
dim = 50
obstacles = np.random.rand(1000, dim)
bounds = np.array([[0, 1]] * dim)

# Optional: explicit engine configuration (backend, thresholds, ...)
index = ToroidalNN(seed=42, backend="rust")

# Initialize Space-Filling Sampler (LHS)
lhs_sampler = LHCSampler(random_state=42)

# Run ESA (returns ONLY the new points)
# search_mode='radius' activates the dense physical interaction model
new_points = esa(
    obstacles, 
    bounds, 
    n=500, 
    index=index,
    init_sampler=lhs_sampler,  # Set custom LHS sampler
    search_mode='radius',      # Use radius instead of k-NN
    radius=None,               # None = Auto-compute based on density
    batch_size=100, 
    epochs=256
)

```

## Algorithms

**ESA (Electrostatic Search Algorithm)** treats existing points as fixed charged particles and new points as free moving charges.

1. **k-NN Mode**: Points are repelled by their  nearest neighbors. Good for maintaining local uniformity.
2. **Radius Mode (New)**: Points are repelled by **all** neighbors within a specific cutoff radius . This mimics real electrostatic fields and prevents "tunneling" in high-density regions.

**Force Functions** (all evaluated on the distance normalised by the
interaction radius, so their parameters are dimension-free):

* `gaussian`: Smooth, short-range repulsion. **Default** — best mean
  dispersion in the benchmark.
* `softened_inverse`: Standard electrostatic repulsion (Coulomb-like).
* `linear`: Simple linear drop-off (Hookean spring), hard cutoff at the radius.
* `cauchy`: Heavy-tailed distribution for global separation.

**Repulsion is local.** Only the nearest few neighbours matter: further
ones add an isotropic pressure that moves points without improving
separation. The default `k` is therefore capped (`ess.K_LOCAL`, 5) rather
than growing as `2d+1` — with a growing `k`, a 64-dimensional design has
every point interacting with a quarter of the whole set, which collapses
the one-dimensional marginals and leaves packing *worse* than random.

## Measuring a design

Uniformity metrics do not survive high dimension equally well, so
`ess.utils` offers the ones appropriate to each regime:

| function | what it measures | use when |
| --- | --- | --- |
| `toroidal_clark_evans` | nearest-neighbour regularity on the torus; calibrated (random = 1), bounded by $2/\Gamma(1+1/d)$ | $d \lesssim 16$ |
| `wrap_around_discrepancy` | periodic $L_2$ discrepancy — deviation from uniform over every wrap-around box | any $d$ |
| `projection_discrepancy` | the same, averaged over 1-D / 2-D projections; fixed scale in any ambient dimension | high $d$ (DoE effect sparsity) |
| `calculate_grid_coverage`, `calculate_min_pairwise_distance` | occupancy and separation | any $d$ |

`calculate_clark_evans_index` (Euclidean, box) is kept for backwards
compatibility but is **not** recommended: it carries an uncorrected edge
bias (a random design scores 1.42 at $d = 64$, not 1) and rewards designs
that pile points against the domain walls.

## Benchmark

`examples/benchmark_dispersion.py` runs the calibration behind the
defaults — force-law selection, a tuning grid, and the main sweep over
$d \in \{2,\dots,64\}$ with the number of points scaled to the dimension:

```bash
python examples/benchmark_dispersion.py --phase all --seeds 10
```

## Documentation

This library is documented using Google-style docstrings.

You can access the full documentation online [here](https://mariolpantunes.github.io/ess/).

To generate the documentation locally using [pdoc](https://pdoc.dev/):

```bash
pdoc --math -d google -o docs src/ess \
    --logo assets/ess_logo.svg \
    --favicon assets/ess_logo.svg

```

## Authors

* **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
