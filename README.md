# ![ESS Logo](assets/ess_logo.svg) Empty Space Search (ESS)

![PyPI - Version](https://img.shields.io/pypi/v/EmptySpaceSearch)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/EmptySpaceSearch)
![GitHub License](https://img.shields.io/github/license/mariolpantunes/ess)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mariolpantunes/ess/main.yml)
![GitHub last commit](https://img.shields.io/github/last-commit/mariolpantunes/ess)

**ESS** is a high-performance Python library that implements the Empty Space Algorithm (ESA), a novel method for generating spatially diverse point distributions.
It simulates electrostatic repulsive forces to "relax" new points into the empty spaces of a high-dimensional domain, making it ideal for sampling, coverage optimization, and exploratory data analysis.

## Features

* **Empty Space Algorithm (ESA)**: Uses physics-inspired repulsive forces (Gaussian, Softened Inverse, etc.) to maximize the separation between points.
* **Toroidal Geometry (New in v0.4.0)**: The relaxation runs on the unit torus $[0, 1)^d$ under the toroidal L1 metric — opposite faces are identified, so there are no walls, no clipping, and no edge-clumping artifacts by construction.
* **Toroidal $L^p$, $0 < p \le 1$ (New in v0.8.0)**: `esa(..., p=0.5)` runs the neighbour search, the interaction radius, the step schedule, the placement and the fitness terrain estimate under the toroidal $L^p$ quasi-norm, where a point is near the points that agree with it on most axes. The default is $p = 1$. The exponent decides which points interact and how strongly; the push itself is along $\operatorname{sign}(r)$ of the displacement at every $p$ (the exact gradient at $p = 1$, a measured choice below it). Requires `torann>=0.6.0`.
* **Radius-Based Interactions**: Supports physical range searches (interacting with *all* neighbors within a radius) alongside standard k-NN, with automatic L1-ball radius estimation for high-dimensional spaces.
* **Single Scalable Engine**: Neighbour search is [torann](https://github.com/mariolpantunes/torann) — exact brute force at small N, toroidal LSH above its threshold, chosen internally; per-epoch coordinate updates are native (no index rebuilds).
* **Single-Parameter Own-Gap Step & Post-Relaxation Re-Placement**: Each epoch moves a point by its own nearest-neighbour distance divided by $t + C$ (`C=12.0` by default) along the unit force direction, eliminating separate learning-rate, decay and step-cap knobs. After convergence, the lowest-scoring fraction `w` (`w=0.05` by default, 5%) is offered a best-candidate re-placement (`_smart_init`), accepted only if it improves clearance.
* **Robust Early Stopping**: Convergence is detected on what the relaxation is for: the run stops when the nearest-neighbour distance of the points being placed (its 5% quantile, read off the neighbour query) and, when guided, the spread-scaled mean field elevation stop improving for `patience` (default 12) epochs.
* **High-Dimensional Evaluation Metrics**: Evaluates iterative void-filling on the toroidal distance from each newly placed point to its nearest other point (`toroidal_separation`, shared with [torann](https://github.com/mariolpantunes/torann), normalised by the fair-share radius `geometry.radius_for_count`), alongside wrap-around and projection discrepancies (`wrap_around_discrepancy`, `projection_discrepancy`, normalised so 1.0 = random).
* **Smart Initialization**: Uses a sequential "Best Candidate" sampling strategy to seed new batches in the most promising void regions without collapsing.
* **Space-Based Terrain Guidance**: Given the `fitness` of the measured points (**lower = closer to the ground = better**), the space itself acts as an inverse-distance terrain (`attraction.InverseDistance`) built once via buffered leave-one-out cross-validation (`attraction.choose_k`, selecting the optimal neighbourhood count $k$ and field `trust` in $[0, 1]$). During relaxation, uphill movement along the terrain slope (climbing towards higher fitness/cost) is damped without injecting point-source pulls that could collapse the search.

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

* Python >= 3.12
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
# search_mode defaults to 'auto', which picks k-NN or radius from the
# dimension; naming one overrides that, as here
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

### Guided ESS: terrain guidance toward promising regions

Pure repulsion treats every empty region as equally worth probing. Given the
measured `fitness` of the existing points (**lower = closer to the ground = better**,
passed as a keyword argument; from `fitness` onwards all arguments are
keyword-only), the space itself acts as a terrain that guides both initial
placement and relaxation:

```python
new_points = esa(
    samples, bounds,
    fitness=objective_values,           # lower = closer to the ground = better
    n=60,                               # defaults to max(1, len(samples)) if omitted
    p=1.0,                              # toroidal L^p metric exponent (0 < p <= 1)
    C=12.0,                             # own-gap step schedule: near_i / (t + C)
    w=0.05,                             # 5% post-relaxation worst-point re-placement
    fitness_weight=1.0,                 # uphill damping & placement weight
)
```

**How the terrain works:**

1. **Static per call, intensive, and IDW-only**: Only the measured anchors carry
   true objective values, interpolated by inverse-distance weighting
   (`attraction.InverseDistance`, the only model that survives high dimensionality
   without fitting under-determined coefficients). At the start of `esa`,
   `attraction.choose_k` runs a buffered leave-one-out cross-validation (at
   buffer $R_1$, the fair-share radius) to choose the optimal neighbourhood count
   $k \in \{8, 32, 128, M\}$ and compute the field's rank-correlation
   `trust` in $[0, 1]$. At low dimension it picks local neighbourhoods ($k = 8$);
   at high dimension ($d \ge 16$) it widens to all anchors so the slope stays
   informative despite distance concentration.
2. **Uphill terrain damping (`damp`)**: No point exerts an attractive pull on
   another point (which would collapse when points accumulate in good regions).
   Instead, particles repel each other and the field slope $\nabla f(x)$
   damps the component of the repulsive force that pushes **uphill** (towards
   higher cost / worse expected fitness), while movement **downhill** towards
   the ground is unbraked.
3. **Neutral on flat or uninformative landscapes**: A constant `fitness`
   vector or a deceptive landscape with `trust == 0` produces zero damping and
   reproduces pure repulsion bit-for-bit.

## Algorithms

**ESA (Empty Space Algorithm)** treats existing points as fixed charged particles and new points as free moving charges.

1. **k-NN Mode**: Points are repelled by their  nearest neighbors. Good for maintaining local uniformity.
2. **Radius Mode**: Points are repelled by **all** neighbors within a specific cutoff radius. This mimics real electrostatic fields and prevents "tunneling" in high-density regions.
3. **Auto (default, new in v0.7.2)**: the mode is chosen from the
   dimension. Below `geometry.LOW_DIM` (10) k-NN wins on optimizer outcome
   and is no more expensive; from there up radius mode wins on quality, and
   from `D = 40` on cost as well (0.90x the k-NN time at `D = 40`, 0.73x at
   `D = 100`). `ess.neighbourhood_for(dim, n_points)` is that policy on its
   own, and it answers the neighbour *count* too, so the default path takes
   no radius tuning at any dimension. Naming a mode still gets that mode.

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

ESS is an iterative space-exploration method: given existing points (anchors), it places and relaxes new points into unexplored regions. `ess.utils` and `ess.geometry` provide metrics for both iterative clearance and global uniformity:

| function | what it measures | primary role |
| --- | --- | --- |
| `toroidal_separation` | smallest toroidal $L_1$ gap from each new point to its nearest other point (defined in `torann.metrics`) | **iterative clearance / coverage** (normalise by `radius_for_count(d, n_total, target=1)`) |
| `wrap_around_discrepancy` | deviation from uniform over every wrap-around box, full dimension | global uniformity diagnostic |
| `projection_discrepancy` | the same, averaged over 1-D / 2-D coordinate projections; fixed scale in any ambient $d$ | marginal uniformity diagnostic |
| `expected_discrepancy` | the null both discrepancies are divided by, so 1.0 = as uniform as random | — |
| `euclidean_separation`, `calculate_grid_coverage` | non-wrapping separation, and grid occupancy | provenance only |

### Division of labour with torann

`torann` answers **geometric** questions — which points are near which, and how
far apart. It owns the neighbour index and the *definitions* of toroidal point
metrics, each being one exact k-NN scan. ESS owns everything about **purpose**:
how new points are generated and relaxed into empty regions, and how they are
evaluated.

The line was drawn after the alternative failed. With a metric defined in one
project and chosen in another, a rename in one silently outlived the other: two
benchmark scripts kept asking for a removed key and died in their reporting
*after completing every run*, and a third printed a lower-is-better discrepancy
under a higher-is-better heading — reporting a 10x improvement as a 10x
regression — for weeks, without ever failing. `test/test_examples.py` now runs
every example end to end, because importing them does not catch a bad key
inside a report function.

**Evaluating iterative re-exploration vs. one-shot uniformity.** Because ESS
iteratively adds points around existing anchors (both across relaxation epochs
and across repeated rounds), one-shot designs like LHS or Sobol are not
comparable baselines: they lay out a fixed count at once and ignore existing
points. At high dimension the volume is too large for a modest point budget to
eliminate global holes, so iterative exploration is evaluated first on the
clearance of the newly placed points to their nearest other point
(`toroidal_separation`, normalised by the fair-share radius
`R1 = geometry.radius_for_count(d, n_total, target=1)`), and second on the
expected fitness of the regions discovered when guided.

For full-design uniformity diagnostics across different geometries, divide
`wrap_around_discrepancy` or `projection_discrepancy` by
`expected_discrepancy(n, s)`: 1.0 is as uniform as random, and lower is better.
`calculate_grid_coverage` saturates above $d \approx 8$ and cannot be built
past $d \approx 20$. Clark-Evans has been **removed**.

## Benchmark

`examples/benchmark_dispersion.py` runs the calibration behind the
defaults — force-law selection, a tuning grid, and the main sweep over
$d \in \{2,\dots,64\}$ with the number of points scaled to the dimension:

```bash
python examples/benchmark_dispersion.py --phase all --seeds 10
```

## Development

The checks in CI also run at commit time:

```bash
pip install pre-commit
pre-commit install
```

That gates each commit on ruff, basedpyright, vulture and the unit tests --
the same four the GitHub Action runs, reading the same `pyproject.toml`, so a
commit that passes locally passes there. `pre-commit run --all-files` checks
the tree without committing.

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
