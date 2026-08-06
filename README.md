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
* **Radius-Based Interactions**: Supports physical range searches (interacting with *all* neighbors within a radius) alongside standard k-NN, with automatic L1-ball radius estimation for high-dimensional spaces.
* **Single Scalable Engine**: Neighbour search is [torann](https://github.com/mariolpantunes/torann) — exact brute force at small N, toroidal LSH above its threshold, chosen internally; per-epoch coordinate updates are native (no index rebuilds).
* **Robust Early Stopping**: Convergence is detected on the force field itself (plateau of the largest net force, learning-rate-decoupled), typically stopping in tens of epochs instead of hundreds.
* **High-Dimensional Metrics**: Includes robust coverage metrics (Maximin, Clark-Evans Index, Sparse Grid Coverage) optimized for dimensions > 32D.
* **Smart Initialization**: Uses a vectorized "Best Candidate" sampling strategy to seed new batches in the most promising void regions.
* **Guided Attraction (New in v0.5.0)**: Given the attractiveness of the measured points, placement and relaxation both balance repulsion against a pull toward promising regions, with the collapse condition checked rather than discovered.

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

### Guided ESS: attraction toward what is worth exploring (New in v0.5.0)

Pure repulsion treats every empty region as equally worth probing. Given a
measurement of how *attractive* the existing points are, the search becomes a
force balance instead: points repel by distance and are attracted by quality.

```python
new_points = esa(
    samples, bounds, n=60,
    attractiveness=-objective_values,   # higher is more attractive
    attraction_weight=0.5,              # pull against the repulsion
    attraction_metric='cauchy',         # must decay slower than the repulsion
    attraction_kwargs={'power': 1.0},
    att_model='fourier',                # how unmeasured positions are estimated
)
```

`attractiveness` is only ever known for the points whose objective has been
paid for, so a candidate's is estimated. `att_model` picks how:

| model | coefficients | how they are obtained |
|---|---|---|
| `idw` | none | inverse-distance over the `k_att` nearest measured points |
| `fourier` | `2d+1` | least squares, ridge-regularised |
| `projection` | `2d` | correlation against the basis, James-Stein shrunk |
| `detrended` | `2d+1` | the Fourier fit plus IDW on its residual |
| `auto` | — | picks by whether the solve is identifiable |

**Why a trigonometric basis.** The space is a torus, so a model of it has to be
periodic — a polynomial is discontinuous at the seam and would assert a
gradient across a boundary the space does not have. The periodic analogue of a
quadratic well is the von Mises density, whose logarithm is
`κ cos(θ − μ) = (κ cos μ) cos θ + (κ sin μ) sin θ` — one first-harmonic term per
axis. A first-harmonic model *is* an additive log-von-Mises field. Higher
harmonics buy narrower and multimodal wells, at `2·harmonics·d` coefficients.

**Which to use.** `fourier` solves for its coefficients, so it needs more
measured points than unknowns; `projection` correlates instead, which stays
defined at any count. Held-out error, normalised so 1.0 is what predicting the
mean scores:

| truth | ridge | projection | idw |
|---|---|---|---|
| additive, `d=8`, `M=300` | **0.31** | 0.36 | 0.48 |
| additive, `d=100`, `M=30` | **0.78** | 0.80 | 0.80 |
| non-additive, `d=32`, `M=120` | 1.08 | **0.81** | 0.82 |
| non-additive, `d=100`, `M=300` | 1.27 | **0.81** | 0.84 |

Above 1.0 means worse than abstaining. Least squares commits hard, which pays
when the basis matches the truth and costs when it does not; the shrunk
projection and the interpolation both hedge. Four of the eight objectives in
the downstream benchmark are non-separable, which is where that matters.

**Custom models.** Subclass `AttractionModel`, implement `fit` and `at`, and
pass the instance:

```python
class MyModel(ess.AttractionModel):
    def fit(self, positions, values, confidence):
        ...
        return self

    def at(self, positions):
        return ...

ess.esa(samples, bounds, n=60, attractiveness=-scores, att_model=MyModel())
```

`att_model` also takes a tuned built-in, e.g. `ess.HarmonicRidge(harmonics=3)`.

Two guards are checked rather than left to a run. An attraction that
out-pulls repulsion at contact collapses every free point onto its most
attractive neighbour, and the plateau detector would report that as
convergence — so `attraction_weight * F_att(0) < F_rep(0)` is enforced, which
refuses weights at or above 2.5 for the default pair. And attraction only
overcomes repulsion *somewhere* if it decays more slowly, so using one law for
both sides warns: two proportional forces can never cross, and the attraction
merely scales the repulsion down instead of pulling.

`placement_weight` separates the two stages. Placement picks where a point
starts; the relaxation decides where it settles. Passing `None` pairs them,
which is the sensible default; setting it apart from `attraction_weight` lets
guided placement and guided relaxation be measured separately.

The estimate is a fitted function, so it improves with the number of *measured*
points and nothing else. At `d=100` the model carries 201 coefficients and
needs roughly 640 points to reach the error `d=8` reaches with 160 — a budget
of 60 is what a weak high-dimensional attraction actually measures, and no work
on the estimator changes that.

## Algorithms

**ESA (Empty Space Algorithm)** treats existing points as fixed charged particles and new points as free moving charges.

1. **k-NN Mode**: Points are repelled by their  nearest neighbors. Good for maintaining local uniformity.
2. **Radius Mode (New)**: Points are repelled by **all** neighbors within a specific cutoff radius. This mimics real electrostatic fields and prevents "tunneling" in high-density regions.

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

| function | what it measures | rank designs with it? |
| --- | --- | --- |
| `wrap_around_discrepancy` | deviation from uniform over every wrap-around box, full dimension | **yes** |
| `projection_discrepancy` | the same, averaged over 1-D / 2-D coordinate projections; fixed scale in any ambient $d$ | **yes** |
| `expected_discrepancy` | the null both are divided by, so 1.0 = as uniform as random | — |
| `toroidal_separation` | the smallest toroidal $L_1$ gap in the set | diagnostic only |
| `euclidean_separation`, `calculate_grid_coverage` | non-wrapping separation, and grid occupancy | provenance only |

**Rank designs with the two discrepancies.** They measure deviation from
uniformity and reference no point metric at all, so no choice of geometry
can flatter an arm that happened to optimise it. Divide by
`expected_discrepancy(n, s)` and the scale is fixed: 1.0 is as uniform as
random, lower is better, and above 1.0 is worse than random — a real and
observed failure mode, not a rounding artefact.

The separations are raw distances, so they are only meaningful inside one
fixed geometry; ranking an $L_1$-optimised design against an
$L_2$-optimised one with either asks which is better at the thing one of
them optimised. `calculate_grid_coverage` saturates and inverts above
$d \approx 8$ and cannot be built past $d \approx 20$.

Clark-Evans has been **removed**. It has meaning only divided by an
expected nearest-neighbour distance, which makes it a statistic about a
*metric* rather than about uniformity — so it cannot compare designs that
optimised different geometries, which is most of the comparisons worth
making. It was also blunt where it was valid: across a change under which
2-D projection discrepancy moved six-fold, it moved 1.4%, and it scored a
design that is worse than random in its projections as 24% better than
random.

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
