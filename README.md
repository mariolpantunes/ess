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
* **Radius-Based Interactions (New in v0.3.0)**: Supports physical range searches (interacting with *all* neighbors within a radius) alongside standard k-NN, with automatic radius estimation for high-dimensional spaces.
* **Repulsive Boundaries**: Implements "soft walls" that exert restorative forces at domain edges, preventing the edge-clumping artifacts common in hard-clipped optimization.
* **Scalable Architecture**:
    * **NumpyNN**: Vectorized pure NumPy implementation with norm caching. Efficient for N < 5,000.
    * **FaissHNSW**: Wraps [Faiss](https://github.com/facebookresearch/faiss) HNSW graphs for logarithmic scaling on large datasets (N > 50,000).
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

* Python >= 3.12
* numpy
* faiss-cpu

## Usage

### Basic Example

Generate 100 new points in a 2D space using the default settings (Auto-Radius + Repulsive Walls):

```python
import numpy as np
import ess

# Define existing points (e.g., obstacles)
obstacles = np.array([[0.5, 0.5]]) 
bounds = np.array([[0, 1], [0, 1]])

# Generate 100 new points
# 'ess' returns the combined set (obstacles + new points)
result = ess.ess(obstacles, bounds, n=100, seed=42, border_strategy='repulsive')

print(f"Total points: {len(result)}")
```

### Advanced Usage with Faiss & Radius Search

For large datasets, explicitly use the `FaissHNSWFlatNN` backend and the new physics-based radius mode:

```python
import numpy as np
from ess import esa, FaissHNSWFlatNN

# 1000 existing points in 50 dimensions
dim = 50
obstacles = np.random.rand(1000, dim)
bounds = np.array([[0, 1]] * dim)

# Initialize HNSW Engine for speed
nn_engine = FaissHNSWFlatNN(dimension=dim, seed=42)

# Run ESA (returns ONLY the new points)
# search_mode='radius' activates the dense physical interaction model
new_points = esa(
    obstacles, 
    bounds, 
    n=500, 
    nn_instance=nn_engine,
    search_mode='radius',  # Use radius instead of k-NN
    radius=None,           # None = Auto-compute based on density
    batch_size=100, 
    epochs=256
)
```

## Algorithms

**ESA (Empty Space Algorithm)** treats existing points as fixed charged particles and new points as free moving charges.

1. **k-NN Mode**: Points are repelled by their  nearest neighbors. Good for maintaining local uniformity.
2. **Radius Mode (New)**: Points are repelled by **all** neighbors within a specific cutoff radius. This mimics real electrostatic fields and prevents "tunneling" in high-density regions.

**Force Functions**:

* `softened_inverse`: Standard electrostatic repulsion (Coulomb-like).
* `gaussian`: Smooth, short-range repulsion.
* `linear`: Simple linear drop-off (Hookean spring).
* `cauchy`: Heavy-tailed distribution for global separation.

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
