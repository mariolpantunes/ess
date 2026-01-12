# Empty Space Search (ESS)

![PyPI - Version](https://img.shields.io/pypi/v/EmptySpaceSearch)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/EmptySpaceSearch)
![GitHub License](https://img.shields.io/github/license/mariolpantunes/ess)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mariolpantunes/ess/main.yml)
![GitHub last commit](https://img.shields.io/github/last-commit/mariolpantunes/ess)

**ESS** is a high-performance Python library that implements the Electrostatic Search Algorithm (ESA), a novel method for generating spatially diverse point distributions. It simulates electrostatic repulsive forces to "relax" new points into the empty spaces of a high-dimensional domain, making it ideal for sampling, coverage optimization, and exploratory data analysis.

## Features

* **Electrostatic Search Algorithm (ESA)**: Uses physics-inspired repulsive forces (Gaussian, Softened Inverse, etc.) to maximize the separation between points.
* **Scalable Batch Processing**: Optimized to handle large datasets by processing new points in efficient batches, preventing memory overload and "clumping" artifacts.
* **Dual NN Architecture**:
    * **NumpyNN**: A pure NumPy implementation optimized for vectorization, ideal for small-to-medium datasets (< 5,000 points) and systems where external C++ dependencies are undesirable.
    * **FaissNN**: A high-performance implementation using [Faiss](https://github.com/facebookresearch/faiss) (CPU-only) for scaling to larger datasets.
* **High-Dimensional Metrics**: Includes robust coverage metrics (Maximin, Clark-Evans Index, Sparse Grid Coverage) that work effectively even in high-dimensional spaces (> 32D).
* **Smart Initialization**: Uses a "Best Candidate" sampling strategy to seed new batches in the most promising void regions before optimization begins.

> **Note:** The library is designed to be compliant with modern Python 3.12+ standards.

## Installation

The library can be installed directly from GitHub by adding the following line to your `requirements.txt` file:

```text
git+[https://github.com/mariolpantunes/ess@main#egg=ess](https://github.com/mariolpantunes/ess@main#egg=ess)
```

**Requirements:**

* Python >= 3.12
* numpy
* faiss-cpu

## Usage

### Basic Example

Generate 100 new points in a 2D space [0, 1] x [0, 1] using the default settings:

```python
import numpy as np
import ess.ess as ess

# Define existing points (e.g., obstacles)
obstacles = np.array([[0.5, 0.5]]) 
bounds = np.array([[0, 1], [0, 1]])

# Generate 100 new points
# If nn_instance is None, it defaults to NumpyNN
result = ess.ess(obstacles, bounds, n=100, seed=42)

print(f"Total points: {len(result)}")
```

### Advanced Usage with Faiss

For larger datasets, explicitly use the `FaissNN` backend:

```python
from ess.ess import ess
from ess.nn import FaissNN
import numpy as np

# 1000 existing points in 50 dimensions
dim = 50
obstacles = np.random.rand(1000, dim)
bounds = np.array([[0, 1]] * dim)

# Initialize FaissNN
nn_engine = FaissNN(dimension=dim, seed=42)

# Run ESS with batching
new_points = ess(
    obstacles, 
    bounds, 
    n=500, 
    nn_instance=nn_engine, 
    batch_size=100, 
    epochs=256
)
```

## Algorithms

1. **ESA (Electrostatic Search Algorithm)**:
The core algorithm. It treats existing points as fixed charged particles and new points as free moving charges. The new points are iteratively pushed away from neighbors until they settle in the "valleys" of the potential field (the empty spaces).

* **Forces Available**:
* `gaussian`: Smooth, short-range repulsion (default).
* `softened_inverse`: Strong near-field repulsion, computationally cheaper.
* `linear`: Simple linear drop-off within a radius.
* `cauchy`: Heavy-tailed distribution.

## Documentation

This library is documented using Google-style docstrings.

To generate the documentation locally using [pdoc](https://pdoc.dev/), run the following command:

```bash
pdoc --math -d google -o docs src/ess
```

## Authors

* **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
