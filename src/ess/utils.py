import logging
import math

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
"""logging.Logger: Module-level logger for debugging ESA optimization steps."""


def _get_nearest_neighbor_distances(
    points: np.ndarray, batch_size: int = 500
) -> np.ndarray:
    """
    Computes the Euclidean distance to the nearest neighbor (excluding self) for each point.

    This method employs a vectorized, chunked approach to calculate the distance matrix
    without allocating the full $N \\times N$ array, making it memory-efficient for large datasets.
    It utilizes the squared Euclidean distance expansion:

    $ ||\\mathbf{A} - \\mathbf{B}||^2 = ||\\mathbf{A}||^2 + ||\\mathbf{B}||^2 - 2 \\mathbf{A} \\cdot \\mathbf{B}^T $

    where $\\mathbf{A}$ is a batch of points and $\\mathbf{B}$ is the full set. The distances
    are computed, the self-interaction (diagonal) is masked with infinity, and the minimum
    value along the row is extracted.

    Args:
        points (np.ndarray): The input coordinate array of shape $(N, D)$.
        batch_size (int): The number of rows to process simultaneously to control memory usage.

    Returns:
        np.ndarray: An array of shape $(N,)$ containing the scalar distance to the closest neighbor for every point.
    """
    n_points = points.shape[0]
    min_dists = np.zeros(n_points)

    # Pre-compute squared magnitudes for the full set
    # Shape: (1, N)
    all_sq = np.sum(points**2, axis=1, keepdims=True).T

    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        chunk = points[i:end]

        # Expansion: ||A - B||^2 = ||A||^2 + ||B||^2 - 2 A.B^T
        # A (chunk): (B, D)
        # B (all):   (N, D)

        chunk_sq = np.sum(chunk**2, axis=1, keepdims=True)  # (B, 1)

        # Dot product: (B, D) @ (D, N) -> (B, N)
        dot_prod = np.dot(chunk, points.T)

        # Broadcasting: (B, 1) + (1, N) - (B, N)
        dist_sq = chunk_sq + all_sq - 2 * dot_prod

        # Numerical stability
        dist_sq = np.maximum(dist_sq, 0.0)
        dists = np.sqrt(dist_sq)

        # Mask self-distance (which is 0.0 at diagonal indices) with infinity
        # Chunk row 'r' corresponds to global index 'i + r'
        for r in range(end - i):
            global_idx = i + r
            dists[r, global_idx] = np.inf

        min_dists[i:end] = np.min(dists, axis=1)

    return min_dists


def calculate_grid_coverage(
    points: np.ndarray, bounds: np.ndarray, grid: int | tuple | list
) -> float:
    """
    Calculates the spatial coverage ratio by discretizing the domain into a grid.

    This function maps continuous coordinates to discrete grid indices to determine how many
    hyper-rectangles (cells) contain at least one point. It uses a sparse tracking method
    (via `np.unique`) rather than dense array allocation, enabling support for high-dimensional spaces.

    For a point $\\mathbf{p}$ in dimension $d$ with bounds $[L_d, U_d]$ and $b_d$ bins, the index is:

    $ \\text{idx}_d = \\left\\lfloor \\frac{p_d - L_d}{U_d - L_d} \\times b_d \\right\\rfloor $

    The coverage ratio $C$ is defined as:

    $ C = \\frac{N_{\\text{occupied}}}{\\prod_{i=1}^D b_i} $

    Args:
        points (np.ndarray): The coordinate array of shape $(N, D)$.
        bounds (np.ndarray): The domain boundaries of shape $(D, 2)$, where column 0 is min and 1 is max.
        grid (int | tuple | list): The grid resolution. If an integer, it is applied to all dimensions.
            If a sequence, it specifies the number of bins per specific dimension.

    Returns:
        float: The fraction of the total grid volume covered by the points, in the range $[0.0, 1.0]$.
    """
    num_dims = points.shape[1]

    # 1. Parse Grid Configuration
    if isinstance(grid, int):
        bins = np.array([grid] * num_dims, dtype=np.int64)
    else:
        bins = np.array(grid, dtype=np.int64)
        if len(bins) != num_dims:
            raise ValueError(f"grid len must match dims {num_dims}")

    # 2. Calculate Total Theoretical Cells
    total_cells = 1
    for b in bins:
        total_cells *= int(b)

    if total_cells == 0:
        return 0.0

    # 3. Compute Bin Indices for Each Point (Sparse Approach)
    min_vals = bounds[:, 0]
    max_vals = bounds[:, 1]

    widths = max_vals - min_vals
    # Avoid division by zero
    widths[widths == 0] = 1.0

    bin_widths = widths / bins

    # Calculate indices: floor( (x - min) / width )
    raw_indices = np.floor((points - min_vals) / bin_widths).astype(np.int64)

    # Clip indices to [0, bins-1]
    clipped_indices = np.clip(raw_indices, 0, bins - 1)

    # 4. Count Unique Occupied Cells
    # np.unique with axis=0 finds unique rows (unique cell coordinates)
    unique_cells = np.unique(clipped_indices, axis=0)
    occupied_count = unique_cells.shape[0]

    return float(occupied_count) / float(total_cells)


def calculate_min_pairwise_distance(points: np.ndarray) -> float:
    """
    Calculates the global minimum distance between any two distinct points in the set.

    This metric is effectively the "separation" distance of the distribution. It corresponds
    to the result of the Maximin criterion.

    $ d_{\\min} = \\min_{i, j, i \\neq j} ||\\mathbf{x}_i - \\mathbf{x}_j|| $

    It is implemented efficiently by computing the nearest neighbor distance for every point
    and taking the minimum of those values.

    Args:
        points (np.ndarray): The coordinate array of shape $(N, D)$.

    Returns:
        float: The minimum distance found between any pair of points. Returns 0.0 if $N < 2$.
    """
    if len(points) < 2:
        return 0.0

    # Use helper to get distance to nearest neighbor for all points
    min_dists = _get_nearest_neighbor_distances(points)

    # The result is the minimum of all nearest neighbor distances
    return float(np.min(min_dists))


def calculate_clark_evans_index(
    points: np.ndarray, bounds: np.ndarray | None = None
) -> float:
    """
    Computes the Clark-Evans Nearest Neighbor Index ($R$) for $D$-dimensional space.

    The index compares the observed mean nearest-neighbor distance to the expected distance
    in a Poisson (random) distribution.

    The index is calculated as $R = \\frac{\\bar{r}_A}{\\bar{r}_E}$, where $\\bar{r}_A$ is the
    mean observed distance. The expected distance $\\bar{r}_E$ for density $\\rho = N/V$
    in $D$ dimensions is derived from the volume of a $D$-dimensional unit ball ($V_D$):

    $ \\bar{r}_E = \\frac{\\Gamma(1/D + 1)}{(\\rho \\cdot V_D)^{1/D}} \\quad \\text{where} \\quad V_D = \\frac{\\pi^{D/2}}{\\Gamma(D/2 + 1)} $

    Interpretation:
    * $R < 1$: Aggregated (clustered) distribution.
    * $R = 1$: Random (Poisson) distribution.
    * $R > 1$: Regular (dispersed/uniform) distribution.

    Args:
        points (np.ndarray): The coordinate array of shape $(N, D)$.
        bounds (np.ndarray | None): The boundaries of the domain $(D, 2)$ used to calculate volume.
            If `None`, the volume is estimated using the bounding box of the provided points.

    Returns:
        float: The Clark-Evans index $R$.
    """
    if len(points) < 2:
        return 0.0

    dim = points.shape[1]
    n = len(points)

    # 1. Mean Observed Distance
    # Calculate NN distance for every point
    nn_dists = _get_nearest_neighbor_distances(points)
    mean_obs_dist = np.mean(nn_dists)

    # 2. Mean Expected Distance (Random)
    if bounds is not None:
        volume = np.prod(bounds[:, 1] - bounds[:, 0])
    else:
        # Estimate volume via bounding box of points
        min_p = np.min(points, axis=0)
        max_p = np.max(points, axis=0)
        volume = np.prod(max_p - min_p)

    if volume <= 0:
        return 0.0

    rho = n / volume

    # Volume of unit ball in D dims: pi^(D/2) / Gamma(D/2 + 1)
    # math.gamma is standard library, replaces scipy.special.gamma
    gamma_val = math.gamma(dim / 2.0 + 1.0)
    vol_unit = (math.pi ** (dim / 2.0)) / gamma_val

    # Expected NN distance for Poisson process in D dimensions
    # Formula: Gamma(1/D + 1) / ( (Volume_unit_ball * rho)^(1/D) )
    numerator = math.gamma(1.0 / dim + 1.0)
    denominator = (vol_unit * rho) ** (1.0 / dim)

    expected_dist = numerator / denominator

    return float(mean_obs_dist / expected_dist)
