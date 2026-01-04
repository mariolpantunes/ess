import logging
import math
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def _get_nearest_neighbor_distances(points: np.ndarray, batch_size: int = 500) -> np.ndarray:
    """
    Computes the distance to the nearest neighbor (excluding self) for each point.
    
    Uses a memory-efficient chunked approach with vectorized operations to replace
    KDTree dependency.

    Args:
        points (np.ndarray): Shape (N, D).
        batch_size (int): Number of query points to process at once to manage memory.

    Returns:
        np.ndarray: Shape (N,). Distance to the nearest neighbor for each point.
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
        
        chunk_sq = np.sum(chunk**2, axis=1, keepdims=True) # (B, 1)
        
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


def calculate_grid_coverage(points: np.ndarray, 
                            bounds: np.ndarray, 
                            grid: int | tuple | list) -> float:
    """
    Calculates the percentage of grid cells covered by at least one point using a sparse method.

    This implementation supports high dimensions (>32) by only tracking occupied cells
    rather than allocating the full grid.

    Args:
        points (np.ndarray): Shape (N, D).
        bounds (np.ndarray): Shape (D, 2).
        grid (int | tuple | list): Number of bins per dimension.

    Returns:
        float: Percentage of covered cells (0.0 to 1.0).
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
    Calculates the minimum distance between any two distinct points (Maximin criterion).

    A higher value indicates better separation (packing) of points. 
    Uses a custom NumPy implementation to avoid scipy dependencies.

    Args:
        points (np.ndarray): Shape (N, D).

    Returns:
        float: The minimum pairwise distance found.
    """
    if len(points) < 2:
        return 0.0
    
    # Use helper to get distance to nearest neighbor for all points
    min_dists = _get_nearest_neighbor_distances(points)
    
    # The result is the minimum of all nearest neighbor distances
    return float(np.min(min_dists))


def calculate_clark_evans_index(points: np.ndarray, 
                                bounds: np.ndarray | None = None) -> float:
    """
    Calculates the Clark-Evans Nearest Neighbor Index (R).

    R < 1 : Clustered distribution.
    R = 1 : Random distribution (Poisson).
    R > 1 : Dispersed (Uniform) distribution.

    Args:
        points (np.ndarray): Shape (N, D).
        bounds (np.ndarray | None): Optional bounds to calculate volume accurately.
            If None, volume is estimated from the points' bounding box.

    Returns:
        float: The R index.
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
    numerator = math.gamma(1.0/dim + 1.0)
    denominator = (vol_unit * rho) ** (1.0/dim)
    
    expected_dist = numerator / denominator
    
    return float(mean_obs_dist / expected_dist)
    