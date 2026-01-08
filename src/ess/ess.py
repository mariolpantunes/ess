import collections.abc
import logging
import math

import numpy as np

import ess.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --- Force Functions ---


def gaussian_force(d: np.ndarray, sigma: float = 0.2, alpha: float = 2.0) -> np.ndarray:
    """
    Computes a Gaussian repulsion force.

    F = alpha * exp(-d^2 / 2sigma^2)

    Args:
        d (np.ndarray): Array of distances.
        sigma (float): Spread of the Gaussian.
        alpha (float): Peak magnitude.

    Returns:
        np.ndarray: Force magnitudes.
    """
    s2 = 2 * (sigma * sigma)
    return alpha * np.exp(-(d * d) / s2)


def softened_inverse_force(
    d: np.ndarray, epsilon: float = 0.05, alpha: float = 0.01
) -> np.ndarray:
    """
    Computes a softened inverse square repulsion.

    F = alpha / (d^2 + epsilon^2)

    Args:
        d (np.ndarray): Array of distances.
        epsilon (float): Softening parameter to avoid division by zero.
        alpha (float): Magnitude scaling factor.

    Returns:
        np.ndarray: Force magnitudes.
    """
    return alpha * (1.0 / ((d * d) + (epsilon * epsilon)))


def linear_force(d: np.ndarray, R: float = 0.5) -> np.ndarray:
    """
    Computes a linear repulsive force within a radius R.

    F = max(0, 1 - d/R)

    Args:
        d (np.ndarray): Array of distances.
        R (float): Radius of influence.

    Returns:
        np.ndarray: Force magnitudes.
    """
    return np.maximum(0.0, 1.0 - (d / R))


def cauchy_force(d: np.ndarray) -> np.ndarray:
    """
    Computes a Cauchy-distribution based repulsion.

    F = 1 / (1 + d^2)

    Args:
        d (np.ndarray): Array of distances.

    Returns:
        np.ndarray: Force magnitudes.
    """
    return 1.0 / (1.0 + (d * d))


METRIC_REGISTRY = {
    "gaussian": gaussian_force,
    "softened_inverse": softened_inverse_force,
    "linear": linear_force,
    "cauchy": cauchy_force,
}


# --- Helpers ---


def _scale(
    arr: np.ndarray,
    min_val: np.ndarray | np.number | float | int | None = None,
    max_val: np.ndarray | np.number | float | int | None = None,
) -> tuple[np.ndarray, np.ndarray | np.number | float | int,
    np.ndarray | np.number | float | int]:
    """
    Scales the input array to the [0, 1] range.

    Computes the minimum and maximum values along axis 0 if not provided,
    and scales the array accordingly. Handles cases where min equals max
    (constant dimension) by avoiding division by zero.

    Args:
        arr (np.ndarray): Input array of shape (N, D).
        min_val (np.ndarray | np.number | None): Optional pre-computed minimums. Shape (D,).
        max_val (np.ndarray | np.number | None): Optional pre-computed maximums. Shape (D,).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - **scaled_arr**: The scaled array in [0, 1].
            - **min_val**: The minimum values used for scaling.
            - **max_val**: The maximum values used for scaling.
    """
    used_min_val = np.min(arr, axis=0) if min_val is None else min_val
    used_max_val = np.max(arr, axis=0) if max_val is None else max_val

    denom = used_max_val - used_min_val
    denom = np.where(denom == 0, 1.0, denom)

    return (arr - used_min_val) / denom, used_min_val, used_max_val


def _inv_scale(
    scl_arr: np.ndarray,
    min_val: np.ndarray | np.number | float | int,
    max_val: np.ndarray | np.number | float | int
) -> np.ndarray:
    """
    Restores the scaled array to its original range.

    Args:
        scl_arr (np.ndarray): Scaled array in [0, 1].
        min_val (np.ndarray | np.number): Minimum values of the original data.
        max_val (np.ndarray | np.number): Maximum values of the original data.

    Returns:
        np.ndarray: The unscaled array in the original data range.
    """
    return scl_arr * (max_val - min_val) + min_val


def _compute_forces(
    active_batch: np.ndarray,
    neighbor_coords: np.ndarray,
    dists_batch: np.ndarray,
    metric_fn: collections.abc.Callable,
    **metric_kwargs,
) -> np.ndarray:
    """
    Computes the total force vector acting on each point in the active batch.

    Aggregates repulsive forces from all k-nearest neighbors based on the
    provided metric function.

    Args:
        active_batch (np.ndarray): Shape (M, D). Coordinates of points being moved.
        neighbor_coords (np.ndarray): Shape (M, k, D). Coordinates of neighbors.
        dists_batch (np.ndarray): Shape (M, k). Euclidean distances.
        metric_fn (collections.abc.Callable): Function that maps distance to force magnitude.
        **metric_kwargs: Arguments for the metric function.

    Returns:
        np.ndarray: Shape (M, D). Resultant force vectors.
    """
    # Safety: Avoid division by zero
    dists_safe = np.maximum(dists_batch, 1e-12)

    # 1. Magnitude
    magnitudes = metric_fn(dists_safe, **metric_kwargs)

    # 2. Vector Calculation
    # Force = Magnitude * (Direction Vector)
    # Direction Vector = (Point - Neighbor) / Distance
    coeffs = magnitudes / dists_safe  # Shape (M, k)

    # diff = active - neighbor
    diff = active_batch[:, np.newaxis, :] - neighbor_coords

    # Broadcast coefficients
    force_vecs = diff * coeffs[:, :, np.newaxis]

    # Sum neighbors
    return np.sum(force_vecs, axis=1)


def _smart_init(
    bounds_01: np.ndarray,
    nn_instance: nn.NearestNeighbors,
    n_new: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates new points using Best Candidate Sampling.

    Candidates are generated uniformly and checked against existing static points.
    The candidate with the largest distance to its nearest neighbor is chosen.

    Args:
        bounds_01 (np.ndarray): Normalized [0, 1] bounds.
        nn_instance (nn.NearestNeighbors): The NN index containing obstacles.
        n_new (int): Number of points to generate.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Shape (n_new, D). The initialized points.
    """
    dim = bounds_01.shape[1]
    new_samples = []

    # number of candidates to test per new point
    n_candidates = 15

    for _ in range(n_new):
        candidates = rng.uniform(
            bounds_01[:, 0], bounds_01[:, 1], (n_candidates, dim)
        ).astype(np.float32)

        # Check against STATIC points
        _, dists = nn_instance.query_external(candidates, k=1)
        dists = dists.flatten()

        best_idx = np.argmax(dists)
        new_samples.append(candidates[best_idx])

    return np.array(new_samples, dtype=np.float32)


# --- Core Logic ---


def _esa(
    samples: np.ndarray,
    bounds: np.ndarray,
    nn_instance: nn.NearestNeighbors,
    *,
    n: int,
    epochs: int = 512,
    lr: float = 0.01,
    decay: float = 0.9,
    batch_size: int = 50,
    k: int | None = None,
    tol: float = 1e-3,
    metric_fn: collections.abc.Callable = softened_inverse_force,
    seed: int = 42,
    **metric_kwargs,
) -> np.ndarray:
    """
    Internal execution loop for the Electrostatic Search Algorithm.

    Implements the batch-wise optimization strategy:
    1. Scale data to [0, 1].
    2. Load existing points into NN static index.
    3. Iterate in batches to generate 'n' new points.
       a. Smart Init candidates.
       b. Optimize active batch using repulsive forces.
       c. Freeze batch (move to static).
    4. Inverse scale results.

    Args:
        samples (np.ndarray): Existing points.
        bounds (np.ndarray): Boundary box.
        nn_instance (nn.NearestNeighbors): NN implementation instance.
        n (int): Number of points to generate.
        epochs (int): Max optimization steps per batch.
        lr (float): Initial learning rate.
        decay (float): Learning rate decay per step.
        batch_size (int): Points to optimize simultaneously.
        k (int | None): Neighbors to consider. Defaults to 2*dim + 1.
        tol (float): Convergence tolerance.
        metric_fn (collections.abc.Callable): Force function.
        seed (int): Random seed.
        **metric_kwargs: Arguments for the metric function.

    Returns:
        np.ndarray: The generated points (unscaled).
    """

    # Scaling
    min_val = bounds[:, 0]
    max_val = bounds[:, 1]
    scaled_samples, _, _ = _scale(samples, min_val, max_val)
    scaled_samples = scaled_samples.astype(np.float32)

    dim = samples.shape[1]
    rng = np.random.default_rng(seed)

    # Setup NN
    nn_instance.clear()
    nn_instance.add_static(scaled_samples)

    search_k = 0
    if k is None:
        search_k = (2 * dim) + 1
    else:
        search_k = k

    generated_points = []
    num_batches = math.ceil(n / batch_size)
    bounds_01 = np.array([[0, 1]] * dim)

    logger.debug(f"Starting ESA: {n} points, {num_batches} batches.")

    for _ in range(num_batches):
        current_n = min(batch_size, n - len(generated_points))
        if current_n <= 0:
            break

        # 1. Smart Initialization
        active_batch = _smart_init(bounds_01, nn_instance, current_n, rng)
        nn_instance.set_active(active_batch)

        current_lr = lr

        # 2. Optimization Loop
        for _ in range(epochs):
            # Query neighbors
            indices, dists = nn_instance.query_active(k=search_k)

            # Map indices to coordinates for vector calc
            # We construct a view of all data: Static + Active
            combined_data = np.vstack((scaled_samples, active_batch))

            # Sanity check
            if np.max(indices) >= len(combined_data):
                logger.error("NN returned indices out of bounds.")
                break

            neighbor_coords = combined_data[indices]  # Shape (M, k, D)

            # Compute Forces
            force_vecs = _compute_forces(
                active_batch, neighbor_coords, dists, metric_fn, **metric_kwargs
            )

            # Update
            prev_pos = active_batch.copy()
            active_batch += force_vecs * current_lr
            np.clip(active_batch, 0.0, 1.0, out=active_batch)

            nn_instance.set_active(active_batch)

            # Convergence
            move_dist = np.linalg.norm(active_batch - prev_pos, axis=1)
            if np.max(move_dist) < tol:
                break

            current_lr *= decay

        # 3. Consolidate Batch
        nn_instance.consolidate()
        scaled_samples = np.vstack((scaled_samples, active_batch))
        generated_points.append(active_batch)

    all_generated = np.vstack(generated_points)
    return _inv_scale(all_generated, min_val, max_val)


# --- Core Logic ---


def esa(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    n: int,
    nn_instance: nn.NearestNeighbors | None = None,
    epochs: int = 512,
    lr: float = 0.01,
    decay: float = 0.9,
    batch_size: int = 50,
    k: int | None = None,
    tol: float = 1e-3,
    metric: str | collections.abc.Callable = "gaussian",
    seed: int = 42,
    **metric_kwargs,
) -> np.ndarray:
    """
    Electrostatic Search Algorithm (ESA).

    Generates 'n' spatially diverse points by simulating electrostatic repulsion
    against existing 'samples'.

    This method returns ONLY the new generated points.

    Args:
        samples (np.ndarray): Existing points.
        bounds (np.ndarray): Boundary box.
        n (int): Number of points to generate.
        nn_instance (nn.NearestNeighbors | None): NN implementation instance.
                                                  Created if None.
        epochs (int): Max optimization steps per batch.
        lr (float): Initial learning rate.
        decay (float): Learning rate decay per step.
        batch_size (int): Points to optimize simultaneously.
        k (int | None): Neighbors to consider. Defaults to 2*dim + 1.
        tol (float): Convergence tolerance.
        metric (str | collections.abc.Callable): Force function name or callable.
        seed (int): Random seed.
        **metric_kwargs: Arguments for the metric function.

    Returns:
        np.ndarray: The generated points (unscaled).
    """

    # 1. Input Sanitization
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples).astype(np.float32)

    # 2. Metric Resolution
    if isinstance(metric, str):
        metric_fn = METRIC_REGISTRY.get(metric.lower())
        if metric_fn is None:
            raise ValueError(f"Unknown metric '{metric}'")
    else:
        metric_fn = metric

    # 3. Scaling
    min_val = bounds[:, 0]
    max_val = bounds[:, 1]
    scaled_samples, _, _ = _scale(samples, min_val, max_val)
    scaled_samples = scaled_samples.astype(np.float32)

    dim = samples.shape[1]
    rng = np.random.default_rng(seed)

    # 4. NN Setup
    if nn_instance is None:
        logger.debug(f"No NN instance provided. Defaulting to NumpyNN(dim={dim}).")
        nn_instance = nn.NumpyNN(dimension=dim, seed=seed)

    nn_instance.clear()
    nn_instance.add_static(scaled_samples)

    search_k = 0
    if k is None:
        search_k = (2 * dim) + 1
    else:
        search_k = k

    # 5. Batch Generation
    generated_points = []
    num_batches = math.ceil(n / batch_size)
    bounds_01 = np.array([[0, 1]] * dim)

    logger.debug(f"Starting ESA: {n} points, {num_batches} batches.")

    for _ in range(num_batches):
        current_n = min(batch_size, n - len(generated_points))
        if current_n <= 0:
            break

        # 5a. Smart Initialization
        active_batch = _smart_init(bounds_01, nn_instance, current_n, rng)
        nn_instance.set_active(active_batch)

        current_lr = lr

        # 5b. Optimization Loop
        for _ in range(epochs):
            # Query neighbors
            indices, dists = nn_instance.query_active(k=search_k)

            # Map indices to coordinates
            # View of all data: Static + Active
            combined_data = np.vstack((scaled_samples, active_batch))

            if np.max(indices) >= len(combined_data):
                logger.error("NN returned indices out of bounds.")
                break

            neighbor_coords = combined_data[indices]  # Shape (M, k, D)

            # Compute Forces
            force_vecs = _compute_forces(
                active_batch, neighbor_coords, dists, metric_fn, **metric_kwargs
            )

            # Update
            prev_pos = active_batch.copy()
            active_batch += force_vecs * current_lr
            np.clip(active_batch, 0.0, 1.0, out=active_batch)

            nn_instance.set_active(active_batch)

            # Convergence
            move_dist = np.linalg.norm(active_batch - prev_pos, axis=1)
            if np.max(move_dist) < tol:
                break

            current_lr *= decay

        # 5c. Consolidate Batch
        nn_instance.consolidate()
        scaled_samples = np.vstack((scaled_samples, active_batch))
        generated_points.append(active_batch)

    if not generated_points:
        return np.empty((0, dim))

    all_generated = np.vstack(generated_points)
    return _inv_scale(all_generated, min_val, max_val)


def ess(
    samples: np.ndarray | list,
    bounds: np.ndarray,
    *,
    n: int,
    nn_instance: nn.NearestNeighbors | None = None,
    epochs: int = 2048,
    lr: float = 0.001,
    decay: float = 0.99,
    batch_size: int = 50,
    seed: int = 42,
    tol: float = 1e-3,
    metric: str | collections.abc.Callable = "gaussian",
    **kwargs,
) -> np.ndarray:
    """
    Electrostatic Search Strategy (ESS).

    A high-level wrapper that runs ESA and returns the combined dataset
    (original samples + new points).

    Args:
        samples (np.ndarray | list): Existing points (obstacles). Shape (N, D).
        bounds (np.ndarray): Bounding box. Shape (D, 2).
        n (int): Number of new points to generate.
        nn_instance (nn.NearestNeighbors | None): NN instance.
        epochs (int): Max optimization epochs.
        lr (float): Initial learning rate.
        decay (float): Decay factor.
        batch_size (int): Batch size.
        seed (int): Random seed.
        tol (float): Tolerance.
        metric (str | collections.abc.Callable): Metric name or function.
        **kwargs: Metric arguments.

    Returns:
        np.ndarray: Array containing original samples + n generated points.
    """

    if not isinstance(samples, np.ndarray):
        samples = np.array(samples).astype(np.float32)

    new_points = esa(
        samples=samples,
        bounds=bounds,
        n=n,
        nn_instance=nn_instance,
        epochs=epochs,
        lr=lr,
        decay=decay,
        batch_size=batch_size,
        k=None,  # Let ESA determine K
        tol=tol,
        metric=metric,
        seed=seed,
        **kwargs,
    )

    return np.concatenate((samples, new_points), axis=0)
