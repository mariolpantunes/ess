import logging

import numpy as np

import ess.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _scale(arr, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(arr, axis=0)
    if max_val is None:
        max_val = np.max(arr, axis=0)

    # Avoid division by zero for constant dimensions
    denom = max_val - min_val
    denom[denom == 0] = 1.0

    scl_arr = (arr - min_val) / denom
    return scl_arr, min_val, max_val


def _inv_scale(scl_arr, min_val, max_val):
    """Internal helper to unscale array."""
    return scl_arr * (max_val - min_val) + min_val


def _force(sigma, d):
    """
    Computes a simplified repulsive force.

    Used by the legacy implementation. Includes safety clips
    to prevent numerical instability.

    Args:
        sigma (float): Scaling factor derived from mean distance.
        d (np.ndarray): Distances.

    Returns:
        np.ndarray: Force magnitudes.
    """
    safe_d = np.maximum(d, 1e-9)
    ratio = sigma / safe_d
    ratio = np.minimum(ratio, 10.0)

    attrac = ratio**6
    attrac = np.minimum(attrac, 1e5)

    # Derivative-like calculation
    term = (2 * (attrac**2)) - attrac
    return np.abs(6 * term / safe_d)


def _elastic(es, neighbors, neighbors_dist):
    """
    Computes directional elastic forces.

    Args:
        es (np.ndarray): Current point coordinates.
        neighbors (np.ndarray): Neighbor coordinates.
        neighbors_dist (np.ndarray): Distances to neighbors.

    Returns:
        np.ndarray: Force vector.
    """
    sigma = np.mean(neighbors_dist) / 5.0

    forces = _force(sigma, neighbors_dist)

    diff = es - neighbors
    safe_dist = np.maximum(neighbors_dist[:, np.newaxis], 1e-9)
    vecs = diff / safe_dist

    direc = np.sum(vecs * forces[:, np.newaxis], axis=0)
    return direc


def _empty_center(coor, data, neigh, *, lr: float, epochs: int, bounds: np.ndarray):
    """
    Optimizes a single point using the Empty Center Strategy.

    Args:
        coor (np.ndarray): Initial coordinate.
        data (np.ndarray): Existing points.
        neigh (nn.NearestNeighbors): NN index.
        lr (float): Step size (unused in this specific logic, uses fixed movestep).
        epochs (int): Max iterations.
        bounds (np.ndarray): Scaled bounds [0, 1].

    Returns:
        np.ndarray: Optimized coordinate.
    """
    movestep = lr if lr is not None else 0.01

    k_req = min(data.shape[1] + 1, neigh.total_count)

    for i in range(epochs):
        # query_external replaces the old query() method
        # It queries the static points (data)
        # k = dim + 1 is a heuristic for sufficient neighbors
        adjs_, distances_ = neigh.query_static(coor, k=k_req)

        direc = _elastic(coor, data[adjs_[0]], distances_[0])
        mag = np.linalg.norm(direc)

        if mag < 1e-7:
            break

        direc /= mag
        coor += direc * movestep

        np.clip(coor, bounds[:, 0], bounds[:, 1], out=coor)

    return coor


def _esa_01(samples, bounds, n: int | None = None, seed: int | None = None):
    """
    ESA Version 0.1:
    Generates 'n' points. Each point is optimized INDEPENDENTLY against the original 'samples'.
    New points do not repel each other during generation.
    """
    min_val = bounds[:, 0]
    max_val = bounds[:, 1]
    scaled_samples, _, _ = _scale(samples, min_val, max_val)
    n_value = n if n is not None else samples.shape[0]

    # Initialize NN with original samples
    seed_value = seed if seed is not None else 42
    neigh = nn.NumpyNN(dimension=scaled_samples.shape[1], seed=seed_value)
    neigh.add_static(scaled_samples)

    rng = np.random.default_rng(seed_value)
    coors = rng.uniform(0, 1, (n_value, scaled_samples.shape[1])).astype(np.float32)
    logger.debug(f"Coors({n_value}, {scaled_samples.shape[1]})\n{coors}")

    scaled_bounds = np.array([[0, 1]] * scaled_samples.shape[1])
    movestep = 0.01
    es_params = []
    logger.debug(f"Samples\n{scaled_samples}")

    for c in coors:
        # Optimize c against samples
        es_param = _empty_center(
            c.reshape(1, -1),
            scaled_samples,
            neigh,
            lr=movestep,
            epochs=100,
            bounds=scaled_bounds,
        )
        es_params.append(es_param[0])

    logger.debug(f"Params({len(es_params)})\n{es_params}")

    rv = np.array(es_params)
    rv = _inv_scale(rv, min_val=min_val, max_val=max_val)

    logger.debug(f"RV({rv.shape})\n{rv}")

    return rv


def _esa_02(
    samples,
    bounds,
    *,
    n: int | None = None,
    epochs: int = 100,
    lr: float = 0.01,
    seed: int | None = None,
):
    """
    Legacy implementation of ESA (Version 0.2).

    Uses a sequential approach (one point at a time) rather than batching.

    Args:
        samples (np.ndarray): Existing points.
        bounds (np.ndarray): Bounding box.
        n (int): Number of points to add.
        epochs (int): Iterations per point.
        lr (float): Learning rate.
        seed (int): Random seed.

    Returns:
        np.ndarray: The generated points (unscaled).
    """
    min_val = bounds[:, 0]
    max_val = bounds[:, 1]
    samples, _, _ = _scale(samples, min_val, max_val)
    n_value = n if n is not None else samples.shape[0]

    seed_value = seed if seed is not None else 42
    neigh = nn.NumpyNN(dimension=samples.shape[1], seed=seed_value)
    neigh.add_static(samples)

    coors = np.random.uniform(0, 1, (n_value, samples.shape[1]))
    es_params = []

    scaled_bounds = np.array([[0, 1]] * samples.shape[1])

    for c in coors:
        es_param = _empty_center(
            c.reshape(1, -1), samples, neigh, lr=lr, epochs=epochs, bounds=scaled_bounds
        )
        es_params.append(es_param[0])

        samples = np.vstack((samples, es_param))
        neigh.add_static(es_param)

    rv = np.array(es_params)
    rv = _inv_scale(rv, min_val=min_val, max_val=max_val)
    return rv
