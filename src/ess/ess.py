import collections.abc
import logging
import math

import numpy as np

import ess.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
"""logging.Logger: Module-level logger for debugging ESA optimization steps."""


# --- Configuration Constants ---
KNN_SWITCH_THRESHOLD = 4096
"""int: The point count threshold for switching k-NN search engines.

If $N_{total} \\le$ this value, the algorithm uses a brute-force `NumpyNN` engine
(exact, faster for small N). Above this, it switches to `FaissHNSWFlatNN`
(approximate, faster for large N).
"""

RADIUS_SWITCH_THRESHOLD = 50000
"""int: The point count threshold for switching Radius search engines.

For radius searches, dense matrix operations (`NumpyNN`) are often faster than
HNSW range search due to implementation overhead, up to a fairly large N.
Above this value, the memory cost of the dense matrix becomes prohibitive.
"""


# --- Force Functions ---
def gaussian_force(d: np.ndarray, sigma: float = 0.2, alpha: float = 2.0) -> np.ndarray:
    """
    Computes a Gaussian repulsion force based on distance.

    This function models a short-range repulsive force that decays exponentially with distance,
    resembling a Gaussian distribution. It is useful for creating "soft" exclusion zones
    around particles. The force magnitude $F$ is calculated as:

    $ F(d) = \\alpha \\cdot \\exp\\left(-\\frac{d^2}{2\\sigma^2}\\right) $

    where $d$ is the distance, $\\sigma$ controls the width of the repulsion (standard deviation),
    and $\\alpha$ is the peak magnitude at $d=0$.

    Args:
        d (np.ndarray): Array of pairwise distances between points.
        sigma (float): The spread parameter $\\sigma$. Larger values increase the range of influence.
        alpha (float): The maximum force magnitude $\\alpha$ (at zero distance).

    Returns:
        np.ndarray: An array of force magnitudes corresponding to the input distances.
    """
    s2 = 2 * (sigma * sigma)
    return alpha * np.exp(-(d * d) / s2)


def softened_inverse_force(
    d: np.ndarray, epsilon: float = 0.1, alpha: float = 0.1
) -> np.ndarray:
    """
    Computes a softened inverse-square repulsion force.

    This function mimics electrostatic or gravitational repulsion but introduces a softening
    parameter $\\epsilon$ to prevent numerical singularities (division by zero) when particles
    overlap ($d \\to 0$). The force decays algebraically rather than exponentially.

    The magnitude is given by:

    $ F(d) = \\frac{\\alpha}{d^2 + \\epsilon^2} $

    This ensures that the maximum force is bounded at $\\alpha / \\epsilon^2$.

    Args:
        d (np.ndarray): Array of distances.
        epsilon (float): Softening parameter $\\epsilon$. Prevents infinite forces at $d=0$.
        alpha (float): Magnitude scaling factor $\\alpha$.

    Returns:
        np.ndarray: An array of force magnitudes.
    """
    return alpha * (1.0 / ((d * d) + (epsilon * epsilon)))


def linear_force(d: np.ndarray, R: float = 0.5) -> np.ndarray:
    """
    Computes a linear repulsive force that falls to zero at a specific radius.

    This force models a simple linear spring compression. It provides a constant stiffness
    repulsion within a defined radius $R$ and is zero elsewhere. This creates a clear
    "cutoff" for interactions.

    The formula is:

    $ F(d) = \\max\\left(0, 1 - \\frac{d}{R}\\right) $

    Args:
        d (np.ndarray): Array of distances.
        R (float): The cutoff radius $R$. Forces are zero for $d \\ge R$.

    Returns:
        np.ndarray: An array of normalized force magnitudes in $[0, 1]$.
    """
    return np.maximum(0.0, 1.0 - (d / R))


def cauchy_force(d: np.ndarray) -> np.ndarray:
    """
    Computes a long-tailed repulsion based on the Cauchy distribution.

    This function provides a heavy-tailed force distribution, which can be useful for
    maintaining global separation between points as the force does not decay as rapidly
    as a Gaussian.

    The magnitude is defined as:

    $ F(d) = \\frac{1}{1 + d^2} $

    Args:
        d (np.ndarray): Array of distances.

    Returns:
        np.ndarray: An array of force magnitudes.
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
) -> tuple[
    np.ndarray,
    np.ndarray | np.number | float | int,
    np.ndarray | np.number | float | int,
]:
    """
    Normalizes the input array to the unit hypercube $[0, 1]^D$.

    Min-max scaling is performed column-wise (per dimension). If explicit bounds are not
    provided, they are inferred from the data. The transformation for a value $x$ is:

    $ x' = \\frac{x - x_{min}}{x_{max} - x_{min}} $

    This function handles constant dimensions (where $x_{max} = x_{min}$) by setting the
    denominator to 1.0 to avoid division by zero.

    Args:
        arr (np.ndarray): Input data array of shape $(N, D)$.
        min_val (np.ndarray | np.number | None): Optional pre-computed minimum values.
            If None, computed from `arr`.
        max_val (np.ndarray | np.number | None): Optional pre-computed maximum values.
            If None, computed from `arr`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - **scaled_arr**: The normalized data in range $[0, 1]$.
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
    max_val: np.ndarray | np.number | float | int,
) -> np.ndarray:
    """
    Restores scaled data from $[0, 1]^D$ back to its original domain.

    This applies the inverse of the min-max normalization:

    $ x = x' \\cdot (x_{max} - x_{min}) + x_{min} $

    Args:
        scl_arr (np.ndarray): Scaled input array in $[0, 1]$.
        min_val (np.ndarray | np.number): The minimum values of the original domain.
        max_val (np.ndarray | np.number): The maximum values of the original domain.

    Returns:
        np.ndarray: The array projected back into the original bounds.
    """
    return scl_arr * (max_val - min_val) + min_val


def _smart_init(
    bounds_01: np.ndarray,
    nn_instance: nn.NearestNeighbors,
    n_new: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Initializes new points using a vectorized Best Candidate Sampling strategy.

    Instead of placing points purely randomly, this method generates a pool of candidate
    positions for each required new point and selects the one that is furthest from
    the existing set of static points.

    **Algorithm:**
    1. For $N$ requested points, generate $N \\times k$ uniform candidates (where $k=15$).
    2. Compute the distance $d_i$ from every candidate to its nearest static neighbor.
    3. For each of the $N$ slots, select the candidate $c^*$ such that:
        $ c^* = \\arg\\max_{c \\in \\text{candidates}} (\\min_{p \\in \\text{static}} ||c - p||) $
    4. Apply a small jitter $\\xi \\sim U(-10^{-3}, 10^{-3})$ to avoid perfect overlaps.

    Args:
        bounds_01 (np.ndarray): Normalized boundaries $[0, 1]$.
        nn_instance (nn.NearestNeighbors): The index containing static points.
        n_new (int): Number of points to initialize.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Initial positions for the new points.
    """
    dim = bounds_01.shape[0]
    n_candidates = 15

    # 1. Generate ALL candidates at once
    # Shape: (n_new * n_candidates, D)
    total_candidates = n_new * n_candidates
    candidates = rng.uniform(
        bounds_01[:, 0], bounds_01[:, 1], (total_candidates, dim)
    ).astype(np.float32)

    # 2. Query NN once for all candidates
    # We only care about distance to the nearest STATIC point
    _, dists = nn_instance.query_static(candidates, k=1)
    dists = dists.flatten()  # Shape (total_candidates,)

    # 3. Reshape to separate candidates per new point
    # Shape: (n_new, n_candidates)
    dists_reshaped = dists.reshape(n_new, n_candidates)

    # Shape: (n_new, n_candidates, dim)
    candidates_reshaped = candidates.reshape(n_new, n_candidates, dim)

    # 4. Find index of best candidate for each new point
    best_indices = np.argmax(dists_reshaped, axis=1)

    # 5. Gather the best candidates
    # Advanced indexing: pick the best candidate for each row
    row_indices = np.arange(n_new)
    best_samples = candidates_reshaped[row_indices, best_indices]

    jitter = rng.uniform(-1e-3, 1e-3, size=best_samples.shape).astype(np.float32)
    return np.clip(best_samples + jitter, 1e-5, 1.0 - 1e-5)


def _compute_radius_heuristic(bounds: np.ndarray, n_points: int) -> float:
    """
    Derives an adaptive interaction radius based on domain volume and point density.

    In high-dimensional spaces, a fixed radius is often inappropriate. This heuristic
    estimates the "Mean Inter-Particle Distance" assuming an ideal packing and scales
    it to define a local neighborhood.

    The estimated volume per point $V_p$ is:
    $ V_p = \\frac{\\prod (side\\_lengths)}{N_{points}} $

    The average neighbor distance $R_{avg}$ is then approximated as:
    $ R_{avg} \\approx (V_p)^{1/D} $

    The returned radius is $R = 1.25 \\times R_{avg}$, capped by the domain size.

    Args:
        bounds (np.ndarray): Domain boundaries of shape $(D, 2)$.
        n_points (int): Total number of points (static + active) in the system.

    Returns:
        float: A recommended radius for force interactions.
    """
    dim = bounds.shape[0]
    sides = bounds[:, 1] - bounds[:, 0]
    max_side = np.max(sides)

    logger.debug(f"Dim {dim} sides {sides} max side {max_side}")

    # Estimate volume per point
    log_vol = np.sum(np.log(sides + 1e-9))
    log_vol_per_point = log_vol - np.log(n_points)

    # Mean distance to nearest neighbor in ideal packing
    avg_neighbor_dist = np.exp(log_vol_per_point / dim)

    # FIXED FACTOR: 2.5
    # Previously, this scaled with sqrt(D), which exploded the radius in High-D.
    # 2.5 ensures we cover the immediate neighborhood without grabbing the entire world.
    radius = avg_neighbor_dist * 1.25

    domain_cap = max_side * 0.25
    radius = min(radius, domain_cap)

    return min(radius, np.linalg.norm(sides))


def _compute_wall_forces(points, bounds, strategy="repulsive", radius=0.1):
    """
    Computes boundary containment forces using a soft linear spring model.

    If `strategy` is 'repulsive', this function applies a restorative force to points
    approaching the domain boundaries. The boundary acts as a Hookean spring with
    stiffness $k = 1/R$.

    For a point $x$ and boundary $B$, if the distance $d = |x - B| < R$:
    $ F_{wall} = k \\cdot (R - d) $ directed inward.

    Args:
        points (np.ndarray): Current point coordinates.
        bounds (np.ndarray): Domain boundaries.
        strategy (str, optional): Strategy name. Returns zero forces if not 'repulsive'.
        radius (float, optional): The distance from the wall at which the force activates.

    Returns:
        np.ndarray: Wall force vectors.
    """
    forces = np.zeros_like(points)
    if strategy != "repulsive":
        return forces

    bounds_min = bounds[:, 0].reshape(1, -1)
    bounds_max = bounds[:, 1].reshape(1, -1)

    dist_min = np.maximum(points - bounds_min, 1e-9)
    dist_max = np.maximum(bounds_max - points, 1e-9)

    mask_min = dist_min < radius
    mask_max = dist_max < radius

    stiffness = 1.0 / (radius + 1e-9)

    if np.any(mask_min):
        forces[mask_min] += stiffness * (radius - dist_min[mask_min])

    if np.any(mask_max):
        forces[mask_max] -= stiffness * (radius - dist_max[mask_max])

    return forces


def _compute_radius_forces(
    active_view: np.ndarray,
    all_data: np.ndarray,
    nn_instance: nn.NearestNeighbors,
    radius: float,
    metric_fn: collections.abc.Callable,
    batch_start_idx: int,
    rng: np.random.Generator,
    **metric_kwargs,
) -> np.ndarray:
    """
    Computes net repulsive forces using a Dense Matrix / Radius Search approach.

    This method calculates forces for a batch of active points against all neighbors
    within a given `radius`. It utilizes vector broadcasting to compute interactions
    efficiently and handles singularities.

    **Mathematical Operation:**
    Let $P_i$ be an active point and $P_j$ be a neighbor. The displacement is
    $\\vec{r}_{ij} = P_i - P_j$. The total force on $P_i$ is:

    $ \\vec{F}_i = \\sum_{j \\in N(i), j \\neq i} \\frac{\\vec{r}_{ij}}{||\\vec{r}_{ij}||} \\cdot f(||\\vec{r}_{ij}||) $

    **Key Steps:**
    1. **Range Search:** Find all pairs $(i, j)$ where $||P_i - P_j|| < R$.
    2. **Self-Masking:** Explicitly zero out interactions where $i = j$.
    3. **Collision Handling:** If $||\\vec{r}_{ij}|| \\approx 0$ (stacking), a random noise vector is injected to break symmetry.
    4. **Force Accumulation:** Forces are summed via matrix operations.

    Args:
        active_view (np.ndarray): The batch of points currently being optimized.
        all_data (np.ndarray): Reference to the complete dataset (static + active).
        nn_instance (nn.NearestNeighbors): Helper for range queries.
        radius (float): Interaction cutoff radius.
        metric_fn (Callable): The scalar force magnitude function $f(d)$.
        batch_start_idx (int): Global index offset for the active batch.
        rng (np.random.Generator): Random generator for collision noise.
        **metric_kwargs: Arguments passed to `metric_fn`.

    Returns:
        np.ndarray: The net force vectors for the active batch.
    """
    all_dists, valid_mask = nn_instance.range_search(radius)

    if not np.any(valid_mask):
        return np.zeros_like(active_view)

    # --- 1. INDEX-BASED SELF MASKING ---
    # The matrix columns [0..Total] map 1:1 to all_data indices.
    # The active batch starts at 'batch_start_idx'.
    n_active = active_view.shape[0]

    # Create diagonal coordinates: (0, start), (1, start+1), ...
    rows = np.arange(n_active)
    cols = batch_start_idx + rows

    # Ensure we don't go out of bounds (sanity check)
    valid_cols_mask = cols < all_dists.shape[1]
    rows = rows[valid_cols_mask]
    cols = cols[valid_cols_mask]

    # Mask out Self
    valid_mask[rows, cols] = False

    # --- 2. COLLISION DETECTION ---
    # After masking self, any remaining d < epsilon is a STACKED NEIGHBOR.
    epsilon = 1e-9
    is_stacked_interaction = (all_dists < epsilon) & valid_mask
    particles_stacking = np.any(is_stacked_interaction, axis=1)

    # --- 3. FORCE CALCULATION ---
    forces_mag = metric_fn(all_dists, **metric_kwargs) * valid_mask
    safe_dists = np.maximum(all_dists, epsilon)
    coeffs = forces_mag / safe_dists

    n_valid = all_dists.shape[1]
    valid_data = all_data[:n_valid]

    sum_coeffs = np.sum(coeffs, axis=1, keepdims=True)
    term1 = active_view * sum_coeffs
    term2 = np.dot(coeffs, valid_data)

    forces = term1 - term2

    if np.any(particles_stacking):
        # Generate noise for the shape of forces
        noise = rng.uniform(-1.0, 1.0, size=forces.shape).astype(np.float32)

        # MASK: Zero out noise for particles that are NOT stacking
        # Broadcasting: (Batch, Dim) * (Batch, 1)
        noise *= particles_stacking[:, np.newaxis]

        forces += noise

    return forces


def _compute_knn_forces(
    active_view: np.ndarray,
    all_data: np.ndarray,
    nn_instance: nn.NearestNeighbors,
    k_value: int,
    metric_fn: collections.abc.Callable,
    rng: np.random.Generator,
    batch_start_idx: int,
    batch_end_idx: int,
    **metric_kwargs,
) -> np.ndarray:
    """
    Computes net repulsive forces using a k-Nearest Neighbors (k-NN) approach.

    Unlike the radius approach, this method interacts with a fixed number ($k$) of
    nearest neighbors, regardless of distance. This adapts well to varying densities.

    **Mathematical Operation:**
    The summation logic mirrors `_compute_radius_forces`, but the neighborhood set
    $N(i)$ is defined by rank order of distance rather than absolute distance threshold:

    $ N(i) = \\{ j \\mid \\text{rank}(||P_i - P_j||) \\le k \\} $

    It includes specific logic to handle cases where the returned neighbor indices
    refer to the point itself (self-interaction) or overlap perfectly (stacking).

    Args:
        active_view (np.ndarray): The batch of active points.
        all_data (np.ndarray): The full dataset.
        nn_instance (nn.NearestNeighbors): Helper for k-NN queries.
        k_value (int): Number of neighbors to consider.
        metric_fn (Callable): Force magnitude function.
        rng (np.random.Generator): Random generator.
        batch_start_idx (int): Global start index of the batch.
        batch_end_idx (int): Global end index of the batch.
        **metric_kwargs: Arguments passed to `metric_fn`.

    Returns:
        np.ndarray: The net force vectors for the active batch.
    """
    indices, dists = nn_instance.query_nn(k=k_value)

    if np.max(indices) >= batch_end_idx:
        indices = np.clip(indices, 0, batch_end_idx - 1)

    # --- 1. INDEX-BASED SELF MASKING ---
    # We check: indices[i, j] == (batch_start_idx + i)
    global_idxs = np.arange(active_view.shape[0]) + batch_start_idx

    # Broadcast comparison: (M, K) vs (M, 1)
    is_self = indices == global_idxs[:, None]

    neighbor_coords = all_data[indices]
    disp_vecs = active_view[:, np.newaxis, :] - neighbor_coords
    norms = np.linalg.norm(disp_vecs, axis=2, keepdims=True)

    # --- 2. COLLISION DETECTION ---
    # Any distance < epsilon that is NOT self is a collision (stacking)
    is_stacked = (norms < 1e-9) & (~is_self[:, :, None])

    # Replace zero vectors with random unit vectors
    if np.any(is_stacked):
        random_dirs = rng.standard_normal(size=disp_vecs.shape, dtype=np.float32)
        rnd_norms = np.linalg.norm(random_dirs, axis=2, keepdims=True)
        random_dirs /= rnd_norms + 1e-9

        # Inject random direction where stacking occurred
        disp_vecs = np.where(is_stacked, random_dirs, disp_vecs)
        norms = np.where(is_stacked, 1.0, norms)

    # --- 3. FORCE CALCULATION ---
    forces_mag = metric_fn(dists, **metric_kwargs)

    # Explicitly zero out self-force magnitude
    forces_mag[is_self] = 0.0

    # Safe Norm division
    safe_norms = np.maximum(norms, 1e-9)
    force_vectors = (disp_vecs / safe_norms) * forces_mag[:, :, None]

    return np.sum(force_vectors, axis=1)


# --- Core Logic ---
def _esa(
    samples: np.ndarray,
    bounds: np.ndarray,
    nn_instance: nn.NearestNeighbors,
    *,
    n: int,
    epochs: int = 512,
    lr: float = 0.01,
    search_mode: str = "k_nn",
    decay: float = 0.9,
    batch_size: int = 50,
    k: int | None = None,
    radius: float | None = None,
    tol: float = 1e-3,
    metric_fn: collections.abc.Callable = softened_inverse_force,
    border_strategy: str = "clip",
    seed: int | np.random.Generator | None = None,
    **metric_kwargs,
) -> np.ndarray:
    """
    Executes the Empty Space Algorithm (ESA) optimization loop.

    This function performs the core iterative position updates. It treats the problem
    as a physics simulation where new points (charged particles) are introduced into
    a field of static points and repel each other to maximize spacing.

    **Process:**
    1. **Scale:** Map input domain to unit hypercube $[0, 1]^D$.
    2. **Batching:** Process $N$ new points in mini-batches to manage complexity.
    3. **Optimization:** For each epoch $t$:
        - Calculate total force $\\vec{F}_{total} = \\vec{F}_{particle} + \\vec{F}_{wall}$.
        - Update positions: $\\mathbf{x}_{t+1} = \\mathbf{x}_t + \\eta_t \\cdot \\vec{F}_{total}$.
        - Decay learning rate: $\\eta_{t+1} = \\eta_t \\cdot \\gamma$.
        - Check convergence: Stop if $||\\mathbf{x}_{t+1} - \\mathbf{x}_t|| < \\text{tol}$.
    4. **Consolidation:** Once a batch converges, it becomes "static" for subsequent batches.
    5. **Unscale:** Map results back to the original domain.

    Args:
        samples (np.ndarray): Existing static points.
        bounds (np.ndarray): Domain boundaries.
        nn_instance (nn.NearestNeighbors): Index structure.
        n (int): Number of points to generate.
        epochs (int): Maximum update steps.
        lr (float): Initial step size $\\eta_0$.
        search_mode (str): 'k_nn' or 'radius'.
        decay (float): Learning rate decay factor $\\gamma$.
        batch_size (int): Size of optimization groups.
        k (int | None): Neighbors for k-NN mode.
        radius (float | None): Interaction radius.
        tol (float): Convergence tolerance.
        metric_fn (Callable): Force function.
        border_strategy (str): 'clip' or 'repulsive'.
        seed (int | np.random.Generator | None): Random seed or Generator instance.
            If None, a new Generator is created with entropy.
            If int, it seeds a new Generator.
            If np.random.Generator, it is used directly.
        **metric_kwargs: Metric parameters.

    Returns:
        np.ndarray: The generated points in the original coordinate system.
    """
    # 1. Scaling & Pre-allocation
    min_val = bounds[:, 0]
    max_val = bounds[:, 1]
    scaled_samples, _, _ = _scale(samples, min_val, max_val)
    scaled_samples = scaled_samples.astype(np.float32)
    dim = samples.shape[1]
    total_points = samples.shape[0] + n
    
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    all_data = np.empty((total_points, dim), dtype=np.float32)
    all_data[: samples.shape[0]] = scaled_samples.astype(np.float32)
    cursor = samples.shape[0]

    # 2. Setup NN
    nn_instance.clear()
    nn_instance.add_static(all_data[:cursor])

    # 3. Parameter Defaults
    k_value = k if k is not None else (2 * dim) + 1
    # Note: Radius heuristic logic moved to wrapper 'esa'
    radius_value = radius if radius is not None else 0.1

    # 4. Batch Processing
    num_batches = math.ceil(n / batch_size)
    bounds_01 = np.array([[0, 1]] * dim)

    logger.debug(
        f"Starting ESA: {n} points, {num_batches} batches. Mode={search_mode}. Border Strategy={border_strategy}"
    )
    for _ in range(num_batches):
        # Calculate batch size (handle last partial batch)
        remaining = n - (cursor - samples.shape[0])
        current_n = min(batch_size, remaining)
        if current_n <= 0:
            break

        # Define the memory slice for this batch in the Master Buffer
        batch_start = cursor
        batch_end = cursor + current_n

        # A. Smart Initialization
        active_batch_init = _smart_init(bounds_01, nn_instance, current_n, rng)
        all_data[batch_start:batch_end] = active_batch_init

        # Create a VIEW of the master buffer for optimization
        # Modifying 'active_view' modifies 'all_data' in place.
        active_view = all_data[batch_start:batch_end]

        nn_instance.set_active(active_view)
        current_lr = lr

        # B. Optimization Loop
        for _ in range(epochs):
            # Coordinate Retrieval:
            # We need neighbors from the "Valid" part of the buffer.
            # When NN returns indices, they map directly to rows in 'all_data'.

            # --- Force Calculation ---
            if search_mode == "radius":
                particle_forces = _compute_radius_forces(
                    active_view,
                    all_data,
                    nn_instance,
                    radius_value,
                    metric_fn,
                    batch_start_idx=batch_start,
                    rng=rng,
                    **metric_kwargs,
                )
            else:
                particle_forces = _compute_knn_forces(
                    active_view,
                    all_data,
                    nn_instance,
                    k_value,
                    metric_fn,
                    rng=rng,
                    batch_start_idx=batch_start,
                    batch_end_idx=batch_end,
                    **metric_kwargs,
                )

            # --- Wall Forces (Elastic Borders) ---
            wall_forces = _compute_wall_forces(
                active_view, bounds_01, strategy=border_strategy, radius=radius_value
            )

            # --- Total Force & Update ---
            total_force = particle_forces + wall_forces

            # Save previous position for convergence check
            prev_pos = active_view.copy()
            # Move (In-Place Update of Master Buffer via View)
            active_view += total_force * current_lr
            # Apply Constraints (Clip)
            # We always clip to ensure validity, even with repulsive walls
            np.clip(active_view, 0.0, 1.0, out=active_view)

            nn_instance.set_active(active_view)

            # --- Early Stopping ---
            move_dist = np.linalg.norm(active_view - prev_pos, axis=1)
            if np.max(move_dist) < tol:
                break

            current_lr *= decay

        # C. Consolidate Batch
        # The batch is already in 'all_data' at the correct position.
        # Tell NN to treat these indices as Static now.
        nn_instance.consolidate()

        # Advance cursor
        cursor = batch_end

    # 5. Inverse Scaling & Return
    # We only return the generated portion (from len(samples) onwards)
    generated_slice = all_data[len(samples) : cursor]
    return _inv_scale(generated_slice, min_val, max_val)


def esa(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    n: int,
    nn_instance: nn.NearestNeighbors | None = None,
    epochs: int = 1024,
    lr: float = 0.01,
    search_mode: str = "k_nn",
    decay: float = 0.9,
    batch_size: int = 50,
    k: int | None = None,
    radius: float | None = None,
    tol: float = 1e-3,
    metric: str | collections.abc.Callable = "softened_inverse",
    border_strategy: str = "clip",
    seed: int | np.random.Generator | None = None,
    **metric_kwargs,
) -> np.ndarray:
    """
    Wrapper for the Empty Space Algorithm (ESA) that returns only generated points.

    This acts as the public API for the algorithm. It handles infrastructure setup
    that the internal loop `_esa` requires, specifically:
    1. **Heuristic Calculation:** Auto-computes interaction radius if not provided.
    2. **Metric Scaling:** Scales force parameters (like $\\sigma$) relative to the radius.
    3. **Engine Selection:** Automatically chooses between a Dense Numpy engine (small $N$)
        and an Approximate Nearest Neighbor (HNSW) engine (large $N$) for performance.

    Args:
        samples (np.ndarray): Existing points to avoid.
        bounds (np.ndarray): Bounding box constraints.
        n (int): Number of new points to create.
        nn_instance (nn.NearestNeighbors | None): Optional custom NN engine.
        epochs (int): Max iterations.
        lr (float): Initial learning rate.
        search_mode (str): 'k_nn' (adaptive) or 'radius' (fixed range).
        decay (float): LR decay per epoch.
        batch_size (int): Optimization batch size.
        k (int | None): Neighbors to use (if mode is 'k_nn').
        radius (float | None): Force radius (if mode is 'radius').
        tol (float): Early stopping tolerance.
        metric (str | Callable): Name of force function or callable object.
        border_strategy (str): 'clip' (hard stop) or 'repulsive' (soft walls).
        seed (int | np.random.Generator | None): Random seed or Generator instance.
            If None, a new Generator is created with entropy.
            If int, it seeds a new Generator.
            If np.random.Generator, it is used directly.
        **metric_kwargs: Arguments for the metric function.

    Returns:
        np.ndarray: An array of size $(n, D)$ containing only the newly generated points.
    """

    if not isinstance(samples, np.ndarray):
        samples = np.array(samples).astype(np.float32)

    if isinstance(metric, str):
        metric_name = metric.lower()
        metric_fn = METRIC_REGISTRY.get(metric_name)
        if metric_fn is None:
            raise ValueError(f"Unknown metric '{metric}'")
    else:
        metric_fn = metric
        metric_name = getattr(metric, "__name__", "unknown")

    # 1. AUTO-COMPUTE Radius (if needed)
    # We do this here so we can use it to auto-scale metric parameters.
    final_radius = radius
    if search_mode == "radius" and radius is None:
        total_capacity = samples.shape[0] + n

        # Calculate heuristic in Unit Space
        # Bounds are just needed for dimensionality and side ratios
        unit_bounds = np.array([[0.0, 1.0]] * samples.shape[1])
        final_radius = _compute_radius_heuristic(unit_bounds, total_capacity)
        logger.debug(f"Computed Heuristic Radius: {final_radius:.4f}")

    # 2. AUTO-SCALE Metric Parameters
    # Ensure sigma/R matches the search radius for physical consistency
    if search_mode == "radius" and final_radius is not None:
        if metric_name == "gaussian" and "sigma" not in metric_kwargs:
            metric_kwargs["sigma"] = final_radius / 3.0
        elif metric_name == "linear" and "R" not in metric_kwargs:
            metric_kwargs["R"] = final_radius

    # 3. NN Factory with DUAL THRESHOLDS
    if nn_instance is None:
        dim = samples.shape[1]
        total = samples.shape[0] + n

        # Select Threshold based on Search Mode
        if search_mode == "radius":
            threshold = RADIUS_SWITCH_THRESHOLD
        else:
            threshold = KNN_SWITCH_THRESHOLD

        if total > threshold:
            logger.debug(f"Using FAISS (HNSW) engine. (N={total} > {threshold})")
            nn_instance = nn.FaissHNSWFlatNN(dimension=dim)
        else:
            logger.debug(f"Using NUMPY (Dense) engine. (N={total} <= {threshold})")
            nn_instance = nn.NumpyNN(dimension=dim)

    return _esa(
        samples=samples,
        bounds=bounds,
        nn_instance=nn_instance,
        metric_fn=metric_fn,
        n=n,
        epochs=epochs,
        lr=lr,
        decay=decay,
        batch_size=batch_size,
        k=k,
        radius=final_radius,
        search_mode=search_mode,
        border_strategy=border_strategy,
        tol=tol,
        seed=seed,
        **metric_kwargs,
    )


def ess(
    samples: np.ndarray | list,
    bounds: np.ndarray,
    *,
    n: int,
    nn_instance: nn.NearestNeighbors | None = None,
    epochs: int = 1024,
    lr: float = 0.01,
    search_mode: str = "k_nn",
    decay: float = 0.95,
    batch_size: int = 50,
    k: int | None = None,
    radius: float | None = None,
    tol: float = 1e-3,
    metric: str | collections.abc.Callable = "softened_inverse",
    border_strategy: str = "clip",
    seed: int | np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    """
    High-level Empty Space Strategy (ESS) wrapper returning the full dataset.

    This convenience function runs the ESA algorithm and concatenates the results
    with the original input samples. It is useful for iterative design of experiments
    or sequential sampling workflows where the complete set of points is needed.

    $$ \\text{Result} = \\text{samples} \\cup \\text{ESA}(\\text{samples}, \\dots) $$

    Args:
        samples (np.ndarray | list): Existing points.
        bounds (np.ndarray): Domain boundaries.
        n (int): Number of new points to generate.
        nn_instance (nn.NearestNeighbors | None): NN engine.
        epochs (int): Optimization steps.
        lr (float): Learning rate.
        search_mode (str): Neighborhood search strategy.
        decay (float): Decay rate.
        batch_size (int): Batch size.
        k (int | None): Neighbors count.
        radius (float | None): Interaction radius.
        tol (float): Convergence tolerance.
        metric (str | Callable): Force metric.
        border_strategy (str): Boundary handling.
        seed (int | np.random.Generator | None): Random seed or Generator instance.
            If None, a new Generator is created with entropy.
            If int, it seeds a new Generator.
            If np.random.Generator, it is used directly.
        **kwargs: Additional metric args.

    Returns:
        np.ndarray: A combined array of shape $(N + n, D)$ containing original and new points.
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
        radius=radius,
        border_strategy=border_strategy,
        search_mode=search_mode,
        batch_size=batch_size,
        k=None,  # Let ESA determine K
        tol=tol,
        metric=metric,
        seed=seed,
        **kwargs,
    )

    return np.concatenate((samples, new_points), axis=0)
