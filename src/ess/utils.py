import logging
import warnings

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
"""logging.Logger: Module-level logger for debugging ESA optimization steps."""


def _euclidean_nn_distances(
    points: np.ndarray, batch_size: int = 500
) -> np.ndarray:
    """
    Computes the **Euclidean, non-toroidal** distance to the nearest neighbor
    (excluding self) for each point. The wrap-aware counterpart is inside
    `toroidal_separation`; this one is what makes `euclidean_separation` a
    box metric.

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

    Warning:
        **Saturates and inverts above roughly $d = 8$**, and cannot be built
        much past $d \approx 20$ at all ($b^D$ cells). Measured at $d = 8$:
        LHS 0.988 against ESS 0.981 and uniform 0.981 — it ranks LHS above
        ESS and cannot separate ESS from random, because once cells greatly
        outnumber points every design occupies one cell each. Useful as a 2-D
        or 3-D illustration; not a quality metric. Use
        `projection_discrepancy` or `wrap_around_discrepancy`.

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


def euclidean_separation(points: np.ndarray) -> float:
    r"""Separation of a design under **plain Euclidean distance, no wrap**.

    The counterpart of `toroidal_separation`, in the other geometry. Which
    one you want is decided by what produced the points: ESS relaxes on the
    torus under $L_1$, so `toroidal_separation` is the one that measures
    what it optimised. This one treats the domain as a box, and a pair
    straddling a seam reads as far apart when it is adjacent — on four
    points with one such pair it reports 0.633 where the toroidal version
    returns 0.020.

    Warning:
        Retained for provenance: figures recorded against the pre-toroidal
        implementation used this. **Do not use it to rank designs.** It is
        an $L_2$ statistic, so comparing arms that optimised different
        geometries with it asks which is better at $L_2$ spacing when one
        of them optimised exactly that — rigged by construction rather than
        merely noisy. For design quality use `projection_discrepancy` or
        `wrap_around_discrepancy`, which reference no point metric at all
        and so cannot be gamed by the geometry an arm was optimised in.

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
    min_dists = _euclidean_nn_distances(points)

    # The result is the minimum of all nearest neighbor distances
    return float(np.min(min_dists))


def expected_discrepancy(n: int, dim: int) -> float:
    r"""Expected wrap-around $L_2$ discrepancy of a random uniform design.

    Taking the expectation of `wrap_around_discrepancy` over $n$ i.i.d.
    uniform points gives a closed form, because the diagonal terms
    contribute $(3/2)^d$ and every off-diagonal pair contributes
    $(4/3)^d$ in expectation:

    $$ \mathbb{E}[WD^2] = \frac{(3/2)^d - (4/3)^d}{n} $$

    Dividing a measured discrepancy by this value yields a scale-free
    score where $1$ means "as uniform as random" and smaller is better —
    the only form that stays comparable across dimensions, since the raw
    discrepancy grows like $(3/2)^d$.

    Args:
        n (int): Number of design points.
        dim (int): Dimensionality $d$ of the design (or of the
            projection being scored).

    Returns:
        float: The expected discrepancy of a random uniform design.
    """
    return ((1.5**dim) - ((4.0 / 3.0) ** dim)) / n


def wrap_around_discrepancy(points: np.ndarray, chunk: int = 128) -> float:
    r"""Wrap-around $L_2$ discrepancy — uniformity measured on the torus.

    The wrap-around discrepancy (Hickernell) compares the empirical
    distribution of a design to the uniform one over *every* periodic
    box, and admits the closed form

    $$ WD^2(X) = -\left(\tfrac{4}{3}\right)^{d} + \frac{1}{n^2}
       \sum_{i=1}^{n} \sum_{j=1}^{n} \prod_{k=1}^{d}
       \left[ \tfrac{3}{2} - |x_{ik}-x_{jk}|\,(1 - |x_{ik}-x_{jk}|) \right] $$

    The per-dimension kernel is invariant under $u \mapsto 1-u$, which
    is exactly what makes the measure *periodic*: it identifies opposite
    faces of the cube, so it scores a design on the same torus the
    relaxation optimizes. Unlike nearest-neighbour statistics (see
    `euclidean_separation`) it does not rely on distance
    contrasts and therefore stays meaningful when concentration of
    measure flattens pairwise distances in high dimension.

    Lower is better. Divide by `expected_discrepancy` to obtain a
    scale-free score against the random-uniform baseline.

    Note:
        Cost is $O(n^2 d)$; the pairwise products are evaluated in row
        chunks so peak memory stays at ``chunk * n * d``.

    Args:
        points (np.ndarray): Design of shape $(N, D)$, assumed to live in
            $[0, 1)^D$ (values are reduced modulo 1).
        chunk (int): Rows per block in the pairwise accumulation.

    Returns:
        float: The squared wrap-around discrepancy $WD^2$.
    """
    pts = np.mod(np.asarray(points, dtype=np.float64), 1.0)
    n, dim = pts.shape

    total = 0.0
    for start in range(0, n, chunk):
        block = pts[start : start + chunk]
        delta = np.abs(block[:, None, :] - pts[None, :, :])
        total += float(np.prod(1.5 - delta * (1.0 - delta), axis=2).sum())

    return -((4.0 / 3.0) ** dim) + total / (n * n)


def projection_discrepancy(
    points: np.ndarray,
    order: int = 2,
    max_projections: int = 200,
    seed: int = 0,
) -> float:
    r"""Mean wrap-around discrepancy over low-dimensional projections.

    The recommended uniformity criterion for high-dimensional designs.
    Full-dimensional measures lose discriminative power as $d$ grows —
    every design looks alike once the space is mostly empty — while the
    *effect sparsity* principle says what actually matters is that each
    factor, and each pair of factors, be well covered. This scores

    $$ \overline{WD^2_s} = \frac{1}{|\mathcal{S}|}
       \sum_{S \in \mathcal{S}} WD^2\!\left(X_{S}\right), \qquad |S| = s $$

    where $X_S$ is the design restricted to the coordinate subset $S$
    and $\mathcal{S}$ is the set of $s$-element subsets (subsampled when
    there are more than `max_projections` of them).

    Because each projection has fixed dimension $s$, the result keeps a
    fixed scale no matter how large the ambient $d$ is, and remains
    directly comparable to `expected_discrepancy(n, s)`. Lower is better.

    Args:
        points (np.ndarray): Design of shape $(N, D)$.
        order (int): Projection dimension $s$ (1 for marginals, 2 for
            pairwise plots).
        max_projections (int): Cap on how many subsets are averaged;
            subsets are sampled without replacement beyond this.
        seed (int): Seed for that subsampling, so the score is
            reproducible.

    Returns:
        float: Mean $WD^2$ over the scored projections.
    """
    import itertools

    pts = np.mod(np.asarray(points, dtype=np.float64), 1.0)
    dim = pts.shape[1]
    if order > dim:
        raise ValueError(f"order {order} exceeds dimensionality {dim}")

    subsets = list(itertools.combinations(range(dim), order))
    if len(subsets) > max_projections:
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(subsets), max_projections, replace=False)
        subsets = [subsets[i] for i in picked]

    return float(
        np.mean([wrap_around_discrepancy(pts[:, list(s)]) for s in subsets])
    )


def toroidal_separation(points: np.ndarray) -> float:
    r"""Separation of a design in the metric ESS optimizes.

    $$ d_{\min} = \min_{i \ne j} d_{L_1}^{tor}(x_i, x_j) $$

    the shortest distance between any two points, wrap-around included —
    the maximin criterion, and the metric that keeps discriminating when
    nearest-neighbour *means* flatten out in high dimension (at $d = 32$,
    $n = 4000$: ESS 5.52 against 3.94 for uniform and 3.92 for LHS,
    where the coverage radius differs by under 1%).

    Note:
        Not the same quantity as `euclidean_separation`, which
        is Euclidean and ignores the wrap: on four points with one pair
        straddling the seam it reports 0.633 where this returns 0.020.
        Prefer this one for anything on the torus.

    Args:
        points (np.ndarray): Design of shape $(N, D)$, reduced modulo 1.

    Returns:
        float: The minimum pairwise distance, or 0.0 if $N < 2$.
    """
    from torann.brute import exact_knn

    pts = np.mod(np.asarray(points, dtype=np.float64), 1.0)
    if pts.shape[0] < 2:
        return 0.0
    _, dists = exact_knn(pts, pts, 2)
    return float(np.min(dists[:, 1]))


# --- Deprecated aliases ------------------------------------------------------
#
# The old names said what was computed but not *in which geometry*, which is
# the thing that decides whether a number means anything: this module holds
# two Clark-Evans indices and two separations, one pair Euclidean and one
# toroidal, and the pre-2026-08 names gave no way to tell them apart at a call
# site. They stay reachable because 0.3.1 is on PyPI and these were public.

def calculate_min_pairwise_distance(points: np.ndarray) -> float:
    """Deprecated alias of `euclidean_separation`.

    .. deprecated::
        The name does not say which geometry it measures. Use
        `euclidean_separation`, or `toroidal_separation` if the points came
        off the torus — which, for anything ESS produced, they did.
    """
    warnings.warn(
        "calculate_min_pairwise_distance is deprecated: it is the Euclidean, "
        "non-toroidal separation. Use euclidean_separation, or "
        "toroidal_separation for points on the torus.",
        DeprecationWarning, stacklevel=2,
    )
    return euclidean_separation(points)
