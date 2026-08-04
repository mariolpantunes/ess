import logging
import math
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
    `toroidal_separation`; this one is what makes `euclidean_separation` and
    `euclidean_clark_evans` box metrics.

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
        implementation used this. **Do not use it to rank designs**, and
        never to compare designs from different geometries — see
        `toroidal_clark_evans` for why that is rigged rather than merely
        noisy. For design quality prefer `projection_discrepancy` or
        `wrap_around_discrepancy`, which reference no point metric at all.

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


def euclidean_clark_evans(
    points: np.ndarray, bounds: np.ndarray | None = None
) -> float:
    r"""Clark-Evans index under **plain Euclidean distance, no wrap**.

    The counterpart of `toroidal_clark_evans`, in the other geometry, and
    subject to every caveat recorded there — it is an $L_2$ statistic in
    the same way that one is an $L_1$ statistic, so it cannot arbitrate
    between designs that optimised different metrics, and it is blunt
    besides. Retained for provenance, because figures recorded against the
    pre-toroidal implementation used it.

    **For anything on the torus use `toroidal_clark_evans`; for design
    quality use `projection_discrepancy`.**

    Computes the Clark-Evans Nearest Neighbor Index ($R$) for $D$-dimensional space.

    The index compares the observed mean nearest-neighbor distance to the expected distance
    in a Poisson (random) distribution.

    The index is calculated as $R = \\frac{\\bar{r}_A}{\\bar{r}_E}$, where $\\bar{r}_A$ is the
    mean observed distance. The expected distance $\\bar{r}_E$ for density $\\rho = N/V$
    in $D$ dimensions is derived from the volume of a $D$-dimensional unit ball ($V_D$):

    $ \\bar{r}_E = \\frac{\\Gamma(1/D + 1)}{(\\rho \\cdot V_D)^{1/D}}
      \\quad \\text{where} \\quad
      V_D = \\frac{\\pi^{D/2}}{\\Gamma(D/2 + 1)} $

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
    nn_dists = _euclidean_nn_distances(points)
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
    `euclidean_clark_evans`) it does not rely on distance
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


#: Cache for `expected_nn_toroidal_l1`, keyed by ``(n, dim)``. The quadrature
#: costs a few FFTs, and the benchmarks ask for the same shapes repeatedly.
_NULL_CACHE: dict[tuple[int, int], float] = {}


def expected_nn_toroidal_l1(n: int, dim: int) -> float:
    r"""Mean nearest-neighbour toroidal $L_1$ distance of $n$ uniform points.

    The null hypothesis Clark-Evans divides by. Computed exactly rather
    than asymptotically, which matters above $d \approx 8$.

    On the torus each coordinate distance $|u_i|_{\circ}$ of a uniform
    point is itself uniform on $[0, 1/2]$, so the distance to a uniform
    point is a sum of $d$ i.i.d. $U(0, 1/2)$ variables — an Irwin-Hall
    variable halved. With the other $n - 1$ points independent,

    $$ V(t) = P(S \le t), \qquad P(R > t) = (1 - V(t))^{n-1}, \qquad
       \mathbb{E}[R] = \int_0^{d/2} (1 - V(t))^{n-1} \, dt $$

    and this is exact for a fixed-$n$ uniform design. $V$ comes from
    convolving the one-coordinate density $d$ times by FFT, which is
    stable where the closed-form Irwin-Hall alternating sum is not.

    The Poisson asymptotic
    $\mathbb{E}[r] = \Gamma(1 + 1/d)(d!/n)^{1/d}/2$ substitutes
    $\exp(-nV)$ for $(1-V)^{n-1}$ and uses $V(t) = (2t)^d/d!$, the
    volume of an $L_1$ ball *that fits inside the torus*. Both premises
    fail in high dimension — at $d = 32$ the mean nearest-neighbour
    distance is $\approx 4.9$, so the ball has wrapped around in every
    coordinate. Measured bias of the asymptotic against uniform samples
    (5 seeds), which is exactly the error it puts into the index:

    | $d$ | 2 | 4 | 8 | 16 | 32 | 64 |
    | --- | --- | --- | --- | --- | --- | --- |
    | $n = 256$ | +2.1% | −0.5% | +0.2% | +3.3% | +8.2% | +13.6% |
    | $n = 4000$ | −0.1% | −0.0% | −0.1% | +1.1% | +5.0% | +10.3% |

    With this null the same samples score 0.995–1.003 at every cell.

    Args:
        n (int): Number of points.
        dim (int): Dimensionality $d$.

    Returns:
        float: The expected nearest-neighbour distance under uniformity.
    """
    key = (int(n), int(dim))
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    if n < 2:
        return 0.0
    poisson = (math.gamma(1.0 + 1.0 / dim)
               * math.exp((math.lgamma(dim + 1) - math.log(n)) / dim) / 2.0)
    # The integrand decays over a range of order E[R] itself, so the grid has
    # to be fine relative to that, not to the support of S.
    step = min(2.0e-4, poisson / 400.0)
    m = int(round(dim * 0.5 / step)) + 1
    size = 1 << int(math.ceil(math.log2(2 * m)))
    half = int(round(0.5 / step)) + 1
    dens = np.zeros(size)
    dens[:half] = 2.0                       # U(0, 1/2) has density 2
    dens[0] = dens[half - 1] = 1.0          # trapezoid end weights
    dens *= step
    pdf = np.fft.irfft(np.fft.rfft(dens) ** dim, size)[:m]
    cdf = np.clip(np.cumsum(pdf), 0.0, 1.0)
    out = float(np.trapezoid(np.power(1.0 - cdf, n - 1), dx=step))
    _NULL_CACHE[key] = out
    return out


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


def toroidal_clark_evans(points: np.ndarray) -> float:
    r"""Clark-Evans index under the toroidal $L_1$ metric.

    The counterpart of `euclidean_clark_evans` measured in the
    geometry the relaxation actually optimizes. Because the torus has no
    boundary, the estimator needs no edge correction — the Euclidean
    box version is biased upward (a random uniform design scores 1.17 at
    $d=8$ and 1.42 at $d=64$ instead of 1), and it rewards designs that
    pile points onto the domain faces.

    The index is the observed mean nearest-neighbour distance divided by
    its value under uniformity, `expected_nn_toroidal_l1`. Values above 1
    mean more regular than random.

    Warning:
        **Never rank designs that optimize different geometries with
        this.** It divides by an $L_1$ null, so it is an $L_1$ statistic:
        scoring an $L_1$-optimised arm against an $L^{0.75}$- or
        $L_2$-optimised one asks which is better at $L_1$ spacing when
        one arm optimised exactly that. Rigged by construction, not
        merely noisy. This covers the force *direction* as much as the
        metric — a kernel pushing along $\operatorname{sign}(\delta)$
        descends $L_1$, one pushing along $\delta/\lVert\delta\rVert_2$
        descends $L_2$. Use `projection_discrepancy` or
        `wrap_around_discrepancy`: they measure deviation from
        uniformity and reference no point metric at all.

        It is also **blunt, not merely biased**. Measured at $d = 32$,
        $n = 128$, 6 seeds, over exactly those two force directions,
        this index moved 1.4% (1.2406 to 1.2575) across a change under
        which mean 2-D projection discrepancy moved *six-fold* (1.331 to
        0.224) — and the worse arm scored 1.331 against a random-uniform
        0.989, i.e. **worse than random**, while this index called it
        24% better than random. Within a single fixed geometry it stays
        meaningful; as an arbiter of design quality it is not.

    Note:
        Like every nearest-neighbour statistic this loses discriminative
        power once concentration of measure flattens the distance
        distribution; above roughly $d = 32$ prefer
        `projection_discrepancy`, or `toroidal_separation`, which still
        separates ESS from LHS at $d = 32$ (and carries the same $L_1$
        bias — it is the minimum toroidal $L_1$ gap). The attainable maximum is
        $2/\Gamma(1+1/d)$ (2.257 in 2D), reached by a perfect $L_1$
        lattice packing — the *diagonal* lattice, since $L_1$ balls are
        diamonds and diamonds tile; the axis-aligned grid reaches only
        1.599. That ceiling is itself a Poisson-asymptotic figure, so
        against the exact null a perfect lattice can sit a hair above it
        (2.2574 vs 2.2568 at $n = 128$).

    Args:
        points (np.ndarray): Design of shape $(N, D)$, reduced modulo 1.

    Returns:
        float: The toroidal Clark-Evans index.
    """
    from torann.brute import exact_knn

    pts = np.mod(np.asarray(points, dtype=np.float64), 1.0)
    n, dim = pts.shape
    if n < 2:
        return 0.0

    _, dists = exact_knn(pts, pts, 2)
    return float(np.mean(dists[:, 1]) / expected_nn_toroidal_l1(n, dim))


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


def calculate_clark_evans_index(
    points: np.ndarray, bounds: np.ndarray | None = None
) -> float:
    """Deprecated alias of `euclidean_clark_evans`.

    .. deprecated::
        The name does not say which geometry it measures. Use
        `euclidean_clark_evans`, or `toroidal_clark_evans` on the torus —
        and read the warning on either before ranking anything with it.
    """
    warnings.warn(
        "calculate_clark_evans_index is deprecated: it is the Euclidean, "
        "non-toroidal index. Use euclidean_clark_evans, or "
        "toroidal_clark_evans for points on the torus.",
        DeprecationWarning, stacklevel=2,
    )
    return euclidean_clark_evans(points, bounds)
