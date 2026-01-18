import abc
import logging
import typing

import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
"""logging.Logger: Module-level logger for debugging ESA optimization steps."""


class NearestNeighbors(abc.ABC):
    """
    Abstract Base Class for Nearest Neighbors implementations tailored for ESA.

    This interface defines the contract for hybrid neighbor search engines that manage
    two sets of points:
    1. **Static Set:** Fixed points (anchors/obstacles) that accumulate over time.
    2. **Active Set:** A transient batch of points currently being optimized.

    The implementations must support efficient queries of Active points against the
    union of both sets: $P_{query} \\in Active \\to \\{P_{static} \\cup P_{active}\\}$.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    @abc.abstractmethod
    def add_static(self, points: np.ndarray) -> None:
        """
        Adds points to the persistent static set.

        These points represent obstacles or previously consolidated solutions that the active
        batch must avoid.

        Args:
            points (np.ndarray): Array of shape $(N, D)$ to add to the index.
        """
        pass

    @abc.abstractmethod
    def set_active(self, points: np.ndarray) -> None:
        """
        Sets the current batch of active points for optimization.

        These points are transient and updated iteratively. Setting a new active batch
        replaces the previous one.

        Args:
            points (np.ndarray): Array of shape $(M, D)$ representing the moving particles.
        """
        pass

    @abc.abstractmethod
    def consolidate(self) -> None:
        """
        Merges the current active points into the static set.

        This is typically called when the optimization of the current batch converges.
        The active points become permanent obstacles for future batches.

        $ S_{new} = S_{old} \\cup A_{current} $
        """
        pass

    @abc.abstractmethod
    def query_nn(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the $k$-nearest neighbors for the ACTIVE batch against the full set (Static + Active).

        For each point $x_i$ in the active set, this computes:
        $ N(x_i) = \\{ y \\in (S \\cup A) \\mid \\text{rank}(||x_i - y||) \\le k \\} $

        The results serve as the basis for k-NN repulsive forces.

        Args:
            k (int): The number of neighbors to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - **Indices**: Shape $(M, k)$, global indices of neighbors.
                - **Distances**: Shape $(M, k)$, Euclidean distances to neighbors.
        """
        pass

    @abc.abstractmethod
    def query_static(
        self, query_points: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the $k$-nearest neighbors for arbitrary query points against the STATIC set only.

        Used mainly for initialization heuristics where we only care about distance to
        established obstacles, ignoring other nascent points.

        Args:
            query_points (np.ndarray): The query coordinates $(Q, D)$.
            k (int): Number of neighbors.

        Returns:
            tuple[np.ndarray, np.ndarray]: Indices and distances.
        """
        pass

    @abc.abstractmethod
    def range_search(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds all neighbors within a fixed `radius` for the ACTIVE batch.

        This returns a dense matrix representation suitable for vectorized force calculation.
        For active point $x_i$ and potential neighbor $y_j$:

        $ M_{ij} = \\begin{cases} ||x_i - y_j|| & \\text{if } ||x_i - y_j|| < R \\\\ 0 & \\text{otherwise} \\end{cases} $

        Args:
            radius (float): The cutoff distance $R$.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - **Distances**: Dense matrix $(M, N_{total})$. Zero where no neighbor exists.
                - **Mask**: Boolean matrix $(M, N_{total})$. True where $d < R$.
        """
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Resets the internal state, clearing both static and active sets.
        """
        pass

    @property
    @abc.abstractmethod
    def total_count(self) -> int:
        """
        Returns the total number of points currently managed (Static + Active).
        """
        pass


class NumpyNN(NearestNeighbors):
    """
    Pure NumPy implementation using vectorized broadcasting and caching.

    This class is optimized for small to medium datasets ($N < 5000$) where the overhead
    of building tree structures or HNSW graphs outweighs the $O(N^2)$ cost of brute-force
    matrix operations.

    **Optimization Mechanism:**
    It utilizes the squared Euclidean distance expansion to leverage BLAS Level 3 optimizations
    (matrix multiplication):
    $ ||A - B||^2 = ||A||^2 + ||B||^2 - 2A \\cdot B^T $

    The term $||B||^2$ (static points squared norms) is cached and incrementally updated
    to avoid recomputation during iterative optimization steps.
    """

    def __init__(self, dimension: int):
        super().__init__(dimension)
        self._static: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._static_sq: np.ndarray = np.empty((0, 1), dtype=np.float32)
        self._active: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def add_static(self, points: np.ndarray) -> None:
        """
        Adds points to the static set and updates the squared-norm cache.

        Args:
            points (np.ndarray): Points to add $(N, D)$.
        """
        if points.shape[1] != self.dimension:
            raise ValueError(f"Dim mismatch: {points.shape[1]} vs {self.dimension}")

        points = points.astype(np.float32)
        self._static = np.vstack((self._static, points))

        # Update cache: ||B||^2
        points_sq = np.sum(points**2, axis=1, keepdims=True)
        self._static_sq = np.vstack((self._static_sq, points_sq))

    def set_active(self, points: np.ndarray) -> None:
        """
        Sets the active batch buffer.

        Args:
            points (np.ndarray): Active points $(M, D)$.
        """
        if points.shape[1] != self.dimension:
            raise ValueError(f"Dim mismatch: {points.shape[1]} vs {self.dimension}")
        self._active = points.astype(np.float32)

    def consolidate(self) -> None:
        """
        Transfers active points to the static array and updates the norm cache.
        """
        if self._active.shape[0] > 0:
            self.add_static(self._active)
            self._active = np.empty((0, self.dimension), dtype=np.float32)

    def clear(self) -> None:
        """
        Reallocates empty buffers for static and active sets.
        """
        self._static = np.empty((0, self.dimension), dtype=np.float32)
        self._static_sq = np.empty((0, 1), dtype=np.float32)
        self._active = np.empty((0, self.dimension), dtype=np.float32)

    @property
    def total_count(self) -> int:
        """
        The sum of rows in the static and active buffers.
        """
        return self._static.shape[0] + self._active.shape[0]

    def _sq_dist_matrix_cached(
        self, A: np.ndarray, B: np.ndarray, B_sq: np.ndarray
    ) -> np.ndarray:
        """Computes ||A - B||^2 using cached ||B||^2."""
        A_sq = np.sum(A**2, axis=1, keepdims=True)
        # A_sq + B_sq.T - 2AB
        return A_sq + B_sq.T - 2 * np.dot(A, B.T)

    def _sq_dist_matrix_uncached(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Standard ||A - B||^2 computation."""
        A_sq = np.sum(A**2, axis=1, keepdims=True)
        B_sq = np.sum(B**2, axis=1, keepdims=True)
        return A_sq + B_sq.T - 2 * np.dot(A, B.T)

    def _compute_full_sq_dists(self) -> np.ndarray:
        """Computes merged squared distances from Active to (Static + Active)."""
        n_active = self._active.shape[0]
        n_static = self._static.shape[0]

        # 1. Active vs Static (Using Cache)
        if n_static > 0:
            dists_s_sq = self._sq_dist_matrix_cached(
                self._active, self._static, self._static_sq
            )
        else:
            dists_s_sq = np.empty((n_active, 0), dtype=np.float32)

        # 2. Active vs Active (Uncached)
        dists_a_sq = self._sq_dist_matrix_uncached(self._active, self._active)

        # 3. Merge
        full_dists_sq = np.hstack((dists_s_sq, dists_a_sq))
        return np.maximum(full_dists_sq, 0.0)

    def query_nn(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs exact k-NN search using dense matrix operations.

        **Algorithm:**
        1. Compute full distance matrix $D^2$ using cached norms ($M \\times N$).
        2. Use `np.argpartition` (Introselect) to find the top-$k$ elements in $O(N)$ time.
        3. Perform a small sort on the $k$ elements to return them in order.

        Args:
            k (int): Number of neighbors.

        Returns:
            tuple[np.ndarray, np.ndarray]: Indices and distances.
        """
        full_dists_sq = self._compute_full_sq_dists()
        n_samples = full_dists_sq.shape[0]
        k = min(k, full_dists_sq.shape[1])

        # 1. Argpartition to find top-k (unsorted)
        # Time: O(N * Total_Count)
        unsorted_top_k_idx = np.argpartition(full_dists_sq, k - 1, axis=1)[:, :k]

        # 2. Extract values (Optimized indexing)
        # Create row indices: [[0,0,..], [1,1,..], ...]
        row_indices = np.arange(n_samples)[:, None]

        # Fancy indexing is often faster than take_along_axis for simple 2D arrays
        top_dists_sq = full_dists_sq[row_indices, unsorted_top_k_idx]

        # 3. Sort the top-k (Small sort: k * log k)
        # We only sort the small k window, not the full array
        local_sort_order = np.argsort(top_dists_sq, axis=1)

        # 4. Final Gather
        final_indices = unsorted_top_k_idx[row_indices, local_sort_order]
        final_dists_sq = top_dists_sq[row_indices, local_sort_order]

        return final_indices, np.sqrt(final_dists_sq)

    def query_static(
        self, query_points: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Queries the static set using cached squared norms.

        Args:
            query_points (np.ndarray): Query inputs.
            k (int): Neighbors count.

        Returns:
            tuple[np.ndarray, np.ndarray]: Indices and distances.
        """
        if self._static.shape[0] == 0:
            return (
                np.zeros((len(query_points), k), dtype=int),
                np.full((len(query_points), k), np.inf),
            )

        k = min(k, self._static.shape[0])
        # Use cached static norms
        dist_sq = self._sq_dist_matrix_cached(
            query_points, self._static, self._static_sq
        )
        dist_sq = np.maximum(dist_sq, 0.0)

        if k == 1:
            final_idx = np.argmin(dist_sq, axis=1)[:, None]
            final_dists = np.sqrt(np.take_along_axis(dist_sq, final_idx, axis=1))
            return final_idx, final_dists

        part_idx = np.argpartition(dist_sq, k - 1, axis=1)[:, :k]
        top_dists_sq = np.take_along_axis(dist_sq, part_idx, axis=1)

        sort_idx = np.argsort(top_dists_sq, axis=1)
        final_dists = np.sqrt(np.take_along_axis(top_dists_sq, sort_idx, axis=1))
        final_idx = np.take_along_axis(part_idx, sort_idx, axis=1)

        return final_idx, final_dists

    def range_search(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the full distance matrix and applies a boolean mask.

        Because `NumpyNN` is designed for dense regimes, this returns the complete
        $M \\times N$ distance matrix and a boolean mask indicating valid neighbors ($d < R$).
        This avoids the overhead of sparse formatting when connectivity is high.

        Args:
            radius (float): Cutoff radius.

        Returns:
            tuple[np.ndarray, np.ndarray]: (Distances, Mask).
        """
        # 1. Compute Full Matrix
        full_dists_sq = self._compute_full_sq_dists()

        # 2. Compute Mask
        radius_sq = radius * radius
        mask = full_dists_sq < radius_sq

        # 3. Return (Dists, Mask) - No 'nonzero' or sorting required
        # Return sqrt distances for metric compatibility
        return np.sqrt(full_dists_sq), mask


class FaissBaseNN(NearestNeighbors):
    """
    Base wrapper for Faiss index implementations.

    This class handles the hybrid logic of querying a specialized Faiss index (for the
    large Static set) and combining it with brute-force calculations for the (small)
    Active set.

    **Hybrid Query Logic:**
    Since Faiss indices are often immutable or expensive to rebuild frequently, the
    Active set is kept separate.
    $ N(x) = \\text{SelectTopK}( \\text{FaissQuery}(x, S) \\cup \\text{BruteForce}(x, A) ) $
    """

    def __init__(self, index_static: faiss.Index, dimension: int):
        super().__init__(dimension)
        self._active: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._static_count = 0
        self._index_static: typing.Any = index_static

    def add_static(self, points: np.ndarray) -> None:
        """
        Adds vectors to the underlying Faiss index.

        Args:
            points (np.ndarray): Data to index.
        """
        data = np.ascontiguousarray(points.astype(np.float32))
        if data.shape[1] != self.dimension:
            raise ValueError(f"Dim mismatch: {data.shape[1]} vs {self.dimension}")
        self._index_static.add(data)
        self._static_count += data.shape[0]

    def set_active(self, points: np.ndarray) -> None:
        """
        Updates the transient active batch.
        """
        if points.shape[1] != self.dimension:
            raise ValueError("Dim mismatch")
        self._active = np.ascontiguousarray(points.astype(np.float32))

    def consolidate(self) -> None:
        """
        Adds the current active batch to the Faiss index and clears the active buffer.
        """
        if self._active.shape[0] > 0:
            self.add_static(self._active)
            self._active = np.empty((0, self.dimension), dtype=np.float32)

    def clear(self) -> None:
        """
        Resets the Faiss index and local counters.
        """
        self._index_static.reset()
        self._static_count = 0
        self._active = np.empty((0, self.dimension), dtype=np.float32)

    @property
    def total_count(self) -> int:
        """
        Total points in Faiss index + active buffer size.
        """
        return self._static_count + self._active.shape[0]

    def _merge_active_active_knn(
        self, dists_s, idxs_s, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        n_active = self._active.shape[0]

        # Active-Active (Brute Force)
        A_sq = np.sum(self._active**2, axis=1, keepdims=True)
        dists_a_sq = A_sq + A_sq.T - 2 * np.dot(self._active, self._active.T)
        dists_a_sq = np.maximum(dists_a_sq, 0.0)

        indices_a = (
            np.broadcast_to(np.arange(n_active), (n_active, n_active))
            + self._static_count
        )

        # Merge
        full_dists = np.hstack((dists_s, dists_a_sq))
        full_dists = np.maximum(full_dists, 0.0)
        full_idxs = np.hstack((idxs_s, indices_a))

        # Sort everything and take top k
        k_final = min(k, full_dists.shape[1])
        sort_order = np.argsort(full_dists, axis=1)[:, :k_final]

        # Apply shuffle
        final_dists_sq = np.take_along_axis(full_dists, sort_order, axis=1)
        final_idxs = np.take_along_axis(full_idxs, sort_order, axis=1)

        return final_idxs, np.sqrt(final_dists_sq)

    def query_nn(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a hybrid k-NN search.

        1. **Static Search:** Queries the Faiss index for $k$ neighbors.
        2. **Active Search:** Computes exact distances against the active batch (brute-force).
        3. **Merge:** Concatenates results and selects the global top-$k$.

        Args:
            k (int): Neighbors count.

        Returns:
            tuple[np.ndarray, np.ndarray]: Global indices and distances.
        """
        n_active = self._active.shape[0]

        # 1. Query Static
        k_s = min(k, self._static_count)
        if k_s > 0:
            dists_s, idxs_s = self._index_static.search(self._active, k_s)
            dists_s = np.maximum(dists_s, 0.0)
        else:
            dists_s = np.empty((n_active, 0), dtype=np.float32)
            idxs_s = np.empty((n_active, 0), dtype=int)

        return self._merge_active_active_knn(dists_s, idxs_s, k)

    def query_static(
        self, query_points: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Directly queries the underlying Faiss index.

        Args:
            query_points (np.ndarray): Query inputs.
            k (int): Neighbors count.

        Returns:
            tuple[np.ndarray, np.ndarray]: Indices and distances.
        """
        if self._static_count == 0:
            return (
                np.zeros((len(query_points), k), dtype=int),
                np.full((len(query_points), k), np.inf),
            )

        q_data = np.ascontiguousarray(query_points.astype(np.float32))
        dists, idxs = self._index_static.search(q_data, k)
        return idxs, np.sqrt(np.maximum(dists, 0.0))

    def _merge_active_active_range(
        self, lims_s, D_s, I_s, radius: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_active = self._active.shape[0]
        radius_sq = radius * radius

        # Active-Active (Brute Force)
        A_sq = np.sum(self._active**2, axis=1, keepdims=True)
        dists_a_sq = A_sq + A_sq.T - 2 * np.dot(self._active, self._active.T)
        dists_a_sq = np.maximum(dists_a_sq, 0.0)

        mask_a = dists_a_sq < radius_sq

        final_D_list = []
        final_I_list = []

        indices_a_base = np.arange(n_active) + self._static_count

        for i in range(n_active):
            s_start, s_end = lims_s[i], lims_s[i + 1]
            d_s = D_s[s_start:s_end]
            i_s = I_s[s_start:s_end]

            a_mask = mask_a[i]
            d_a = dists_a_sq[i][a_mask]
            i_a = indices_a_base[a_mask]

            d_combined = np.concatenate((d_s, d_a))
            i_combined = np.concatenate((i_s, i_a))

            final_D_list.append(np.sqrt(np.maximum(d_combined, 0.0)))
            final_I_list.append(i_combined)

        lens = [len(x) for x in final_D_list]
        lims = np.zeros(n_active + 1, dtype=int)
        lims[1:] = np.cumsum(lens)

        if sum(lens) > 0:
            D = np.concatenate(final_D_list)
            Idx = np.concatenate(final_I_list)
        else:
            D = np.array([], dtype=np.float32)
            Idx = np.array([], dtype=int)

        return lims, D, Idx

    def range_search(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a hybrid range search returning a dense matrix.

        This method is critical for performance in the physics simulation. It merges:
        1. **Sparse Static Results:** `index.range_search()` returns a packed list of neighbors.
            These are scattered into a dense matrix.
        2. **Dense Active Results:** Brute-force calculation for the active batch is filled
            into the remaining columns.

        The result is a unified dense matrix $(M, N_{total})$ compatible with the vector
        force equations.

        Args:
            radius (float): Cutoff radius.

        Returns:
            tuple[np.ndarray, np.ndarray]: (Distances Matrix, Validity Mask).
        """
        n_active = self._active.shape[0]
        n_static = self._static_count
        n_total = n_static + n_active

        # 1. Allocate Unified Buffers (Batch x Total)
        # Columns [0 : n_static] -> Static Neighbors
        # Columns [n_static : n_total] -> Active Neighbors
        dists_matrix = np.zeros((n_active, n_total), dtype=np.float32)
        mask_matrix = np.zeros((n_active, n_total), dtype=bool)

        # --- PART A: Static Neighbors (Faiss) ---
        if n_static > 0:
            lims, D, Idx = self._index_static.range_search(
                self._active, radius * radius
            )
            if len(D) > 0:
                counts = np.diff(lims.astype(np.int64))
                row_indices = np.repeat(np.arange(n_active), counts)
                col_indices = Idx
                dists_matrix[row_indices, col_indices] = np.sqrt(D)
                mask_matrix[row_indices, col_indices] = True

        # --- PART B: Active-Active Neighbors (Numpy) ---
        # Compute distances for the active batch against itself
        # Shape: (Batch, Batch)
        A_sq = np.sum(self._active**2, axis=1, keepdims=True)
        dists_sq_active = A_sq + A_sq.T - 2 * np.dot(self._active, self._active.T)

        # Avoid self-interaction and respect radius
        # Note: We rely on the caller (physics engine) to handle epsilon/division-by-zero,
        # but we must ensure self-interaction (dist=0) is masked out.
        radius_sq = radius * radius
        mask_active = (dists_sq_active < radius_sq) & (dists_sq_active > 1e-9)

        # Place into the right-side of the matrix
        start_col = n_static
        end_col = n_total
        dists_matrix[:, start_col:end_col] = np.sqrt(np.maximum(dists_sq_active, 0.0))
        mask_matrix[:, start_col:end_col] = mask_active

        return dists_matrix, mask_matrix


class FaissFlatL2NN(FaissBaseNN):
    """
    Wrapper for `faiss.IndexFlatL2`.

    This provides exact nearest neighbor search using brute-force comparison within Faiss.
    It is generally faster than `NumpyNN` for medium datasets ($N > 5000$) due to C++
    optimizations and multithreading, but slower than HNSW.
    """

    def __init__(self, dimension: int):
        super().__init__(faiss.IndexFlatL2(dimension), dimension)


class FaissHNSWFlatNN(FaissBaseNN):
    """
    Wrapper for `faiss.IndexHNSWFlat`.

    Uses Hierarchical Navigable Small World graphs for approximate nearest neighbor search.
    This offers logarithmic scaling $O(\\log N)$, making it suitable for very large datasets
    ($N > 50,000$).

    Args:
        dimension (int): Dimensionality of the data.
        M (int): Number of edges per node in the graph (default 32). Higher $M$ improves
            recall at the cost of memory and build time.
    """

    def __init__(self, dimension: int, M: int = 32):
        super().__init__(faiss.IndexHNSWFlat(dimension, M), dimension)
