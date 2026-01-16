import abc
import logging
import typing

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class NearestNeighbors(abc.ABC):
    """
    Abstract Base Class for Nearest Neighbors implementations tailored for ESA.
    """

    def __init__(self, dimension: int, seed: int = 42):
        self.dimension = dimension
        self.seed = seed

    @abc.abstractmethod
    def add_static(self, points: np.ndarray) -> None:
        """Adds points to the static set (anchors/obstacles)."""
        pass

    @abc.abstractmethod
    def set_active(self, points: np.ndarray) -> None:
        """Sets the current batch of active points."""
        pass

    @abc.abstractmethod
    def consolidate(self) -> None:
        """Merges current active points into the static set."""
        pass

    @abc.abstractmethod
    def query_nn(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds k nearest neighbors for the CURRENT ACTIVE batch against (Static + Active).

        Returns:
            tuple: (indices, distances) where shape is (M, k).
            The point itself (distance 0) is guaranteed to be at index 0.
        """
        pass

    @abc.abstractmethod
    def query_static(
        self, query_points: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds k nearest neighbors for ARBITRARY points against the STATIC set only.
        """
        pass

    @abc.abstractmethod
    def range_search(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds all neighbors within `radius` for the CURRENT ACTIVE batch against (Static + Active).
        """
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Resets the internal state."""
        pass

    @property
    @abc.abstractmethod
    def total_count(self) -> int:
        pass


class NumpyNN(NearestNeighbors):
    """
    Pure NumPy implementation using vectorized broadcasting.
    Optimized with caching for norms and implicit indexing.
    """

    def __init__(self, dimension: int, seed: int = 42):
        super().__init__(dimension, seed)
        self._static: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        # Cache squared norms of static points to avoid re-computing in loop
        self._static_sq: np.ndarray = np.empty((0, 1), dtype=np.float32)
        self._active: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def add_static(self, points: np.ndarray) -> None:
        if points.shape[1] != self.dimension:
            raise ValueError(f"Dim mismatch: {points.shape[1]} vs {self.dimension}")
        
        points = points.astype(np.float32)
        self._static = np.vstack((self._static, points))
        
        # Update cache: ||B||^2
        points_sq = np.sum(points**2, axis=1, keepdims=True)
        self._static_sq = np.vstack((self._static_sq, points_sq))

    def set_active(self, points: np.ndarray) -> None:
        if points.shape[1] != self.dimension:
            raise ValueError(f"Dim mismatch: {points.shape[1]} vs {self.dimension}")
        self._active = points.astype(np.float32)

    def consolidate(self) -> None:
        if self._active.shape[0] > 0:
            self.add_static(self._active)
            self._active = np.empty((0, self.dimension), dtype=np.float32)

    def clear(self) -> None:
        self._static = np.empty((0, self.dimension), dtype=np.float32)
        self._static_sq = np.empty((0, 1), dtype=np.float32)
        self._active = np.empty((0, self.dimension), dtype=np.float32)

    @property
    def total_count(self) -> int:
        return self._static.shape[0] + self._active.shape[0]

    def _sq_dist_matrix_cached(self, A: np.ndarray, B: np.ndarray, B_sq: np.ndarray) -> np.ndarray:
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
            dists_s_sq = self._sq_dist_matrix_cached(self._active, self._static, self._static_sq)
        else:
            dists_s_sq = np.empty((n_active, 0), dtype=np.float32)

        # 2. Active vs Active (Uncached)
        dists_a_sq = self._sq_dist_matrix_uncached(self._active, self._active)
        
        # 3. Merge
        full_dists_sq = np.hstack((dists_s_sq, dists_a_sq))
        return np.maximum(full_dists_sq, 0.0)

    def query_nn(self, k: int) -> tuple[np.ndarray, np.ndarray]:
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
        if self._static.shape[0] == 0:
            return (
                np.zeros((len(query_points), k), dtype=int),
                np.full((len(query_points), k), np.inf),
            )

        k = min(k, self._static.shape[0])
        # Use cached static norms
        dist_sq = self._sq_dist_matrix_cached(query_points, self._static, self._static_sq)
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
        Returns the full dense distance matrix and a validity mask.
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
    """Base class for Faiss implementations."""

    def __init__(
        self,
        index_static: faiss.Index,
        dimension: int,
        seed: int = 42,
    ):
        super().__init__(dimension, seed)
        self._active: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._static_count = 0
        self._index_static: typing.Any = index_static

    def add_static(self, points: np.ndarray) -> None:
        data = np.ascontiguousarray(points.astype(np.float32))
        if data.shape[1] != self.dimension:
            raise ValueError(f"Dim mismatch: {data.shape[1]} vs {self.dimension}")
        self._index_static.add(data)
        self._static_count += data.shape[0]

    def set_active(self, points: np.ndarray) -> None:
        if points.shape[1] != self.dimension:
            raise ValueError("Dim mismatch")
        self._active = np.ascontiguousarray(points.astype(np.float32))

    def consolidate(self) -> None:
        if self._active.shape[0] > 0:
            self.add_static(self._active)
            self._active = np.empty((0, self.dimension), dtype=np.float32)

    def clear(self) -> None:
        self._index_static.reset()
        self._static_count = 0
        self._active = np.empty((0, self.dimension), dtype=np.float32)

    @property
    def total_count(self) -> int:
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
        Optimized Path: Returns a DENSE (Batch, Total) matrix.
        
        This method internally merges:
        1. Static Neighbors (retrieved via Faiss Sparse -> Scattered to Dense)
        2. Active Neighbors (computed via Numpy Brute Force -> Filled in Dense)
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
            lims, D, Idx = self._index_static.range_search(self._active, radius * radius)
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
    """Faiss IndexFlatL2."""

    def __init__(self, dimension: int, seed: int = 42):
        super().__init__(faiss.IndexFlatL2(dimension), dimension, seed)


class FaissHNSWFlatNN(FaissBaseNN):
    """Faiss IndexHNSWFlat."""

    def __init__(self, dimension: int, seed: int = 42, M: int = 32):
        super().__init__(faiss.IndexHNSWFlat(dimension, M), dimension, seed)
