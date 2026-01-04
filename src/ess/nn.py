import abc
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class NearestNeighbors(abc.ABC):
    """
    Abstract Base Class for Nearest Neighbors implementations tailored for ESA.

    This class distinguishes between 'static' points (anchors/obstacles) and 
    'active' points (currently moving particles). This split allows for 
    optimizations where the large static set is indexed once, while the small 
    active set is updated frequently.
    """

    def __init__(self, dimension: int, seed: int = 42):
        """
        Initializes the NearestNeighbors index.

        Args:
            dimension (int): The dimensionality of the points (D).
            seed (int): Random seed for reproducibility (where applicable).
        """
        self.dimension = dimension
        self.seed = seed

    @abc.abstractmethod
    def add_static(self, points: np.ndarray) -> None:
        """
        Adds points to the static set (anchors).

        These points are considered fixed obstacles that do not move during
        the current optimization batch.

        Args:
            points (np.ndarray): A (N, D) array of coordinates.
        """
        pass

    @abc.abstractmethod
    def set_active(self, points: np.ndarray) -> None:
        """
        Sets the current batch of active points.

        Replaces any previously active points. These are the particles 
        currently being optimized by the algorithm.

        Args:
            points (np.ndarray): A (M, D) array of coordinates.
        """
        pass

    @abc.abstractmethod
    def consolidate(self) -> None:
        """
        Merges the current active points into the static set.

        This is typically called at the end of an optimization batch, freezing
        the recently optimized points so they become obstacles for the next batch.
        """
        pass

    @abc.abstractmethod
    def query_active(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds k nearest neighbors for each ACTIVE point.

        Neighbors are drawn from the union of the Static set and the Active set.
        The results are used to calculate repulsive forces acting on the active points.

        Args:
            k (int): The number of neighbors to find.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - **indices** (np.ndarray): Shape (M, k). Indices of neighbors. 
                  Indices 0..N_static-1 refer to static points. 
                  Indices N_static..Total-1 refer to active points.
                - **distances** (np.ndarray): Shape (M, k). Euclidean distances.
        """
        pass
    
    @abc.abstractmethod
    def query_external(self, query_points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds k nearest neighbors for arbitrary external points against the STATIC set only.

        This is primarily used during the initialization phase (Smart Init) to find 
        starting positions far from existing obstacles.

        Args:
            query_points (np.ndarray): Shape (Q, D). Points to query.
            k (int): Number of neighbors.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - **indices** (np.ndarray): Shape (Q, k). Indices into the static set.
                - **distances** (np.ndarray): Shape (Q, k). Euclidean distances.
        """
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Resets the internal state, removing all static and active points.
        """
        pass
    
    @property
    @abc.abstractmethod
    def total_count(self) -> int:
        """
        Returns the total number of points currently tracked (static + active).

        Returns:
            int: Total count.
        """
        pass


class NumpyNN(NearestNeighbors):
    """
    Pure NumPy implementation using vectorized broadcasting.

    Best suited for small to medium datasets (< 5000 points) where the overhead
    of building a complex index (like HNSW) outweighs the brute-force cost.
    """

    def __init__(self, dimension: int, seed: int = 42):
        """
        Initializes the NumpyNN.

        Args:
            dimension (int): Dimensionality of data.
            seed (int): Random seed.
        """
        super().__init__(dimension, seed)
        self._static: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._active: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def add_static(self, points: np.ndarray) -> None:
        """Adds points to static set."""
        if points.shape[1] != self.dimension:
            raise ValueError(f"Dim mismatch: {points.shape[1]} vs {self.dimension}")
        self._static = np.vstack((self._static, points.astype(np.float32)))

    def set_active(self, points: np.ndarray) -> None:
        """Sets active points."""
        if points.shape[1] != self.dimension:
            raise ValueError(f"Dim mismatch: {points.shape[1]} vs {self.dimension}")
        self._active = points.astype(np.float32)

    def consolidate(self) -> None:
        """Moves active points to static."""
        if self._active.shape[0] > 0:
            self.add_static(self._active)
            self._active = np.empty((0, self.dimension), dtype=np.float32)

    def clear(self) -> None:
        """Clears all data."""
        self._static = np.empty((0, self.dimension), dtype=np.float32)
        self._active = np.empty((0, self.dimension), dtype=np.float32)

    @property
    def total_count(self) -> int:
        """Total point count."""
        return self._static.shape[0] + self._active.shape[0]

    def query_external(self, query_points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Queries against static points only."""
        if self._static.shape[0] == 0:
            return (np.zeros((len(query_points), k), dtype=int), 
                    np.full((len(query_points), k), np.inf))
        
        return self._compute_knn(query_points, self._static, k)

    def query_active(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Queries active points against (Static U Active)."""
        n_active = self._active.shape[0]
        n_static = self._static.shape[0]
        
        # 1. Active vs Static
        if n_static > 0:
            dists_s_sq = self._sq_dist_matrix(self._active, self._static)
            indices_s = np.broadcast_to(np.arange(n_static), (n_active, n_static))
        else:
            dists_s_sq = np.empty((n_active, 0), dtype=np.float32)
            indices_s = np.empty((n_active, 0), dtype=int)

        # 2. Active vs Active (M, M)
        dists_a_sq = self._sq_dist_matrix(self._active, self._active)
        dists_a_sq = np.maximum(dists_a_sq, 0.0)
        # Offset indices by static count
        indices_a = np.broadcast_to(np.arange(n_active), (n_active, n_active)) + n_static

        # 3. Concatenate
        full_dists_sq = np.hstack((dists_s_sq, dists_a_sq))
        full_indices = np.hstack((indices_s, indices_a))

        # 4. Sort and Select Top K
        k = min(k, full_dists_sq.shape[1])
        
        # argpartition for partial sort (faster)
        part_idx = np.argpartition(full_dists_sq, k-1, axis=1)[:, :k]
        
        # Gather the top k
        final_dists_sq = np.take_along_axis(full_dists_sq, part_idx, axis=1)
        final_indices = np.take_along_axis(full_indices, part_idx, axis=1)
        
        # Final Sort within K
        sort_order = np.argsort(final_dists_sq, axis=1)
        final_dists_sq = np.take_along_axis(final_dists_sq, sort_order, axis=1)
        final_indices = np.take_along_axis(final_indices, sort_order, axis=1)

        return final_indices, np.sqrt(final_dists_sq)

    def _sq_dist_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Computes the squared Euclidean distance matrix between A and B.
        Formula: ||A - B||^2 = ||A||^2 + ||B||^2 - 2AB^T
        """
        A_sq = np.sum(A**2, axis=1, keepdims=True)
        B_sq = np.sum(B**2, axis=1, keepdims=True)
        return A_sq + B_sq.T - 2 * np.dot(A, B.T)
        
    def _compute_knn(self, query: np.ndarray, target: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Internal helper for exact k-NN."""
        k = min(k, target.shape[0])
        dist_sq = self._sq_dist_matrix(query, target)
        dist_sq = np.maximum(dist_sq, 0.0)
        
        idx = np.argpartition(dist_sq, k-1, axis=1)[:, :k]
        top_dists = np.take_along_axis(dist_sq, idx, axis=1)
        
        sort_idx = np.argsort(top_dists, axis=1)
        final_dists = np.sqrt(np.take_along_axis(top_dists, sort_idx, axis=1))
        final_idx = np.take_along_axis(idx, sort_idx, axis=1)
        
        return final_idx, final_dists


class FaissNN(NearestNeighbors):
    """
    Faiss-based implementation for high-performance similarity search (CPU Only).

    This implementation uses a persistent `IndexFlatL2` for the static points.
    Active points are queried against this index and against themselves using 
    brute-force (since the active batch is small), and the results are merged.
    """
    
    def __init__(self, dimension: int, seed: int = 42):
        """
        Initializes the FaissNN.

        Args:
            dimension (int): Dimensionality.
            seed (int): Random seed (unused by IndexFlatL2 but kept for interface).
        """
        super().__init__(dimension, seed)
        
        # Strict CPU only
        self._index_static = faiss.IndexFlatL2(dimension)
        
        self._active: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._static_count = 0

    def add_static(self, points: np.ndarray) -> None:
        """Adds points to the Faiss static index."""
        data = np.ascontiguousarray(points.astype(np.float32))
        if data.shape[1] != self.dimension:
            raise ValueError(f"Dim mismatch: {data.shape[1]} vs {self.dimension}")
        self._index_static.add(data)
        self._static_count += data.shape[0]

    def set_active(self, points: np.ndarray) -> None:
        """Sets the active batch."""
        if points.shape[1] != self.dimension:
            raise ValueError("Dim mismatch")
        self._active = np.ascontiguousarray(points.astype(np.float32))

    def consolidate(self) -> None:
        """Adds active points to the static index and clears active set."""
        if self._active.shape[0] > 0:
            self.add_static(self._active)
            self._active = np.empty((0, self.dimension), dtype=np.float32)

    def clear(self) -> None:
        """Resets the Faiss index."""
        self._index_static.reset()
        self._static_count = 0
        self._active = np.empty((0, self.dimension), dtype=np.float32)

    @property
    def total_count(self) -> int:
        """Total point count."""
        return self._static_count + self._active.shape[0]

    def query_external(self, query_points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Queries external points against the static index."""
        if self._static_count == 0:
            return (np.zeros((len(query_points), k), dtype=int), 
                    np.full((len(query_points), k), np.inf))
        
        # Faiss search expects (x, k)
        q_data = np.ascontiguousarray(query_points.astype(np.float32))
        dists, idxs = self._index_static.search(q_data, k)
        
        return idxs, np.sqrt(dists)

    def query_active(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Queries active points against Static index and Self.
        
        Strategy:
        1. Search Static Index (Faiss).
        2. Compute Active-Active distances (Numpy/Brute-force).
        3. Merge, Sort, and Select Top-K.
        """
        n_active = self._active.shape[0]
        
        # 1. Query Static
        k_s = min(k, self._static_count)
        if k_s > 0:
            dists_s, idxs_s = self._index_static.search(self._active, k_s)
        else:
            dists_s = np.empty((n_active, 0), dtype=np.float32)
            idxs_s = np.empty((n_active, 0), dtype=int)

        # 2. Query Active (Self)
        A_sq = np.sum(self._active**2, axis=1, keepdims=True)
        dists_a_sq = A_sq + A_sq.T - 2 * np.dot(self._active, self._active.T)
        dists_a_sq = np.maximum(dists_a_sq, 0.0)
        
        # Indices for active points need to be offset by static_count
        indices_a = np.broadcast_to(np.arange(n_active), (n_active, n_active)) + self._static_count

        # 3. Merge
        # Note: dists_s from Faiss is squared L2 distance (IndexFlatL2 default)
        full_dists = np.hstack((dists_s, dists_a_sq)) 
        full_idxs = np.hstack((idxs_s, indices_a))
        
        # 4. Sort Top K
        k_final = min(k, full_dists.shape[1])
        
        part_idx = np.argpartition(full_dists, k_final-1, axis=1)[:, :k_final]
        
        final_dists_sq = np.take_along_axis(full_dists, part_idx, axis=1)
        final_idxs = np.take_along_axis(full_idxs, part_idx, axis=1)
        
        sort_order = np.argsort(final_dists_sq, axis=1)
        final_dists_sq = np.take_along_axis(final_dists_sq, sort_order, axis=1)
        final_idxs = np.take_along_axis(final_idxs, sort_order, axis=1)
        
        return final_idxs, np.sqrt(final_dists_sq)