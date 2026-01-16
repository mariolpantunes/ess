# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"
__license__ = "MIT"


import typing
import unittest

import numpy as np

import ess


class SharedNNTests(unittest.TestCase):
    """
    Shared test logic for NN implementations.
    Verifies API compliance, shape correctness, and basic geometric logic.
    """

    nn_class: typing.Any = None

    def setUp(self):
        if self.nn_class is None:
            self.skipTest("SharedNNTests should not be run directly.")

        self.dim = 3
        self.model = self.nn_class(dimension=self.dim, seed=42)

        # Standard orthogonal setup
        self.static_points = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32
        )

        self.active_points = np.array(
            [[0.1, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float32
        )

    def test_lifecycle_and_consolidate(self):
        """Test adding static, setting active, and merging them."""
        self.assertEqual(self.model.total_count, 0)

        self.model.add_static(self.static_points)
        self.assertEqual(self.model.total_count, 3)

        self.model.set_active(self.active_points)
        self.assertEqual(self.model.total_count, 5)  # 3 Static + 2 Active

        # Test Consolidate: Active should move to Static
        self.model.consolidate()
        self.assertEqual(self.model.total_count, 5)

        # After consolidate, active buffer should be empty
        # We verify this by running a query; it should process 0 items if we didn't set new active
        # But set_active updates internal state. Let's clear and re-add to verify count persistence.
        self.model.set_active(np.array([[99, 99, 99]], dtype=np.float32))
        self.assertEqual(self.model.total_count, 6)

        self.model.clear()
        self.assertEqual(self.model.total_count, 0)

    def test_query_static_basic(self):
        """Test searching active/query points against STATIC set only."""
        self.model.add_static(self.static_points)

        # Query exactly at [0,0,0] -> Should match static[0]
        query = np.zeros((1, self.dim), dtype=np.float32)
        indices, dists = self.model.query_static(query, k=1)

        self.assertEqual(indices[0][0], 0)
        self.assertAlmostEqual(dists[0][0], 0.0, places=5)

    def test_query_nn_mixed(self):
        """Test searching active points against (Static + Active)."""
        self.model.add_static(self.static_points)
        # Add a point very close to the second active point
        complex_active = np.vstack([self.active_points, [5.1, 5.0, 5.0]])
        self.model.set_active(complex_active)

        # Indices in mixed space:
        # Static: 0, 1, 2
        # Active: 3 (0.1,0,0), 4 (5,5,5), 5 (5.1,5,5)

        indices, dists = self.model.query_nn(k=3)

        # 1. Point 3 (0.1,0,0) -> Closest should be self (dist 0) and Static[0] (dist 0.1)
        self.assertAlmostEqual(dists[0][0], 0.0, places=5)  # Self
        # Check neighbor is 0
        self.assertIn(0, indices[0])

        # 2. Point 5 (5.1,5,5) -> Closest is Self(5), then Active(4)
        neighbors_of_5 = indices[2]
        self.assertIn(4, neighbors_of_5)  # Should find the other active point

    def test_range_search_dense_matrix(self):
        """
        Test that range_search returns a full dense matrix (Active x Total).
        Correctness check for the ESS physics engine format.
        """
        self.model.add_static(self.static_points)  # 3 points
        self.model.set_active(self.active_points)  # 2 points

        # Total points = 5
        # Active 0: [0.1, 0, 0] -> Dist to Static[0] is 0.1
        # Radius 0.2 should capture Static[0] but NOT Static[1] (dist 10)

        dists, mask = self.model.range_search(radius=0.2)

        # Shape check: (N_Active, N_Total)
        self.assertEqual(dists.shape, (2, 5))
        self.assertEqual(mask.shape, (2, 5))

        # Check Row 0 (Active 0)
        # Col 0 (Static 0): Should be True, Dist ~0.1
        self.assertTrue(mask[0, 0])
        self.assertAlmostEqual(dists[0, 0], 0.1, places=4)

        # Col 1 (Static 1): Should be False (too far)
        self.assertFalse(mask[0, 1])
        # Value doesn't strictly matter if mask is False, but usually 0 or distance

        # Active-Active Interaction
        # Row 0 vs Row 1: Dist is large (~8.6), should be False
        self.assertFalse(mask[0, 4])

    def test_high_dimensions(self):
        """Verify functionality in high dimensions (e.g., 64D)."""
        dim = 64
        model = self.nn_class(dimension=dim, seed=123)

        # Random data
        rng = np.random.default_rng(123)
        static = rng.random((100, dim)).astype(np.float32)
        active = rng.random((10, dim)).astype(np.float32)

        model.add_static(static)
        model.set_active(active)

        # Test KNN
        idx, dist = model.query_nn(k=5)
        self.assertEqual(idx.shape, (10, 5))

        # Test Range
        d_mat, mask = model.range_search(radius=10.0)  # Large radius
        self.assertEqual(d_mat.shape, (10, 110))

    def test_empty_handling(self):
        """Test behavior with 0 static points."""
        self.model.clear()
        self.model.set_active(self.active_points)

        # Query NN with no static points -> Should return self + other active
        idx, dist = self.model.query_nn(k=2)
        self.assertEqual(idx.shape, (2, 2))

        # Query Static with no static points -> Should handle gracefully
        idx_s, dist_s = self.model.query_static(self.active_points, k=1)
        # Should return indices 0 and inf distance (or similar sentinel)
        self.assertTrue(np.all(np.isinf(dist_s)))

    def test_range_search_no_neighbors(self):
        """Test range search with very small radius."""
        self.model.add_static(self.static_points)
        self.model.set_active(self.active_points)

        dists, mask = self.model.range_search(radius=1e-6)

        # Only self-loops might exist if not masked, but physics engine handles self-loops via mask.
        # Ensure standard neighbors are false.
        self.assertFalse(mask[0, 1])  # Static 1 is far

        # Depending on implementation, diagonal (self) might be masked or not.
        # NumpyNN masks strictly < radius. Self dist is 0. 0 < 1e-6 is True.
        # But ESS physics loop usually ignores self via checks.
        # Let's just ensure no *other* points are found.
        mask_no_self = mask.copy()
        # Row 0 is index 3 in total list.
        mask_no_self[0, 3] = False
        self.assertFalse(np.any(mask_no_self[0]))


class TestNumpyNN(SharedNNTests):
    nn_class = ess.NumpyNN


class TestFaissFlatL2NN(SharedNNTests):
    nn_class = ess.FaissFlatL2NN


class TestFaissHNSWFlatNN(SharedNNTests):
    nn_class = ess.FaissHNSWFlatNN


if __name__ == "__main__":
    unittest.main()
