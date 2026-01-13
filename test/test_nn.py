# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.1"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"
__license__ = "MIT"


import typing
import unittest

import numpy as np

import ess.nn as nn


class SharedNNTests(unittest.TestCase):
    """
    Shared test logic for NN implementations.
    """

    nn_class: typing.Any = None

    def setUp(self):
        if self.nn_class is None:
            self.skipTest("SharedNNTests should not be run directly.")

        self.dim = 3
        self.model = self.nn_class(dimension=self.dim, seed=42)

        self.static_points = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32
        )

        self.active_points = np.array(
            [[0.1, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float32
        )

    def test_lifecycle(self):
        self.assertEqual(self.model.total_count, 0)
        self.model.add_static(self.static_points)
        self.assertEqual(self.model.total_count, 3)
        self.model.set_active(self.active_points)
        self.assertEqual(self.model.total_count, 5)
        self.model.consolidate()
        self.assertEqual(self.model.total_count, 5)
        self.model.clear()
        self.assertEqual(self.model.total_count, 0)

    def test_query_static(self):
        """Test searching against static points only."""
        self.model.add_static(self.static_points)
        query = np.zeros((1, self.dim), dtype=np.float32)
        indices, dists = self.model.query_static(query, k=1)
        self.assertEqual(indices[0][0], 0)
        self.assertAlmostEqual(dists[0][0], 0.0, places=5)

    def test_query_nn(self):
        """Test searching for active points against all."""
        self.model.add_static(self.static_points)
        complex_active = np.vstack([self.active_points, [5.1, 5.0, 5.0]])
        self.model.set_active(complex_active)

        # query_nn (previously query_active)
        indices, dists = self.model.query_nn(k=3)

        # 1. Self check (Point 0)
        self.assertAlmostEqual(dists[0][0], 0.0, places=6)

        # 2. Check logic: Point 2 (5.1) should find Point 1 (5.0)
        # Static Count = 3. Point 1 has index 3+1 = 4.
        # Neighbors of Point 2: Self(5), Active1(4), Static...
        neighbors_of_2 = indices[2]
        self.assertIn(4, neighbors_of_2)

    def test_range_search(self):
        """Test Range Search (Radius)."""
        self.model.add_static(self.static_points)
        # Active point [0.1, 0, 0]
        # Neighbors within R=1.0: Self (0.0) and Static[0] (0.1)
        self.model.set_active(self.active_points[0:1])

        lims, D, _ = self.model.range_search(radius=1.0)

        self.assertEqual(lims[0], 0)
        self.assertEqual(lims[1], 2)  # 2 neighbors found
        self.assertIn(0.0, D)  # Self
        self.assertTrue(np.any(np.isclose(D, 0.1, atol=1e-5)))

    def test_stability_coincident_points(self):
        self.model.add_static(np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
        active = np.array([[1.0, 1.0, 1.0], [1.0 + 1e-9, 1.0, 1.0]], dtype=np.float32)

        self.model.set_active(active)
        indices, dists = self.model.query_nn(k=2)

        self.assertFalse(np.any(np.isnan(dists)))
        self.assertAlmostEqual(dists[0][0], 0.0, places=6)

    def test_empty_query_safety(self):
        query = np.zeros((1, self.dim))
        indices, dists = self.model.query_static(query, k=1)
        self.assertTrue(np.isinf(dists[0][0]))


class TestNumpyNN(SharedNNTests):
    nn_class = nn.NumpyNN


class TestFaissFlatL2NN(SharedNNTests):
    nn_class = nn.FaissFlatL2NN


class TestFaissHNSWFlatNN(SharedNNTests):
    nn_class = nn.FaissHNSWFlatNN


if __name__ == "__main__":
    unittest.main()
