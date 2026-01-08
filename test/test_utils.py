# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'

import unittest
import numpy as np

from ess import utils
from ess.ess import _scale, _inv_scale

class TestUtils(unittest.TestCase):

    def test_scaling_inverse(self):
        """Test that scale -> inv_scale returns original data."""
        original = np.array([
            [10.0, 20.0],
            [20.0, 40.0],
            [15.0, 30.0]
        ])

        scaled, min_v, max_v = _scale(original)

        # Check scaled bounds
        self.assertTrue(np.all(scaled >= 0.0))
        self.assertTrue(np.all(scaled <= 1.0))
        self.assertTrue(np.allclose(np.min(scaled, axis=0), 0.0))
        self.assertTrue(np.allclose(np.max(scaled, axis=0), 1.0))

        restored = _inv_scale(scaled, min_v, max_v)
        self.assertTrue(np.allclose(original, restored))

    def test_scaling_constant_dimension(self):
        """Test scaling when a dimension has 0 variance."""
        data = np.array([[10, 5], [20, 5]]) # Dim 1 is constant 5
        scaled, min_v, max_v = _scale(data)

        # Should not divide by zero (denom becomes 1.0)
        # Scaled col 1 should be 0.0 (value - min) / 1.0 = 0.0
        self.assertTrue(np.all(scaled[:, 1] == 0.0))

    def test_scaling_scalars(self):
            """
            Test scaling with explicit scalar min/max values.
            This covers the corner case where min/max are not arrays.
            """
            # 1D array case (scalars naturally occur here)
            data_1d = np.array([10, 20, 30])
            # Explicitly pass scalar limits
            scaled, min_v, max_v = _scale(data_1d, min_val=0, max_val=40)

            # Expect: (10-0)/40=0.25, (20-0)/40=0.5, (30-0)/40=0.75
            expected = np.array([0.25, 0.5, 0.75])
            self.assertTrue(np.allclose(scaled, expected))

            # Verify that scalars were accepted and returned
            self.assertEqual(min_v, 0)
            self.assertEqual(max_v, 40)

            # 2D case with scalar limits (broadcasting)
            data_2d = np.array([[10, 10], [20, 20]])
            # Min=0, Max=20 applies to ALL dimensions if passed as scalar
            scaled_2d, _, _ = _scale(data_2d, min_val=0, max_val=20)

            self.assertTrue(np.allclose(scaled_2d, [[0.5, 0.5], [1.0, 1.0]]))

    def test_grid_coverage_2d(self):
        """Test basic 2D grid coverage calculation."""
        bounds = np.array([[0, 10], [0, 10]])
        # Points in 2 different corners of a 2x2 grid
        points = np.array([
            [2, 2], # Bottom-Left
            [8, 8]  # Top-Right
        ])

        # Grid 2x2 = 4 cells. We occupy 2. Coverage = 0.5
        coverage = utils.calculate_grid_coverage(points, bounds, grid=2)
        self.assertAlmostEqual(coverage, 0.5)

    def test_grid_coverage_sparse_high_dim(self):
        """Test that high-dimensional grids use sparse logic (no memory error)."""
        dim = 50
        n_points = 100
        bounds = np.array([[0, 1]] * dim)
        points = np.random.uniform(0, 1, (n_points, dim))

        # If this was dense, 10^50 bins would crash RAM.
        # Sparse should run instantly.
        try:
            coverage = utils.calculate_grid_coverage(points, bounds, grid=10)
        except MemoryError:
            self.fail("calculate_grid_coverage raised MemoryError on high dims")

        # Since space is huge, every point likely occupies a unique cell
        # Coverage ~= 100 / 10^50 (approx 0.0)
        self.assertGreaterEqual(coverage, 0.0)

    def test_min_pairwise_distance(self):
        """Test Maximin metric."""
        points = np.array([[0, 0], [0, 1], [1, 0]]) # Triangle
        # Dists: (0,0)-(0,1)=1; (0,0)-(1,0)=1; (0,1)-(1,0)=sqrt(2)
        # Min dist = 1.0
        d = utils.calculate_min_pairwise_distance(points)
        self.assertAlmostEqual(d, 1.0)

        # Corner case: < 2 points
        self.assertEqual(utils.calculate_min_pairwise_distance(np.zeros((1,2))), 0.0)

    def test_clark_evans_index(self):
        """Test Clark-Evans: Clustered vs Uniform."""
        bounds = np.array([[0, 10], [0, 10]])

        # Clustered Points (all at 0,0)
        clustered = np.zeros((10, 2))
        r_clustered = utils.calculate_clark_evans_index(clustered, bounds)
        self.assertLess(r_clustered, 1.0)

        # Uniform-ish (Grid)
        x = np.linspace(0, 10, 4)
        y = np.linspace(0, 10, 4)
        xx, yy = np.meshgrid(x, y)
        uniform = np.column_stack((xx.ravel(), yy.ravel()))

        r_uniform = utils.calculate_clark_evans_index(uniform, bounds)
        # R should be near 1 or > 1 (Regular)
        self.assertGreater(r_uniform, 0.8)

if __name__ == '__main__':
    unittest.main()
