# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.2'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'

import unittest
import numpy as np
from ess import utils
from ess.ess import _scale, _inv_scale, METRIC_REGISTRY

class TestUtils(unittest.TestCase):

    def test_scaling_lifecycle(self):
        """Test scale -> inv_scale round trip."""
        original = np.array([[10.0, 20.0], [20.0, 40.0], [15.0, 30.0]])
        scaled, min_v, max_v = _scale(original)
        
        self.assertTrue(np.all(scaled >= 0.0))
        self.assertTrue(np.all(scaled <= 1.0))
        
        restored = _inv_scale(scaled, min_v, max_v)
        self.assertTrue(np.allclose(original, restored))

    def test_scaling_degenerate(self):
        """Test scaling constant dimensions and scalars."""
        # Constant dimension
        data = np.array([[5, 5], [5, 10]])
        scaled, _, _ = _scale(data)
        self.assertTrue(np.all(scaled[:, 0] == 0.0)) # Denom was 0 -> handled
        
        # Scalars passed as min/max
        arr = np.array([10, 20, 30])
        s, _, _ = _scale(arr, min_val=0, max_val=40)
        self.assertTrue(np.allclose(s, [0.25, 0.5, 0.75]))

    def test_grid_coverage(self):
        """Test grid coverage logic."""
        bounds = np.array([[0, 10], [0, 10]])
        points = np.array([[2, 2], [8, 8]]) # 2 points
        
        # 2x2 Grid -> 4 cells. Points in different cells. Coverage 0.5.
        cov = utils.calculate_grid_coverage(points, bounds, grid=2)
        self.assertAlmostEqual(cov, 0.5)
        
        # High Dim Sparse Test
        dim = 64
        pts_hd = np.random.rand(10, dim)
        bounds_hd = np.array([[0, 1]] * dim)
        # Should not crash
        cov_hd = utils.calculate_grid_coverage(pts_hd, bounds_hd, grid=3)
        self.assertGreater(cov_hd, 0.0)

    def test_metrics_maximin_clarkevans(self):
        """Test distribution metrics."""
        # Triangle (Regular)
        points = np.array([[0,0], [1,0], [0.5, 0.866]])
        d = utils.calculate_min_pairwise_distance(points)
        self.assertAlmostEqual(d, 1.0, places=3)
        
        # Clark Evans
        # Clustered
        clust = np.zeros((10, 2))
        ce_c = utils.calculate_clark_evans_index(clust)
        self.assertLess(ce_c, 1.0)
        
        # Random/Uniform check (Statistical, loose bounds)
        uni = np.random.rand(100, 2) * 10
        ce_u = utils.calculate_clark_evans_index(uni)
        self.assertGreater(ce_u, 0.5)

    def test_force_functions(self):
        """Verify force function behaviors."""
        d = np.array([0.0, 1.0, 100.0])
        
        # 1. Gaussian: High at 0, Low at 100
        f_gauss = METRIC_REGISTRY['gaussian'](d, sigma=1.0, alpha=1.0)
        self.assertAlmostEqual(f_gauss[0], 1.0)
        self.assertLess(f_gauss[2], 1e-5)
        
        # 2. Linear: High at 0, Zero > R
        f_lin = METRIC_REGISTRY['linear'](d, R=1.0)
        self.assertAlmostEqual(f_lin[0], 1.0)
        self.assertAlmostEqual(f_lin[1], 0.0) # At radius R=1, force is 0
        self.assertEqual(f_lin[2], 0.0)
        
        # 3. Softened Inverse: Finite at 0, Decays slowly
        f_inv = METRIC_REGISTRY['softened_inverse'](d, epsilon=0.1, alpha=1.0)
        self.assertLess(f_inv[0], 101.0) # 1 / 0.01 = 100
        self.assertGreater(f_inv[2], 0.0) # Never zero

if __name__ == '__main__':
    unittest.main()