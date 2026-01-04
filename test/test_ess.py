# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'

import unittest
import numpy as np

import ess.ess as ess
import ess.nn as nn

class TestESS(unittest.TestCase):
    
    def setUp(self):
        self.samples = np.array([[0.5, 0.5]], dtype=np.float32)
        self.bounds = np.array([[0, 1], [0, 1]])
        self.n_new = 10

    def test_ess_basic_execution(self):
        """Test basic execution returns correct shape and bounds."""
        result = ess.ess(self.samples, self.bounds, n=self.n_new, seed=42)
        self.assertEqual(result.shape, (11, 2))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_ess_factory_numpy(self):
        """Test that default behavior (no NN provided) works."""
        res = ess.ess(self.samples, self.bounds, n=5, nn_instance=None)
        self.assertEqual(len(res), 6)

    def test_ess_explicit_nn(self):
        """Test passing an explicit NN instance."""
        my_nn = nn.NumpyNN(dimension=2)
        res = ess.ess(self.samples, self.bounds, n=5, nn_instance=my_nn)
        self.assertEqual(len(res), 6)

    def test_ess_custom_metric(self):
        """Test using a different force metric."""
        # Test by string name
        res_str = ess.ess(self.samples, self.bounds, n=5, metric='linear')
        self.assertEqual(len(res_str), 6)
        
        # Test by callable
        def my_force(d): return 1.0 / (d + 1.0)
        res_call = ess.ess(self.samples, self.bounds, n=5, metric=my_force)
        self.assertEqual(len(res_call), 6)

    def test_ess_batching(self):
        """Test that batching logic works when n > batch_size."""
        # 15 points, batch 5 -> 3 batches
        res = ess.ess(self.samples, self.bounds, n=15, batch_size=5, epochs=10)
        self.assertEqual(len(res), 16)

    def test_invalid_metric_error(self):
        with self.assertRaises(ValueError):
            ess.ess(self.samples, self.bounds, n=5, metric='invalid_metric')

    def test_zero_epochs(self):
        """Test that 0 epochs returns the Smart Init points immediately."""
        res = ess.ess(self.samples, self.bounds, n=5, epochs=0)
        self.assertEqual(len(res), 6)

if __name__ == '__main__':
    unittest.main()
