# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'

import unittest
import numpy as np

import ess.nn as nn

class SharedNNTests(unittest.TestCase):
    """
    Shared test logic for NN implementations.
    Concrete classes define self.nn_class.
    """
    nn_class = None

    def setUp(self):
        if self.nn_class is None:
            self.skipTest("SharedNNTests should not be run directly.")
        
        self.dim = 3
        # Use a specific seed for reproducibility
        self.model = self.nn_class(dimension=self.dim, seed=42)
        
        # Standard test data
        self.static_points = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0]
        ], dtype=np.float32)
        
        self.active_points = np.array([
            [0.1, 0.0, 0.0],  # Close to static[0]
            [5.0, 5.0, 5.0]   # Far away
        ], dtype=np.float32)

    def test_lifecycle(self):
        """Test add_static -> set_active -> consolidate -> clear cycle."""
        # 1. Initial State
        self.assertEqual(self.model.total_count, 0)

        # 2. Add Static
        self.model.add_static(self.static_points)
        self.assertEqual(self.model.total_count, 3)

        # 3. Set Active
        self.model.set_active(self.active_points)
        self.assertEqual(self.model.total_count, 5) # 3 static + 2 active

        # 4. Consolidate (Active becomes Static)
        self.model.consolidate()
        self.assertEqual(self.model.total_count, 5)
        
        # Verify active is empty by setting new active and checking count
        self.model.set_active(np.zeros((1, self.dim)))
        self.assertEqual(self.model.total_count, 6) # 5 static + 1 new active

        # 5. Clear
        self.model.clear()
        self.assertEqual(self.model.total_count, 0)

    def test_query_external(self):
        """Test querying static points from outside (Smart Init usage)."""
        self.model.add_static(self.static_points)
        
        # Query point exactly at [0,0,0]
        query = np.zeros((1, self.dim), dtype=np.float32)
        indices, dists = self.model.query_external(query, k=1)
        
        self.assertEqual(indices[0][0], 0) # Should match index 0
        self.assertAlmostEqual(dists[0][0], 0.0, places=5)

    def test_query_active_logic(self):
        """
        Test that active points find neighbors in BOTH static and active sets.
        """
        self.model.add_static(self.static_points)
        
        # active[0] is (0.1, 0, 0), very close to static[0] (0, 0, 0)
        # active[1] is (5, 5, 5)
        # We add a third active point close to active[1] to test active-active search
        complex_active = np.vstack([
            self.active_points, 
            [5.1, 5.0, 5.0] # Index 2 in active list (Global index 3+2=5)
        ])
        self.model.set_active(complex_active)

        # Query k=3 to ensure we capture: [Self, Nearest Static, Nearest Active]
        indices, dists = self.model.query_active(k=3)
        
        # -- Check Active Point 0 (0.1, 0, 0) --
        neighbors_0 = indices[0]
        self.assertIn(0, neighbors_0)
        # Verify distance to Static Point 0 is ~0.1
        idx_in_knn = np.where(neighbors_0 == 0)[0][0]
        self.assertAlmostEqual(dists[0][idx_in_knn], 0.1, places=5)

        # -- Check Active Point 2 (5.1, 5.0, 5.0) --
        # Nearest should be Active Point 1 (5.0, 5.0, 5.0) -> Global Index 3+1 = 4
        # Distance should be 0.1
        # Depending on sort order, check if neighbor is correct
        neighbors_of_2 = indices[2]
        self.assertIn(4, neighbors_of_2) 

    def test_dimension_mismatch(self):
        """Test error handling for wrong dimensions."""
        wrong_dim_points = np.zeros((5, self.dim + 1))
        
        with self.assertRaises(ValueError):
            self.model.add_static(wrong_dim_points)
            
        with self.assertRaises(ValueError):
            self.model.set_active(wrong_dim_points)

    def test_empty_query_safety(self):
        """Ensure querying empty index returns safe values (inf/zeros)."""
        # No static points
        query = np.zeros((1, self.dim))
        indices, dists = self.model.query_external(query, k=1)
        
        self.assertTrue(np.isinf(dists[0][0]))
        self.assertEqual(indices[0][0], 0)


class TestNumpyNN(SharedNNTests):
    nn_class = nn.NumpyNN

class TestFaissNN(SharedNNTests):
    nn_class = nn.FaissNN

if __name__ == '__main__':
    unittest.main()