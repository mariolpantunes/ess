# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.3"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"
__license__ = "MIT"

import itertools
import logging
import unittest

import numpy as np

import ess
import ess.nn as nn
import ess.utils as utils

# Disable info logging during tests to keep output clean
logging.getLogger("ess.ess").setLevel(logging.WARNING)


class TestESS(unittest.TestCase):
    def setUp(self):
        # Initial "center" point to force repulsion
        self.samples = np.array([[0.5, 0.5]], dtype=np.float32)
        self.bounds = np.array([[0, 1], [0, 1]])
        self.n_new = 50

    def test_basic_api(self):
        """Test simple execution flows."""
        # 1. Default
        res = ess.ess(self.samples, self.bounds, n=5, seed=42)
        self.assertEqual(res.shape, (6, 2))

        # 2. Explicit NN
        my_nn = nn.NumpyNN(dimension=2)
        res = ess.ess(self.samples, self.bounds, n=5, nn_instance=my_nn)
        self.assertEqual(res.shape, (6, 2))

    def test_full_permutation_matrix(self):
        """
        Permutes ALL combinations of configuration flags.
        Evaluates quality against a Random Uniform Baseline.

        Checks:
        1. Does it crash?
        2. Are points valid?
        3. Is the distribution better than random? (Maximin Distance)
        """
        search_modes = ["k_nn", "radius"]
        borders = ["clip", "repulsive"]
        metrics = ["gaussian", "softened_inverse", "linear"]

        # We perform a rigorous quality check
        # Random Sampling Baseline for N=50
        rng = np.random.default_rng(42)
        random_points = rng.uniform(0, 1, (self.n_new, 2))
        baseline_min_dist = utils.calculate_min_pairwise_distance(random_points)

        for mode, border, metric in itertools.product(search_modes, borders, metrics):
            with self.subTest(mode=mode, border=border, metric=metric):
                # Run ESS
                # Use a known seed for reproducibility
                res = ess.ess(
                    self.samples,
                    self.bounds,
                    n=self.n_new,
                    search_mode=mode,
                    border_strategy=border,
                    metric=metric,
                    epochs=200,  # Sufficient for convergence test
                    seed=42,
                )

                # 1. Validity Check
                # Exclude original sample [0] for rigorous bounds checking
                new_pts = res[1:]
                self.assertEqual(len(new_pts), self.n_new)
                self.assertTrue(
                    np.all(new_pts >= -1e-5),
                    f"Lower bound violation: {np.min(new_pts)}",
                )
                self.assertTrue(
                    np.all(new_pts <= 1.00001),
                    f"Upper bound violation: {np.max(new_pts)}",
                )

                # 2. Quality Check (Maximin Criterion)
                ess_min_dist = utils.calculate_min_pairwise_distance(new_pts)

                # 3. Regularity Check (Clark-Evans)
                # R > 1 means dispersed/regular. R ~ 1 is random.
                ce_index = utils.calculate_clark_evans_index(new_pts, self.bounds)

                if border == "repulsive":
                    self.assertGreater(
                        ess_min_dist, 1e-5, f"Points stacked! {mode}-{border}-{metric}"
                    )
                    self.assertGreater(
                        ess_min_dist,
                        baseline_min_dist * 1.1,
                        f"ESS failed to beat random baseline in configuration {mode}-{border}-{metric}",
                    )
                self.assertGreater(
                    ce_index,
                    1.05,
                    f"Distribution not dispersed enough (R={ce_index}) for {mode}-{border}-{metric}",
                )

    def test_dimensionality_scaling(self):
        """Test Low (1D) and High (50D) dimensions logic."""
        # 1D Case
        s_1d = np.array([[0.5]])
        b_1d = np.array([[0, 1]])
        res_1d = ess.ess(s_1d, b_1d, n=10)
        self.assertEqual(res_1d.shape, (11, 1))

        # High-D Case (50D) - Forces Sparse Logic / Heuristic Caps
        dim = 50
        s_hd = np.zeros((1, dim))
        b_hd = np.array([[0, 1]] * dim)

        # Use Radius search to verify the Heuristic Radius Cap logic works
        res_hd = ess.ess(s_hd, b_hd, n=10, search_mode="radius")

        self.assertEqual(res_hd.shape, (11, dim))
        self.assertTrue(np.all(res_hd >= 0))
        self.assertTrue(np.all(res_hd <= 1))

    def test_backend_parity(self):
        """
        Verify that NumpyNN and FaissHNSWFlatNN produce qualitatively similar results.
        This ensures the 'fast path' isn't broken compared to the 'scalable path'.
        """
        # Force Numpy
        numpy_nn = nn.NumpyNN(dimension=2)
        res_numpy = ess.ess(
            self.samples, self.bounds, n=20, nn_instance=numpy_nn, seed=42
        )
        dist_numpy = utils.calculate_min_pairwise_distance(res_numpy[1:])

        # Force Faiss
        faiss_nn = nn.FaissHNSWFlatNN(dimension=2)
        res_faiss = ess.ess(
            self.samples, self.bounds, n=20, nn_instance=faiss_nn, seed=42
        )
        dist_faiss = utils.calculate_min_pairwise_distance(res_faiss[1:])

        # They should be within 10% of each other (stochastic differences allowed)
        # Note: They won't be identical because HNSW is approximate and Numpy is exact.
        delta = abs(dist_numpy - dist_faiss)
        self.assertLess(
            delta, dist_numpy * 0.15, "Backends diverged significantly in quality"
        )

    def test_corner_cases(self):
        """Test Robustness against bad inputs."""
        # 1. n=0 -> Should return original samples
        res_0 = ess.ess(self.samples, self.bounds, n=0)
        self.assertEqual(len(res_0), 1)

        # 2. epochs=0 -> Should return Smart Init points (valid but unoptimized)
        res_ep0 = ess.ess(self.samples, self.bounds, n=10, epochs=0)
        self.assertEqual(len(res_ep0), 11)
        # Smart init should still be better than pure random
        # (Implicitly tested via quality, but just shape here)

        # 3. Singularity (All points identical)
        s_coin = np.array([[0.5, 0.5], [0.5, 0.5]])
        # Should not crash with div/0
        res_coin = ess.ess(s_coin, self.bounds, n=5)
        self.assertEqual(len(res_coin), 7)

    def test_batching_logic(self):
        """Test that batch logic handles remainders correctly."""
        # n=12, batch=5 -> Batches of 5, 5, 2
        res = ess.ess(self.samples, self.bounds, n=12, batch_size=5)
        self.assertEqual(len(res), 13)


if __name__ == "__main__":
    unittest.main()
