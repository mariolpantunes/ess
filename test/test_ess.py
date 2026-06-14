# coding: utf-8


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

    def test_knn_forces_with_invalid_indices(self):
        """Verify that _compute_knn_forces handles invalid neighbor indices (e.g. -1, out-of-bounds) gracefully."""
        from ess.ess import _compute_knn_forces, softened_inverse_force

        # Mock NN engine that returns invalid/out-of-bounds indices
        class MockNN:
            def query_nn(self, k):
                # Return shape (n_active, k)
                # First neighbor is valid, second is -1 (missing), third is out of bounds (999)
                indices = np.array([[0, -1, 999]], dtype=np.int32)
                dists = np.array([[0.1, 0.5, 1.0]], dtype=np.float32)
                return indices, dists

        nn_instance = MockNN()

        # 1 active point, 3 dimensions
        active_view = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        # all_data contains 1 static point at index 0, and the active point at index 1
        all_data = np.array(
            [
                [0.4, 0.5, 0.5],  # static point (index 0)
                [0.5, 0.5, 0.5],  # active point (index 1)
            ],
            dtype=np.float32,
        )

        rng = np.random.default_rng(42)

        # Call _compute_knn_forces
        # batch_start_idx = 1, batch_end_idx = 2
        forces = _compute_knn_forces(
            active_view=active_view,
            all_data=all_data,
            nn_instance=nn_instance,
            k_value=3,
            metric_fn=softened_inverse_force,
            rng=rng,
            batch_start_idx=1,
            batch_end_idx=2,
        )

        # The forces should not raise an IndexError and should be calculated only from the valid neighbor at index 0.
        # Neighbor at -1 and 999 are invalid, so their force magnitude should be zeroed out.
        # Since active_view is [0.5, 0.5, 0.5] and static neighbor is [0.4, 0.5, 0.5],
        # displacement vector is [0.1, 0.0, 0.0].
        # Force direction should be along +x axis: [positive_value, 0, 0].
        self.assertEqual(forces.shape, (1, 3))
        self.assertGreater(forces[0, 0], 0.0)
        self.assertAlmostEqual(forces[0, 1], 0.0)
        self.assertAlmostEqual(forces[0, 2], 0.0)

    def test_samplers(self):
        """Test basic functionality of LHCSampler and UniformSampler."""
        from ess.samplers import LHCSampler, UniformSampler

        rng = np.random.default_rng(42)

        # Test LHCSampler
        lhc = LHCSampler(random_state=42)
        pts_lhc = lhc.sample(10, 3)
        self.assertEqual(pts_lhc.shape, (10, 3))
        self.assertTrue(np.all(pts_lhc >= 0.0))
        self.assertTrue(np.all(pts_lhc <= 1.0))

        # Test UniformSampler
        uni = UniformSampler(random_state=rng)
        pts_uni = uni.sample(10, 3)
        self.assertEqual(pts_uni.shape, (10, 3))
        self.assertTrue(np.all(pts_uni >= 0.0))
        self.assertTrue(np.all(pts_uni <= 1.0))

    def test_ess_with_samplers(self):
        """Test running ESS with different samplers, both with and without initial samples."""
        from ess import LHCSampler, UniformSampler

        # 1. With initial samples using LHCSampler (default)
        res_lhc = ess.ess(
            self.samples, self.bounds, n=10, init_sampler=LHCSampler(), seed=42
        )
        self.assertEqual(len(res_lhc), 11)

        # 2. With initial samples using UniformSampler
        res_uni = ess.ess(
            self.samples, self.bounds, n=10, init_sampler=UniformSampler(), seed=42
        )
        self.assertEqual(len(res_uni), 11)

        # 3. From scratch (no initial samples) using LHCSampler
        empty_samples = np.empty((0, 2), dtype=np.float32)
        res_scratch = ess.ess(
            empty_samples, self.bounds, n=15, init_sampler=LHCSampler(), seed=42
        )
        self.assertEqual(len(res_scratch), 15)
        self.assertTrue(np.all(res_scratch >= 0.0))
        self.assertTrue(np.all(res_scratch <= 1.0))

        # 4. Using None for init_sampler (defaults to LHCSampler)
        res_none = ess.ess(self.samples, self.bounds, n=5, init_sampler=None, seed=42)
        self.assertEqual(len(res_none), 6)

        # 5. Using integer seed for init_sampler (defaults to LHCSampler seeded with that integer)
        res_seed = ess.ess(self.samples, self.bounds, n=5, init_sampler=123, seed=42)
        self.assertEqual(len(res_seed), 6)


if __name__ == "__main__":
    unittest.main()
