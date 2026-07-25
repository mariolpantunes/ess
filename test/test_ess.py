# coding: utf-8

"""End-to-end tests of the toroidal ESA/ESS pipeline."""

import itertools
import logging
import unittest

import numpy as np

import ess
import ess.utils as utils
from torann import available_backends

# Disable info logging during tests to keep output clean
logging.getLogger("ess.ess").setLevel(logging.WARNING)
logging.getLogger("torann").setLevel(logging.WARNING)


class TestESS(unittest.TestCase):
    def setUp(self):
        # Initial "center" point to force repulsion
        self.samples = np.array([[0.5, 0.5]])
        self.bounds = np.array([[0, 1], [0, 1]])
        self.n_new = 50

    def test_basic_api(self):
        """Simple execution flows: defaults and a custom torann index."""
        res = ess.ess(self.samples, self.bounds, n=5, seed=42)
        self.assertEqual(res.shape, (6, 2))

        my_index = ess.ToroidalNN(seed=0, backend="python")
        res = ess.ess(self.samples, self.bounds, n=5, index=my_index, seed=42)
        self.assertEqual(res.shape, (6, 2))
        self.assertEqual(my_index.n_points, 6)

    def test_full_permutation_matrix(self):
        """Permutes search modes x force laws and checks, for every cell:
        validity in [0, 1], separation above the random baseline (Maximin)
        and a dispersed distribution (Clark-Evans R > 1.05)."""
        search_modes = ["k_nn", "radius"]
        metrics = ["gaussian", "softened_inverse", "linear", "cauchy"]

        rng = np.random.default_rng(42)
        random_points = rng.uniform(0, 1, (self.n_new, 2))
        baseline_min_dist = utils.calculate_min_pairwise_distance(random_points)

        for mode, metric in itertools.product(search_modes, metrics):
            with self.subTest(mode=mode, metric=metric):
                res = ess.ess(
                    self.samples,
                    self.bounds,
                    n=self.n_new,
                    search_mode=mode,
                    metric=metric,
                    seed=42,
                )
                new_pts = res[1:]
                self.assertEqual(len(new_pts), self.n_new)
                self.assertTrue(np.all(new_pts >= 0.0), np.min(new_pts))
                self.assertTrue(np.all(new_pts <= 1.0), np.max(new_pts))

                min_dist = utils.calculate_min_pairwise_distance(new_pts)
                self.assertGreater(
                    min_dist,
                    baseline_min_dist * 1.1,
                    f"ESS failed to beat random baseline for {mode}-{metric}",
                )

                ce_index = utils.calculate_clark_evans_index(new_pts, self.bounds)
                self.assertGreater(
                    ce_index,
                    1.05,
                    f"Distribution not dispersed enough (R={ce_index}) "
                    f"for {mode}-{metric}",
                )

    def test_from_scratch(self):
        """No initial samples: the first batch anchors only against itself."""
        empty = np.empty((0, 2))
        res = ess.ess(empty, self.bounds, n=30, seed=42)
        self.assertEqual(res.shape, (30, 2))
        self.assertTrue(np.all(res >= 0.0) and np.all(res <= 1.0))
        ce_index = utils.calculate_clark_evans_index(res, self.bounds)
        self.assertGreater(ce_index, 1.05)

    def test_seam_interaction(self):
        """Toroidal geometry: static anchors hugging the x=1 edge must
        repel new points from the x=0 side too (the faces are glued), so
        the strip just inside x=0 stays sparser than the middle."""
        anchors = np.column_stack(
            [np.full(20, 0.995), np.linspace(0.025, 0.975, 20)]
        )
        new = ess.esa(anchors, self.bounds, n=60, seed=1)
        near_seam = int(np.sum(new[:, 0] < 0.05))  # within 0.05 of the glued edge
        middle = int(np.sum(np.abs(new[:, 0] - 0.5) < 0.05))  # same-width strip
        self.assertLessEqual(near_seam, middle)

    def test_non_unit_bounds(self):
        """Scaling round-trip: results live inside the original domain."""
        bounds = np.array([[-5.0, 5.0], [100.0, 200.0]])
        samples = np.array([[0.0, 150.0]])
        res = ess.ess(samples, bounds, n=25, seed=42)
        self.assertEqual(res.shape, (26, 2))
        self.assertTrue(np.all(res[:, 0] >= -5.0) and np.all(res[:, 0] <= 5.0))
        self.assertTrue(np.all(res[:, 1] >= 100.0) and np.all(res[:, 1] <= 200.0))

    def test_dimensionality_scaling(self):
        """Low (1D) and high (50D) dimensions."""
        s_1d = np.array([[0.5]])
        b_1d = np.array([[0, 1]])
        res_1d = ess.ess(s_1d, b_1d, n=10, seed=1)
        self.assertEqual(res_1d.shape, (11, 1))
        # 1D relaxation on the circle: gaps should be fairly even
        gaps = np.diff(np.sort(res_1d[:, 0]))
        self.assertGreater(float(gaps.min()), 0.01)

        dim = 50
        s_hd = np.zeros((1, dim))
        b_hd = np.array([[0, 1]] * dim)
        res_hd = ess.ess(s_hd, b_hd, n=10, search_mode="radius", seed=1)
        self.assertEqual(res_hd.shape, (11, dim))
        self.assertTrue(np.all(res_hd >= 0) and np.all(res_hd <= 1))
        # the d/4 radius cap must not starve the neighbourhood: points moved
        self.assertGreater(
            utils.calculate_min_pairwise_distance(res_hd[1:]), 1.0
        )

    def test_backend_parity(self):
        """python and rust torann backends give the same quality class."""
        if "rust" not in available_backends():
            self.skipTest("rust backend not installed")
        dists = {}
        for backend in ("python", "rust"):
            idx = ess.ToroidalNN(seed=0, backend=backend)
            res = ess.ess(self.samples, self.bounds, n=20, index=idx, seed=42)
            dists[backend] = utils.calculate_min_pairwise_distance(res[1:])
        delta = abs(dists["python"] - dists["rust"])
        self.assertLess(delta, dists["python"] * 0.15, dists)

    def test_corner_cases(self):
        """Robustness against degenerate inputs."""
        # 1. n=0 -> only the original samples come back
        res_0 = ess.ess(self.samples, self.bounds, n=0)
        self.assertEqual(len(res_0), 1)

        # 2. epochs=0 -> smart-init points, valid but unoptimized
        res_ep0 = ess.ess(self.samples, self.bounds, n=10, epochs=0, seed=1)
        self.assertEqual(len(res_ep0), 11)
        self.assertTrue(np.all(res_ep0 >= 0.0) and np.all(res_ep0 <= 1.0))

        # 3. Singularity (all initial points identical) must not crash
        s_coin = np.array([[0.5, 0.5], [0.5, 0.5]])
        res_coin = ess.ess(s_coin, self.bounds, n=5, seed=1)
        self.assertEqual(len(res_coin), 7)

        # 4. Unknown metric name
        with self.assertRaises(ValueError):
            ess.ess(self.samples, self.bounds, n=5, metric="no_such_law")

    def test_run_stats(self):
        """The optional stats sink reports epochs, EMA and the radius."""
        stats = {}
        res = ess.esa(
            self.samples, self.bounds, n=12, batch_size=5, seed=42, stats=stats
        )
        self.assertEqual(len(res), 12)
        self.assertEqual(len(stats["batch_epochs"]), 3)  # batches 5, 5, 2
        self.assertEqual(len(stats["batch_force_ema"]), 3)
        self.assertEqual(stats["epochs_total"], sum(stats["batch_epochs"]))
        self.assertGreater(stats["radius"], 0.0)
        self.assertTrue(all(e >= 1 for e in stats["batch_epochs"]))

    def test_batching_logic(self):
        """Remainder batches: n=12, batch=5 -> 5, 5, 2."""
        res = ess.ess(self.samples, self.bounds, n=12, batch_size=5, seed=1)
        self.assertEqual(len(res), 13)

    def test_callable_metric(self):
        """A custom log-space force law is accepted as-is."""

        def my_law(d, **kwargs):
            return -2.0 * d  # exponential decay in log-space

        res = ess.ess(self.samples, self.bounds, n=15, metric=my_law, seed=42)
        self.assertEqual(len(res), 16)
        self.assertGreater(
            utils.calculate_min_pairwise_distance(res[1:]), 0.01
        )

    def test_samplers(self):
        """Basic functionality of LHCSampler and UniformSampler."""
        from ess.samplers import LHCSampler, UniformSampler

        lhc = LHCSampler(random_state=42)
        pts_lhc = lhc.sample(10, 3)
        self.assertEqual(pts_lhc.shape, (10, 3))
        self.assertTrue(np.all(pts_lhc >= 0.0) and np.all(pts_lhc <= 1.0))

        uni = UniformSampler(random_state=np.random.default_rng(42))
        pts_uni = uni.sample(10, 3)
        self.assertEqual(pts_uni.shape, (10, 3))
        self.assertTrue(np.all(pts_uni >= 0.0) and np.all(pts_uni <= 1.0))

    def test_ess_with_samplers(self):
        """ESS accepts sampler instances, ints, and None."""
        from ess import LHCSampler, UniformSampler

        res_lhc = ess.ess(
            self.samples, self.bounds, n=10, init_sampler=LHCSampler(), seed=42
        )
        self.assertEqual(len(res_lhc), 11)

        res_uni = ess.ess(
            self.samples, self.bounds, n=10, init_sampler=UniformSampler(), seed=42
        )
        self.assertEqual(len(res_uni), 11)

        empty_samples = np.empty((0, 2))
        res_scratch = ess.ess(
            empty_samples, self.bounds, n=15, init_sampler=LHCSampler(), seed=42
        )
        self.assertEqual(len(res_scratch), 15)
        self.assertTrue(np.all(res_scratch >= 0.0) and np.all(res_scratch <= 1.0))

        res_none = ess.ess(self.samples, self.bounds, n=5, init_sampler=None, seed=42)
        self.assertEqual(len(res_none), 6)

        res_seed = ess.ess(self.samples, self.bounds, n=5, init_sampler=123, seed=42)
        self.assertEqual(len(res_seed), 6)


if __name__ == "__main__":
    unittest.main()
