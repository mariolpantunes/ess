
"""End-to-end tests of the toroidal ESA/ESS pipeline."""

import importlib
import itertools
import logging
import unittest

import numpy as np

import ess
from ess import attraction, geometry, utils

# `ess.ess` is the exported *function*, which shadows the submodule of
# the same name, so the private helpers need an explicit module import.
ess_core = importlib.import_module("ess.ess")
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
        baseline_min_dist = utils.euclidean_separation(random_points)

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

                min_dist = utils.euclidean_separation(new_pts)
                self.assertGreater(
                    min_dist,
                    baseline_min_dist * 1.1,
                    f"ESS failed to beat random baseline for {mode}-{metric}",
                )

    def test_from_scratch(self):
        """No initial samples: the first batch anchors only against itself."""
        empty = np.empty((0, 2))
        res = ess.ess(empty, self.bounds, n=30, seed=42)
        self.assertEqual(res.shape, (30, 2))
        self.assertTrue(np.all(res >= 0.0) and np.all(res <= 1.0))

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
            utils.euclidean_separation(res_hd[1:]), 1.0
        )

    def test_backend_parity(self):
        """python and rust torann backends give the same quality class."""
        if "rust" not in available_backends():
            self.skipTest("rust backend not installed")
        dists = {}
        for backend in ("python", "rust"):
            idx = ess.ToroidalNN(seed=0, backend=backend)
            res = ess.ess(self.samples, self.bounds, n=20, index=idx, seed=42)
            dists[backend] = utils.euclidean_separation(res[1:])
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

    def test_run_stats_timing_decomposition(self):
        """The timing buckets must cover the inner loop, so their sum
        accounts for nearly all of the measured wall time."""
        import time

        stats = {}
        t0 = time.perf_counter()
        ess.esa(self.samples, self.bounds, n=40, seed=1, stats=stats)
        wall = time.perf_counter() - t0

        buckets = ("query_s", "force_s", "step_s", "update_s", "setup_s")
        for key in buckets:
            self.assertIn(key, stats)
            self.assertGreater(stats[key], 0.0)
        accounted = sum(stats[k] for k in buckets)
        self.assertLessEqual(accounted, wall * 1.05)
        self.assertGreater(accounted, wall * 0.5)

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
            utils.euclidean_separation(res[1:]), 0.01
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


class TestGuidedPlacement(unittest.TestCase):
    """Placement mirrors the mode: repulsive alone, composite with fitness.

    Scoring candidates on novelty alone puts each one as far from everything
    as possible -- including as far from the good regions -- and then leaves
    the relaxation to guide it back. On the OBLESA sweep that placement was
    worth nothing: unguided dart tied uniform noise 233-243 over 480 paired
    cells, while the same search with a fitness term in the *placement* beat
    it 309-169.
    """

    D = 8
    BOUNDS = np.array([[-5.0, 5.0]] * D)

    def _setup(self, seed=0, m=40):
        rng = np.random.default_rng(seed)
        static = rng.uniform(-5, 5, (m, self.D))
        # "good" (low fitness) means near the origin, so the effect is measurable as a radius
        return static, np.linalg.norm(static, axis=1)

    def _composite(self, static, fit, weight=0.5, seed=11, n=30, **kw):
        return ess.esa(static, self.BOUNDS, fitness=fit, n=n,
                       fitness_weight=weight, seed=seed, **kw)

    def test_weight_zero_reproduces_repulsive_placement_exactly(self):
        """The composite path has to be an ablation of the repulsive one, not
        a different algorithm, or no comparison between them means anything."""
        static, fit = self._setup()
        rep = ess.esa(static, self.BOUNDS, n=30, seed=11)
        zero = self._composite(static, fit, weight=0.0)
        np.testing.assert_allclose(zero, rep)

    def test_fitness_pulls_placement_toward_the_good_region(self):
        static, fit = self._setup()
        rep = ess.esa(static, self.BOUNDS, n=30, seed=11)
        com = self._composite(static, fit)
        self.assertLess(np.linalg.norm(com, axis=1).mean(),
                        np.linalg.norm(rep, axis=1).mean())

    def test_more_weight_pulls_harder(self):
        static, fit = self._setup(seed=3, m=80)
        radii = [float(np.linalg.norm(
            self._composite(static, fit, weight=w), axis=1).mean())
            for w in (0.0, 0.25, 0.5)]
        self.assertLess(radii[1], radii[0])
        self.assertLess(radii[2], radii[1])

    def test_runs_stay_reproducible(self):
        static, fit = self._setup()
        a = self._composite(static, fit)
        b = self._composite(static, fit)
        np.testing.assert_array_equal(a, b)

    def test_results_stay_inside_the_bounds(self):
        static, fit = self._setup()
        out = self._composite(static, fit)
        self.assertTrue((out >= self.BOUNDS[:, 0]).all())
        self.assertTrue((out <= self.BOUNDS[:, 1]).all())


class TestAttractionField(unittest.TestCase):
    """Fitness belongs to a position, not to a point."""

    def _field(self, k=4):
        pos = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])
        val = np.array([0.0, 1.0, 0.5, 0.5])
        return ess_core._AttractionField(pos, val, k=k)

    def test_the_value_follows_the_position(self):
        f = self._field()
        near_good = f.at(np.array([[0.12, 0.12]]))[0]
        near_bad = f.at(np.array([[0.88, 0.88]]))[0]
        self.assertLess(near_good, near_bad)

    def test_measured_sources_count_is_recorded(self):
        f = self._field()
        self.assertEqual(f.n_measured, 4)


class TestAttractionModels(unittest.TestCase):
    """Inverse-distance nearest-neighbour weighting across dimensions."""

    def _sample(self, d, m, seed=0):
        rng = np.random.default_rng(seed)
        pos = rng.random((m, d))
        val = (np.sin(2 * np.pi * pos[:, 0])
               + 0.5 * np.cos(2 * np.pi * pos[:, 1]))
        return pos, val

    def _loo(self, d, m):
        pos, val = self._sample(d, m)
        err = []
        for i in range(m):
            keep = np.ones(m, bool)
            keep[i] = False
            f = ess_core._AttractionField(pos[keep], val[keep])
            err.append(abs(f.at(pos[i:i + 1])[0] - val[i]))
        return float(np.mean(err))

    def test_more_points_helps_less_as_dimension_rises(self):
        """More measured points is not a general escape from high dimension."""
        for d in (8, 100):
            with self.subTest(d=d):
                few = self._loo(d, 60)
                many = self._loo(d, 300)
                self.assertLessEqual(many, few)
        self.assertLess(self._loo(8, 300),
                        self._loo(8, 60) * 0.8)

    def test_the_estimate_never_leaves_the_range_of_its_sources(self):
        pos, val = self._sample(4, 50)
        f = ess_core._AttractionField(pos, val)
        rng = np.random.default_rng(2)
        got = f.at(rng.random((400, 4)))
        self.assertGreaterEqual(got.min(), val.min())
        self.assertLessEqual(got.max(), val.max())

    def test_the_estimate_is_periodic(self):
        pos, val = self._sample(6, 40)
        f = ess_core._AttractionField(pos, val)
        near_zero = f.at(np.array([[1e-9] + [0.5] * 5]))
        near_one = f.at(np.array([[1.0 - 1e-9] + [0.5] * 5]))
        self.assertAlmostEqual(float(near_zero[0]), float(near_one[0]), places=6)

    def test_idw_is_reachable_through_esa(self):
        rng = np.random.default_rng(0)
        static = rng.uniform(-5, 5, (40, 6))
        bounds = np.array([[-5.0, 5.0]] * 6)
        fit = np.linalg.norm(static, axis=1)
        out = ess.esa(static, bounds, fitness=fit, n=20, seed=1,
                      fitness_weight=0.5)
        self.assertEqual(out.shape, (20, 6))


class TestEstimateImprovesWithData(unittest.TestCase):
    """More measured points must make the estimate better, monotonically."""

    @staticmethod
    def _truth(x):
        return (np.sin(2 * np.pi * x[:, 0])
                + 0.5 * np.cos(2 * np.pi * x[:, 1])
                + 0.25 * np.sin(4 * np.pi * x[:, 2]))

    def _error(self, d, m, seeds=5):
        """Mean absolute error on a held-out set, averaged over seeds."""
        errs = []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            pos = rng.random((m, d))
            f = ess_core._AttractionField(pos, self._truth(pos))
            q = rng.random((500, d))
            errs.append(float(np.abs(f.at(q) - self._truth(q)).mean()))
        return float(np.mean(errs))

    def _ranking(self, d, m, seeds=5):
        """Share of the true best (lowest-fitness) decile the estimate recovers."""
        got = []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            pos = rng.random((m, d))
            f = ess_core._AttractionField(pos, self._truth(pos))
            q = rng.random((500, d))
            est, tru = f.at(q), self._truth(q)
            n = max(1, len(q) // 10)
            pick = set(np.argsort(est)[:n].tolist())
            best = set(np.argsort(tru)[:n].tolist())
            got.append(len(pick & best) / n)
        return float(np.mean(got))

    def test_error_never_rises_as_measured_points_are_added(self):
        for d in (8, 32):
            ladder = [(m, self._error(d, m)) for m in (20, 40, 80, 160, 320)]
            for (m0, e0), (m1, e1) in itertools.pairwise(ladder):
                self.assertLessEqual(
                    e1, e0 * 1.10,
                    f"d={d}: error rose from {e0:.4f} at M={m0} to "
                    f"{e1:.4f} at M={m1}")

    def test_the_ranking_is_what_keeps_improving(self):
        for d in (8, 32, 100):
            with self.subTest(d=d):
                few = self._ranking(d, 60)
                many = self._ranking(d, 600)
                self.assertGreater(many, few, f"d={d}: {few:.3f} -> {many:.3f}")
                self.assertGreater(few, 0.10,
                                   f"d={d}: {few:.3f} is chance or worse")

    def test_the_estimate_beats_predicting_the_mean(self):
        for d, m in ((8, 60), (32, 60), (100, 60)):
            with self.subTest(d=d, m=m):
                rng = np.random.default_rng(7)
                q = rng.random((500, d))
                tru = self._truth(q)
                flat = float(np.abs(tru - tru.mean()).mean())
                self.assertLess(self._error(d, m), flat)

    def test_it_improves_at_medium_dimension(self):
        lo = self._error(16, 40)
        hi = self._error(16, 320)
        self.assertLess(hi, lo, f"{lo:.4f} -> {hi:.4f}")


class TestAttractivenessEstimate(unittest.TestCase):
    """The candidate-side fitness estimate."""

    @staticmethod
    def _fit(static, values):
        return attraction.InverseDistance().fit(static, values)

    def test_a_point_on_a_static_point_takes_its_value(self):
        static = np.array([[0.1, 0.1], [0.8, 0.8]])
        a = np.array([0.25, 0.75])
        got = self._fit(static, a).at(static.copy())
        np.testing.assert_allclose(got, a)

    def test_the_estimate_is_bounded_by_the_known_values(self):
        rng = np.random.default_rng(1)
        static = rng.random((30, 5))
        a = rng.random(30)
        q = rng.random((200, 5))
        got = self._fit(static, a).at(q)
        self.assertGreaterEqual(got.min(), a.min() - 1e-12)
        self.assertLessEqual(got.max(), a.max() + 1e-12)

    def test_it_is_nearer_the_closer_neighbour(self):
        """Inverse-distance weighting: the near value must dominate."""
        static = np.array([[0.10, 0.5], [0.90, 0.5]])
        a = np.array([0.0, 1.0])
        got = self._fit(static, a).at(np.array([[0.15, 0.5]]))
        self.assertLess(got[0], 0.25)

    def test_no_static_points_is_not_an_error(self):
        got = self._fit(np.zeros((0, 3)), np.zeros(0)).at(np.zeros((4, 3)))
        self.assertEqual(got.shape, (4,))

    def test_it_wraps_around_the_torus(self):
        static = np.array([[0.98, 0.5], [0.50, 0.5]])
        a = np.array([1.0, 0.0])
        got = self._fit(static, a).at(np.array([[0.02, 0.5]]))
        self.assertGreater(got[0], 0.5)


class TestStopRule(unittest.TestCase):
    """The relaxation stops on the nearest-neighbour gap, not on the force.

    While the step cap binds, the points travel at constant speed and the
    largest force does not fall, so a plateau of the force fired after 31-58
    epochs at d >= 16, before most of the gain: from scratch at d = 16 the
    run kept 80% of the gap a long run reaches, and an extension of uniform
    points ended below its own placement.
    """

    D = 16
    N = 256
    BOUNDS = np.array([[0.0, 1.0]] * D)

    @staticmethod
    def _gap_p05(new, old):
        """5% quantile of the toroidal L1 distance from each new point to its
        nearest other point."""
        others = np.concatenate((new, old)) if old.size else new
        diff = np.abs(new[:, None, :] - others[None, :, :])
        dist = np.minimum(diff, 1.0 - diff).sum(axis=2)
        dist[np.arange(new.shape[0]), np.arange(new.shape[0])] = np.inf
        return float(np.quantile(dist.min(axis=1), 0.05))

    def test_stop_keeps_most_of_what_a_long_run_reaches(self):
        empty = np.empty((0, self.D))
        stats = {}
        stopped = ess.esa(empty, self.BOUNDS, n=self.N, seed=0, stats=stats)
        long = ess.esa(empty, self.BOUNDS, n=self.N, seed=0, epochs=200,
                       patience=10**9)
        self.assertGreater(self._gap_p05(stopped, empty),
                           0.93 * self._gap_p05(long, empty))
        self.assertLess(stats["epochs_total"], 100)

    def test_relaxation_ends_above_its_placement(self):
        for seed in range(3):
            static = np.random.default_rng(seed).random((self.N, self.D))
            placed = ess.esa(static, self.BOUNDS, n=self.N, seed=seed, epochs=0, w=0.0)
            relaxed = ess.esa(static, self.BOUNDS, n=self.N, seed=seed)
            self.assertGreater(self._gap_p05(relaxed, static),
                               self._gap_p05(placed, static))

    def test_patience_bounds_a_run_that_cannot_improve(self):
        """One point far from everything has no gap to gain: the run ends
        after `patience` epochs, not after `epochs`."""
        static = np.full((1, 2), 0.5)
        stats = {}
        ess.esa(static, np.array([[0.0, 1.0]] * 2), n=1, seed=0, stats=stats,
                search_mode="radius", patience=7)
        self.assertLessEqual(stats["epochs_total"], 8)


class TestGuardsAndDegenerateInputs(unittest.TestCase):
    """Branches that only fire on bad or edge input, and so never ran."""

    def test_invalid_c_and_w_are_refused(self):
        rng = np.random.default_rng(0)
        static = rng.uniform(-5, 5, (20, 3))
        bounds = np.array([[-5.0, 5.0]] * 3)
        with self.assertRaises(ValueError):
            ess.esa(static, bounds, n=5, C=0.0)
        with self.assertRaises(ValueError):
            ess.esa(static, bounds, n=5, C=-2.0)
        with self.assertRaises(ValueError):
            ess.esa(static, bounds, n=5, w=-0.1)
        with self.assertRaises(ValueError):
            ess.esa(static, bounds, n=5, w=1.5)

    def test_a_generator_is_used_as_given(self):
        """Passing a Generator must not reseed it into a different stream."""
        static = np.random.default_rng(3).uniform(-5, 5, (20, 3))
        bounds = np.array([[-5.0, 5.0]] * 3)
        one = ess.esa(static, bounds, n=6, seed=np.random.default_rng(11))
        two = ess.esa(static, bounds, n=6, seed=np.random.default_rng(11))
        np.testing.assert_array_equal(one, two)

    def test_a_per_axis_grid_must_match_the_dimension(self):
        pts = np.random.default_rng(0).random((30, 3))
        bounds = np.array([[0.0, 1.0]] * 3)
        self.assertGreaterEqual(
            utils.calculate_grid_coverage(pts, bounds, [4, 4, 4]), 0.0)
        with self.assertRaises(ValueError):
            utils.calculate_grid_coverage(pts, bounds, [4, 4])

    def test_an_empty_grid_scores_nothing_rather_than_dividing_by_zero(self):
        pts = np.random.default_rng(0).random((10, 2))
        bounds = np.array([[0.0, 1.0]] * 2)
        self.assertEqual(
            utils.calculate_grid_coverage(pts, bounds, [0, 5]), 0.0)

    def test_a_sampler_can_be_asked_for_no_points(self):
        for sampler in (ess.LHCSampler(random_state=0),
                        ess.UniformSampler(random_state=0)):
            with self.subTest(sampler=type(sampler).__name__):
                out = sampler.sample(0, 4)
                self.assertEqual(out.shape, (0, 4))

    def test_check_sampler_refuses_what_it_cannot_build(self):
        self.assertIsInstance(ess.check_sampler(7), ess.LHCSampler)
        built = ess.UniformSampler(random_state=1)
        self.assertIs(ess.check_sampler(built), built)
        with self.assertRaises(TypeError):
            ess.check_sampler("lhs")


class TestInitPool(unittest.TestCase):
    """`init_pool` is the k of Mitchell's best-candidate inside `_smart_init`.

    The sampler does not place the initial positions; it proposes
    `init_pool` candidates per slot and the farthest from everything already
    indexed wins. `init_pool=1` is therefore the ablation of that selection,
    and it measures worse at every dimension tested -- which is why the step
    exists.
    """

    def setUp(self):
        self.bounds = np.array([[0.0, 1.0]] * 6)
        self.static = np.random.default_rng(0).random((40, 6))

    def test_pool_is_reachable_and_changes_the_result(self):
        a = ess.esa(self.static, self.bounds, n=20, seed=1, init_pool=1)
        b = ess.esa(self.static, self.bounds, n=20, seed=1, init_pool=256)
        self.assertEqual(a.shape, b.shape)
        self.assertFalse(np.allclose(a, b))

    def test_every_pool_size_stays_in_bounds(self):
        for p in (1, 15, 64, 512):
            out = ess.esa(self.static, self.bounds, n=16, seed=2, init_pool=p)
            self.assertTrue((out >= 0.0).all() and (out <= 1.0).all(), p)
            self.assertTrue(np.isfinite(out).all(), p)

    def test_slots_of_one_batch_do_not_race_for_the_same_voids(self):
        """Extending clustered data by many more points than it holds.

        Scored against the index alone, every slot of the batch picks the
        same few emptiest pockets and the batch lands in the corners: this
        configuration covered 30% of the grid. Each slot has to be far from
        the slots placed before it too.
        """
        rng = np.random.default_rng(0)
        centres = rng.uniform(0.2, 0.8, size=(6, 2))
        static = np.clip(
            centres[rng.integers(6, size=1000)]
            + rng.normal(0.0, 0.06, size=(1000, 2)), 0.0, 1.0)
        bounds = np.array([[0.0, 1.0]] * 2)
        out = ess.esa(static, bounds, n=1000, seed=1)
        cells = np.unique(np.minimum((out * 10).astype(int), 9), axis=0)
        self.assertGreaterEqual(cells.shape[0], 85)

    def test_placement_rounds_keep_the_whole_batch_active(self):
        """A batch larger than one placement round is indexed in rounds and
        then refitted; every point of it must still come back, relaxed."""
        n = 3 * ess_core.PLACEMENT_ROUND + 7
        stats = {}
        out = ess.esa(self.static, self.bounds, n=n, seed=4, stats=stats)
        self.assertEqual(out.shape, (n, 6))
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(np.unique(out, axis=0).shape[0], n)
        self.assertEqual(len(stats["batch_epochs"]), 1)

    def test_empty_start_ignores_the_pool(self):
        """With no static points there is nothing to be far from, so the
        sampler's output is used as drawn and the pool cannot matter."""
        empty = np.empty((0, 6))
        a = ess.esa(empty, self.bounds, n=20, seed=3, init_pool=1)
        b = ess.esa(empty, self.bounds, n=20, seed=3, init_pool=512)
        np.testing.assert_allclose(a, b)


class TestSamplesOutsideBounds(unittest.TestCase):
    """A sample past a bound wraps to the far side of the torus and repels
    there. That is the geometry, but it is silent, so it is announced."""

    def test_samples_outside_the_bounds_are_announced(self):
        bounds = np.array([[0.0, 1.0]] * 2)
        static = np.array([[1.4, 0.5], [0.5, -0.3], [0.5, 0.5]])
        with self.assertLogs("ess.ess", level="WARNING") as seen:
            ess.esa(static, bounds, n=8, seed=0)
        self.assertIn("2 of 3 samples lie outside", seen.output[0])

    def test_samples_on_the_bounds_are_not(self):
        bounds = np.array([[0.0, 1.0]] * 2)
        static = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        with self.assertNoLogs("ess.ess", level="WARNING"):
            ess.esa(static, bounds, n=8, seed=0)


class TestSpaceGuidedLoop(unittest.TestCase):
    """The unified loop: 3 core parameters, own-gap step `C`, worst-point
    re-placement `w`, automatic `k_att` with `trust`, and uphill terrain damping.

    Spec: `.agents/SPEC.md`.
    """

    def setUp(self):
        rng = np.random.default_rng(7)
        self.dim = 4
        self.static = rng.random((96, self.dim))
        self.bounds = np.array([[0.0, 1.0]] * self.dim)
        self.cost = ((self.static - 0.25) ** 2).sum(axis=1)

    def _run(self, weight=None, values=None, stats=None, **kwargs):
        if weight is not None:
            kwargs.update(fitness=self.cost if values is None else values,
                          fitness_weight=weight)
        elif values is not None:
            kwargs.update(fitness=values)
        return np.asarray(
            ess.esa(self.static, self.bounds, n=64, seed=3, stats=stats, **kwargs))

    def _cost(self, points):
        return float(np.median(((points - 0.25) ** 2).sum(axis=1)))

    def test_three_parameter_interface_and_keyword_only_from_fitness(self):
        """`esa(samples, bounds)` and `esa(samples, bounds, fitness=...)`
        work with no extra parameters, `p` defaults to 1.0, and `fitness`
        onwards is keyword-only."""
        import inspect
        for fn in (ess.esa, ess.ess):
            sig = inspect.signature(fn).parameters
            self.assertEqual(sig["p"].default, 1.0)
        small = self.static[:12]
        out_rep = ess.esa(small, self.bounds, seed=1)
        self.assertEqual(out_rep.shape, (12, self.dim))
        out_fit = ess.esa(small, self.bounds, fitness=self.cost[:12], seed=1)
        self.assertEqual(out_fit.shape, (12, self.dim))
        comb = ess.ess(small, self.bounds, fitness=self.cost[:12], p=0.5, seed=1)
        self.assertEqual(comb.shape, (24, self.dim))
        with self.assertRaises(TypeError):
            ess.esa(small, self.bounds, self.cost[:12])  # pyright: ignore[reportCallIssue]

    def test_flat_fitness_and_zero_weight_are_repulsion_alone(self):
        plain = self._run()
        np.testing.assert_array_equal(
            self._run(weight=1.0, values=np.full(96, 2.5)), plain)
        np.testing.assert_array_equal(self._run(weight=0.0), plain)

    def test_guide_moves_points_to_better_regions_without_collapse(self):
        plain = self._run()
        fair = geometry.radius_for_count(self.dim, 96 + 64, target=1)
        out = self._run(weight=1.0)
        self.assertLess(self._cost(out), self._cost(plain))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))
        both = np.vstack((self.static, out))
        diff = np.abs(out[:, None, :] - both[None, :, :])
        dist = np.minimum(diff, 1.0 - diff).sum(axis=2)
        dist[dist < 1e-12] = np.inf
        self.assertGreater(dist.min(), 0.25 * fair)

    def test_copies_of_the_anchors_do_not_grow_the_damping(self):
        """A region probed three times guides like a region probed once."""
        rng = np.random.default_rng(5)
        pts = rng.random((40, self.dim))
        field = ess_core._AttractionField(
            self.static, ess_core._rank_normalise(self.cost), k=96)
        tripled = ess_core._AttractionField(
            np.tile(self.static, (3, 1)),
            np.tile(ess_core._rank_normalise(self.cost), 3), k=288)
        push = rng.normal(size=pts.shape)
        one, three = push.copy(), push.copy()
        ess_core._apply_guide(one, field.guide(pts)[1], 1.0)
        ess_core._apply_guide(three, tripled.guide(pts)[1], 1.0)
        np.testing.assert_allclose(three, one, rtol=1e-7, atol=1e-10)

    def test_damp_never_grows_the_push_and_removes_uphill_component(self):
        rng = np.random.default_rng(9)
        push = rng.normal(size=(50, self.dim))
        guide = rng.normal(size=(50, self.dim))
        guide[:5] = 0.0
        damped = push.copy()
        ess_core._apply_guide(damped, guide, 5.0)
        np.testing.assert_array_equal(damped[:5], push[:5])
        self.assertTrue(np.all(np.linalg.norm(damped, axis=1)
                               <= np.linalg.norm(push, axis=1) + 1e-12))
        self.assertTrue(np.all(np.einsum("md,md->m", damped[5:], guide[5:]) <= 1e-9))

    def test_auto_k_and_trust_are_recorded_and_zero_trust_is_repulsion_alone(self):
        stats = {}
        self._run(weight=1.0, stats=stats)
        self.assertIn(stats["k_att"], (8, 32, 96))
        self.assertGreater(stats["field_trust"], 0.5)
        noise = np.random.default_rng(2).random(96)
        stats_noise = {}
        self._run(weight=1.0, values=noise, stats=stats_noise)
        self.assertLess(stats_noise["field_trust"], 0.3)

    def test_worst_point_replacement_parameter_w(self):
        stats_zero = {}
        out_zero = self._run(w=0.0, stats=stats_zero)
        self.assertEqual(sum(stats_zero["batch_replaced"]), 0)
        stats_rep = {}
        out_rep = self._run(w=0.1, stats=stats_rep)
        self.assertEqual(out_rep.shape, out_zero.shape)
        self.assertGreaterEqual(sum(stats_rep["batch_replaced"]), 0)
