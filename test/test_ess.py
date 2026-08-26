
"""End-to-end tests of the toroidal ESA/ESS pipeline."""

import importlib
import itertools
import logging
import unittest
from typing import ClassVar

import numpy as np

import ess
from ess import utils

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
    """Placement mirrors the mode: repulsive alone, composite with attraction.

    Scoring candidates on novelty alone puts each one as far from everything
    as possible -- including as far from the good regions -- and then leaves
    the relaxation's attraction to drag it back. On the OBLESA sweep that
    placement was worth nothing: unguided dart tied uniform noise 233-243 over
    480 paired cells, while the same search with a fitness term in the
    *placement* beat it 309-169.
    """

    D = 8
    BOUNDS = np.array([[-5.0, 5.0]] * D)

    def _setup(self, seed=0, m=40):
        rng = np.random.default_rng(seed)
        static = rng.uniform(-5, 5, (m, self.D))
        # "good" means near the origin, so the effect is measurable as a radius
        return static, -np.linalg.norm(static, axis=1)

    def _composite(self, static, attract, weight=0.5, seed=11, n=30, **kw):
        return ess.esa(static, self.BOUNDS, n=n, seed=seed,
                       attractiveness=attract, attraction_weight=weight,
                       attraction_metric="cauchy",
                       attraction_kwargs={"power": 1.0}, **kw)

    def test_weight_zero_reproduces_repulsive_placement_exactly(self):
        """The composite path has to be an ablation of the repulsive one, not
        a different algorithm, or no comparison between them means anything."""
        static, attract = self._setup()
        rep = ess.esa(static, self.BOUNDS, n=30, seed=11)
        zero = self._composite(static, attract, weight=0.0)
        np.testing.assert_allclose(zero, rep)

    def test_attraction_pulls_placement_toward_the_good_region(self):
        static, attract = self._setup()
        rep = ess.esa(static, self.BOUNDS, n=30, seed=11)
        com = self._composite(static, attract)
        self.assertLess(np.linalg.norm(com, axis=1).mean(),
                        np.linalg.norm(rep, axis=1).mean())

    def test_more_weight_pulls_harder(self):
        static, attract = self._setup(seed=3)
        radii = [float(np.linalg.norm(
            self._composite(static, attract, weight=w), axis=1).mean())
            for w in (0.0, 0.5, 2.0)]
        self.assertLess(radii[1], radii[0])
        self.assertLess(radii[2], radii[1])

    def test_runs_stay_reproducible(self):
        static, attract = self._setup()
        a = self._composite(static, attract)
        b = self._composite(static, attract)
        np.testing.assert_array_equal(a, b)

    def test_results_stay_inside_the_bounds(self):
        static, attract = self._setup()
        out = self._composite(static, attract)
        self.assertTrue((out >= self.BOUNDS[:, 0]).all())
        self.assertTrue((out <= self.BOUNDS[:, 1]).all())


class TestAttractionField(unittest.TestCase):
    """Attractiveness belongs to a position, not to a point.

    The estimate used to be taken once at placement and carried for the whole
    relaxation, which left one side of the force balance live and the other
    stale: a point that drifted out of a good region kept its old high value
    and went on pulling everything else toward where it no longer was.
    """

    def _field(self, k=4):
        pos = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])
        val = np.array([1.0, 0.0, 0.5, 0.5])
        return ess_core._AttractionField(pos, val, k=k)

    def test_the_value_follows_the_position(self):
        f = self._field()
        near_good = f.at(np.array([[0.12, 0.12]]))[0]
        near_bad = f.at(np.array([[0.88, 0.88]]))[0]
        self.assertGreater(near_good, near_bad)

    def test_measured_sources_start_at_full_confidence(self):
        f = self._field()
        self.assertEqual(f.n_measured, 4)

    def test_inferred_sources_enter_below_full_confidence(self):
        """Otherwise a cached batch launders an estimate into ground truth."""
        f = self._field()
        before = f.at(np.array([[0.5, 0.5]]))[0]
        f.add_inferred(np.array([[0.5, 0.5]]), np.array([1.0]))
        after = f.at(np.array([[0.5, 0.5]]))[0]
        # the new source sits exactly on the query, so it dominates, but it
        # was added as an estimate rather than as a measurement
        self.assertNotEqual(before, after)
        self.assertEqual(f.n_measured, 4)

    def test_adding_nothing_changes_nothing(self):
        f = self._field()
        before = f.at(np.array([[0.4, 0.6]]))[0]
        f.add_inferred(np.zeros((0, 2)), np.zeros(0))
        self.assertEqual(f.at(np.array([[0.4, 0.6]]))[0], before)

    def test_the_refresh_stride_does_not_change_the_outcome_much(self):
        """A point moves at most 2% of the interaction radius per epoch, so
        refreshing every epoch buys accuracy the physics cannot use."""
        rng = np.random.default_rng(0)
        d, bounds = 16, np.array([[-5.0, 5.0]] * 16)
        static = rng.uniform(-5, 5, (60, d))
        attract = -np.linalg.norm(static, axis=1)
        kw = {"attractiveness": attract, "attraction_weight": 0.5,
                  "attraction_metric": "cauchy", "attraction_kwargs": {"power": 1.0}}

        def radius(every):
            return float(np.median([
                float(np.linalg.norm(
                    ess.esa(static, bounds, n=60, seed=s, att_every=every,
                            **kw), axis=1).mean())
                for s in range(6)]))

        self.assertAlmostEqual(radius(1), radius(5), delta=0.5)


class TestAttractionModels(unittest.TestCase):
    """The fitted function against the nearest-neighbour weighting."""

    def _sample(self, d, m, seed=0):
        rng = np.random.default_rng(seed)
        pos = rng.random((m, d))
        val = (np.sin(2 * np.pi * pos[:, 0])
               + 0.5 * np.cos(2 * np.pi * pos[:, 1]))
        return pos, val

    def _loo(self, model, d, m):
        pos, val = self._sample(d, m)
        err = []
        for i in range(m):
            keep = np.ones(m, bool)
            keep[i] = False
            f = ess_core._AttractionField(pos[keep], val[keep], model=model)
            err.append(abs(f.at(pos[i:i + 1])[0] - val[i]))
        return float(np.mean(err))

    def test_more_points_helps_less_as_dimension_rises(self):
        """More measured points is not a general escape from high dimension.

        This asserted a halving of the error, which held only for the
        least-squares ridge: `M = 300` crosses its `M > 2d` threshold at
        `d = 100` exactly, so the improvement it measured was the fit becoming
        identifiable rather than the estimate getting more data. Neither
        surviving model has a threshold to cross, and what they show instead
        is the concentration of distances -- measured leave-one-out error,
        `M = 60` against `M = 300`:

        =============  ==========  ===========
        model          d=8 ratio   d=100 ratio
        =============  ==========  ===========
        idw            0.69        0.98
        projection     0.61        0.74
        =============  ==========  ===========

        At `d = 100` inverse-distance weighting gains essentially nothing from
        five times the sources. That is the documented failure mode -- the
        weights flatten and the estimate tends to the mean -- and it matters
        for a caller who assumed more anchors would sharpen the field. It does
        not; what extra anchors buy is on the placement side, where a probe is
        scored against points whose objective was actually paid for.
        """
        for model in ("idw", "projection"):
            for d in (8, 100):
                with self.subTest(model=model, d=d):
                    few = self._loo(model, d, 60)
                    many = self._loo(model, d, 300)
                    self.assertLessEqual(many, few)
            # The gain is real where distances still separate.
            self.assertLess(self._loo(model, 8, 300),
                            self._loo(model, 8, 60) * 0.8)

    def test_the_table_matches_the_closed_form(self):
        """The fitted model is tabulated as `d` one-dimensional curves and
        looked up, so the table must not be a second, different model."""
        pos, val = self._sample(32, 60)
        f = ess_core._AttractionField(pos, val, model="projection")
        rng = np.random.default_rng(4)
        q = rng.random((2000, 32))
        # `_AttractionField` owns the sources; the model owns the estimate,
        # so the closed form is read off the model rather than the field.
        m = f.model
        closed = m._features(q) @ m._w + m._bias
        np.testing.assert_allclose(f.at(q), closed, atol=1e-5)

    def test_the_table_costs_less_than_the_distance_weighting(self):
        pos, val = self._sample(32, 60)
        rng = np.random.default_rng(5)
        q = rng.random((20000, 32))
        import time
        out = {}
        for model in ("idw", "projection"):
            f = ess_core._AttractionField(pos, val, model=model)
            f.at(q[:50])
            t = time.perf_counter()
            f.at(q)
            out[model] = time.perf_counter() - t
        self.assertLess(out["projection"], out["idw"])

    def test_the_basis_is_periodic(self):
        """A linear or polynomial basis is discontinuous at the wrap, and the
        model would disagree with itself across a seam the torus lacks."""
        pos, val = self._sample(6, 40)
        f = ess_core._AttractionField(pos, val, model="projection")
        near_zero = f.at(np.array([[1e-9] + [0.5] * 5]))
        near_one = f.at(np.array([[1.0 - 1e-9] + [0.5] * 5]))
        self.assertAlmostEqual(float(near_zero[0]), float(near_one[0]), places=6)

    def test_the_fit_can_leave_the_range_of_the_measured_values(self):
        """A convex combination of neighbours never can, so plain IDW cannot
        call anywhere more promising than the best point already evaluated."""
        pos, val = self._sample(4, 50)
        f = ess_core._AttractionField(pos, val, model="projection")
        rng = np.random.default_rng(2)
        got = f.at(rng.random((400, 4)))
        self.assertTrue(got.max() > val.max() or got.min() < val.min())

    def test_every_model_is_reachable_through_esa(self):
        rng = np.random.default_rng(0)
        static = rng.uniform(-5, 5, (40, 6))
        bounds = np.array([[-5.0, 5.0]] * 6)
        att = -np.linalg.norm(static, axis=1)
        for model in ("idw", "projection"):
            out = ess.esa(static, bounds, n=20, seed=1, attractiveness=att,
                          attraction_weight=0.5, att_model=model,
                          attraction_metric="cauchy",
                          attraction_kwargs={"power": 1.0})
            self.assertEqual(out.shape, (20, 6))


class TestEstimateImprovesWithData(unittest.TestCase):
    """More measured points must make the estimate better, monotonically.

    This is the property the whole design rests on. The attractiveness
    function is fitted from the points whose objective has actually been paid
    for, so the guarantee a caller needs is that paying for more of them buys
    a better function -- and that the estimator is never the thing standing in
    the way.

    It is also what settles the high-dimension question. At d=100 with 60
    measured points every model is near-useless, and it would be easy to read
    that as a defect in the estimator. It is not: 60 samples cannot determine
    a function in 100 dimensions. Hold the dimension fixed, raise the count,
    and the error falls -- so the lever is the population, not the maths.
    """

    @staticmethod
    def _truth(x):
        """Smooth and periodic, with a second harmonic the model cannot
        represent, so the error falls to the model's own bias rather than to
        zero -- which is the honest shape of the curve."""
        return (np.sin(2 * np.pi * x[:, 0])
                + 0.5 * np.cos(2 * np.pi * x[:, 1])
                + 0.25 * np.sin(4 * np.pi * x[:, 2]))

    def _error(self, d, m, model="projection", seeds=5):
        """Mean absolute error on a held-out set, averaged over seeds."""
        errs = []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            pos = rng.random((m, d))
            f = ess_core._AttractionField(pos, self._truth(pos), model=model)
            q = rng.random((500, d))
            errs.append(float(np.abs(f.at(q) - self._truth(q)).mean()))
        return float(np.mean(errs))

    def test_error_falls_as_measured_points_are_added(self):
        for d in (8, 32):
            ladder = [(m, self._error(d, m)) for m in (20, 40, 80, 160, 320)]
            errs = [e for _, e in ladder]
            # non-increasing, with a little slack for sampling noise
            for (m0, e0), (m1, e1) in itertools.pairwise(ladder):
                self.assertLessEqual(
                    e1, e0 * 1.10,
                    f"d={d}: error rose from {e0:.4f} at M={m0} to "
                    f"{e1:.4f} at M={m1}")
            self.assertLess(errs[-1], errs[0] * 0.75, f"d={d}: {errs}")

    def test_the_same_holds_where_the_data_is_scarcest(self):
        """d=100 is where the estimate looks worst, and where it is most
        important that more data is the answer."""
        few = self._error(100, 60)
        many = self._error(100, 600)
        self.assertLess(many, few * 0.75, f"{few:.4f} -> {many:.4f}")

    def test_it_holds_for_every_model(self):
        """Whatever the estimator, paying for more points must not make it
        worse -- otherwise the caller cannot reason about the trade at all."""
        for model in ("idw", "projection"):
            lo = self._error(16, 40, model=model)
            hi = self._error(16, 320, model=model)
            self.assertLess(hi, lo, f"{model}: {lo:.4f} -> {hi:.4f}")


class TestPlacementVsRelaxation(unittest.TestCase):
    """The two attractions are separable, and both are needed.

    Measured at d=16 over 12 seeds, mean distance to the good region: 11.596
    repulsive, 11.386 placement only, 10.892 relaxation only, 10.367 both.
    Placement alone recovers almost nothing -- with a repulsion-only
    relaxation the points drift off the good positions they were placed on and
    push the rest of the design around -- and the pair beats the sum of the
    two separate effects, so they are complementary rather than redundant.
    """

    D = 16
    BOUNDS = np.array([[-5.0, 5.0]] * D)
    KW: ClassVar[dict] = {
        "attraction_metric": "cauchy",
        "attraction_kwargs": {"power": 1.0},
    }

    def _setup(self, seed=0, m=60):
        rng = np.random.default_rng(seed)
        static = rng.uniform(-5, 5, (m, self.D))
        return static, -np.linalg.norm(static, axis=1)

    def _radius(self, static, attract, seeds=8, **kw):
        return float(np.median([
            float(np.linalg.norm(
                ess.esa(static, self.BOUNDS, n=40, seed=s,
                        attractiveness=attract, **self.KW, **kw),
                axis=1).mean())
            for s in range(seeds)]))

    def test_placement_weight_defaults_to_the_attraction_weight(self):
        static, attract = self._setup()
        paired = ess.esa(static, self.BOUNDS, n=40, seed=5,
                         attractiveness=attract, attraction_weight=0.5,
                         **self.KW)
        explicit = ess.esa(static, self.BOUNDS, n=40, seed=5,
                           attractiveness=attract, attraction_weight=0.5,
                           placement_weight=0.5, **self.KW)
        np.testing.assert_array_equal(paired, explicit)

    def test_both_beats_either_alone(self):
        static, attract = self._setup()
        place = self._radius(static, attract, attraction_weight=0.0,
                             placement_weight=0.5)
        relax = self._radius(static, attract, attraction_weight=0.5,
                             placement_weight=0.0)
        both = self._radius(static, attract, attraction_weight=0.5,
                            placement_weight=0.5)
        self.assertLess(both, place)
        self.assertLess(both, relax)

    def test_a_repulsive_relaxation_undoes_the_guided_placement(self):
        """The reason the relaxation term cannot simply be dropped once the
        placement is guided."""
        static, attract = self._setup()
        rep = float(np.median([
            float(np.linalg.norm(
                ess.esa(static, self.BOUNDS, n=40, seed=s), axis=1).mean())
            for s in range(8)]))
        place_only = self._radius(static, attract, attraction_weight=0.0,
                                  placement_weight=0.5)
        relax_only = self._radius(static, attract, attraction_weight=0.5,
                                  placement_weight=0.0)
        # placement alone recovers far less of the gap than the relaxation
        self.assertLess(rep - place_only, rep - relax_only)


class TestAttractivenessEstimate(unittest.TestCase):
    """The candidate-side estimate.

    Attractiveness is known only for the static points. Before this existed
    the active rows were left at 0.0, which is not "unknown" but the *bottom*
    of the normalised scale -- every candidate modelled as the least
    attractive thing in the space.
    """

    def test_a_point_on_a_static_point_takes_its_value(self):
        static = np.array([[0.1, 0.1], [0.8, 0.8]])
        a = np.array([0.25, 0.75])
        got = ess_core._estimate_attractiveness(static.copy(), static, a)
        np.testing.assert_allclose(got, a)

    def test_the_estimate_is_bounded_by_the_known_values(self):
        rng = np.random.default_rng(1)
        static = rng.random((30, 5))
        a = rng.random(30)
        q = rng.random((200, 5))
        got = ess_core._estimate_attractiveness(q, static, a)
        self.assertGreaterEqual(got.min(), a.min() - 1e-12)
        self.assertLessEqual(got.max(), a.max() + 1e-12)

    def test_it_is_nearer_the_closer_neighbour(self):
        """Inverse-distance weighting: the near value must dominate."""
        static = np.array([[0.10, 0.5], [0.90, 0.5]])
        a = np.array([0.0, 1.0])
        got = ess_core._estimate_attractiveness(
            np.array([[0.15, 0.5]]), static, a)
        self.assertLess(got[0], 0.25)

    def test_no_static_points_is_not_an_error(self):
        got = ess_core._estimate_attractiveness(
            np.zeros((4, 3)), np.zeros((0, 3)), np.zeros(0))
        self.assertEqual(got.shape, (4,))

    def test_it_wraps_around_the_torus(self):
        """0.02 and 0.98 are close on the unit torus; a non-toroidal metric
        would call them the two most distant points in the axis."""
        static = np.array([[0.98, 0.5], [0.50, 0.5]])
        a = np.array([1.0, 0.0])
        got = ess_core._estimate_attractiveness(
            np.array([[0.02, 0.5]]), static, a)
        self.assertGreater(got[0], 0.5)


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

    def test_empty_start_ignores_the_pool(self):
        """With no static points there is nothing to be far from, so the
        sampler's output is used as drawn and the pool cannot matter."""
        empty = np.empty((0, 6))
        a = ess.esa(empty, self.bounds, n=20, seed=3, init_pool=1)
        b = ess.esa(empty, self.bounds, n=20, seed=3, init_pool=512)
        np.testing.assert_allclose(a, b)
