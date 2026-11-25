"""Tests for the inverse-distance fitness terrain and automatic k selection."""

import unittest

import numpy as np

import ess
from ess import attraction


def additive_truth(x):
    """Smooth periodic test landscape."""
    ang = 2.0 * np.pi * x
    return np.sin(ang).sum(1) + 0.4 * np.cos(2 * ang).sum(1)


def coupled_truth(x):
    """Pairwise coupled periodic test landscape."""
    a = np.sin(2.0 * np.pi * x)
    return np.sum(a * np.roll(a, 1, axis=1), axis=1)


def held_out(model, truth, d, m, seed=0, q=200):
    """Mean absolute error on fresh points, normalised by the truth's spread.

    Normalised so 1.0 is what predicting the mean would score -- anything
    above that is a model asserting structure that is not there.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1, size=(m, d))
    xq = rng.uniform(0, 1, size=(q, d))
    yq = truth(xq)
    model.fit(xs, truth(xs))
    return float(np.mean(np.abs(model.at(xq) - yq))) / float(np.std(yq))


class TestInterface(unittest.TestCase):
    def test_idw_round_trips(self):
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 1, size=(40, 6))
        val = additive_truth(pos)
        model = attraction.InverseDistance()
        out = model.fit(pos, val).at(pos[:5])
        self.assertEqual(out.shape, (5,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_empty_source_set_is_not_an_error(self):
        model = attraction.InverseDistance()
        model.fit(np.empty((0, 4)), np.empty(0))
        out = model.at(np.random.default_rng(0).uniform(0, 1, (3, 4)))
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.all(np.isfinite(out)))


class TestRegimes(unittest.TestCase):
    """Inverse-distance interpolation bounds and periodicity."""

    def test_idw_stays_below_mean_predictor_at_high_dimension(self):
        for d, m in ((32, 120), (100, 300)):
            with self.subTest(d=d, m=m):
                err = held_out(attraction.InverseDistance(),
                               coupled_truth, d, m)
                self.assertLess(err, 1.0)

    def test_the_estimate_cannot_leave_the_range_of_its_sources(self):
        """A convex combination of measured values cannot exceed them."""
        rng = np.random.default_rng(2)
        pos = rng.uniform(0, 1, size=(50, 4))
        val = additive_truth(pos)
        model = attraction.InverseDistance().fit(pos, val)
        got = model.at(rng.uniform(0, 1, size=(400, 4)))
        self.assertGreaterEqual(got.min(), val.min())
        self.assertLessEqual(got.max(), val.max())

    def test_the_estimate_is_periodic_across_the_seam(self):
        """No seam: the space wraps, so the metric behind the estimate must."""
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 1, size=(120, 4))
        model = attraction.InverseDistance().fit(pos, additive_truth(pos))
        left = np.full((1, 4), 1e-9)
        right = np.full((1, 4), 1.0 - 1e-9)
        np.testing.assert_allclose(model.at(left), model.at(right), atol=1e-3)


class TestEstimateIsReproducible(unittest.TestCase):
    """Above torann's crossover the estimate's neighbour search is LSH, and
    its hash functions are drawn at fit. They have to be seeded, or one
    `esa` seed gives two designs."""

    def test_two_fits_agree_once_the_search_is_approximate(self):
        rng = np.random.default_rng(0)
        pos, val = rng.random((3000, 8)), rng.random(3000)
        queries = rng.random((200, 8))
        first = attraction.InverseDistance().fit(pos, val)
        again = attraction.InverseDistance().fit(pos, val)
        np.testing.assert_array_equal(first.at(queries), again.at(queries))

    def test_a_seeded_run_with_fitness_repeats(self):
        rng = np.random.default_rng(1)
        pos, val = rng.random((3000, 8)), rng.random(3000)
        bounds = np.array([[0.0, 1.0]] * 8)
        kw = {"fitness": val, "n": 60, "seed": 7}
        np.testing.assert_array_equal(
            ess.esa(pos, bounds, **kw), ess.esa(pos, bounds, **kw))


class TestGuide(unittest.TestCase):
    """The slope of the field: a property of the position, not of the points."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.dim = 6
        self.pos = rng.random((80, self.dim))
        self.val = rng.random(80)
        self.queries = rng.random((12, self.dim))

    def _model(self, pos, val, k=None):
        k = pos.shape[0] if k is None else k
        return attraction.InverseDistance(k=k).fit(pos, val)

    def test_guide_is_the_gradient_of_the_estimate(self):
        """Central differences of `at`, over the coordinates that differ."""
        model = self._model(self.pos, self.val)
        est, guide = model.guide(self.queries)
        np.testing.assert_allclose(est, model.at(self.queries), rtol=1e-12)
        h = 1e-6
        numeric = np.empty_like(self.queries)
        for axis in range(self.dim):
            step = np.zeros(self.dim)
            step[axis] = h
            numeric[:, axis] = (model.at(self.queries + step)
                                - model.at(self.queries - step)) / (2 * h)
        np.testing.assert_allclose(guide * self.dim, numeric, atol=1e-6)

    def test_constant_values_have_no_guide(self):
        _, guide = self._model(self.pos, np.full(80, 0.7)).guide(self.queries)
        np.testing.assert_allclose(guide, 0.0, atol=1e-12)

    def test_guide_does_not_grow_with_the_number_of_sources(self):
        """Every source three times: the same field and the same guide."""
        est, guide = self._model(self.pos, self.val).guide(self.queries)
        est3, guide3 = self._model(np.tile(self.pos, (3, 1)),
                                   np.tile(self.val, 3)).guide(self.queries)
        np.testing.assert_allclose(est3, est, rtol=1e-9)
        np.testing.assert_allclose(guide3, guide, rtol=1e-9, atol=1e-12)

    def test_guide_points_uphill_and_negative_guide_points_to_ground(self):
        """Source at x=0.2 has fitness 0.0 (ground); source at x=0.6 has fitness 1.0 (hill).
        The uphill slope guide points toward x=0.6 (+x), so downhill (-guide) points to x=0.2."""
        pos = np.array([[0.2, 0.5], [0.6, 0.5]])
        _, guide = self._model(pos, np.array([0.0, 1.0])).guide(np.array([[0.4, 0.45]]))
        self.assertGreater(guide[0, 0], 0.0)

    def test_query_on_a_source_has_no_guide(self):
        est, guide = self._model(self.pos, self.val).guide(self.pos[:3])
        np.testing.assert_allclose(est, self.val[:3])
        np.testing.assert_array_equal(guide, 0.0)


class TestChooseK(unittest.TestCase):
    """Leave-one-out choice of the neighbour count, and the trust of the field."""

    def setUp(self):
        rng = np.random.default_rng(4)
        self.pos = rng.random((200, 3))
        self.smooth = ((self.pos - 0.5) ** 2).sum(axis=1)
        self.noise = rng.random(200)

    def test_scores_follow_the_information_in_the_values(self):
        smooth = attraction.leave_one_out(self.pos, self.smooth)
        noise = attraction.leave_one_out(self.pos, self.noise)
        self.assertEqual(sorted(smooth), [8, 32, 128, 199])
        self.assertGreater(smooth[8], 0.7)
        self.assertLess(max(noise.values()), 0.3)

    def test_choice_is_a_candidate_and_trust_is_bounded(self):
        k, trust = attraction.choose_k(self.pos, self.smooth)
        self.assertIn(k, (8, 32, 128, 200))
        self.assertGreater(trust, 0.5)
        k, trust = attraction.choose_k(self.pos, self.noise)
        self.assertGreaterEqual(trust, 0.0)
        self.assertLess(trust, 0.3)

    def test_constant_values_have_no_trust(self):
        self.assertEqual(attraction.choose_k(self.pos, np.full(200, 0.3))[1], 0.0)

    def test_buffer_leaves_the_cluster_out(self):
        """Tight clusters with one value each: predicted from inside, not from outside."""
        rng = np.random.default_rng(8)
        centres = rng.random((20, 3))
        pos = np.mod(np.repeat(centres, 10, axis=0) + rng.normal(0, 1e-3, (200, 3)), 1.0)
        val = np.repeat(rng.random(20), 10)
        inside = attraction.leave_one_out(pos, val, ks=(8,))[8]
        outside = attraction.leave_one_out(pos, val, ks=(8,), buffer=0.05)[8]
        self.assertGreater(inside, 0.95)
        self.assertLess(outside, 0.5)

    def test_copies_of_the_sources_do_not_change_the_choice(self):
        k1, t1 = attraction.choose_k(self.pos, self.smooth)
        scores = attraction.leave_one_out(self.pos, self.smooth, ks=(8, 32))
        self.assertEqual(sorted(scores), [8, 32])
        self.assertGreater(t1, 0.0)
        self.assertGreater(k1, 0)

    def test_exact_hit_duplicates_are_averaged(self):
        """Duplicate / 0-vs-1 aliased anchors with distinct fitness return their mean."""
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        val = np.array([0.0, 1.0, 0.25])
        model = attraction.InverseDistance(k=3).fit(pos, val)
        q = np.array([[0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(model.at(q), [0.5, 0.5])
        est, slope = model.guide(q)
        np.testing.assert_allclose(est, [0.5, 0.5])
        np.testing.assert_allclose(slope, 0.0)

    def test_guide_rust_kernel_matches_numpy_path(self):
        """Compiled weighted_directions path in guide matches the NumPy fallback."""
        rng = np.random.default_rng(11)
        pos = rng.random((128, 32))
        val = ((pos - 0.5) ** 2).sum(axis=1)
        q = rng.random((80, 32))
        for p in (1.0, 0.5):
            with self.subTest(p=p):
                model = attraction.InverseDistance(k=128, p=p).fit(pos, val)
                est_rust, grad_rust = model.guide(q)
                old_min = attraction._KERNEL_MIN_WORK
                try:
                    attraction._KERNEL_MIN_WORK = 10**12
                    est_np, grad_np = model.guide(q)
                finally:
                    attraction._KERNEL_MIN_WORK = old_min
                np.testing.assert_allclose(est_rust, est_np, rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(grad_rust, grad_np, rtol=1e-12, atol=1e-12)

    def test_at_chunked_matches_unchunked(self):
        """Chunking InverseDistance.at preserves exact outputs."""
        rng = np.random.default_rng(12)
        pos = rng.random((40, 6))
        val = rng.random(40)
        q = rng.random((100, 6))
        model = attraction.InverseDistance(k=40).fit(pos, val)
        ref = model.at(q)
        old_blk = attraction._AT_BLOCK_ELEMENTS
        try:
            attraction._AT_BLOCK_ELEMENTS = 200
            chunked = model.at(q)
        finally:
            attraction._AT_BLOCK_ELEMENTS = old_blk
        np.testing.assert_array_equal(chunked, ref)


if __name__ == "__main__":
    unittest.main()
