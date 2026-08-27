
"""Tests for the attractiveness models."""

import unittest

import numpy as np

import ess
from ess import attraction


def additive_truth(x):
    """A truth the additive basis can represent exactly."""
    ang = 2.0 * np.pi * x
    return np.sin(ang).sum(1) + 0.4 * np.cos(2 * ang).sum(1)


def coupled_truth(x):
    """A truth it cannot: pairwise products have no additive decomposition."""
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
    model.fit(xs, truth(xs), np.ones(m))
    return float(np.mean(np.abs(model.at(xq) - yq))) / float(np.std(yq))


class TestInterface(unittest.TestCase):
    def test_every_registered_model_round_trips(self):
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 1, size=(40, 6))
        val = additive_truth(pos)
        for name in attraction.MODELS:
            with self.subTest(model=name):
                model = attraction.get_model(name)
                self.assertIsInstance(model, attraction.AttractionModel)
                out = model.fit(pos, val, np.ones(40)).at(pos[:5])
                self.assertEqual(out.shape, (5,))
                self.assertTrue(np.all(np.isfinite(out)))

    def test_get_model_accepts_name_class_and_instance(self):
        built = attraction.InverseDistance(k=3)
        self.assertIs(attraction.get_model(built), built)
        self.assertIsInstance(attraction.get_model("idw"),
                              attraction.InverseDistance)
        self.assertIsInstance(
            attraction.get_model(attraction.InverseDistance),
            attraction.InverseDistance)

    def test_get_model_forwards_only_what_the_class_takes(self):
        """`_AttractionField` passes every knob; each model takes its own."""
        idw = attraction.get_model("idw", k=3, power=1.5, nonsense=8)
        self.assertEqual((idw.k, idw.power), (3, 1.5))

    def test_get_model_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            attraction.get_model("no-such-model")
        with self.assertRaises(TypeError):
            attraction.get_model(42)

    def test_empty_source_set_is_not_an_error(self):
        for name in ("idw",):
            with self.subTest(model=name):
                model = attraction.get_model(name)
                model.fit(np.empty((0, 4)), np.empty(0), np.empty(0))
                out = model.at(np.random.default_rng(0).uniform(0, 1, (3, 4)))
                self.assertEqual(out.shape, (3,))
                self.assertTrue(np.all(np.isfinite(out)))


class TestCustomModel(unittest.TestCase):
    class Constant(attraction.AttractionModel):
        def fit(self, positions, values, confidence):
            self.value = float(np.max(values))
            return self

        def at(self, positions):
            return np.full(len(positions), self.value)

    def test_a_user_model_drives_a_real_run(self):
        rng = np.random.default_rng(0)
        d = 6
        samples = rng.uniform(0, 1, size=(30, d))
        bounds = np.array([[0.0, 1.0]] * d)
        placed = ess.esa(
            samples, bounds, n=8,
            attractiveness=-np.sum((samples - 0.3) ** 2, axis=1),
            attraction_weight=0.5, attraction_metric="cauchy",
            attraction_kwargs={"power": 1.0},
            att_model=self.Constant(), seed=1,
        )
        self.assertEqual(placed.shape, (8, d))
        self.assertTrue(np.all(np.isfinite(placed)))


class TestRegimes(unittest.TestCase):
    """Which model to trust is a question about the data, not a preference."""

    def test_the_hedging_models_never_do(self):
        for name in ("idw",):
            for d, m in ((32, 120), (100, 300)):
                with self.subTest(model=name, d=d, m=m):
                    err = held_out(attraction.get_model(name),
                                   coupled_truth, d, m)
                    self.assertLess(err, 1.0)

    def test_the_estimate_cannot_leave_the_range_of_its_sources(self):
        """The guarantee the whole default rests on.

        A convex combination of measured values cannot exceed them, so the
        field can never pull a probe toward a number nobody observed. The
        removed parametric fits could and did -- that was the argument *for*
        them, and it is why they were dangerous when the basis was wrong.
        """
        rng = np.random.default_rng(2)
        pos = rng.uniform(0, 1, size=(50, 4))
        val = additive_truth(pos)
        model = attraction.InverseDistance().fit(pos, val, np.ones(50))
        got = model.at(rng.uniform(0, 1, size=(400, 4)))
        self.assertGreaterEqual(got.min(), val.min())
        self.assertLessEqual(got.max(), val.max())

    def test_the_estimate_is_periodic_across_the_seam(self):
        """No seam: the space wraps, so the metric behind the estimate must."""
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 1, size=(120, 4))
        model = attraction.InverseDistance().fit(
            pos, additive_truth(pos), np.ones(120))
        left = np.full((1, 4), 1e-9)
        right = np.full((1, 4), 1.0 - 1e-9)
        np.testing.assert_allclose(model.at(left), model.at(right), atol=1e-3)


if __name__ == "__main__":
    unittest.main()


class TestRadiusMode(unittest.TestCase):
    """`InverseDistance(search_mode='radius')`.

    The two modes differ in what they hold fixed -- k-NN the neighbour
    *count*, radius the *volume* -- so these assert the consequences of that
    difference rather than that the numbers match.
    """

    def setUp(self):
        rng = np.random.default_rng(4)
        self.pos = rng.random((200, 10))
        self.val = rng.random(200)
        self.conf = np.ones(200)
        self.q = rng.random((50, 10))

    def _fit(self, **kw):
        return attraction.InverseDistance(k=8, **kw).fit(
            self.pos, self.val, self.conf)

    def test_the_default_radius_tracks_the_k_nn_estimate(self):
        """Auto targets `k` neighbours, so the two modes start from the same
        neighbourhood and should agree closely -- if they did not, the
        default radius would be calibrated to something else."""
        a = self._fit().at(self.q)
        b = self._fit(search_mode="radius").at(self.q)
        self.assertGreater(float(np.corrcoef(a, b)[0, 1]), 0.85)

    def test_a_wide_radius_collapses_toward_the_mean(self):
        """Averaging over nearly everything is averaging: the estimate must
        lose its spread, which is the check that the radius is really
        widening the neighbourhood and not being ignored."""
        narrow = self._fit(search_mode="radius").at(self.q)
        wide = self._fit(search_mode="radius", radius=0.95).at(self.q)
        self.assertLess(float(wide.std()), float(narrow.std()))

    def test_an_empty_neighbourhood_falls_back_to_the_mean(self):
        """A radius small enough to find nothing is legal, not an error: the
        honest estimate for a position nothing has measured near is the mean
        of what has been measured."""
        out = self._fit(search_mode="radius", radius=1e-9).at(self.q)
        np.testing.assert_allclose(out, self.val.mean())

    def test_a_query_on_a_source_takes_its_value(self):
        """The exact-hit path is shared, so radius mode must not lose it."""
        out = self._fit(search_mode="radius").at(self.pos[:5])
        np.testing.assert_allclose(out, self.val[:5])

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            attraction.InverseDistance(search_mode="bogus")
        with self.assertRaises(ValueError):
            attraction.InverseDistance(search_mode="radius", radius=1.5).fit(
                self.pos, self.val, self.conf)


class TestRadiusModeThroughEsa(unittest.TestCase):
    """`att_search_mode` reaches the model, and is independent of the
    repulsion's `search_mode`."""

    def _run(self, **kw):
        rng = np.random.default_rng(2)
        bounds = np.tile([0.0, 1.0], (8, 1))
        samples = rng.random((40, 8))
        return ess.esa(samples, bounds, n=12, epochs=40, seed=1,
                       attractiveness=rng.random(40), **kw)

    def test_every_combination_of_the_two_modes_runs(self):
        for repel in ("k_nn", "radius"):
            for attract in ("k_nn", "radius"):
                with self.subTest(search_mode=repel, att_search_mode=attract):
                    out = self._run(search_mode=repel, att_search_mode=attract)
                    self.assertEqual(out.shape, (12, 8))
                    self.assertTrue(np.isfinite(out).all())

    def test_the_two_modes_are_genuinely_independent(self):
        """Changing one must move the result while the other is held: if the
        arms coincided, the 2x2 would not be measuring two things."""
        base = self._run(search_mode="k_nn", att_search_mode="k_nn")
        att = self._run(search_mode="k_nn", att_search_mode="radius")
        rep = self._run(search_mode="radius", att_search_mode="k_nn")
        self.assertFalse(np.allclose(base, att))
        self.assertFalse(np.allclose(base, rep))
