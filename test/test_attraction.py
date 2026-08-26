
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
