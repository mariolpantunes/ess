
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
        built = attraction.HarmonicProjection(harmonics=3)
        self.assertIs(attraction.get_model(built), built)
        self.assertIsInstance(attraction.get_model("projection"),
                              attraction.HarmonicProjection)
        self.assertIsInstance(
            attraction.get_model(attraction.InverseDistance),
            attraction.InverseDistance)

    def test_get_model_forwards_only_what_the_class_takes(self):
        """`_AttractionField` passes every knob; each model takes its own."""
        idw = attraction.get_model("idw", k=3, power=1.5, bins=8)
        self.assertEqual((idw.k, idw.power), (3, 1.5))
        proj = attraction.get_model("projection", k=3, power=1.5, bins=8)
        self.assertEqual(proj.bins, 8)

    def test_get_model_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            attraction.get_model("no-such-model")
        with self.assertRaises(TypeError):
            attraction.get_model(42)

    def test_empty_source_set_is_not_an_error(self):
        for name in ("idw", "projection"):
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
        for name in ("projection", "idw"):
            for d, m in ((32, 120), (100, 300)):
                with self.subTest(model=name, d=d, m=m):
                    err = held_out(attraction.get_model(name),
                                   coupled_truth, d, m)
                    self.assertLess(err, 1.0)

    def test_projection_survives_where_the_solve_is_underdetermined(self):
        """`M` well under `2d`: the regime `HarmonicRidge` cannot serve.

        Unshrunk, this scored 1.93 -- twice as bad as predicting the mean --
        because 2d coefficients each carry noise and they accumulate.
        """
        err = held_out(attraction.HarmonicProjection(), additive_truth, 100, 30)
        self.assertLess(err, 1.0)

class TestHarmonicInternals(unittest.TestCase):
    def test_table_agrees_with_the_closed_form(self):
        """The table is the same model, not a cheaper one that disagrees."""
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 1, size=(200, 5))
        model = attraction.HarmonicProjection(harmonics=2)
        model.fit(pos, additive_truth(pos), np.ones(200))
        query = rng.uniform(0, 1, size=(50, 5))
        tabulated = model.at(query)
        closed = model._features(query) @ model._w + model._bias
        np.testing.assert_allclose(tabulated, closed, atol=1e-5)

    def test_more_harmonics_fit_a_second_harmonic_better(self):
        def second_only(x):
            return np.cos(4.0 * np.pi * x).sum(1)

        one = held_out(attraction.HarmonicProjection(harmonics=1),
                       second_only, 6, 400)
        two = held_out(attraction.HarmonicProjection(harmonics=2),
                       second_only, 6, 400)
        self.assertLess(two, one)

    def test_estimate_is_periodic_across_the_seam(self):
        """No seam: the space wraps, so the model must too."""
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 1, size=(120, 4))
        model = attraction.HarmonicProjection().fit(
            pos, additive_truth(pos), np.ones(120))
        left = np.full((1, 4), 1e-9)
        right = np.full((1, 4), 1.0 - 1e-9)
        np.testing.assert_allclose(model.at(left), model.at(right), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
