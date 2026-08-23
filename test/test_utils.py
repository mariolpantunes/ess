

import unittest
import warnings

import numpy as np

from ess import utils
from ess.ess import METRIC_REGISTRY, _inv_scale, _scale
from ess.utils import (
    expected_discrepancy,
    projection_discrepancy,
    toroidal_separation,
    wrap_around_discrepancy,
)


class TestUtils(unittest.TestCase):
    def test_scaling_lifecycle(self):
        """Test scale -> inv_scale round trip."""
        original = np.array([[10.0, 20.0], [20.0, 40.0], [15.0, 30.0]])
        scaled, min_v, max_v = _scale(original)

        self.assertTrue(np.all(scaled >= 0.0))
        self.assertTrue(np.all(scaled <= 1.0))

        restored = _inv_scale(scaled, min_v, max_v)
        self.assertTrue(np.allclose(original, restored))

    def test_scaling_degenerate(self):
        """Test scaling constant dimensions and scalars."""
        # Constant dimension
        data = np.array([[5, 5], [5, 10]])
        scaled, _, _ = _scale(data)
        self.assertTrue(np.all(scaled[:, 0] == 0.0))  # Denom was 0 -> handled

        # Scalars passed as min/max
        arr = np.array([10, 20, 30])
        s, _, _ = _scale(arr, min_val=0, max_val=40)
        self.assertTrue(np.allclose(s, [0.25, 0.5, 0.75]))

    def test_grid_coverage(self):
        """Test grid coverage logic."""
        bounds = np.array([[0, 10], [0, 10]])
        points = np.array([[2, 2], [8, 8]])  # 2 points

        # 2x2 Grid -> 4 cells. Points in different cells. Coverage 0.5.
        cov = utils.calculate_grid_coverage(points, bounds, grid=2)
        self.assertAlmostEqual(cov, 0.5)

        # High Dim Sparse Test
        dim = 64
        pts_hd = np.random.rand(10, dim)
        bounds_hd = np.array([[0, 1]] * dim)
        # Should not crash
        cov_hd = utils.calculate_grid_coverage(pts_hd, bounds_hd, grid=3)
        self.assertGreater(cov_hd, 0.0)

    def test_separation_in_both_geometries(self):
        """Both separations, and the gap between them.

        This replaces a test that had been gutted: when Clark-Evans left the
        metric panel its half of the body went with it, leaving two assigned
        variables, no assertion and no blank line before the next method. It
        passed for weeks while checking one third of what its name claimed.
        """
        # Equilateral triangle, side 1: no wrap involved, so both agree.
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
        self.assertAlmostEqual(utils.euclidean_separation(points), 1.0,
                               places=3)

        # Across the seam the two must disagree, and that is the whole
        # reason the toroidal one exists.
        seam = np.array([[0.99, 0.5], [0.01, 0.5]])
        self.assertAlmostEqual(utils.toroidal_separation(seam), 0.02,
                               places=9)
        self.assertAlmostEqual(utils.euclidean_separation(seam), 0.98,
                               places=9)

        # Clustered points are separated by ~0; spread ones are not.
        clustered = np.zeros((10, 2))
        self.assertAlmostEqual(utils.toroidal_separation(clustered), 0.0)
        spread = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        self.assertGreater(utils.toroidal_separation(spread), 0.4)

    def test_force_functions(self):
        """Verify force function behaviors."""
        d = np.array([0.0, 1.0, 100.0])

        # 1. Gaussian: High at 0, Low at 100
        f_gauss = METRIC_REGISTRY["gaussian"](d, sigma=1.0, alpha=1.0)
        self.assertAlmostEqual(f_gauss[0], 0.0)
        self.assertLess(f_gauss[2], -100.0)

        # 2. Linear: High at 0, cut off at the radius (d_hat >= 1)
        f_lin = METRIC_REGISTRY["linear"](d, alpha=1.0)
        self.assertAlmostEqual(f_lin[0], 0.0)
        self.assertAlmostEqual(f_lin[1], np.log(1e-9))
        self.assertAlmostEqual(f_lin[2], np.log(1e-9))

        # 3. Softened Inverse: Finite at 0, Decays slowly
        f_inv = METRIC_REGISTRY["softened_inverse"](d, epsilon=0.1, alpha=1.0)
        self.assertLess(f_inv[0], 5.0)  # log(10) ~ 2.3
        self.assertTrue(np.isfinite(f_inv[2]))  # Never negative infinity


class TestDiscrepancy(unittest.TestCase):
    """The wrap-around discrepancy is the high-dimensional criterion:
    periodic (so it scores the torus the relaxation optimizes) and free
    of the distance contrasts that collapse in high dimension."""

    def test_random_matches_its_expected_value(self):
        """A uniform sample scores ~1 against `expected_discrepancy`."""
        for dim in (2, 8, 64):
            n = 256
            vals = [
                wrap_around_discrepancy(np.random.default_rng(s).random((n, dim)))
                / expected_discrepancy(n, dim)
                for s in range(3)
            ]
            self.assertAlmostEqual(float(np.mean(vals)), 1.0, delta=0.25)

    def test_invariant_under_toroidal_shift(self):
        """'Wrap-around' means exactly this: rigidly shifting a design
        around the torus cannot change its uniformity."""
        P = np.random.default_rng(0).random((64, 5))
        base = wrap_around_discrepancy(P)
        for shift in (0.37, 0.5, 0.99):
            self.assertAlmostEqual(
                wrap_around_discrepancy(np.mod(P + shift, 1.0)), base, places=9
            )

    def test_regular_grid_beats_random(self):
        """Lower is better: an equispaced 1-D design is far more uniform."""
        grid = (np.arange(64) / 64.0)[:, None]
        rand = np.random.default_rng(0).random((64, 1))
        self.assertLess(wrap_around_discrepancy(grid), wrap_around_discrepancy(rand))

    def test_chunking_does_not_change_the_value(self):
        P = np.random.default_rng(1).random((97, 4))
        self.assertAlmostEqual(
            wrap_around_discrepancy(P, chunk=8),
            wrap_around_discrepancy(P, chunk=1000),
            places=10,
        )

    def test_projection_discrepancy_detects_lhs_stratification(self):
        """LHS is built for uniform 1-D margins; the projection score
        must see that even in high dimension, where the full-dimensional
        measure cannot."""
        from ess.samplers import LHCSampler

        n, dim = 256, 32
        lhs = LHCSampler(random_state=0).sample(n, dim)
        rand = np.random.default_rng(0).random((n, dim))
        base = expected_discrepancy(n, 1)
        self.assertLess(projection_discrepancy(lhs, 1) / base, 0.1)
        self.assertGreater(projection_discrepancy(rand, 1) / base, 0.5)

    def test_toroidal_separation_sees_the_wrap(self):
        """The pair straddling the seam is 0.02 apart, not 0.98."""
        pts = np.array([[0.01, 0.5], [0.99, 0.5], [0.5, 0.1], [0.5, 0.9]])
        self.assertAlmostEqual(toroidal_separation(pts), 0.02, places=9)
        # the Euclidean version measures a different geometry entirely
        self.assertGreater(utils.euclidean_separation(pts), 0.6)

    def test_toroidal_separation_matches_all_pairs(self):
        pts = np.random.default_rng(5).random((256, 3))
        d = np.abs(pts[:, None, :] - pts[None, :, :])
        d = np.minimum(d, 1.0 - d).sum(-1)
        np.fill_diagonal(d, np.inf)
        self.assertEqual(toroidal_separation(pts), float(d.min()))

    def test_projection_order_validated(self):
        P = np.random.default_rng(0).random((16, 3))
        with self.assertRaises(ValueError):
            projection_discrepancy(P, order=4)


if __name__ == "__main__":
    unittest.main()


class TestMetricGeometryIsInTheName(unittest.TestCase):
    """`utils` holds two Clark-Evans indices and two separations, one pair
    Euclidean and one toroidal. The old names said what was computed but not
    in which geometry, which is the thing that decides whether a number means
    anything — so the names now carry it, and the old ones are deprecated
    rather than removed, because 0.3.1 published them.
    """

    def setUp(self):
        # one pair straddling the seam: adjacent on the torus, far apart in
        # a box. The whole point of distinguishing the two.
        self.seam = np.array([[0.01, 0.5], [0.99, 0.5],
                              [0.50, 0.2], [0.50, 0.8]])

    def test_the_two_geometries_disagree_and_should(self):
        box = utils.euclidean_separation(self.seam)
        tor = utils.toroidal_separation(self.seam)
        self.assertGreater(box, 0.5)     # the seam pair looks distant
        self.assertLess(tor, 0.05)       # ...and is in fact adjacent
        self.assertGreater(box, 10 * tor)

    def test_deprecated_aliases_warn_and_agree(self):
        for old, new, args in (
            (utils.calculate_min_pairwise_distance, utils.euclidean_separation, ()),
        ):
            with self.assertWarns(DeprecationWarning):
                got = old(self.seam, *args)
            self.assertEqual(got, new(self.seam, *args))

    def test_deprecation_message_names_the_replacement(self):
        """A warning that does not say what to use instead is noise."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            utils.calculate_min_pairwise_distance(self.seam)
        msg = str(w[0].message)
        self.assertIn("euclidean_separation", msg)
        self.assertIn("toroidal_separation", msg)
