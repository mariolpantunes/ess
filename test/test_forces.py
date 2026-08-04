# coding: utf-8

"""Unit tests for the toroidal force kernel, force laws and heuristics.

These pin down the geometry that makes the torann-backed ESS correct:
the seam-aware displacement wrap, the missing-neighbour padding
convention, the tie-breaking noise, and the L1-on-torus radius
heuristic.
"""

import inspect
import math
import unittest

import numpy as np

from ess.ess import (
    NEIGHBOUR_TARGET,
    esa,
    METRIC_REGISTRY,
    _compute_forces,
    _l1_radius_heuristic,
    _pad_ragged,
    _row_blocks,
    softened_inverse_force,
)


class TestForceLaws(unittest.TestCase):
    def test_all_laws_decrease_with_distance(self):
        """Repulsion must weaken monotonically with normalised distance."""
        d = np.linspace(0.0, 3.0, 61)
        for name, fn in METRIC_REGISTRY.items():
            log_f = fn(d)
            self.assertTrue(
                np.all(np.diff(log_f) <= 1e-12), f"{name} is not decreasing"
            )
            self.assertTrue(np.all(np.isfinite(log_f[:1])), f"{name} infinite at 0")

    def test_laws_are_order_one_near_the_radius(self):
        """Calibration: F(3/4) within [0.1, 10] so lr defaults are shared.
        (Evaluated inside the radius because the linear law cuts off
        exactly at 1.)"""
        for name, fn in METRIC_REGISTRY.items():
            f_near_r = float(np.exp(fn(np.array([0.75]))[0]))
            self.assertGreater(f_near_r, 0.1, name)
            self.assertLess(f_near_r, 10.0, name)

    def test_laws_tolerate_extra_kwargs(self):
        """The kernel forwards **metric_kwargs; unknown keys must not raise."""
        for fn in METRIC_REGISTRY.values():
            fn(np.array([0.5]), unused_key=123)


class TestComputeForces(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def _forces(self, active, all_data, ids, dists, radius=0.2):
        return _compute_forces(
            np.asarray(active, dtype=np.float64),
            np.asarray(all_data, dtype=np.float64),
            np.asarray(ids, dtype=np.int64),
            np.asarray(dists, dtype=np.float64),
            radius,
            softened_inverse_force,
            self.rng,
        )

    def test_repulsion_points_away_from_neighbour(self):
        """A single neighbour on the left pushes the point right."""
        active = [[0.5, 0.5]]
        all_data = [[0.4, 0.5], [0.5, 0.5]]
        f = self._forces(active, all_data, [[0]], [[0.1]])
        self.assertGreater(f[0, 0], 0.0)
        self.assertAlmostEqual(f[0, 1], 0.0)

    def test_seam_wrap_pushes_the_short_way(self):
        """Neighbour at 0.98 vs point at 0.02: toroidal distance is 0.04,
        so the push is +x (away across the seam), not -x (across the
        whole domain, as the naive displacement would give)."""
        active = [[0.02, 0.5]]
        all_data = [[0.98, 0.5], [0.02, 0.5]]
        f = self._forces(active, all_data, [[0]], [[0.04]])
        self.assertGreater(f[0, 0], 0.0)
        self.assertAlmostEqual(f[0, 1], 0.0)

    def test_missing_neighbours_are_ignored(self):
        """Padded rows (-1 / inf) contribute nothing; all-missing = zero."""
        active = [[0.5, 0.5]]
        all_data = [[0.4, 0.5], [0.5, 0.5]]
        f = self._forces(active, all_data, [[-1, -1]], [[np.inf, np.inf]])
        np.testing.assert_array_equal(f, np.zeros((1, 2)))

        # one valid + one missing == just the valid one
        f_pair = self._forces(active, all_data, [[0, -1]], [[0.1, np.inf]])
        f_solo = self._forces(active, all_data, [[0]], [[0.1]])
        np.testing.assert_allclose(f_pair, f_solo, rtol=1e-12)

    def test_stacked_points_get_a_tie_breaking_push(self):
        """Exactly coincident neighbours must yield a non-zero force."""
        active = [[0.5, 0.5]]
        all_data = [[0.5, 0.5], [0.5, 0.5]]
        f = self._forces(active, all_data, [[0]], [[0.0]])
        self.assertGreater(float(np.linalg.norm(f)), 0.0)

    def test_force_magnitude_is_capped(self):
        """The log-sum-exp restore is capped at 1e3 per point."""
        active = [[0.5, 0.5]]
        all_data = [[0.4999, 0.5], [0.5, 0.5]]
        f = self._forces(active, all_data, [[0]], [[1e-4]], radius=1.0)
        self.assertLessEqual(float(np.linalg.norm(f)), 1000.0 * (1.0 + 1e-9))


class TestRowBlocking(unittest.TestCase):
    """The kernel is chunked so a wide neighbour list (radius mode in a
    sparse high-dimensional regime returns the whole design) cannot
    allocate gigabytes."""

    def test_blocks_cover_every_row_exactly_once(self):
        for rows, width, dim in ((10, 3, 2), (1000, 1000, 64), (1, 1, 1)):
            blocks = list(_row_blocks(rows, width, dim))
            self.assertEqual(blocks[0][0], 0)
            self.assertEqual(blocks[-1][1], rows)
            for (_, prev_stop), (start, _) in zip(blocks, blocks[1:]):
                self.assertEqual(prev_stop, start)

    def test_working_set_is_bounded(self):
        from ess.ess import _FORCE_BLOCK_ELEMENTS

        rows, width, dim = 4096, 4096, 64  # would be 8 GB unchunked
        for start, stop in _row_blocks(rows, width, dim):
            self.assertLessEqual((stop - start) * width * dim, _FORCE_BLOCK_ELEMENTS)

    def test_chunking_does_not_change_the_forces(self):
        """Same result whichever block size the budget implies."""
        import sys

        # `ess.ess` the attribute is the public function, not this module
        core = sys.modules["ess.ess"]

        rng = np.random.default_rng(0)
        active = rng.random((40, 3))
        all_data = rng.random((60, 3))
        ids = rng.integers(0, 60, size=(40, 7))
        dists = rng.random((40, 7)) * 0.3

        original = core._FORCE_BLOCK_ELEMENTS
        try:
            core._FORCE_BLOCK_ELEMENTS = 10**9
            whole = core._compute_forces(
                active, all_data, ids, dists, 0.2,
                softened_inverse_force, np.random.default_rng(1),
            )
            core._FORCE_BLOCK_ELEMENTS = 21  # forces 1-row blocks
            chunked = core._compute_forces(
                active, all_data, ids, dists, 0.2,
                softened_inverse_force, np.random.default_rng(1),
            )
        finally:
            core._FORCE_BLOCK_ELEMENTS = original
        np.testing.assert_allclose(whole, chunked, rtol=1e-12, atol=1e-12)


class TestPadRagged(unittest.TestCase):
    def test_pads_with_missing_convention(self):
        res = [
            (np.array([3, 1]), np.array([0.1, 0.2])),
            (np.array([], dtype=np.int64), np.array([])),
            (np.array([7]), np.array([0.05])),
        ]
        ids, dists = _pad_ragged(res)
        self.assertEqual(ids.shape, (3, 2))
        np.testing.assert_array_equal(ids[1], [-1, -1])
        self.assertTrue(np.isinf(dists[1]).all())
        self.assertEqual(ids[2, 0], 7)
        self.assertEqual(ids[2, 1], -1)

    def test_all_empty_keeps_valid_shape(self):
        ids, dists = _pad_ragged([(np.array([], dtype=np.int64), np.array([]))])
        self.assertEqual(ids.shape, (1, 1))
        self.assertEqual(ids[0, 0], -1)


class TestRadiusHeuristic(unittest.TestCase):
    """The contract is a *count*, not a distance: the ball should hold
    about `NEIGHBOUR_TARGET` neighbours at every dimension."""

    def test_matches_the_exact_dense_formula(self):
        # dense regime (R <= 1/2): R = 0.5 * (target * d! / n)^(1/d)
        expected = 0.5 * math.sqrt(NEIGHBOUR_TARGET * 2.0 / 100.0)
        self.assertAlmostEqual(_l1_radius_heuristic(2, 100), expected, places=9)

    def test_decreases_with_density(self):
        radii = [_l1_radius_heuristic(5, n) for n in (10, 100, 1000, 10000)]
        self.assertTrue(all(a > b for a, b in zip(radii, radii[1:])))

    def test_never_spans_the_space(self):
        """The toroidal L1 diameter is d/2; a radius near it would make
        every point a neighbour of every other."""
        for dim in (2, 8, 16, 32, 64, 128):
            for n in (256, 1024, 50000):
                r = _l1_radius_heuristic(dim, n)
                self.assertLess(r, dim / 2.0, (dim, n))
                self.assertGreater(r, 0.0, (dim, n))

    def test_holds_about_the_target_count(self):
        """The property that matters, measured on uniform points: the
        old radius-space margin gave 1.6 neighbours at d=2 and 31 at
        d=64; this must stay flat across dimension."""
        from torann.brute import exact_radius

        for dim, n in ((2, 512), (8, 512), (16, 1024), (32, 1024), (64, 1024)):
            r = _l1_radius_heuristic(dim, n)
            pts = np.random.default_rng(0).random((n, dim))
            indptr, _, _ = exact_radius(pts, pts, r, np.arange(n))
            mean_count = float(np.diff(indptr).mean())
            self.assertGreater(mean_count, NEIGHBOUR_TARGET * 0.4, (dim, n, mean_count))
            self.assertLess(mean_count, NEIGHBOUR_TARGET * 2.5, (dim, n, mean_count))

    def test_sparse_regime_uses_the_distance_law(self):
        """Where the L1 ball would exceed the torus, the radius follows
        the CLT quantile around the mean pairwise distance d/4."""
        dim, n = 64, 1024
        r = _l1_radius_heuristic(dim, n)
        self.assertLess(r, dim / 4.0)              # below the mean distance
        self.assertGreater(r, dim / 4.0 - 6.0 * math.sqrt(dim / 48.0))


if __name__ == "__main__":
    unittest.main()


class TestForceDirection(unittest.TestCase):
    """The push direction must be the gradient of the metric measuring it.

    Before this was parameterised the kernel always pushed along
    `delta / ||delta||_2` -- the p=2 gradient -- while `dists` carried
    toroidal L1 magnitudes, so the applied force descended no potential ESS
    defines. `force_p` selects the exponent and defaults to 1, matching the
    metric.
    """

    def setUp(self):
        self.bounds = np.array([[0.0, 1.0]] * 8)
        self.static = np.random.default_rng(0).random((60, 8))

    def test_default_is_the_l1_gradient(self):
        self.assertEqual(
            inspect.signature(esa).parameters["force_p"].default, 1.0)

    def test_p2_is_a_regression_anchor_and_differs_from_the_default(self):
        """`force_p=2.0` exists to reproduce figures recorded before the
        fix, so it must stay reachable *and* stay distinguishable."""
        historic = esa(self.static, self.bounds, n=24, seed=7, force_p=2.0)
        again = esa(self.static, self.bounds, n=24, seed=7, force_p=2.0)
        default = esa(self.static, self.bounds, n=24, seed=7)
        np.testing.assert_array_equal(historic, again)
        self.assertFalse(np.allclose(historic, default))

    def test_every_direction_stays_in_bounds(self):
        for fp in (0.5, 1.0, 2.0):
            out = esa(self.static, self.bounds, n=16, seed=3, force_p=fp)
            self.assertTrue((out >= 0.0).all() and (out <= 1.0).all(), fp)
            self.assertTrue(np.isfinite(out).all(), fp)

    def test_coincident_coordinates_do_not_produce_nan(self):
        """`|r|**(p-1)` diverges as a gap closes; `_GRAD_EPS` floors it."""
        static = np.zeros((12, 8)) + 0.5      # every coordinate shared
        out = esa(static, self.bounds, n=8, seed=1, force_p=0.5)
        self.assertTrue(np.isfinite(out).all())
