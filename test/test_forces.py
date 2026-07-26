# coding: utf-8

"""Unit tests for the toroidal force kernel, force laws and heuristics.

These pin down the geometry that makes the torann-backed ESS correct:
the seam-aware displacement wrap, the missing-neighbour padding
convention, the tie-breaking noise, and the L1-on-torus radius
heuristic.
"""

import unittest

import numpy as np

from ess.ess import (
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
    def test_matches_the_closed_form_in_2d(self):
        # r = 1.25 * 0.5 * (2!/100)^(1/2) = 0.0884
        self.assertAlmostEqual(_l1_radius_heuristic(2, 100), 0.08839, places=4)

    def test_decreases_with_density(self):
        radii = [_l1_radius_heuristic(5, n) for n in (10, 100, 1000, 10000)]
        self.assertTrue(all(a > b for a, b in zip(radii, radii[1:])))

    def test_capped_at_a_quarter_of_the_l1_diameter(self):
        # 2D with 2 points: uncapped 1.25 * 0.5 * (2/2)^(1/2) = 0.625 > 0.5
        self.assertAlmostEqual(_l1_radius_heuristic(2, 2), 2 / 4.0)

    def test_reaches_past_the_mean_nn_distance(self):
        """The radius must not starve the neighbourhood: it has to reach
        at least the ideal-packing distance it was derived from."""
        for dim, n in ((2, 100), (10, 1000), (20, 330), (50, 11)):
            packing = 0.5 * np.exp(
                (np.log(np.arange(1, dim + 1)).sum() - np.log(n)) / dim
            )
            r = _l1_radius_heuristic(dim, n)
            self.assertGreaterEqual(r, min(packing, dim / 4.0) * 0.99, (dim, n))


if __name__ == "__main__":
    unittest.main()
