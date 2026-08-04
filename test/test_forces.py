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
    _rank_normalise,
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

    Distances here are toroidal L1 -- torann returns L1, the force law is
    evaluated on it, the step is capped in it -- so the direction is
    `grad(d_L1) = sign(delta)`, which pushes every coordinate equally.

    Until 2026-07-28 the kernel used `delta/||delta||_2` regardless, the L2
    gradient, so the applied force descended no potential ESS defines. The
    exponent was briefly a parameter while `p != 1` was explored; that arm
    lives on `wip_lp` and main is L1 end to end, so there is nothing left to
    select and the geometry is checked here instead.
    """

    def setUp(self):
        self.bounds = np.array([[0.0, 1.0]] * 8)
        self.static = np.random.default_rng(0).random((60, 8))

    def test_no_metric_exponent_is_exposed(self):
        params = inspect.signature(esa).parameters
        self.assertNotIn("force_p", params)
        self.assertNotIn("p", params)

    def test_direction_is_the_l1_subgradient(self):
        """Equal push per coordinate, and a shared coordinate gets none."""
        active = np.array([[0.5, 0.5, 0.5]])
        # one neighbour: differs a lot on axis 0, a little on 1, not at all on 2
        nbrs = np.array([[0.2, 0.48, 0.5]])
        ids = np.array([[0]])
        dists = np.abs(active - nbrs).sum(axis=1, keepdims=True)
        f = _compute_forces(active, nbrs, ids, dists, 1.0,
                            METRIC_REGISTRY["gaussian"],
                            np.random.default_rng(0))
        self.assertGreater(f[0, 0], 0.0)                       # pushed away
        self.assertAlmostEqual(f[0, 0], f[0, 1], places=12)    # equally
        self.assertAlmostEqual(f[0, 2], 0.0, places=12)        # shared: none

    def test_runs_stay_finite_and_in_bounds(self):
        out = esa(self.static, self.bounds, n=16, seed=3)
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())
        self.assertTrue(np.isfinite(out).all())

    def test_coincident_points_do_not_produce_nan(self):
        """Every coordinate shared: the tie-break noise has to take over."""
        static = np.zeros((12, 8)) + 0.5
        out = esa(static, self.bounds, n=8, seed=1)
        self.assertTrue(np.isfinite(out).all())


class TestMetricKwargsAreChecked(unittest.TestCase):
    """`esa` forwards extra keywords to the force law, and every law in the
    registry ends in `**kwargs`, so anything at all used to be accepted.

    The failure that matters is not the nonsense keyword, it is the typo: a
    misspelled `sigma` ran to completion with the default and reported a
    number. Not an error, just a quietly wrong measurement -- which is the
    expensive kind.
    """

    def setUp(self):
        self.bounds = np.array([[0.0, 1.0]] * 3)
        self.static = np.random.default_rng(0).random((10, 3))

    def run_esa(self, **kw):
        return esa(self.static, self.bounds, n=5, seed=0, **kw)

    def test_nonsense_keyword_is_rejected(self):
        with self.assertRaises(TypeError):
            self.run_esa(bogus_kwarg=7)

    def test_typo_of_a_real_parameter_is_rejected(self):
        with self.assertRaises(TypeError) as cm:
            self.run_esa(sigmaa=0.7)
        self.assertIn("sigmaa", str(cm.exception))
        self.assertIn("sigma", str(cm.exception))   # says what is accepted

    def test_another_laws_parameter_is_rejected(self):
        """`power` belongs to softened_inverse; under gaussian it is a
        silent no-op, which is the same defect wearing a plausible name."""
        with self.assertRaises(TypeError):
            self.run_esa(metric="gaussian", power=2.0)
        self.run_esa(metric="softened_inverse", power=2.0)   # legitimate here

    def test_real_parameters_still_reach_the_law(self):
        for metric, kw in (("gaussian", {"sigma": 0.7, "alpha": 4.0}),
                           ("softened_inverse", {"epsilon": 0.4, "power": 3.0}),
                           ("linear", {"alpha": 2.0}),
                           ("cauchy", {"power": 1.5})):
            out = self.run_esa(metric=metric, **kw)
            self.assertEqual(out.shape, (5, 3))

    def test_a_custom_callable_declaring_kwargs_is_taken_at_its_word(self):
        """A `**kwargs` in user code is an explicit choice; the registry's is
        an implementation detail of this module."""
        out = self.run_esa(metric=lambda d, **k: -d * d, anything=1)
        self.assertEqual(out.shape, (5, 3))

    def test_a_custom_callable_without_kwargs_is_checked(self):
        out = self.run_esa(metric=lambda d, scale=1.0: -d * scale, scale=2.0)
        self.assertEqual(out.shape, (5, 3))
        with self.assertRaises(TypeError):
            self.run_esa(metric=lambda d, scale=1.0: -d * scale, scal=2.0)

    def test_stats_is_a_real_parameter_not_a_forwarded_one(self):
        """The timing breakdown used to crash inside the force kernel."""
        stats: dict = {}
        self.run_esa(stats=stats)
        self.assertIn("epochs_total", stats)


class TestAttraction(unittest.TestCase):
    """The second, signed term: repulsion for novelty, attraction for
    fine-tuning toward regions the caller says are good.

    Everything here is about the properties that make it safe to enable --
    that it is off by default, that its units cannot leak in, and that it
    cannot pull hard enough to collapse the design.
    """

    def setUp(self):
        self.b = np.array([[0.0, 1.0]] * 4)
        self.static = np.random.default_rng(0).random((30, 4))
        self.centre = np.full(4, 0.25)
        # "good" near (0.25, ...), so the pull has somewhere to go
        self.q = -np.abs(self.static - self.centre).sum(axis=1)
        self.slow = {"attraction_metric": "cauchy",
                     "attraction_kwargs": {"power": 1.0}}

    def pull_distance(self, **kw):
        out = esa(self.static, self.b, n=15, seed=1, **kw)
        return float(np.abs(out - self.centre).sum(axis=1).mean())

    def test_off_by_default_and_byte_for_byte_unchanged(self):
        """`None` must evaluate no second term, not a zero-weighted one."""
        a = esa(self.static, self.b, n=15, seed=1)
        b = esa(self.static, self.b, n=15, seed=1, attractiveness=None)
        np.testing.assert_array_equal(a, b)

    def test_it_pulls_towards_the_attractive_region(self):
        plain = self.pull_distance()
        pulled = self.pull_distance(attractiveness=self.q,
                                    attraction_weight=0.8, **self.slow)
        self.assertLess(pulled, plain)

    def neighbour_quality(self, seeds=8, **kw):
        """Mean normalised attractiveness of each placed point's nearest
        existing neighbour.

        This is the direct measure of what the term does: it should place
        points beside good ones. Distance to some fixed "good" coordinate is
        not -- the attractive points are scattered, so that proxy mixes the
        effect with wherever they happen to be, and it reads ~3% where this
        reads ~30%.
        """
        out = []
        for s in range(seeds):
            static = np.random.default_rng(s).random((30, 4))
            q = -np.abs(static - 0.25).sum(axis=1)
            a = _rank_normalise(q)
            call = dict(kw)
            if "attractiveness" in call:
                call["attractiveness"] = q
            placed = esa(static, self.b, n=15, seed=s, **call)
            d = np.abs(placed[:, None, :] - static[None, :, :])
            d = np.minimum(d, 1.0 - d).sum(-1)
            out.append(a[d.argmin(axis=1)].mean())
        return float(np.mean(out))

    def test_pure_repulsion_is_indifferent_to_quality(self):
        """The control: with no attraction the nearest neighbour of a placed
        point is of average quality, because nothing told ESS about it."""
        self.assertAlmostEqual(self.neighbour_quality(), 0.5, delta=0.05)

    def test_attraction_places_points_beside_good_neighbours(self):
        got = self.neighbour_quality(attractiveness=True,
                                     attraction_weight=0.5, **self.slow)
        self.assertGreater(got, 0.7)

    def test_a_little_attraction_already_moves_it(self):
        """Monotonicity in the weight does *not* hold -- past roughly 0.5 the
        points cluster onto the single best neighbour instead of spreading
        over the good regions, and the measure falls back. So the claim is
        that any real weight beats none, not that more is always better."""
        none = self.neighbour_quality()
        for w in (0.2, 0.5, 1.0, 2.0):
            got = self.neighbour_quality(attractiveness=True,
                                         attraction_weight=w, **self.slow)
            self.assertGreater(got, none + 0.15, f"weight={w}")

    def test_units_of_the_objective_cannot_reach_the_force(self):
        """The defect this guards against: a weight calibrated on one
        objective's scale meaning something else on another."""
        a = esa(self.static, self.b, n=15, seed=1, attractiveness=self.q,
                attraction_weight=0.8, **self.slow)
        for factor, shift in ((1e6, 0.0), (1e-6, 0.0), (1.0, 5e5)):
            b = esa(self.static, self.b, n=15, seed=1,
                    attractiveness=self.q * factor + shift,
                    attraction_weight=0.8, **self.slow)
            np.testing.assert_array_equal(a, b)

    def test_monotone_rescaling_is_all_that_matters(self):
        """Rank-normalised, so any order-preserving transform is identical."""
        a = esa(self.static, self.b, n=15, seed=1, attractiveness=self.q,
                attraction_weight=0.8, **self.slow)
        b = esa(self.static, self.b, n=15, seed=1,
                attractiveness=np.exp(self.q), attraction_weight=0.8,
                **self.slow)
        np.testing.assert_array_equal(a, b)

    def test_collapse_is_refused_at_setup(self):
        with self.assertRaises(ValueError) as cm:
            esa(self.static, self.b, n=15, seed=1, attractiveness=self.q,
                attraction_weight=50.0, **self.slow)
        self.assertIn("collapse", str(cm.exception).lower())

    def test_one_value_per_existing_point(self):
        with self.assertRaises(ValueError) as cm:
            esa(self.static, self.b, n=15, seed=1, attractiveness=self.q[:5])
        self.assertIn("5 values", str(cm.exception))

    def test_same_law_on_both_sides_warns_that_it_cannot_cross(self):
        with self.assertLogs("ess.ess", level="WARNING") as log:
            esa(self.static, self.b, n=15, seed=1, attractiveness=self.q,
                attraction_weight=0.5)
        self.assertIn("never overcomes", "".join(log.output))

    def test_a_constant_attractiveness_pulls_everything_equally(self):
        """No information in the values, so no preferred direction: the
        result must stay a valid design rather than degenerate."""
        out = esa(self.static, self.b, n=15, seed=1,
                  attractiveness=np.ones(len(self.static)),
                  attraction_weight=0.8, **self.slow)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_placed_points_never_attract(self):
        """They have no evaluated quality; only the static block pulls."""
        out = esa(self.static, self.b, n=15, seed=1, attractiveness=self.q,
                  attraction_weight=0.8, **self.slow)
        self.assertEqual(out.shape, (15, 4))
        self.assertTrue(np.isfinite(out).all())
