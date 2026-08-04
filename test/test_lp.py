# coding: utf-8

"""Unit tests for the general-`p` metric: force direction, exact engine, radius.

Three pieces have to agree on `p` or the relaxation is not a descent on
anything: the distances the neighbour search returns, the direction the
force kernel pushes along (the gradient of *that* metric), and the radius
those distances are normalised by. These tests pin each one, and pin the
two anchors that let a `p` sweep be trusted — ``force_p=2`` reproducing the
historic behaviour, and ``_LpIndex`` at `p=1` reproducing the indexed path.
"""

import unittest

import numpy as np

from ess.ess import (
    _GRAD_EPS,
    _compute_forces,
    _l1_radius_heuristic,
    _lp_radius_empirical,
    _LpIndex,
    esa,
    softened_inverse_force,
)


class TestForceDirection(unittest.TestCase):
    """`u_l ∝ |δ_l|^(p-1)·sign(δ_l)` — the gradient of `d_p`, up to a
    positive scalar the normalisation removes."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def _force(self, active, all_data, ids, dists, force_p, radius=0.5):
        return _compute_forces(
            np.asarray(active, dtype=np.float64),
            np.asarray(all_data, dtype=np.float64),
            np.asarray(ids, dtype=np.int64),
            np.asarray(dists, dtype=np.float64),
            radius, softened_inverse_force, self.rng, force_p=force_p)

    def _one_neighbour(self, force_p):
        """A neighbour differing a lot in x and a little in y."""
        active = [[0.50, 0.50]]
        all_data = [[0.20, 0.45], [0.50, 0.50]]   # δ = (+0.30, +0.05)
        return self._force(active, all_data, [[0]], [[0.35]], force_p)[0]

    def test_p2_pushes_along_the_coordinate_that_differs_most(self):
        fx, fy = self._one_neighbour(2.0)
        self.assertAlmostEqual(fy / fx, 0.05 / 0.30, places=9)

    def test_p1_pushes_every_coordinate_equally(self):
        fx, fy = self._one_neighbour(1.0)
        self.assertAlmostEqual(fy / fx, 1.0, places=9)

    def test_p_below_one_pushes_hardest_along_the_shared_coordinate(self):
        """The inversion that motivates p < 1 at all: the near-equal axis
        gets the larger push, so coordinate gaps equalise."""
        fx, fy = self._one_neighbour(0.5)
        self.assertGreater(fy, fx)
        # |δ|^(p-1) ratio: (0.05/0.30)^(0.5-1) = sqrt(6)
        self.assertAlmostEqual(fy / fx, np.sqrt(6.0), places=9)

    def test_direction_is_scale_free_in_the_gradient_prefactor(self):
        """d_p^(1-p) only rescales, so it cannot change the direction."""
        for p in (0.25, 0.5, 0.75, 1.0, 2.0):
            f = self._one_neighbour(p)
            g = self._force([[0.50, 0.50]], [[0.20, 0.45], [0.50, 0.50]],
                            [[0]], [[0.35]], p, radius=2.0)[0]
            np.testing.assert_allclose(f / np.linalg.norm(f),
                                       g / np.linalg.norm(g), atol=1e-12)

    def test_seam_wrap_holds_at_every_p(self):
        """Across the seam the push still goes the short way round."""
        for p in (0.25, 0.5, 1.0, 2.0):
            f = self._force([[0.02, 0.5]], [[0.98, 0.5], [0.02, 0.5]],
                            [[0]], [[0.04]], p)
            self.assertGreater(f[0, 0], 0.0, f"p={p}")

    def test_exactly_shared_coordinate_gets_no_push(self):
        """`sign(0) = 0`, so an exactly equal coordinate contributes
        nothing — the symmetric choice among the subgradients at a point
        where `|δ|^(p-1)·sign(δ)` jumps from -inf to +inf. Finite, and
        not a direction the kernel is entitled to invent."""
        f = self._force([[0.5, 0.5]], [[0.3, 0.5], [0.5, 0.5]],
                        [[0]], [[0.2]], 0.5)
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(f[0, 1], 0.0)
        self.assertGreater(f[0, 0], 0.0)

    def test_nearly_shared_coordinate_dominates_but_stays_bounded(self):
        """The case that actually occurs (exact ties have measure zero):
        the near-equal axis takes over, and the floor bounds by how much."""
        f = self._force([[0.5, 0.5]], [[0.3, 0.5 - 1e-6], [0.5, 0.5]],
                        [[0]], [[0.2]], 0.5)
        self.assertTrue(np.isfinite(f).all())
        self.assertGreater(abs(f[0, 1]), abs(f[0, 0]))
        self.assertAlmostEqual(abs(f[0, 1]) / abs(f[0, 0]),
                               (0.2 / 1e-6) ** 0.5, delta=1.0)
        self.assertLessEqual(abs(f[0, 1]) / abs(f[0, 0]),
                             (0.5 / _GRAD_EPS) ** 0.5)

    def test_stacked_neighbours_still_get_a_push_at_every_p(self):
        for p in (0.25, 1.0, 2.0):
            f = self._force([[0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]],
                            [[0]], [[0.0]], p)
            self.assertGreater(float(np.linalg.norm(f)), 0.0, f"p={p}")

    def test_missing_neighbours_contribute_nothing_at_every_p(self):
        for p in (0.5, 1.0, 2.0):
            f = self._force([[0.5, 0.5]], [[0.4, 0.5], [0.5, 0.5]],
                            [[-1]], [[np.inf]], p)
            np.testing.assert_array_equal(f, np.zeros((1, 2)))


class TestLpRadius(unittest.TestCase):
    def test_matches_the_analytic_l1_heuristic_at_p_one(self):
        """Same quantity, two routes: inverting the exact L1 ball volume,
        and measuring the target-th neighbour of a uniform sample."""
        for d in (2, 8, 32):
            for n in (128, 256):
                an = _l1_radius_heuristic(d, n)
                em = _lp_radius_empirical(d, n, 1.0, np.random.default_rng(7))
                self.assertAlmostEqual(em / an, 1.0, delta=0.1,
                                       msg=f"d={d} n={n}")

    def test_absorbs_the_quasinorm_scale_blowup(self):
        """d_p = d^(1/p)·M_p runs away as p falls; an empirical R must
        track it, since the force law only ever sees d_p/R."""
        radii = [_lp_radius_empirical(32, 128, p, np.random.default_rng(7))
                 for p in (1.0, 0.75, 0.5, 0.25)]
        self.assertTrue(all(a < b for a, b in zip(radii, radii[1:])), radii)
        self.assertGreater(radii[-1], 1e4)   # ~1.5e5 at d=32, p=0.25

    def test_positive_and_finite_everywhere(self):
        for d in (2, 40):
            for p in (0.25, 0.5, 1.0, 2.0):
                r = _lp_radius_empirical(d, 64, p, np.random.default_rng(1))
                self.assertTrue(np.isfinite(r) and r > 0.0)


class TestLpIndex(unittest.TestCase):
    """The exact engine must satisfy the slice of the index contract the
    ESA loop uses, and must be indistinguishable from the indexed path at
    `p = 1` — otherwise a `p` sweep measures the engine, not the metric."""

    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.static = self.rng.random((40, 5))
        self.cands = self.rng.random((12, 5))

    def fitted(self, p=1.0):
        return _LpIndex(p).fit(self.static, self.cands, k=4)

    def test_tiers_and_ids(self):
        ix = self.fitted()
        self.assertEqual((ix.n_static, ix.n_candidates, ix.n_points),
                         (40, 12, 52))
        idx, _ = ix.query(k=4)
        self.assertEqual(idx.shape, (12, 4))
        self.assertTrue(((idx >= 0) & (idx < 52)).all())

    def test_default_query_excludes_the_querying_point(self):
        idx, _ = self.fitted().query(k=6)
        self.assertFalse((idx == np.arange(40, 52)[:, None]).any())

    def test_matches_brute_reference_at_p_one(self):
        diff = np.abs(self.cands[:, None, :]
                      - np.vstack([self.static, self.cands])[None, :, :])
        D = np.minimum(diff, 1.0 - diff).sum(-1)
        D[np.arange(12), np.arange(40, 52)] = np.inf
        idx, dst = self.fitted().query(k=4)
        np.testing.assert_allclose(dst, np.sort(D, axis=1)[:, :4], rtol=1e-12)

    def test_update_moves_only_the_candidate_tier(self):
        ix = self.fitted()
        before = ix._arena[:40].copy()
        ix.update(np.zeros((12, 5)))
        np.testing.assert_array_equal(ix._arena[:40], before)
        np.testing.assert_array_equal(ix._arena[40:], np.zeros((12, 5)))

    def test_promote_freezes_and_appends(self):
        ix = self.fitted()
        ix.promote(self.rng.random((7, 5)))
        self.assertEqual((ix.n_static, ix.n_candidates), (52, 7))

    def test_wraps_coordinates_into_the_torus(self):
        ix = _LpIndex(0.5).fit(np.array([[1.25, -0.25]]), k=1)
        np.testing.assert_allclose(ix._arena, [[0.25, 0.75]], atol=1e-12)

    def test_radius_query_returns_one_row_per_query(self):
        rows = self.fitted().query_radius(2.0)
        self.assertEqual(len(rows), 12)
        for ids, dst in rows:
            self.assertEqual(ids.shape, dst.shape)
            self.assertTrue((np.diff(dst) >= -1e-12).all())

    def test_reproduces_the_indexed_path_exactly_at_p_one(self):
        """End to end: same seed, same points, byte for byte. The engine
        choice must not perturb the generator, or every cross-`p`
        comparison varies two things at once."""
        d, n = 6, 64
        b = np.array([[0.0, 1.0]] * d)
        static = np.random.default_rng(11).random((n, d))
        a = esa(static, b, n=n, seed=np.random.default_rng(0), k=5)
        c = esa(static, b, n=n, seed=np.random.default_rng(0), k=5,
                index=_LpIndex(1.0))
        np.testing.assert_array_equal(a, c)


class TestEsaPValidation(unittest.TestCase):
    def test_rejects_non_positive_p(self):
        b = np.array([[0.0, 1.0]] * 3)
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                esa(np.empty((0, 3)), b, n=8, p=bad, seed=0)

    def test_rejects_an_lsh_index_under_a_non_l1_metric(self):
        """torann's hash collides as a function of L1 only, and p < 1 is
        not a metric — there is no approximate path to hand it to."""
        from torann import ToroidalNN
        b = np.array([[0.0, 1.0]] * 3)
        with self.assertRaises(ValueError):
            esa(np.empty((0, 3)), b, n=8, p=0.5, index=ToroidalNN(), seed=0)

    def test_runs_and_stays_on_the_torus_for_p_below_one(self):
        d, n = 8, 32
        b = np.array([[0.0, 1.0]] * d)
        for p in (0.5, 0.75):
            pts = esa(np.empty((0, d)), b, n=n, p=p, k=5,
                      seed=np.random.default_rng(0))
            self.assertEqual(pts.shape, (n, d))
            self.assertTrue(np.isfinite(pts).all())
            self.assertTrue(((pts >= 0.0) & (pts <= 1.0)).all())

    def test_force_p_defaults_to_the_metric_and_can_be_overridden(self):
        d, n = 5, 24
        b = np.array([[0.0, 1.0]] * d)
        kw = dict(n=n, k=5, p=0.5)
        default = esa(np.empty((0, d)), b, seed=np.random.default_rng(0), **kw)
        same = esa(np.empty((0, d)), b, seed=np.random.default_rng(0),
                   force_p=0.5, **kw)
        other = esa(np.empty((0, d)), b, seed=np.random.default_rng(0),
                    force_p=2.0, **kw)
        np.testing.assert_array_equal(default, same)
        self.assertGreater(np.abs(default - other).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
