"""The reference constructions in `ess.baseline`.

These exist to test ESS, so they get tested harder than ESS does: a baseline
that is quietly wrong makes every comparison against it wrong in the same
direction, and nothing else would notice.
"""

import unittest

import numpy as np

from ess.baseline import dart, grid_oracle, random_fill


def torus_l1(a, b):
    """Reference metric, written out here rather than imported, so a change
    to the shared implementation cannot make this agree by construction."""
    diff = np.abs(a[:, None, :] - b[None, :, :])
    return np.minimum(diff, 1.0 - diff).sum(-1)


class TestDart(unittest.TestCase):
    def setUp(self):
        self.b = np.array([[0.0, 1.0]] * 3)
        self.static = np.random.default_rng(0).random((25, 3))

    def test_shape_bounds_and_determinism(self):
        a = dart(self.static, self.b, n=12, seed=4)
        c = dart(self.static, self.b, n=12, seed=4)
        self.assertEqual(a.shape, (12, 3))
        self.assertTrue((a >= 0.0).all() and (a <= 1.0).all())
        np.testing.assert_array_equal(a, c)

    def test_it_beats_a_uniform_draw_at_being_far_away(self):
        """The whole claim: picks are further from the static set than
        chance. Averaged over seeds, since any single draw can be lucky."""
        got, null = [], []
        for s in range(15):
            got.append(torus_l1(dart(self.static, self.b, n=8, seed=s,
                                     k_cand=256), self.static).min(axis=1).mean())
            null.append(torus_l1(random_fill(self.static, self.b, n=8, seed=s),
                                 self.static).min(axis=1).mean())
        self.assertGreater(np.mean(got), np.mean(null) * 1.2)

    def test_more_candidates_is_monotonically_further(self):
        """`k_cand` is the accuracy knob; the quantity it improves must
        actually improve with it."""
        means = []
        for k in (1, 8, 64, 512):
            per_seed = [
                torus_l1(dart(self.static, self.b, n=6, seed=s, k_cand=k),
                         self.static).min(axis=1).mean()
                for s in range(12)
            ]
            means.append(float(np.mean(per_seed)))
        self.assertEqual(means, sorted(means), means)

    def test_sequential_separates_placed_points_better_than_batch(self):
        """Sequential wins on *separation* and loses on *uniformity*, so be
        precise about which is asserted here.

        Batch picks every slot against a fixed reference, so two slots can
        land together; sequential adds each point before choosing the next
        and cannot. That is what this measures -- the worst pair among the
        placed points.

        It does not generalise: on normalised projection discrepancy batch
        beats sequential by 7-14% at every dimension measured, because
        greedy farthest-first chains outward to the extremes. See `dart`.
        """
        seq, bat = [], []
        for s in range(12):
            for out, acc in ((dart(self.static, self.b, n=10, seed=s,
                                   k_cand=128, sequential=True), seq),
                             (dart(self.static, self.b, n=10, seed=s,
                                   k_cand=128, sequential=False), bat)):
                d = torus_l1(out, out)
                np.fill_diagonal(d, np.inf)
                acc.append(d.min())
        self.assertGreater(np.mean(seq), np.mean(bat))

    def test_empty_static_set_is_allowed(self):
        out = dart(np.empty((0, 3)), self.b, n=5, seed=1)
        self.assertEqual(out.shape, (5, 3))

    def test_non_unit_bounds_are_respected(self):
        b = np.array([[-5.0, -1.0], [10.0, 20.0]])
        static = np.random.default_rng(1).uniform(b[:, 0], b[:, 1], size=(10, 2))
        out = dart(static, b, n=7, seed=2)
        self.assertTrue((out >= b[:, 0]).all() and (out <= b[:, 1]).all())

    def test_zero_and_negative_n(self):
        for n in (0, -3):
            self.assertEqual(dart(self.static, self.b, n=n, seed=0).shape, (0, 3))
            self.assertEqual(random_fill(self.static, self.b, n=n).shape, (0, 3))


class TestDartMeasuresTheTorus(unittest.TestCase):
    def test_the_seam_is_not_an_edge(self):
        """A point just inside x=1 must repel candidates near x=0. If the
        metric did not wrap, the emptiest spot would be beside the anchor."""
        b = np.array([[0.0, 1.0], [0.0, 1.0]])
        anchors = np.array([[0.99, 0.25], [0.99, 0.75]])
        out = dart(anchors, b, n=1, seed=0, k_cand=4096)
        # x=0.01 is 0.02 away through the seam, so it must not be chosen
        self.assertGreater(min(out[0, 0], 1.0 - out[0, 0]), 0.1)


class TestGridOracle(unittest.TestCase):
    """The oracle is exhaustive, so it is the ground truth `dart` is
    approximating -- in the metric ESS uses, which is the reason it exists
    rather than a Delaunay/Voronoi construction (those are L2)."""

    def test_it_finds_a_known_hole(self):
        """A lattice with a 2x2 block removed: the answer is inside the gap.

        Distances make it unambiguous — from the middle of the gap the
        nearest surviving point is 0.20 away, where an ordinary cell centre
        is 0.10 from its neighbours.
        """
        b = np.array([[0.0, 1.0], [0.0, 1.0]])
        ax = (np.arange(10) + 0.5) / 10
        lattice = np.array([[x, y] for x in ax for y in ax])
        keep = ~((np.abs(lattice[:, 0] - 0.5) < 0.1)
                 & (np.abs(lattice[:, 1] - 0.5) < 0.1))
        pt, d = grid_oracle(lattice[keep], b, resolution=100)
        self.assertLess(abs(pt[0] - 0.5), 0.06)
        self.assertLess(abs(pt[1] - 0.5), 0.06)
        self.assertGreater(d, 0.15)

    def test_on_a_torus_the_outside_of_a_ring_is_the_bigger_hole(self):
        """Documents a result that is easy to get wrong.

        A ring of points looks like it encloses the emptiest region, and in
        a box it would. On the torus the *outside* of the ring wraps through
        all four seams into one connected area, which is larger: measured,
        the ring centre is 0.350 from the nearest ring point while the
        corner -- the centre of that outside region -- is 0.505. The oracle
        must prefer the corner, and an implementation that answers 0.5, 0.5
        has quietly stopped wrapping.
        """
        b = np.array([[0.0, 1.0], [0.0, 1.0]])
        th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.column_stack([0.5 + 0.35 * np.cos(th),
                                0.5 + 0.35 * np.sin(th)])
        pt, d = grid_oracle(ring, b, resolution=128)
        # near a corner: every coordinate close to 0 or to 1
        self.assertTrue(all(min(v, 1.0 - v) < 0.1 for v in pt), pt)
        self.assertGreater(d, 0.45)

    def test_dart_approaches_the_oracle_as_k_cand_grows(self):
        """`dart` maximises the same quantity by sampling, so its gap must
        approach the exhaustive answer from below."""
        b = np.array([[0.0, 1.0]] * 2)
        static = np.random.default_rng(3).random((30, 2))
        _, best = grid_oracle(static, b, resolution=256)
        ratios = []
        for k in (4, 64, 4096):
            got = [
                torus_l1(dart(static, b, n=1, seed=s, k_cand=k), static).min()
                for s in range(12)
            ]
            ratios.append(float(np.mean(got)) / best)
        self.assertEqual(ratios, sorted(ratios), ratios)
        self.assertLess(ratios[-1], 1.02)      # never beats exhaustion
        self.assertGreater(ratios[-1], 0.9)    # and gets close

    def test_refusing_a_lattice_it_cannot_afford(self):
        b = np.array([[0.0, 1.0]] * 8)
        with self.assertRaises(ValueError) as cm:
            grid_oracle(np.random.default_rng(0).random((5, 8)), b, resolution=64)
        self.assertIn("nodes", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
