import itertools
import unittest

import numpy as np

import support


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9127)

    @staticmethod
    def score(path, logp, costs):
        value = sum(logp[t, state] for t, state in enumerate(path))
        for t in range(1, len(path)):
            matrix = costs if costs.ndim == 2 else costs[t]
            value -= matrix[path[t - 1], path[t]]
        return value

    def test_fixed_transition_optimum(self):
        p = self.rng.dirichlet([1, 1, 1], size=6)
        costs = self.rng.uniform(.2, 2, size=(3, 3))
        predicted = support.viterbi(p, [(0, 6)], costs=costs)
        expected = max(self.score(path, np.log(p), costs)
                       for path in itertools.product(range(3), repeat=6))
        self.assertAlmostEqual(self.score(predicted, np.log(p), costs), expected)

    def test_time_dependent_cost_uses_destination_frame(self):
        p = self.rng.dirichlet([1, 1, 1], size=6)
        costs = self.rng.uniform(0, 4, size=(6, 3, 3))
        predicted = support.viterbi(p, [(0, 6)], costs=costs)
        expected = max(self.score(path, np.log(p), costs)
                       for path in itertools.product(range(3), repeat=6))
        self.assertAlmostEqual(self.score(predicted, np.log(p), costs), expected)

    def test_asymmetric_duration_matches_feasible_optimum(self):
        p = self.rng.dirichlet([1, 1, 1], size=6)
        costs = np.full((3, 3), 1.3)
        np.fill_diagonal(costs, 0)
        for dwell in [(1, 2, 3), (2, 5, 1), (3, 1, 2)]:
            def feasible(path):
                edges = [0] + [i for i in range(1, len(path)) if path[i] != path[i - 1]] + [len(path)]
                return all(b - a >= dwell[path[a]] for a, b in list(zip(edges, edges[1:]))[1:-1])
            expected = max(self.score(path, np.log(p), costs)
                           for path in itertools.product(range(3), repeat=6) if feasible(path))
            predicted = support.duration_decode(p, [(0, 6)], fps=1, seconds=dwell, switch_cost=1.3)
            self.assertAlmostEqual(self.score(predicted, np.log(p), costs), expected)

    def test_halves_reset_even_under_large_switch_cost(self):
        p = np.array([[.99, .005, .005], [.99, .005, .005], [.005, .99, .005], [.005, .99, .005]])
        for prediction in [support.viterbi(p, [(0, 2), (2, 4)], switch_cost=100),
                           support.duration_decode(p, [(0, 2), (2, 4)], seconds=(10, 10, 10), switch_cost=100)]:
            np.testing.assert_array_equal(prediction, [0, 0, 1, 1])

    def test_single_frame_allows_partial_duration(self):
        p = np.array([[.1, .2, .7]])
        np.testing.assert_array_equal(support.duration_decode(p, [(0, 1)], seconds=(2, 3, 5)), [2])

    def test_temporal_window_uses_seconds_without_crossing_blocks(self):
        X = np.zeros((20, 16), np.float32)
        X[:, 2] = np.r_[np.arange(10), np.arange(10) + 1000]
        data = dict(X=X, blocks=[(0, 10), (10, 20)], fps=5)
        out = support.temporal(data, 1)['X']
        self.assertAlmostEqual(float(out[5, 21]), np.sqrt(2), places=5)
        self.assertLess(out[9, 16], 10)
        self.assertGreater(out[10, 16], 999)

    def test_transition_learning_skips_unknowns_and_half_boundary(self):
        data = dict(y=np.array([0, 0, -1, 1, 2, 2], np.int8), blocks=[(0, 4), (4, 6)])
        probability = np.exp(-support.transition_cost(data))
        np.testing.assert_allclose(probability[0], [.5, .25, .25])
        np.testing.assert_allclose(probability[1], [1 / 3] * 3)
        np.testing.assert_allclose(probability[2], [.25, .25, .5])


if __name__ == '__main__':
    unittest.main()
