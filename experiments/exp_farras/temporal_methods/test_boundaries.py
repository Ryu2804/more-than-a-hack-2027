import itertools
import unittest

import numpy as np

import boundaries


def run_edges(codes):
    return np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]


def objective(codes, probability, original, radius, penalty):
    internal = run_edges(codes)[1:-1]
    displacement = (internal - run_edges(original)[1:-1]) / radius
    return (float(np.log(np.maximum(probability, 1e-8))[np.arange(len(codes)), codes].sum())
            - penalty * float(np.sum(displacement ** 2)))


def brute_force(codes, probability, radius, penalty):
    edges = run_edges(codes)
    labels = codes[edges[:-1]]
    best_score = -np.inf
    best_codes = None
    for positions in itertools.combinations(range(1, len(codes)), len(labels) - 1):
        if any(abs(position - old) > radius
               for position, old in zip(positions, edges[1:-1])):
            continue
        candidate_edges = (0, *positions, len(codes))
        candidate = np.empty(len(codes), dtype=np.int8)
        for label, start, end in zip(labels, candidate_edges, candidate_edges[1:]):
            candidate[start:end] = label
        score = objective(candidate, probability, codes, radius, penalty)
        if score > best_score:
            best_score = score
            best_codes = candidate
    return best_codes, best_score


class BoundaryTests(unittest.TestCase):

    def test_exhaustive_tiny_objective_matches_brute_force(self):
        rng = np.random.default_rng(20261116)
        for length in range(1, 6):
            probability = rng.dirichlet((0.7, 1.1, 0.9), size=length)
            for values in itertools.product(range(3), repeat=length):
                codes = np.array(values, dtype=np.int8)
                for config in boundaries.CONFIGS:
                    radius, penalty = config["radius"], config["penalty"]
                    with self.subTest(codes=values, config=config["name"]):
                        expected, optimum = brute_force(codes, probability, radius, penalty)
                        actual = boundaries.relocate(codes, probability, [(0, length)],
                                                     radius, penalty)
                        self.assertAlmostEqual(objective(actual, probability, codes,
                                                         radius, penalty), optimum, places=11)
                        np.testing.assert_array_equal(actual, expected)
                        old_edges, new_edges = run_edges(codes), run_edges(actual)
                        np.testing.assert_array_equal(codes[old_edges[:-1]],
                                                      actual[new_edges[:-1]])
                        self.assertTrue(np.all(np.abs(new_edges - old_edges) <= radius))

    def test_uniform_ties_choose_lexicographically_earliest_boundaries(self):
        codes = np.array([0, 0, 0, 2, 2, 1, 1, 1], dtype=np.int8)
        probability = np.full((len(codes), 3), 1 / 3)
        for radius in (2, 5, 10):
            actual = boundaries.relocate(codes, probability, [(0, len(codes))], radius, 0)
            expected, _ = brute_force(codes, probability, radius, 0)
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(run_edges(boundaries.relocate(
            codes, probability, [(0, len(codes))], 10, 0)), [0, 1, 2, 8])

    def test_quadratic_penalty_changes_optimum(self):
        codes = np.array([0, 0, 1, 1], dtype=np.int8)
        probability = np.array([[.99, .01, 0], [.499, .501, 0],
                                [.01, .99, 0], [.01, .99, 0]])
        free = boundaries.relocate(codes, probability, [(0, 4)], 2, 0)
        penalized = boundaries.relocate(codes, probability, [(0, 4)], 2, .05)
        np.testing.assert_array_equal(free, [0, 1, 1, 1])
        np.testing.assert_array_equal(penalized, codes)
        for penalty, actual in ((0, free), (.05, penalized)):
            _, optimum = brute_force(codes, probability, 2, penalty)
            self.assertAlmostEqual(objective(actual, probability, codes, 2, penalty), optimum)

    def test_joint_dead_boundaries_move_without_changing_run_count(self):
        codes = np.array([0, 0, 2, 2, 1, 1], dtype=np.int8)
        target = np.array([0, 2, 2, 2, 2, 1], dtype=np.int8)
        probability = np.full((len(codes), 3), .005)
        probability[np.arange(len(codes)), target] = .99
        for config in boundaries.CONFIGS:
            actual = boundaries.relocate(codes, probability, [(0, 6)],
                                         config["radius"], config["penalty"])
            np.testing.assert_array_equal(actual, target)
            self.assertEqual(len(run_edges(actual)), len(run_edges(codes)))

    def test_dense_changes_cannot_cross_or_disappear(self):
        for values in ([0, 1, 2, 0, 1], [0, 0, 1, 2, 2, 0]):
            codes = np.array(values, dtype=np.int8)
            probability = np.tile([.001, .998, .001], (len(codes), 1))
            for config in boundaries.CONFIGS:
                actual = boundaries.relocate(codes, probability, [(0, len(codes))],
                                             config["radius"], config["penalty"])
                old_edges, new_edges = run_edges(codes), run_edges(actual)
                np.testing.assert_array_equal(codes[old_edges[:-1]], actual[new_edges[:-1]])
                self.assertTrue(np.all(np.diff(new_edges) >= 1))
                _, optimum = brute_force(codes, probability, config["radius"], config["penalty"])
                self.assertAlmostEqual(objective(actual, probability, codes, config["radius"],
                                                 config["penalty"]), optimum, places=11)

    def test_halves_optimize_independently_with_fixed_endpoints(self):
        codes = np.array([0, 0, 2, 1, 1, 1, 0, 0], dtype=np.int8)
        probability = np.array([[.05, .05, .9], [.05, .05, .9], [.9, .05, .05],
                                [.9, .05, .05], [.9, .05, .05], [.05, .9, .05],
                                [.05, .9, .05], [.05, .9, .05]])
        blocks = [(0, 3), (3, 8)]
        for config in boundaries.CONFIGS:
            actual = boundaries.relocate(codes, probability, blocks,
                                         config["radius"], config["penalty"])
            for start, end in blocks:
                _, optimum = brute_force(codes[start:end], probability[start:end],
                                         config["radius"], config["penalty"])
                old_half, new_half = codes[start:end], actual[start:end]
                np.testing.assert_array_equal(old_half[run_edges(old_half)[:-1]],
                                              new_half[run_edges(new_half)[:-1]])
                self.assertAlmostEqual(objective(actual[start:end], probability[start:end],
                                                 codes[start:end], config["radius"],
                                                 config["penalty"]), optimum)

    def test_constant_and_single_sample_halves_return_fresh_int8_codes(self):
        codes = np.array([2, 0, 0, 0, 1], dtype=np.int64)
        probability = np.tile([.01, .98, .01], (len(codes), 1))
        actual = boundaries.relocate(codes, probability, [(0, 1), (1, 4), (4, 5)], 10, 0)
        np.testing.assert_array_equal(actual, codes)
        self.assertEqual(actual.dtype, np.int8)
        self.assertFalse(np.shares_memory(actual, codes))

    def test_empty_input(self):
        actual = boundaries.relocate(np.array([], dtype=np.int8), np.empty((0, 3)), [], 2, 0)
        self.assertEqual(actual.shape, (0,))
        self.assertEqual(actual.dtype, np.int8)

    def test_zero_probabilities_use_declared_log_floor(self):
        codes = np.array([0, 0, 2, 1, 1], dtype=np.int8)
        probability = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1],
                                [0, 1e-12, 1], [0, 1, 0]], dtype=float)
        for config in boundaries.CONFIGS:
            actual = boundaries.relocate(codes, probability, [(0, 5)],
                                         config["radius"], config["penalty"])
            _, optimum = brute_force(codes, probability, config["radius"], config["penalty"])
            self.assertAlmostEqual(objective(actual, probability, codes, config["radius"],
                                             config["penalty"]), optimum, places=11)

    def test_readonly_inputs_remain_unchanged(self):
        codes = np.array([0, 0, 2, 1, 1], dtype=np.int8)
        probability = np.array([[.9, .05, .05], [.1, .1, .8], [.1, .1, .8],
                                [.1, .8, .1], [.1, .8, .1]])
        blocks = np.array([[0, 5]], dtype=np.int64)
        originals = [value.copy() for value in (codes, probability, blocks)]
        for value in (codes, probability, blocks):
            value.flags.writeable = False
        actual = boundaries.relocate(codes, probability, blocks, 2, .05)
        for value, original in zip((codes, probability, blocks), originals):
            np.testing.assert_array_equal(value, original)
            self.assertFalse(np.shares_memory(actual, value))


    def test_invalid_shapes_labels_and_bounds_fail(self):
        codes = np.array([0, 1], dtype=np.int8)
        probability = np.full((2, 3), 1 / 3)
        for radius in (0, -1, 1.5, np.nan):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                boundaries.relocate(codes, probability, [(0, 2)], radius, 0)
        for penalty in (-1, np.nan, np.inf):
            with self.subTest(penalty=penalty), self.assertRaises(ValueError):
                boundaries.relocate(codes, probability, [(0, 2)], 2, penalty)
        for invalid_codes in (np.array([0, -1]), np.array([[0, 1]])):
            with self.assertRaises(ValueError):
                boundaries.relocate(invalid_codes, probability, [(0, 2)], 2, 0)
        for invalid_probability in (np.ones((2, 2)), np.full((2, 3), np.nan),
                                    np.full((2, 3), -.1)):
            with self.assertRaises(ValueError):
                boundaries.relocate(codes, invalid_probability, [(0, 2)], 2, 0)
        for blocks in ([(0, 1)], [(0, 2), (1, 2)], [(1, 2)]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                boundaries.relocate(codes, probability, blocks, 2, 0)


if __name__ == "__main__":
    unittest.main()
