import unittest

import numpy as np

import core
import m06_short_guard as method


def example(codes, team_probability=None, visibility=None, blocks=None, dead_probability=0.0):
    codes = np.asarray(codes, dtype=np.int8)
    n = len(codes)
    features = np.ones((n, 16), dtype=np.float64)
    features[:, 3] = 0 if visibility is None else visibility
    observations = {
        'X': features,
        'blocks': np.asarray([(0, n)] if blocks is None else blocks, dtype=np.int64),
    }
    p_team = np.full(n, 0.5) if team_probability is None else np.asarray(team_probability)
    prediction = core.from_heads(np.full(n, dead_probability), p_team)
    return codes, observations, prediction


class ShortGuardTests(unittest.TestCase):
    def test_low_confidence_occluded_excursion_corrected_for_both_teams(self):
        for team in (0, 1):
            with self.subTest(team=team):
                codes = [1 - team] * 8 + [team] * 5 + [1 - team] * 8
                confidence = np.array([0.95, 0.20, 0.50, 0.50, 0.55])
                p_team = np.full(len(codes), 0.5)
                p_team[8:13] = confidence if team == 0 else 1 - confidence
                visibility = np.ones(len(codes))
                visibility[8:13] = [0, 0, 1, 0, 0]
                values, observations, prediction = example(codes, p_team, visibility)
                result = method.guard(values, observations, prediction, 1, 0.60)
                np.testing.assert_array_equal(result, np.full(len(codes), 1 - team))
                self.assertEqual(result.dtype, np.dtype(np.int8))

    def test_strong_short_spells_use_conditional_not_joint_probability(self):
        for team in (0, 1):
            with self.subTest(team=team):
                codes = [1 - team] * 8 + [team] * 5 + [1 - team] * 8
                p_team = np.full(len(codes), 0.95 if team == 0 else 0.05)
                values, observations, prediction = example(codes, p_team, dead_probability=0.40)
                result = method.guard(values, observations, prediction, 3, 0.70)
                np.testing.assert_array_equal(result, values)

    def test_duration_limit_includes_exact_boundary_not_next_sample(self):
        for max_s in (1, 2, 3):
            for extra in (0, 1):
                with self.subTest(max_s=max_s, extra=extra):
                    length = max_s * 5 + extra
                    values, observations, prediction = example([0] * 20 + [1] * length + [0] * 20)
                    result = method.guard(values, observations, prediction, max_s, 0.60)
                    expected = values if extra else np.zeros(len(values), dtype=np.int8)
                    np.testing.assert_array_equal(result, expected)

    def test_visibility_requires_strictly_less_than_half_visible(self):
        for visible_count in (1, 2, 3):
            with self.subTest(visible_count=visible_count):
                visibility = np.ones(20)
                visibility[8:12] = 0
                visibility[8:8 + visible_count] = 1
                values, observations, prediction = example([0] * 8 + [1] * 4 + [0] * 8,
                                                             visibility=visibility)
                result = method.guard(values, observations, prediction, 1, 0.60)
                expected = np.zeros(20, dtype=np.int8) if visible_count == 1 else values
                np.testing.assert_array_equal(result, expected)

    def test_confidence_threshold_is_strict_and_changes_behavior(self):
        for confidence in (0.60, 0.70):
            for run_confidence in (confidence, confidence - 0.01):
                with self.subTest(confidence=confidence, run_confidence=run_confidence):
                    values, observations, prediction = example([1] * 8 + [0] * 5 + [1] * 8,
                                                               np.full(21, run_confidence))
                    result = method.guard(values, observations, prediction, 1, confidence)
                    expected = values if run_confidence == confidence else np.ones(21, dtype=np.int8)
                    np.testing.assert_array_equal(result, expected)
        values, observations, prediction = example([1] * 8 + [0] * 5 + [1] * 8, np.full(21, 0.65))
        np.testing.assert_array_equal(method.guard(values, observations, prediction, 1, 0.60), values)
        np.testing.assert_array_equal(method.guard(values, observations, prediction, 1, 0.70), np.ones(21))

    def test_neighbor_support_cannot_cross_half_boundary(self):
        for split in (6, 7, 8):
            with self.subTest(split=split):
                values, observations, prediction = example([0] * 6 + [1] * 2 + [0] * 6,
                                                           blocks=[(0, split), (split, 14)])
                np.testing.assert_array_equal(method.guard(values, observations, prediction, 3, 0.70), values)

    def test_dead_runs_are_not_replaced_or_skipped_for_support(self):
        cases = (
            [0] * 8 + [2] + [1] * 2 + [0] * 8,
            [0] * 8 + [1] * 2 + [2] + [0] * 8,
            [0] * 8 + [2] * 2 + [0] * 8,
        )
        for codes in cases:
            with self.subTest(codes=codes):
                values, observations, prediction = example(codes)
                np.testing.assert_array_equal(method.guard(values, observations, prediction, 3, 0.70), values)

    def test_half_edges_without_two_neighbors_are_preserved(self):
        for codes in ([1] * 2 + [0] * 8, [0] * 8 + [1] * 2, [1] * 2):
            with self.subTest(codes=codes):
                values, observations, prediction = example(codes)
                np.testing.assert_array_equal(method.guard(values, observations, prediction, 3, 0.70), values)

    def test_adjacent_replacements_use_fixed_original_boundaries(self):
        values, observations, prediction = example([0] * 8 + [1] * 2 + [0] * 2 + [1] * 2 + [0] * 8)
        result = method.guard(values, observations, prediction, 1, 0.60)
        np.testing.assert_array_equal(result, [0] * 10 + [1] * 2 + [0] * 10)

    def test_guard_leaves_input_codes_features_and_probabilities_unchanged(self):
        values, observations, prediction = example([0] * 8 + [1] * 5 + [0] * 8)
        original_codes = values.copy()
        original_features = observations['X'].copy()
        original_predictions = {key: value.copy() for key, value in prediction.items()
                                if isinstance(value, np.ndarray)}
        result = method.guard(values, observations, prediction, 1, 0.60)
        np.testing.assert_array_equal(result, np.zeros(len(values)))
        np.testing.assert_array_equal(values, original_codes)
        np.testing.assert_array_equal(observations['X'], original_features)
        for key, original in original_predictions.items():
            np.testing.assert_array_equal(prediction[key], original)
        self.assertFalse(np.shares_memory(result, values))


if __name__ == '__main__':
    unittest.main()
