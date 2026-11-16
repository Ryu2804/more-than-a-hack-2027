import unittest

import numpy as np

import core
import m05_dead_hysteresis as method


class HysteresisTests(unittest.TestCase):
    def test_long_weak_run_without_high_seed_is_live(self):
        values = np.r_[0.1, np.full(20, 0.48), 0.1]
        actual = method.hysteresis_mask(values, [(0, len(values))], 0.35, 0.55)
        np.testing.assert_array_equal(actual, np.zeros(len(values), dtype=bool))

    def test_high_seed_accepts_entire_component_before_and_after_seed(self):
        values = np.array([0.1, 0.36, 0.40, 0.56, 0.39, 0.42, 0.1])
        values.setflags(write=False)
        actual = method.hysteresis_mask(values, [(0, len(values))], 0.35, 0.55)
        np.testing.assert_array_equal(actual, [False, True, True, True, True, True, False])
        np.testing.assert_array_equal(values, [0.1, 0.36, 0.40, 0.56, 0.39, 0.42, 0.1])
        self.assertEqual(actual.dtype, np.dtype(bool))

    def test_low_equality_splits_components_and_does_not_spread_seed(self):
        values = np.array([0.40, 0.40, 0.70, 0.40, 0.40, 0.35,
                           0.40, 0.40, 0.40, 0.40, 0.40])
        actual = method.hysteresis_mask(values, [(0, len(values))], 0.35, 0.55)
        np.testing.assert_array_equal(actual, [True] * 5 + [False] * 6)

    def test_high_equality_never_seeds_but_next_float_does(self):
        for config in method.CONFIGS:
            with self.subTest(config=config['name']):
                high = config['high']
                values = np.full(5, high)
                equal = method.hysteresis_mask(values, [(0, 5)], config['low'], high)
                np.testing.assert_array_equal(equal, [False] * 5)
                values[2] = np.nextafter(high, np.inf)
                above = method.hysteresis_mask(values, [(0, 5)], config['low'], high)
                np.testing.assert_array_equal(above, [True] * 5)

    def test_four_samples_rejected_and_five_samples_retained(self):
        values = np.array([0.40, 0.70, 0.40, 0.40, 0.1,
                           0.40, 0.40, 0.70, 0.40, 0.40])
        actual = method.hysteresis_mask(values, [(0, len(values))], 0.35, 0.55)
        np.testing.assert_array_equal(actual, [False] * 5 + [True] * 5)

    def test_half_boundary_prevents_seeding_in_either_direction(self):
        for seed_index, expected in ((2, [True] * 5 + [False] * 5),
                                     (7, [False] * 5 + [True] * 5)):
            with self.subTest(seed_index=seed_index):
                values = np.full(10, 0.40)
                values[seed_index] = 0.70
                actual = method.hysteresis_mask(values, [(0, 5), (5, 10)], 0.35, 0.55)
                np.testing.assert_array_equal(actual, expected)

    def test_minimum_duration_cannot_accumulate_across_halves(self):
        values = np.array([0.40, 0.70, 0.40, 0.40, 0.70, 0.40])
        actual = method.hysteresis_mask(values, [(0, 3), (3, 6)], 0.35, 0.55)
        np.testing.assert_array_equal(actual, [False] * 6)


    def test_predict_emits_seeded_dead_component_and_keeps_short_live_team(self):
        dead = np.r_[np.full(10, 0.1), np.full(5, 0.48), np.full(5, 0.1),
                     0.40, 0.40, 0.70, 0.40, 0.40, np.full(10, 0.1)]
        short = core.from_heads(dead, np.full(len(dead), 0.99))
        other = core.from_heads(np.full(len(dead), 0.90), np.full(len(dead), 0.01))
        probability_before = short['probability'].copy()
        team_before = short['p_team'].copy()
        features = np.zeros((len(dead), 16))
        features[:, 3] = 1.0
        observations = {'X': features, 'blocks': [(0, len(dead))],
                        'distances': np.full((len(dead), 2), 5.0)}
        actual = method.predict({'short': short, 'old': other, 'boundary': other},
                                observations, {'rate': 0.0, 'dead_share': 0.1},
                                {'name': 'low35_high55', 'low': 0.35, 'high': 0.55})
        expected = np.zeros(len(dead), dtype=np.int8)
        expected[20:25] = 2
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype(np.int8))
        np.testing.assert_array_equal(short['probability'], probability_before)
        np.testing.assert_array_equal(short['p_team'], team_before)


if __name__ == '__main__':
    unittest.main()
