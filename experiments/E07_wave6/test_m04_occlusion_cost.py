import unittest

import numpy as np

import core
import m04_occlusion_cost as method


class OcclusionCostTests(unittest.TestCase):
    @staticmethod
    def observations(visibility, blocks):
        X = np.zeros((len(visibility), 16), dtype=np.int8)
        X[:, 3] = visibility
        return {
            'X': X,
            'blocks': np.asarray(blocks, dtype=np.int64),
            'distances': np.full((len(X), 2), 10.0),
        }

    def test_visible_and_missing_costs(self):
        visible = self.observations(np.ones(12), [(0, 6), (6, 12)])
        missing = self.observations(np.zeros(12), [(0, 6), (6, 12)])
        for config in method.CONFIGS:
            with self.subTest(config=config['name']):
                np.testing.assert_array_equal(method.weights(visible, config), np.ones(12))
                np.testing.assert_array_equal(
                    method.weights(missing, config), np.full(12, config['factor'])
                )

    def test_window_one_preserves_fractional_penalty_and_input(self):
        observations = self.observations([1, 0, 1, 0, 0], [(0, 3), (3, 5)])
        original = observations['X'].copy()
        actual = method.weights(observations, {'factor': 1.25, 'window': 1})
        np.testing.assert_array_equal(actual, [1.0, 1.25, 1.0, 1.25, 1.25])
        self.assertEqual(actual.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(observations['X'], original)

    def test_centered_window_clips_edges_and_resets_at_half(self):
        observations = self.observations([0] * 7 + [1] * 8 + [0] * 4, [(0, 15), (15, 19)])
        visible_mean = np.array([
            0, 0, 1 / 8, 2 / 9, 3 / 10, 4 / 11, 5 / 11, 6 / 11,
            7 / 11, 8 / 11, 8 / 10, 8 / 9, 1, 1, 1, 0, 0, 0, 0,
        ])
        actual = method.weights(observations, {'factor': 1.5, 'window': 11})
        np.testing.assert_allclose(actual, 1.5 - 0.5 * visible_mean, rtol=0, atol=1e-15)

    def test_window_longer_than_half_does_not_mix_halves(self):
        observations = self.observations([1, 1, 0, 0, 0], [(0, 2), (2, 5)])
        actual = method.weights(observations, {'factor': 2.0, 'window': 11})
        np.testing.assert_array_equal(actual, [1, 1, 2, 2, 2])

    def test_unit_cost_matches_frozen_decoder_with_half_and_dead_resets(self):
        observations = self.observations(np.ones(80), [(0, 40), (40, 80)])
        p_team = np.full(80, 0.99)
        p_team[10:40] = 0.01
        p_dead = np.full(80, 0.01)
        p_dead[20:25] = 0.9
        p_dead[55:60] = 0.9
        prediction = core.from_heads(p_dead, p_team)
        original_probability = prediction['probability'].copy()
        stats = {'rate': 18.75, 'dead_share': 0.125}
        expected = np.zeros(80, dtype=np.int8)
        expected[10:40] = 1
        expected[20:25] = 2
        expected[55:60] = 2
        baseline = core.decode(prediction, observations, stats)
        np.testing.assert_array_equal(baseline, expected)
        for config in method.CONFIGS:
            with self.subTest(config=config['name']):
                actual = method.predict({'short': prediction}, observations, stats, config)
                np.testing.assert_array_equal(actual, baseline)
                self.assertEqual(actual.dtype, np.dtype(np.int8))
        np.testing.assert_array_equal(prediction['probability'], original_probability)


if __name__ == '__main__':
    unittest.main()
