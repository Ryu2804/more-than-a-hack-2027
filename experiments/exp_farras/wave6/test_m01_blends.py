import copy
import unittest

import numpy as np

import core
import m01_blends as blends


class BlendTests(unittest.TestCase):
    def bundle(self):
        return {
            'short': core.result([[0.02, 0.08, 0.90], [0.72, 0.18, 0.10]]),
            'old': core.result([[0.81, 0.09, 0.10], [0.02, 0.08, 0.90]]),
            'boundary': core.result([[0.10, 0.80, 0.10], [0.10, 0.10, 0.80]]),
        }

    def test_joint_mixture_weights_live_mass_before_conditioning(self):
        fused = blends.fuse(self.bundle(), {'other': 'old', 'weight': 0.5})
        np.testing.assert_allclose(fused['probability'], [[0.415, 0.085, 0.5], [0.37, 0.13, 0.5]])
        np.testing.assert_allclose(fused['p_dead'], [0.5, 0.5])
        np.testing.assert_allclose(fused['p_team'], [0.83, 0.74])
        self.assertFalse(np.allclose(fused['p_team'], [0.55, 0.5]))


    def test_read_only_inputs_and_config_remain_unchanged(self):
        bundle = self.bundle()
        original = copy.deepcopy(bundle)
        config = copy.deepcopy(blends.CONFIGS[-1])
        original_config = copy.deepcopy(config)
        for prediction in bundle.values():
            for value in prediction.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
        fused = blends.fuse(bundle, config)
        for key, prediction in bundle.items():
            self.assertEqual(set(prediction), set(original[key]))
            for field, value in prediction.items():
                np.testing.assert_equal(value, original[key][field])
                if isinstance(value, np.ndarray):
                    for output in fused.values():
                        if isinstance(output, np.ndarray):
                            self.assertFalse(np.shares_memory(value, output))
        self.assertEqual(config, original_config)

    def test_home_away_swap_commutes_with_joint_fusion(self):
        bundle = self.bundle()
        swapped = {
            key: core.result(prediction['probability'][:, [1, 0, 2]])
            for key, prediction in bundle.items()
        }
        for config in blends.CONFIGS:
            with self.subTest(config=config['name']):
                fused = blends.fuse(bundle, config)
                reverse = blends.fuse(swapped, config)
                np.testing.assert_allclose(reverse['probability'], fused['probability'][:, [1, 0, 2]])
                np.testing.assert_allclose(reverse['p_dead'], fused['p_dead'])
                np.testing.assert_allclose(reverse['p_team'], 1 - fused['p_team'])

    def test_all_dead_joint_rows_stay_finite(self):
        bundle = {key: core.result([[0, 0, 1]]) for key in ('short', 'old', 'boundary')}
        for config in blends.CONFIGS:
            with self.subTest(config=config['name']):
                fused = blends.fuse(bundle, config)
                np.testing.assert_allclose(fused['probability'], [[0, 0, 1]], atol=1e-6)
                self.assertTrue(np.isfinite(fused['p_team']).all())
                self.assertGreater(float(fused['p_dead'][0]), 0.999)

    def test_predict_preserves_half_local_dead_duration(self):
        probability = np.tile([0.81, 0.09, 0.1], (40, 1))
        probability[20:] = [0.09, 0.81, 0.1]
        probability[5:10] = [0.08, 0.02, 0.9]
        probability[16:20] = [0.08, 0.02, 0.9]
        probability[20:24] = [0.02, 0.08, 0.9]
        bundle = {key: core.result(probability) for key in ('short', 'old', 'boundary')}
        observations = {
            'X': np.zeros((40, 16)),
            'blocks': [(0, 20), (20, 40)],
            'distances': np.ones((40, 2)),
        }
        expected = np.array([0] * 5 + [2] * 5 + [0] * 10 + [1] * 20, dtype=np.int8)
        original = copy.deepcopy(observations)
        for config in blends.CONFIGS:
            with self.subTest(config=config['name']):
                actual = blends.predict(bundle, observations, {'rate': 0.0, 'dead_share': 0.3}, config)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.int8)
        np.testing.assert_array_equal(observations['X'], original['X'])
        np.testing.assert_array_equal(observations['distances'], original['distances'])
        self.assertEqual(observations['blocks'], original['blocks'])


if __name__ == '__main__':
    unittest.main()
