import unittest

import numpy as np

import core
import m02_visibility as visibility_method


class VisibilityFusionTests(unittest.TestCase):
    def fixture(self, visibility, blocks):
        count = len(visibility)
        short = np.resize(np.array([[0.72, 0.18, 0.10], [0.09, 0.21, 0.70]]), (count, 3))
        old = np.resize(np.array([[0.12, 0.48, 0.40], [0.48, 0.32, 0.20]]), (count, 3))
        features = np.full((count, 16), 0.93)
        features[:, 3] = visibility
        bundle = {'short': core.result(short), 'old': core.result(old)}
        observations = {
            'X': features,
            'blocks': blocks,
            'distances': np.full((count, 2), 2.0),
        }
        return bundle, observations

    def assert_mixture(self, actual, bundle, short_weight):
        weight = np.asarray(short_weight)
        if weight.ndim:
            weight = weight[:, None]
        expected = (
            weight * bundle['short']['probability']
            + (1 - weight) * bundle['old']['probability']
        )
        np.testing.assert_allclose(actual['probability'], expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual['p_dead'], expected[:, 2], rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            actual['p_team'], expected[:, 0] / expected[:, :2].sum(axis=1), rtol=0, atol=1e-14
        )

    def test_absent_and_visible_use_exact_endpoint_joint_mixtures(self):
        for config in visibility_method.CONFIGS:
            for visible, key in ((0, 'missing_weight'), (1, 'visible_weight')):
                with self.subTest(config=config['name'], visible=visible):
                    bundle, observations = self.fixture(np.full(17, visible), [(0, 17)])
                    actual = visibility_method.fuse(bundle, observations, config)
                    self.assert_mixture(actual, bundle, config[key])
                    if config[key] == 1:
                        np.testing.assert_allclose(
                            actual['probability'], bundle['short']['probability'], rtol=0, atol=1e-14
                        )

    def test_half_boundary_does_not_share_visibility(self):
        bundle, observations = self.fixture([0] * 7 + [1] * 4, [(0, 7), (7, 11)])
        for config in visibility_method.CONFIGS:
            with self.subTest(config=config['name']):
                actual = visibility_method.fuse(bundle, observations, config)
                expected_weight = np.r_[
                    np.full(7, config['missing_weight']),
                    np.full(4, config['visible_weight']),
                ]
                self.assert_mixture(actual, bundle, expected_weight)

    def test_centered_eleven_samples_clip_edges_without_zero_padding(self):
        bundle, observations = self.fixture(
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [(0, 13)]
        )
        expected_visibility = np.array([
            1 / 6, 1 / 7, 1 / 8, 1 / 9, 1 / 10, 1 / 11, 1 / 11,
            2 / 11, 2 / 10, 2 / 9, 2 / 8, 1 / 7, 1 / 6,
        ])
        for config in visibility_method.CONFIGS:
            with self.subTest(config=config['name']):
                actual = visibility_method.fuse(bundle, observations, config)
                weight = config['missing_weight'] + (
                    config['visible_weight'] - config['missing_weight']
                ) * expected_visibility
                self.assert_mixture(actual, bundle, weight)

    def test_inputs_unchanged_and_output_not_aliased(self):
        bundle, observations = self.fixture([0, 1, 0, 1, 1, 0, 0], [(0, 3), (3, 7)])
        inputs = [
            value
            for prediction in bundle.values()
            for value in prediction.values()
            if isinstance(value, np.ndarray)
        ] + [observations['X'], observations['distances']]
        snapshots = [value.copy() for value in inputs]
        original_blocks = list(observations['blocks'])
        for value in inputs:
            value.setflags(write=False)
        for config in visibility_method.CONFIGS:
            original_config = dict(config)
            with self.subTest(config=config['name']):
                actual = visibility_method.fuse(bundle, observations, config)
                for key in ('probability', 'p_dead', 'p_team'):
                    for source in inputs:
                        self.assertFalse(np.shares_memory(actual[key], source))
                    actual[key][...] = 0
                for value, snapshot in zip(inputs, snapshots):
                    np.testing.assert_array_equal(value, snapshot)
                self.assertEqual(observations['blocks'], original_blocks)
                self.assertEqual(config, original_config)


if __name__ == '__main__':
    unittest.main()
