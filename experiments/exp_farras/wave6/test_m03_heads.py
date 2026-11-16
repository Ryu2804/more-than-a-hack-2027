import unittest

import numpy as np

import m03_heads as heads


def prediction(dead, team, dtype=np.float64):
    dead = np.asarray(dead, dtype=dtype)
    team = np.asarray(team, dtype=dtype)
    return {
        'probability': np.column_stack(((1 - dead) * team, (1 - dead) * (1 - team), dead)),
        'p_dead': dead,
        'p_team': team,
        'boundary': np.ones(len(dead), dtype=dtype),
        'has_boundary': True,
    }


def bundle(dead, team, other_dead, other_team, dtype=np.float64):
    return {
        'short': prediction(dead, team, dtype),
        'old': prediction(other_dead, other_team, dtype),
        'boundary': prediction(other_dead, other_team, dtype),
    }


class HeadFusionTests(unittest.TestCase):
    def test_each_dead_mixture_keeps_short_conditional_team(self):
        team = np.array([0.8, 0.2, 0.65, 0.35])
        inputs = {
            'short': prediction([0, 0.2, 0.8, 1], team),
            'old': prediction([0.8, 0.6, 0.4, 0.2], 1 - team),
            'boundary': prediction([0.1, 0.8, 0.2, 0.6], [0.1, 0.9, 0.1, 0.9]),
        }
        expected_dead = {
            'old25': [0.2, 0.3, 0.7, 0.8],
            'old50': [0.4, 0.4, 0.6, 0.6],
            'old75': [0.6, 0.5, 0.5, 0.4],
            'boundary25': [0.025, 0.35, 0.65, 0.9],
            'boundary50': [0.05, 0.5, 0.5, 0.8],
            'boundary75': [0.075, 0.65, 0.35, 0.7],
        }
        for config in heads.CONFIGS:
            with self.subTest(config=config['name']):
                output = heads.fuse(inputs, config)
                dead = np.asarray(expected_dead[config['name']])
                np.testing.assert_allclose(output['p_dead'], dead, atol=1e-14, rtol=0)
                np.testing.assert_allclose(output['p_team'], team, atol=1e-14, rtol=0)
                np.testing.assert_allclose(output['probability'][:, 0], (1 - dead) * team)
                np.testing.assert_allclose(output['probability'][:, 1], (1 - dead) * (1 - team))
                self.assertFalse(output['has_boundary'])

    def test_changing_dead_mass_cannot_change_team_odds(self):
        team = np.array([0.97, 0.03, 0.8, 0.2])
        for config in heads.CONFIGS:
            for dead in (0.0, 0.5, 1.0 - 1e-12, 1.0):
                with self.subTest(config=config['name'], dead=dead):
                    inputs = bundle(np.full(4, dead), team, np.full(4, dead), 1 - team)
                    output = heads.fuse(inputs, config)
                    np.testing.assert_allclose(output['p_team'], team, atol=1e-14, rtol=0)

    def test_zero_live_and_certain_team_edges_remain_finite(self):
        inputs = bundle([0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0])
        expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
        for config in heads.CONFIGS:
            with self.subTest(config=config['name']):
                output = heads.fuse(inputs, config)
                self.assertTrue(np.isfinite(output['probability']).all())
                self.assertTrue(np.isfinite(output['p_dead']).all())
                self.assertTrue(np.isfinite(output['p_team']).all())
                np.testing.assert_allclose(output['probability'], expected, atol=1e-15, rtol=0)
                np.testing.assert_array_equal(output['p_team'], [0, 1, 0, 1])
                np.testing.assert_allclose(output['probability'].sum(axis=1), 1, atol=1e-15, rtol=0)

    def test_float16_live_mass_is_not_rounded_away(self):
        inputs = bundle([0.9995, 0.9995], [0.0001, 0.9995],
                        [1, 1], [1, 0], dtype=np.float16)
        team = inputs['short']['p_team'].astype(np.float64)
        for config in heads.CONFIGS:
            with self.subTest(config=config['name']):
                output = heads.fuse(inputs, config)
                self.assertEqual(output['probability'].dtype, np.dtype(np.float64))
                self.assertTrue(np.all(output['probability'][:, :2] > 0))
                np.testing.assert_allclose(output['p_team'], team, atol=1e-14, rtol=0)
                np.testing.assert_allclose(output['probability'][:, 0],
                                           (1 - output['p_dead']) * team, atol=1e-15, rtol=1e-12)

    def test_home_away_swap_only_swaps_live_columns(self):
        inputs = bundle([0, 0.2, 1, 0.7], [0, 0.8, 0.3, 1],
                        [0, 0.9, 1, 0.1], [0.1, 0.9, 0.6, 0.2])
        swapped = {
            key: prediction(value['p_dead'], 1 - value['p_team'])
            for key, value in inputs.items()
        }
        for config in heads.CONFIGS:
            with self.subTest(config=config['name']):
                normal = heads.fuse(inputs, config)
                reverse = heads.fuse(swapped, config)
                np.testing.assert_allclose(reverse['probability'], normal['probability'][:, [1, 0, 2]],
                                           atol=1e-15, rtol=0)
                np.testing.assert_allclose(reverse['p_team'], 1 - normal['p_team'], atol=1e-15, rtol=0)
                np.testing.assert_array_equal(reverse['p_dead'], normal['p_dead'])

    def test_fusion_does_not_mutate_cached_heads(self):
        inputs = bundle([0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8])
        originals = {key: {field: value.copy() for field, value in source.items()
                           if isinstance(value, np.ndarray)} for key, source in inputs.items()}
        for config in heads.CONFIGS:
            heads.fuse(inputs, config)
        for key, fields in originals.items():
            for field, value in fields.items():
                np.testing.assert_array_equal(inputs[key][field], value)


if __name__ == '__main__':
    unittest.main()
