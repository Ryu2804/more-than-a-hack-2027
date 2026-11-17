import itertools
import unittest

import numpy as np

import decode
import ensemble
import tokens


class CombinedInvariants(unittest.TestCase):
    def test_decoder_matches_exhaustive_centered_optimum(self):
        random = np.random.default_rng(82)
        p = random.dirichlet([1, 1, 1], 6)
        boundary = np.array([.1, .2, .9, .05, .8, .15])
        logit = lambda value: np.log(value) - np.log1p(-value)
        for config in decode.CONFIGS:
            def score(path):
                value = sum(np.log(p[t, state]) for t, state in enumerate(path))
                for t in range(1, len(path)):
                    if path[t] != path[t - 1]:
                        cost = config['lambda_dead'] if 2 in (path[t], path[t - 1]) else config['lambda_team']
                        cost = max(0, cost - config['gamma'] * (logit(boundary[t]) - logit(boundary.mean())))
                        value -= cost
                return value
            result = decode.decode(p, boundary, [(0, 6)], config)
            optimum = max(score(path) for path in itertools.product(range(3), repeat=6))
            self.assertAlmostEqual(score(result), optimum, places=10)

    def test_decoder_resets_periods_and_accepts_zero_mass_classes(self):
        p = np.array([[1., 0, 0], [0, 1., 0]])
        result = decode.decode(p, np.array([0., 1.]), [(0, 1), (1, 2)], decode.CONFIGS[2])
        np.testing.assert_array_equal(result, [0, 1])
        self.assertEqual(decode.decode(np.empty((0, 3)), np.empty(0), [], decode.CONFIGS[0]).dtype, np.int8)
        with self.assertRaises(ValueError):
            decode.decode(np.full((2, 3), 1 / 3), np.zeros(2), [(0, 1)], decode.CONFIGS[0])

    def test_visibility_fusion_uses_observations_and_preserves_inputs(self):
        neural = {'probability': np.array([[.8, .1, .1], [.8, .1, .1]]), 'boundary': np.array([.2, .2])}
        tree = {'probability': np.array([[.1, .8, .1], [.1, .8, .1]]), 'boundary': np.array([.8, .8])}
        x = np.zeros((2, 16)); x[0, 3] = 1
        result = ensemble.combine(neural, tree, {'X': x}, ensemble.CONFIGS[3])
        np.testing.assert_allclose(result['probability'], [[.275, .625, .1], [.625, .275, .1]])
        np.testing.assert_allclose(result['boundary'], [.65, .35])
        np.testing.assert_allclose(neural['probability'][0], [.8, .1, .1])

    def test_token_permutation_and_mirror_involution(self):
        players = np.array([[[.2, .3], [.7, .6], [.25, .4], [.75, .5]]] * 6, dtype=np.float32)
        ball = np.array([[.3, .4]] * 3 + [[np.nan, np.nan]] * 3, dtype=np.float32)
        x = np.zeros((6, 16), dtype=np.float32); x[:3, 3] = 1
        data = dict(players=players, teams=np.array([0, 1, 0, 1]), ball=ball, X=x, blocks=[(0, 3), (3, 6)])
        built = tokens.build(data)
        perm = [3, 1, 2, 0]
        other = tokens.build(dict(data, players=players[:, perm], teams=data['teams'][perm]))
        for key in ('tokens', 'mask', 'glob'):
            np.testing.assert_allclose(built[key], other[key], atol=1e-5)
        state = np.array([0, 1, 2, -1, 0, 1], dtype=np.int8)
        mirrored = tokens.mirror(built['tokens'], built['glob'], state)
        restored = tokens.mirror(*mirrored)
        for expected, actual in zip((built['tokens'], built['glob'], state), restored):
            np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.isfinite(built['tokens']).all())
        self.assertTrue(np.isfinite(built['glob']).all())

    def test_tokens_preserve_control_evidence_and_period_local_speed(self):
        players = np.full((6, 2, 2), .2, dtype=np.float32)
        players[3:] = .8
        x = np.zeros((6, 16), dtype=np.float32)
        x[:, [0, 1, 2, 3, 7, 14, 15]] = [3, 6, 1, 0, 17, 12, 24]
        built = tokens.build(dict(players=players, teams=np.array([0, 1]),
            ball=np.full((6, 2), np.nan), X=x, blocks=[(0, 3), (3, 6)]))
        np.testing.assert_allclose(built['glob'][:, 12:],
            np.tile([.1, .2, .1, 0, .5, .2, .4], (6, 1)), atol=1e-7)
        np.testing.assert_array_equal(built['tokens'][:, :, 2:5], 0)

    def test_target_team_distance_missingness_must_match_cache(self):
        import prepare_local
        players = np.array([[[0., 0.], [np.nan, np.nan]],
                            [[0., 0.], [2., 0.]]], dtype=np.float32)
        ball = np.array([[1., 0.], [1., 0.]], dtype=np.float32)
        cached = np.array([[1., np.inf], [1., 1.]], dtype=np.float32)
        teams = np.array([0, 1], dtype=np.int8)
        audit = prepare_local._validate_target_distances(players, ball, teams, cached)
        self.assertEqual(audit['finite_team_distances_checked'], 3)
        self.assertEqual(audit['unavailable_team_distances_on_visible_frames'], 1)
        wrong = cached.copy(); wrong[0, 1] = 1
        with self.assertRaises(ValueError):
            prepare_local._validate_target_distances(players, ball, teams, wrong)


if __name__ == '__main__':
    unittest.main()
