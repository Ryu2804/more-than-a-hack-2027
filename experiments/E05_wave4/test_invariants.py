import itertools
import unittest

import numpy as np

import decoder
import fusion


class DecoderInvariants(unittest.TestCase):
    def test_two_state_viterbi_matches_exhaustive_optimum(self):
        home = np.array([-.1, -2, -.2, -3, -4, -.1])
        away = np.array([-2, -.1, -3, -.1, -.2, -3])
        for penalty in (.05, .5, 2, 10):
            def score(path):
                emission = sum(home[t] if state == 0 else away[t] for t, state in enumerate(path))
                return emission - penalty * sum(path[t] != path[t - 1] for t in range(1, len(path)))
            result = decoder.viterbi_2state(home, away, penalty)
            self.assertAlmostEqual(score(result), max(score(path) for path in itertools.product((0, 1), repeat=6)))

    def test_dead_duration_does_not_cross_half_boundary(self):
        p = np.full(10, .99)
        np.testing.assert_array_equal(decoder.dead_mask(p, [(0, 5), (5, 10)]), False)
        np.testing.assert_array_equal(decoder.dead_mask(p, [(0, 10)]), True)
        np.testing.assert_array_equal(decoder.dead_mask(p[:9], [(0, 9)]), False)

    def test_live_slivers_do_not_erase_stoppage_or_first_run(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 0, 0], dtype=np.int8)
        result = decoder._merge_short_runs(labels.copy(), np.array([[0, len(labels)]]), 3)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 1])

    def test_rate_grid_accepts_nonmonotonic_counts(self):
        examples = {.1: np.array([0, 1, 0, 0]), 1.: np.array([0, 0, 0, 0]),
                    2.: np.array([0, 0, 1, 1])}
        self.assertEqual(decoder.calibrate_lambda(examples.__getitem__, [(0, 4)], 2, (.1, 1., 2.)), 2.)
        self.assertEqual(decoder.calibrate_lambda(examples.__getitem__, [(0, 4)], 1.5, (.1, 1., 2.)), 1.)

    def test_ball_gate_requires_visible_known_distances(self):
        n = 30
        x = np.zeros((n, 16)); x[:, 3] = 1
        distances = np.tile([5., .1], (n, 1))
        prediction = dict(p_dead=np.full(n, .01), p_team=np.full(n, .99))
        config = dict(next(value for value in decoder.CONFIGS if value['name'] == 'fixed2'), shift=False)
        stats = dict(rate=3., dead_share=.4)
        visible = decoder.decode(prediction, dict(X=x, distances=distances, blocks=[(0, n)]), config, stats)
        np.testing.assert_array_equal(visible['codes'], 1)
        hidden_x = x.copy(); hidden_x[:, 3] = 0
        hidden = decoder.decode(prediction, dict(X=hidden_x, distances=distances, blocks=[(0, n)]), config, stats)
        np.testing.assert_array_equal(hidden['codes'], 0)
        missing = distances.copy(); missing[:, 0] = np.nan
        unknown = decoder.decode(prediction, dict(X=x, distances=missing, blocks=[(0, n)]), config, stats)
        np.testing.assert_array_equal(unknown['codes'], 0)

    def test_empty_and_partial_block_contract(self):
        output = decoder.decode(dict(probability=np.empty((0, 3))), dict(X=np.empty((0, 16)), blocks=[]),
            decoder.CONFIGS[0], dict(rate=3., dead_share=.4))
        self.assertEqual(output['codes'].shape, (0,))
        with self.assertRaises(ValueError):
            decoder.decode(dict(probability=np.full((2, 3), 1 / 3)), dict(X=np.zeros((2, 16)), blocks=[(0, 1)]),
                decoder.CONFIGS[0], dict(rate=3., dead_share=.4))

    def test_probability_fusion_factorizes_conditional_team(self):
        primary = dict(probability=np.array([[.6, .2, .2]]), boundary=np.array([.2]))
        other = dict(probability=np.array([[.1, .1, .8]]), boundary=np.array([.8]))
        result = fusion.combine(primary, other, fusion.CONFIGS[0])
        np.testing.assert_allclose(result['probability'], [[.475, .175, .35]])
        np.testing.assert_allclose(result['p_dead'], [.35])
        np.testing.assert_allclose(result['p_team'], [.475 / .65])
        np.testing.assert_allclose(primary['probability'], [[.6, .2, .2]])
        empty = dict(probability=np.empty((0, 3)))
        self.assertEqual(fusion.combine(empty, empty, fusion.CONFIGS[0])['probability'].shape, (0, 3))

    def test_player_set_uses_opposing_team_not_self(self):
        import features
        players = np.tile(np.array([[[.2, .3], [.3, .4], [.4, .5], [.5, .6]]],
                                   dtype=np.float32), (20, 1, 1))
        data = dict(players=players, ball=np.full((20, 2), np.nan),
                    teams=np.array([0, 0, 1, 1]), X=np.zeros((20, 16)), blocks=[(0, 10), (10, 20)])
        result = features.build(data)
        self.assertTrue(np.all(result['pp'][:, :, :4, 9] > 0))
        permutation = [3, 1, 2, 0]
        changed = features.build(dict(data, players=players[:, permutation],
                                      teams=data['teams'][permutation]))
        np.testing.assert_allclose(changed['player'], result['player'], atol=1e-4)
        np.testing.assert_allclose(changed['pp'][:, :, :4],
                                   result['pp'][:, :, permutation], atol=1e-3)

    def test_feature_windows_and_velocity_reset_at_halves(self):
        import features
        values = np.r_[np.zeros(10), np.ones(10)]
        rolled = features._roll(values, [(0, 10, 1), (10, 20, 2)])
        np.testing.assert_allclose(rolled, values)
        x = np.array([[0.], [.2], [.4], [90.], [90.2], [90.4]], dtype=np.float32)
        speed = features._player_speeds(x, np.zeros_like(x), np.ones_like(x, dtype=bool),
                                        [(0, 3, 1), (3, 6, 2)])
        self.assertTrue(np.isnan(speed[3, 0]))
        np.testing.assert_allclose(speed[[1, 2, 4, 5], 0], 1, atol=1e-4)

    def test_additional_ball_mask_hides_controls_and_player_ball_fields(self):
        import models
        n = 400
        prepared = dict(player=np.ones((2, n, 23), dtype=np.float16),
            ball=np.ones((2, n, 5), dtype=np.float16), valid=np.ones((2, n), dtype=bool),
            pp=np.ones((2, n, 48, 10), dtype=np.float16), ppm=np.ones((2, n, 48), dtype=bool),
            control=np.ones((2, n, 7), dtype=np.float16), blocks=[(0, n)])
        settings = dict(models.CONFIGS[2], batch=2, mask_full=1., noise=0.)
        records = models._prepare_records([dict(_features=prepared, y=np.zeros(n, dtype=np.int8),
                                                _source='idsse', _name='toy')], settings)
        mean, scale = models._player_standardisation(records)
        x, pp, _, _, _ = models._make_batch(records, models._balanced_index(records),
                                            mean, scale, settings, np.random.default_rng(4))
        np.testing.assert_array_equal(x[:, 24:36], 0)
        np.testing.assert_array_equal(pp[:, :, :, 6:9], 0)
        np.testing.assert_array_equal(pp[:, :, :, :6], 1)
        np.testing.assert_array_equal(prepared['control'], 1)

    def test_shipped_weights_reject_in_sample_provider_evaluation(self):
        import reference
        for name in ('J03WQQ', 'skillcorner_2010085'):
            with self.assertRaises(ValueError):
                reference.predict(name)


if __name__ == '__main__':
    unittest.main()
