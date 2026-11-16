import copy
import unittest

import numpy as np

import classic


def moving_observations(n=24, teams=(0,), blocks=None):
    x = 0.2 + np.arange(n) * 0.6 / 105.3
    players = np.empty((n, len(teams), 2), dtype=np.float64)
    players[:, :, 0] = x[:, None]
    players[:, :, 1] = 0.5
    ball = np.column_stack([x + 0.2 / 105.3, np.full(n, 0.5)])
    features = np.zeros((n, 16), dtype=np.float64)
    features[:, 3] = 1
    return dict(players=players, ball=ball, teams=np.asarray(teams), X=features,
                blocks=np.array([[0, n]] if blocks is None else blocks))


def training_matches():
    datasets = []
    for index in range(6):
        n = 48
        features = np.zeros((n, 16), dtype=np.float64)
        features[:, :2] = [0.2, 8.0] if index < 5 else [12.0, 12.0]
        features[:, 2] = (features[:, 1] - features[:, 0]) / 3
        features[:, 3] = 1
        labels = np.full(n, 0 if index < 5 else 2, dtype=np.int8)
        datasets.append(dict(_name=f'match_{index}', X=features, y=labels,
                             blocks=np.array([[0, 20], [20, n]])))
    return datasets


class ClassicTreeTests(unittest.TestCase):
    def test_pool_keeps_all_six_matches_and_offsets_every_half(self):
        datasets = training_matches()
        pooled = classic.pool_training(datasets)
        self.assertEqual(pooled['training_names'], tuple(data['_name'] for data in datasets))
        self.assertEqual(len(pooled['X']), 288)
        for index, data in enumerate(datasets):
            start = index * 48
            np.testing.assert_array_equal(pooled['X'][start:start + 48], data['X'])
            np.testing.assert_array_equal(pooled['y'][start:start + 48], data['y'])
            np.testing.assert_array_equal(pooled['blocks'][index * 2:index * 2 + 2],
                                          data['blocks'] + start)
        mirrored = classic.tree.symmetrize(pooled)
        self.assertEqual(len(mirrored['y']), 576)
        np.testing.assert_array_equal(mirrored['y'][288:528], 1)
        np.testing.assert_array_equal(mirrored['y'][528:], 2)
        np.testing.assert_array_equal(mirrored['blocks'][12:], pooled['blocks'] + 288)

    def test_fit_uses_later_matches_and_predicts_joint_probabilities_without_mutation(self):
        datasets = training_matches()
        snapshots = copy.deepcopy(datasets)
        fitted = classic.tree_fit(datasets)
        np.testing.assert_array_equal(fitted['model'].classes_, [0, 1, 2])
        self.assertEqual(fitted['training_names'], tuple(data['_name'] for data in datasets))
        observations = dict(X=np.concatenate([data['X'][:1] for data in datasets]))
        before = observations['X'].copy()
        prediction = classic.tree_predict(fitted, observations)
        probability = prediction['probability']
        self.assertEqual(probability.shape, (6, 3))
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue((probability >= 0).all())
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        self.assertEqual(int(probability[-1].argmax()), 2)
        self.assertEqual(int(probability[0].argmax()), 0)
        np.testing.assert_array_equal(observations['X'], before)
        for data, snapshot in zip(datasets, snapshots):
            for key in ('X', 'y', 'blocks'):
                np.testing.assert_array_equal(data[key], snapshot[key])

    def test_pool_rejects_missing_or_duplicate_training_matches(self):
        datasets = training_matches()
        with self.assertRaises(ValueError):
            classic.pool_training(datasets[:2])
        datasets[-1]['_name'] = datasets[0]['_name']
        with self.assertRaises(ValueError):
            classic.pool_training(datasets)


class TouchRuleTests(unittest.TestCase):
    def test_speed_is_meters_per_second_and_resets_at_each_half(self):
        observations = moving_observations(12, blocks=[[0, 6], [6, 12]])
        observations['players'][6:, :, 0] += 0.4
        speed = classic.player_speeds(observations['players'], observations['blocks'])
        self.assertTrue(np.isnan(speed[[0, 6]]).all())
        np.testing.assert_allclose(speed[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]], 3.0,
                                   rtol=1e-12, atol=1e-12)
        y_motion = np.zeros((4, 1, 2))
        y_motion[:, 0, 1] = np.arange(4) * 0.4 / 68
        np.testing.assert_allclose(classic.player_speeds(y_motion, [[0, 4]])[1:], 2.0)

    def test_speed_requires_both_finite_endpoints_without_gap_interpolation(self):
        observations = moving_observations(9)
        players = observations['players']
        players[3, 0, 0] = np.nan
        players[6, 0, 1] = np.inf
        speed = classic.player_speeds(players, observations['blocks'])[:, 0]
        self.assertTrue(np.isnan(speed[[0, 3, 4, 6, 7]]).all())
        np.testing.assert_allclose(speed[[1, 2, 5, 8]], 3.0)

    def test_stationary_nearest_player_triggers_five_sample_dead_cue(self):
        observations = moving_observations(12)
        observations['players'][:] = observations['players'][0]
        observations['ball'][:] = observations['ball'][0]
        expected = np.r_[False, np.ones(11, dtype=bool)]
        np.testing.assert_array_equal(classic.dead_cues(observations), expected)
        np.testing.assert_array_equal(classic.touch_predict(observations), np.r_[0, np.full(11, 2)])
        moving = moving_observations(12)
        self.assertFalse(classic.dead_cues(moving).any())
        np.testing.assert_array_equal(classic.touch_predict(moving), np.zeros(12))

    def test_four_slow_samples_are_not_enough(self):
        observations = moving_observations(5)
        observations['players'][:] = observations['players'][0]
        self.assertFalse(classic.dead_cues(observations).any())

    def test_visibility_masks_finite_coordinates_and_breaks_slow_runs(self):
        observations = moving_observations(12)
        observations['X'][:, 3] = 0
        distance, team, speed = classic.nearest_player(observations)
        self.assertTrue(np.isinf(distance).all())
        np.testing.assert_array_equal(team, -1)
        self.assertTrue(np.isnan(speed).all())
        np.testing.assert_array_equal(classic.touch_predict(observations), 2)
        observations['X'][:4, 3] = [np.nan, np.inf, -np.inf, 0.5]
        np.testing.assert_array_equal(classic.touch_predict(observations), 2)
        observations = moving_observations(10)
        observations['players'][:] = observations['players'][0]
        observations['X'][5, 3] = 0
        self.assertFalse(classic.dead_cues(observations).any())

    def test_outside_cue_needs_three_visible_samples_and_keeps_short_dead(self):
        observations = moving_observations()
        observations['ball'][8:11, 0] = 1.01
        observations['ball'][16:18, 1] = -0.01
        expected = np.zeros(24, dtype=bool)
        expected[8:11] = True
        np.testing.assert_array_equal(classic.dead_cues(observations), expected)
        codes = classic.touch_predict(observations)
        np.testing.assert_array_equal(codes[8:11], 2)
        np.testing.assert_array_equal(codes[11:], 0)
        observations['X'][9, 3] = 0
        self.assertFalse(classic.dead_cues(observations).any())

    def test_nonfinite_ball_never_supplies_touch_or_outside_evidence(self):
        observations = moving_observations(12)
        observations['ball'][6:, :] = [np.nan, np.inf]
        distance, team, speed = classic.nearest_player(observations)
        self.assertTrue(np.isinf(distance[6:]).all())
        np.testing.assert_array_equal(team[6:], -1)
        self.assertTrue(np.isnan(speed[6:]).all())
        self.assertFalse(classic.dead_cues(observations).any())
        np.testing.assert_array_equal(classic.touch_predict(observations), 0)

    def test_unknown_team_slots_cannot_touch_or_supply_stationary_cue(self):
        observations = moving_observations(12, teams=(-1, 1, 2, np.nan))
        observations['players'][:, 0] = observations['ball'][0]
        observations['players'][:, 2] = observations['ball'][0]
        observations['players'][:, 3] = observations['ball'][0]
        distance, team, speed = classic.nearest_player(observations)
        np.testing.assert_allclose(distance, 0.2)
        np.testing.assert_array_equal(team, 1)
        np.testing.assert_allclose(speed[1:], 3.0)
        np.testing.assert_array_equal(classic.touch_predict(observations), 1)
        observations['teams'][:] = -1
        np.testing.assert_array_equal(classic.touch_predict(observations), 2)
        self.assertFalse(classic.dead_cues(observations).any())

    def test_metadata_permutation_preserves_rule_and_team_swap_swaps_prediction(self):
        observations = moving_observations(12, teams=(1, -1, 0))
        observations['players'][:, 2, 0] += 0.1
        baseline = classic.touch_predict(observations)
        np.testing.assert_array_equal(baseline, 1)
        order = [2, 0, 1]
        observations['players'] = observations['players'][:, order]
        observations['teams'] = observations['teams'][order]
        np.testing.assert_array_equal(classic.touch_predict(observations), baseline)
        known = np.isin(observations['teams'], (0, 1))
        observations['teams'][known] = 1 - observations['teams'][known]
        np.testing.assert_array_equal(classic.touch_predict(observations), 0)

    def test_hold_starts_at_first_touch_and_does_not_cross_half(self):
        touches = np.full(20, -1)
        touches[2] = 1
        held = classic.carry_team(touches, [[0, 10], [10, 20]])
        np.testing.assert_array_equal(held[:2], 2)
        np.testing.assert_array_equal(held[2:10], 1)
        np.testing.assert_array_equal(held[10:], 2)
        np.testing.assert_array_equal(touches[3:], -1)

    def test_six_sample_hold_survives_but_five_sample_switch_is_absorbed(self):
        touches = np.full(23, -1)
        touches[0], touches[6], touches[11], touches[17] = 0, 1, 0, 1
        held = classic.carry_team(touches, [[0, 23]])
        np.testing.assert_array_equal(held[:17], 0)
        np.testing.assert_array_equal(held[17:], 1)
        np.testing.assert_array_equal(classic.carry_team(np.r_[0, np.full(4, -1)], [[0, 5]]), 2)

    def test_cue_runs_do_not_join_across_half_boundary(self):
        observations = moving_observations(12, blocks=[[0, 6], [6, 12]])
        observations['ball'][4:8, 0] = 1.1
        self.assertFalse(classic.dead_cues(observations).any())
        stationary = moving_observations(10, blocks=[[0, 5], [5, 10]])
        stationary['players'][:] = stationary['players'][0]
        self.assertFalse(classic.dead_cues(stationary).any())

    def test_hidden_labels_and_cached_distances_cannot_change_rule_or_mutate_inputs(self):
        observations = moving_observations(24)
        observations['ball'][:3, 0] += 0.1
        observations['ball'][14:18] = np.nan
        observations['y'] = np.ones(24, dtype=np.int8)
        observations['y25'] = np.ones(120, dtype=np.int8)
        observations['distances'] = np.zeros((24, 2))
        before = copy.deepcopy(observations)
        for value in observations.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        actual = classic.touch_predict(observations)
        expected = np.r_[np.full(3, 2), np.zeros(21)]
        np.testing.assert_array_equal(actual, expected)
        stripped = {key: value for key, value in observations.items()
                    if key not in ('y', 'y25', 'distances')}
        np.testing.assert_array_equal(classic.touch_predict(stripped), expected)
        for key in observations:
            np.testing.assert_array_equal(observations[key], before[key])


if __name__ == '__main__':
    unittest.main()
