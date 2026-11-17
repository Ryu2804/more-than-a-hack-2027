import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

import context


def moving_source(n=220, teams=(0, 1)):
    frame = np.arange(n)
    players = np.zeros((n, len(teams), 3), dtype=np.float64)
    players[:, :, 0] = 20 + frame[:, None] * 0.12 + np.arange(len(teams))[None, :] * 10
    players[:, :, 1] = 30
    players[:, :, 2] = 9999
    ball = players[:, 0, :2].copy()
    ball[:, 0] += 0.2
    return players, ball, np.asarray(teams), np.ones(n, dtype=np.int8)


def prepared_from(match):
    blocks25 = context.half_blocks(match.half)
    parts = [np.arange(a, z, 5, dtype=np.int64) for a, z in blocks25]
    sampled = np.concatenate(parts)
    ends = np.cumsum([len(part) for part in parts])
    starts = np.r_[0, ends[:-1]]
    pitch = np.asarray(match.pitch, dtype=np.float32)
    return dict(players=match.players[sampled, :, :2] / pitch,
                ball=match.ball[sampled] / pitch, teams=match.teams.copy(), sampled=sampled,
                blocks=np.column_stack([starts, ends]), blocks25=np.asarray(blocks25),
                fps=np.asarray(5), n_frames=np.asarray(len(match.ball)))


class ContactTests(unittest.TestCase):
    def test_analytic_dv_and_earliest_equal_peak_in_trailing_five_frames(self):
        ball = np.column_stack([np.array([0, 1, 3, 6, 10, 15]), np.zeros(6)])
        half = np.ones(6, dtype=np.int8)
        geometry = dict(distance=np.array([[99, 99], [9, 8], [7, 6], [5, 4], [2, 9], [8, 7]]),
                        nearest_team=np.array([0, 0, 0, 1, 1, 1]))
        output = context.contact_features(ball, half, np.array([5]), geometry)
        np.testing.assert_allclose(context.ball_impulse(ball, half)[2:], 25)
        np.testing.assert_allclose(output, [[0.2, 0.4, 1.25, 1]])

    def test_impulse_requires_three_visible_positions_in_same_half(self):
        ball = np.column_stack([np.arange(10, dtype=float) ** 2, np.zeros(10)])
        ball[3] = np.nan
        half = np.r_[np.ones(7), np.full(3, 2)]
        impulse = context.ball_impulse(ball, half)
        self.assertTrue(np.isnan(impulse[[0, 1, 3, 4, 5, 7, 8]]).all())
        np.testing.assert_allclose(impulse[[2, 6, 9]], 50)
        half[7:] = 1
        half[6] = 0
        self.assertTrue(np.isnan(context.ball_impulse(ball, half)[[7, 8]]).all())

    def test_distance_window_never_crosses_half_and_missing_defaults_are_finite(self):
        ball = np.zeros((8, 2))
        ball[4:] = np.nan
        half = np.r_[np.ones(4), np.full(4, 2)]
        geometry = dict(distance=np.r_[np.zeros((4, 2)), np.full((4, 2), np.inf)],
                        nearest_team=np.zeros(8, dtype=np.int8))
        output = context.contact_features(ball, half, np.array([4, 7]), geometry)
        np.testing.assert_array_equal(output, [[3, 3, 0, 0], [3, 3, 0, 0]])

    def test_impulse_and_distance_caps(self):
        ball = np.array([[0, 0], [0, 0], [100, 0]], dtype=float)
        geometry = dict(distance=np.full((3, 2), 50), nearest_team=np.full(3, -1))
        np.testing.assert_array_equal(context.contact_features(ball, np.ones(3), [2], geometry),
                                      [[3, 3, 3, 0]])

    def test_metadata_ignores_unknown_slots_and_slot_permutation(self):
        players, ball, teams, half = moving_source(50, (0, 1, -1, 2, np.nan))
        players[:, 2:, :2] = ball[:, None]
        geometry = context.source_geometry(players, ball, teams, half)
        np.testing.assert_allclose(geometry['distance'], np.tile([0.2, 9.8], (50, 1)))
        np.testing.assert_array_equal(geometry['nearest_team'], 0)
        order = [4, 1, 3, 0, 2]
        permuted = context.source_geometry(players[:, order], ball, teams[order], half)
        for key in geometry:
            np.testing.assert_allclose(permuted[key], geometry[key], equal_nan=True)
        unknown = context.source_geometry(players, ball, np.full(5, -1), half)
        self.assertTrue(np.isinf(unknown['distance']).all())
        np.testing.assert_array_equal(unknown['nearest_team'], -1)
        self.assertTrue(np.isnan(unknown['nearest_speed']).all())

    def test_full_features_team_mirror_and_view_involution(self):
        players, ball, teams, half = moving_source()
        half[110:] = 2
        players[110:, :, 0] -= 10
        ball[110:, 0] -= 10
        sampled = np.arange(0, len(ball), 5)
        before = [value.copy() for value in (players, ball, teams, half)]
        output = context.source_features(players, ball, teams, half, (105.3, 68), sampled)
        swapped = context.source_features(players, ball, 1 - teams, half, (105.3, 68), sampled)
        np.testing.assert_array_equal(swapped['control'][0], output['control'][1])
        np.testing.assert_array_equal(swapped['native_codes'], 1 - output['native_codes'])
        np.testing.assert_array_equal(context.mirror_view(output['control'][1]), output['control'][0])
        self.assertEqual(output['control'].shape, (2, 44, 7))
        self.assertEqual(output['control'].dtype, np.float16)
        self.assertEqual(output['native_codes'].dtype, np.int8)
        self.assertTrue(all(np.isscalar(value) for value in output['audit'].values()))
        for value, snapshot in zip((players, ball, teams, half), before):
            np.testing.assert_array_equal(value, snapshot)


class MemoryTests(unittest.TestCase):
    def test_hold_resets_without_future_backfill_and_recency_uses_seconds(self):
        half = np.r_[np.ones(300), np.full(40, 2)]
        touch = np.full(340, -1)
        touch[10] = 0
        held = context.carry_team(touch, half)
        np.testing.assert_array_equal(held[:10], 2)
        np.testing.assert_array_equal(held[10:300], 0)
        np.testing.assert_array_equal(held[300:], 2)
        recency = context.contact_recency(touch, half)
        np.testing.assert_array_equal(recency[:10], 0)
        self.assertEqual(recency[10], 1)
        self.assertAlmostEqual(float(recency[260]), np.exp(-1), places=7)
        np.testing.assert_array_equal(recency[300:], 0)

    def test_carried_run_minimum_is_thirty_frames_not_touch_run_length(self):
        touch = np.full(121, -1)
        touch[[0, 30, 59, 89]] = [0, 1, 0, 1]
        held = context.carry_team(touch, np.ones(121))
        np.testing.assert_array_equal(held[:89], 0)
        np.testing.assert_array_equal(held[89:], 1)
        touch = np.full(90, -1)
        touch[[0, 30, 60]] = [0, 1, 0]
        np.testing.assert_array_equal(context.carry_team(touch, np.ones(90)),
                                      np.repeat([0, 1, 0], 30))
        np.testing.assert_array_equal(context.carry_team(np.r_[0, np.full(28, -1)], np.ones(29)), 2)

    def test_no_touch_keeps_both_memory_channels_zero(self):
        players, ball, teams, half = moving_source(200)
        ball[:, 0] += 3
        ball[50:170] = np.nan
        output = context.source_features(players, ball, teams, half, (105.3, 68), np.arange(0, 200, 5))
        np.testing.assert_array_equal(output['control'][:, :, 4:6], 0)
        np.testing.assert_array_equal(output['native_codes'], 2)

    def test_held_confidence_survives_occlusion_while_recency_decays(self):
        players, ball, teams, half = moving_source(400)
        ball[40:350] = np.nan
        output = context.source_features(players, ball, teams, half, (105.3, 68), np.arange(0, 400, 5))
        np.testing.assert_array_equal(output['control'][0, :, 4], 1)
        self.assertGreater(output['control'][0, 20, 5], output['control'][0, 60, 5])
        np.testing.assert_array_equal(output['native_codes'], 0)


class DeadCueTests(unittest.TestCase):
    def test_speed_is_derived_from_metre_positions_not_provider_column(self):
        players, ball, teams, half = moving_source(200)
        speed = context.player_speeds(players, half)
        np.testing.assert_allclose(speed[1:], 3, atol=1e-11)
        geometry = context.source_geometry(players, ball, teams, half)
        self.assertFalse(context.dead_cues(ball, half, (105.3, 68), geometry).any())
        players[:, :, 2] = -100000
        np.testing.assert_allclose(context.player_speeds(players, half), speed, equal_nan=True)
        players[100] = np.nan
        half[150:] = 2
        speed = context.player_speeds(players, half)
        self.assertTrue(np.isnan(speed[[0, 100, 101, 150]]).all())
        y_players = np.zeros((4, 1, 2))
        y_players[:, 0, 1] = np.arange(4) * 0.12
        np.testing.assert_allclose(context.player_speeds(y_players, np.ones(4))[1:], 3)

    def test_outside_requires_fifteen_and_slow_requires_twenty_five(self):
        ball = np.full((100, 2), 20.0)
        ball[5:19, 0] = -1
        ball[30:45, 1] = 69
        speed = np.full(100, 3.0)
        speed[50:74] = 1
        speed[75:100] = 1
        geometry = dict(nearest_speed=speed, nearest_distance=np.ones(100))
        cue = context.dead_cues(ball, np.ones(100), (105.3, 68), geometry, merge_frames=1)
        expected = np.zeros(100, dtype=bool)
        expected[30:45] = expected[75:100] = True
        np.testing.assert_array_equal(cue, expected)
        ball[37] = np.nan
        ball[85] = np.nan
        self.assertFalse(context.dead_cues(ball, np.ones(100), (105.3, 68), geometry, merge_frames=1).any())

    def test_dead_merge_clears_149_but_preserves_150_and_resets_half(self):
        ball = np.full((320, 2), 20.0)
        ball[:149, 0] = -1
        ball[160:310, 0] = -1
        geometry = dict(nearest_speed=np.full(320, 3.0), nearest_distance=np.ones(320))
        cue = context.dead_cues(ball, np.ones(320), (105.3, 68), geometry)
        expected = np.zeros(320, dtype=bool)
        expected[160:310] = True
        np.testing.assert_array_equal(cue, expected)
        half = np.ones(320)
        half[235:] = 2
        self.assertFalse(context.dead_cues(ball, half, (105.3, 68), geometry).any())

    def test_native_codes_overlay_corrected_cue_not_short_dead_runs(self):
        players, ball, teams, half = moving_source(200)
        players[:, :, :2] = players[0, :, :2]
        ball[:] = ball[0]
        output = context.source_features(players, ball, teams, half, (105.3, 68), np.arange(0, 200, 5))
        np.testing.assert_array_equal(output['control'][0, :, 4], 1)
        np.testing.assert_array_equal(output['control'][0, :, 6], np.r_[0, np.ones(39)])
        np.testing.assert_array_equal(output['native_codes'], np.r_[0, np.full(39, 2)])
        players, ball, teams, half = moving_source(100)
        players[:, :, :2] = players[0, :, :2]
        ball[:] = ball[0]
        output = context.source_features(players, ball, teams, half, (105.3, 68), np.arange(0, 100, 5))
        np.testing.assert_array_equal(output['control'][0, :, 6], 0)
        np.testing.assert_array_equal(output['native_codes'], 0)


class RegenerationTests(unittest.TestCase):
    def test_comparison_rejects_sample_metadata_and_observation_changes(self):
        players, ball, teams, half = moving_source(100)
        half[50:] = 2
        match = SimpleNamespace(players=players, ball=ball, teams=teams, half=half,
                                labels=None, pitch=(105.3, 68))
        prepared = prepared_from(match)
        np.testing.assert_array_equal(context.assert_regeneration(match, prepared), prepared['sampled'])
        for key in ('players', 'ball', 'teams', 'sampled', 'blocks', 'blocks25', 'n_frames', 'fps'):
            changed = copy.deepcopy(prepared)
            changed[key].flat[0] += 0.01 if key in ('players', 'ball') else 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f'Regeneration mismatch: {key}'):
                context.assert_regeneration(match, changed)
        prepared['ball'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Regeneration mismatch: ball'):
            context.assert_regeneration(match, prepared)

    def test_regeneration_preserves_missing_observations_and_invalid_metadata(self):
        players, ball, teams, half = moving_source(40, (0, 1, np.nan))
        players[5, 0, :2] = np.nan
        ball[10] = np.nan
        match = SimpleNamespace(players=players, ball=ball, teams=teams, half=half,
                                labels=None, pitch=(105.3, 68))
        prepared = prepared_from(match)
        context.assert_regeneration(match, prepared)
        prepared['players'][0, 0, 0] += 0.5e-6
        context.assert_regeneration(match, prepared)
        prepared['players'][0, 0, 0] += 2e-6
        with self.assertRaisesRegex(ValueError, 'Regeneration mismatch: players'):
            context.assert_regeneration(match, prepared)

    def test_real_degradation_build_is_label_free_and_does_not_mutate_inputs(self):
        import bridge as b

        players, ball, teams, half = moving_source(6000)
        half[3000:] = 2
        players[:, :, 0] = 20 + (np.arange(6000) % 300)[:, None] * 0.12 + np.arange(2)[None, :] * 10
        players[:, :, 2] = 3
        ball[:, 0] = players[:, 0, 0] + 0.2
        clean = b.reference.Match('synthetic', players.astype(np.float32), ball.astype(np.float32),
                                  teams.astype(np.int8), half, np.arange(6000), None, (105.3, 68))
        regenerated = b.reference.degrade(clean, seed=20261108)
        prepared = prepared_from(regenerated)
        before = copy.deepcopy(prepared)
        original_players, original_ball = clean.players.copy(), clean.ball.copy()
        output = context.build(clean, prepared, 20261108)
        direct = context.source_features(regenerated.players, regenerated.ball, regenerated.teams,
                                         half, clean.pitch, prepared['sampled'])
        np.testing.assert_array_equal(output['control'], direct['control'])
        np.testing.assert_array_equal(output['native_codes'], direct['native_codes'])
        self.assertTrue(output['audit']['regenerated_observations_match'])
        self.assertFalse(output['audit']['label_access'])
        for key in before:
            np.testing.assert_array_equal(prepared[key], before[key])
        np.testing.assert_array_equal(clean.players, original_players)
        np.testing.assert_array_equal(clean.ball, original_ball)
        for label in (0, 1, 2, -1):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'label-free'):
                context.build(replace(clean, labels=np.full(6000, label)), prepared, 20261108)
        for key in ('y', 'y25'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'label-free'):
                context.build(clean, {**prepared, key: object()}, 20261108)
        changed = copy.deepcopy(prepared)
        changed['players'][0, 0, 0] = 100
        with self.assertRaisesRegex(ValueError, 'Regeneration mismatch: players'):
            context.build(clean, changed, 20261108)


if __name__ == '__main__':
    unittest.main()
