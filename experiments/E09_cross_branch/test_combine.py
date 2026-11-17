import copy
import unittest

import numpy as np

import bridge as b
import combine


def prediction(team, dead):
    team, dead = np.broadcast_arrays(np.asarray(team, dtype=float), np.asarray(dead, dtype=float))
    return b.result(np.column_stack(((1 - dead) * team, (1 - dead) * (1 - team), dead)))


def observations(n, blocks=None):
    return dict(X=np.zeros((n, 16)), blocks=np.array([[0, n]] if blocks is None else blocks))


def frozen(value):
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for child in value.values():
            frozen(child)
    return value


class FusionTests(unittest.TestCase):
    def setUp(self):
        rows = {
            'short': [.54, .06, .40], 'setseq': [.02, .18, .80],
            'pp': [.20, .50, .30], 'contact': [.60, .30, .10],
            'memory': [.15, .25, .60], 'combined': [.05, .70, .25],
        }
        self.bundle = {name: b.result(np.tile(row, (24, 1))) for name, row in rows.items()}
        self.observations = observations(24, [[0, 12], [12, 24]])

    def test_every_family_mixes_joint_probabilities_numerically(self):
        expected_weights = {
            'short': {'short': 1}, 'setseq': {'setseq': 1}, 'pp': {'pp': 1},
            'contact': {'contact': 1}, 'memory': {'memory': 1}, 'combined': {'combined': 1},
            'short_setseq25': {'short': .75, 'setseq': .25},
            'short_pp25': {'short': .75, 'pp': .25},
            'short_contact25': {'short': .75, 'contact': .25},
            'short_memory25': {'short': .75, 'memory': .25},
            'short_combined25': {'short': .75, 'combined': .25},
            'triple_contact': {'short': .5, 'setseq': .25, 'contact': .25},
            'triple_combined': {'short': .5, 'setseq': .25, 'combined': .25},
            'quality_combined': {'short': .9, 'combined': .1},
        }
        for name, weights in expected_weights.items():
            with self.subTest(family=name):
                actual = combine.fuse(name, self.bundle, self.observations)
                expected = sum(weight * self.bundle[source]['probability'] for source, weight in weights.items())
                np.testing.assert_allclose(actual['probability'], expected, atol=1e-14)
                np.testing.assert_allclose(actual['probability'].sum(1), 1)
                np.testing.assert_allclose(actual['p_dead'], expected[:, 2])
                np.testing.assert_allclose(actual['p_team'], expected[:, 0] / expected[:, :2].sum(1))
        result = combine.fuse('short_setseq25', self.bundle, self.observations)
        conditional_average = .75 * .9 + .25 * .1
        self.assertGreater(abs(result['p_team'][0] - conditional_average), .05)

    def test_quality_visibility_is_centered_and_half_local(self):
        self.observations['X'][5:12, 3] = 1
        result = combine.fuse('quality_combined', self.bundle, self.observations)
        expected_mean = np.zeros(24)
        for start, end in self.observations['blocks']:
            for index in range(start, end):
                expected_mean[index] = np.mean(self.observations['X'][max(start, index - 5):min(end, index + 6), 3])
        weight = .10 + .30 * expected_mean
        expected = (1 - weight[:, None]) * self.bundle['short']['probability'] + weight[:, None] * self.bundle['combined']['probability']
        np.testing.assert_allclose(result['probability'], expected)
        self.assertAlmostEqual(expected_mean[0], 1 / 6)
        self.assertEqual(weight[11], .4)
        self.assertEqual(weight[12], .1)
        changed = copy.deepcopy(self.observations)
        changed['X'][12:, 3] = 1
        second = combine.fuse('quality_combined', self.bundle, changed)
        np.testing.assert_array_equal(result['probability'][:12], second['probability'][:12])
        np.testing.assert_allclose(second['probability'][12:], .6 * self.bundle['short']['probability'][12:] + .4 * self.bundle['combined']['probability'][12:])

    def test_inputs_stay_immutable_and_metadata_is_recomputed(self):
        for source in self.bundle.values():
            source['p_dead'] = np.ones(24)
            source['p_team'] = np.zeros(24)
            source['has_boundary'] = True
            source['boundary'] = np.linspace(0, 1, 24)
        original = copy.deepcopy(self.bundle)
        original_observations = copy.deepcopy(self.observations)
        frozen(self.bundle)
        frozen(self.observations)
        for name in combine.FAMILIES:
            result = combine.fuse(name, self.bundle, self.observations)
            self.assertFalse(result['has_boundary'])
            self.assertNotIn('boundary', result)
            np.testing.assert_allclose(result['p_dead'], result['probability'][:, 2])
            np.testing.assert_allclose(result['p_team'], result['probability'][:, 0] / result['probability'][:, :2].sum(1))
            self.assertTrue(all(not np.shares_memory(result['probability'], source['probability']) for source in self.bundle.values()))
        for source, saved in original.items():
            for key, value in saved.items():
                np.testing.assert_array_equal(self.bundle[source][key], value)
        for key, value in original_observations.items():
            np.testing.assert_array_equal(self.observations[key], value)

    def test_home_away_swap_preserves_dead_and_reverses_team(self):
        self.observations['X'][:, 3] = np.arange(24) % 2
        swapped = {name: b.result(source['probability'][:, [1, 0, 2]]) for name, source in self.bundle.items()}
        for name in combine.FAMILIES:
            direct = combine.fuse(name, self.bundle, self.observations)
            reverse = combine.fuse(name, swapped, self.observations)
            np.testing.assert_allclose(reverse['probability'], direct['probability'][:, [1, 0, 2]])
            np.testing.assert_allclose(reverse['p_dead'], direct['p_dead'])
            np.testing.assert_allclose(reverse['p_team'], 1 - direct['p_team'])
            self.assertEqual(reverse['has_boundary'], direct['has_boundary'])

    def test_pure_dead_mass_has_neutral_conditional_team(self):
        self.bundle['short'] = dict(probability=np.tile([0., 0., 7.], (24, 1)))
        result = combine.fuse('short', self.bundle, self.observations)
        np.testing.assert_array_equal(result['probability'], np.tile([0., 0., 1.], (24, 1)))
        np.testing.assert_array_equal(result['p_team'], .5)


class HysteresisTests(unittest.TestCase):
    def test_thresholds_are_strict_and_components_extend_both_ways(self):
        values = np.array([.1, .36, .40, .51, .40, .36, .35, .40, .40, .40, .40, .40])
        original = values.copy()
        values.setflags(write=False)
        expected = np.zeros(len(values), dtype=bool)
        expected[1:6] = True
        np.testing.assert_array_equal(combine.hysteresis_mask(values, [[0, len(values)]]), expected)
        np.testing.assert_array_equal(values, original)
        equal = np.full(5, .50)
        np.testing.assert_array_equal(combine.hysteresis_mask(equal, [[0, 5]]), False)
        equal[2] = np.nextafter(.50, np.inf)
        np.testing.assert_array_equal(combine.hysteresis_mask(equal, [[0, 5]]), True)

    def test_five_sample_minimum_cannot_join_across_half(self):
        values = np.full(10, .6)
        np.testing.assert_array_equal(combine.hysteresis_mask(values, [[0, 4], [4, 10]]), [False] * 4 + [True] * 6)
        np.testing.assert_array_equal(combine.hysteresis_mask(values[:8], [[0, 4], [4, 8]]), False)
        np.testing.assert_array_equal(combine.hysteresis_mask(values[:5], [[0, 5]]), True)
        weak = np.r_[.4, .4, .4, .4, .4, .6, .4, .4, .4, .4]
        np.testing.assert_array_equal(combine.hysteresis_mask(weak, [[0, 5], [5, 10]]), [False] * 5 + [True] * 5)


class DecoderTests(unittest.TestCase):
    def test_real_dead45_parity_when_masks_match(self):
        codes = np.r_[np.zeros(15), np.full(10, 2), np.ones(45), np.full(5, 2), np.zeros(25)].astype(np.int8)
        team = np.where(codes == 1, .001, .999)
        dead = np.where(codes == 2, .80, .10)
        output = frozen(prediction(team, dead))
        data = frozen(observations(len(codes), [[0, 50], [50, 100]]))
        stats = dict(rate=12., dead_share=.15)
        actual_baseline = b.fair.decoder.decode(output, data, b.fair.SPEC, stats)['codes']
        np.testing.assert_array_equal(actual_baseline, codes)
        for mode in combine.DECODERS:
            actual = combine.decode(output, data, stats, mode)
            np.testing.assert_array_equal(actual, actual_baseline)
            self.assertEqual(actual.dtype, np.int8)
            changed_prior = combine.decode(output, data, dict(rate=12., dead_share=.99), mode)
            np.testing.assert_array_equal(changed_prior, actual)

    def test_hysteresis_weak_shoulders_change_real_decoder_output(self):
        dead = np.r_[np.full(6, .1), .4, .4, .6, .4, .4, np.full(9, .1)]
        output = prediction(np.full(20, .99), dead)
        data = observations(20)
        stats = dict(rate=0., dead_share=.5)
        baseline = combine.decode(output, data, stats, 'dead45')
        actual = combine.decode(output, data, stats, 'hysteresis')
        np.testing.assert_array_equal(baseline, np.zeros(20, dtype=np.int8))
        expected = np.zeros(20, dtype=np.int8)
        expected[6:11] = 2
        np.testing.assert_array_equal(actual, expected)

    def test_calibration_uses_full_grid_and_counts_after_merge(self):
        decoder = b.fair.decoder
        rng = np.random.default_rng(43)
        team = 1 / (1 + np.exp(-rng.normal(0, 2, 120)))
        dead = np.r_[np.full(57, .1), np.full(5, .8), np.full(58, .1)]
        output = prediction(team, dead)
        data = observations(120, [[0, 60], [60, 120]])
        stats = dict(rate=15., dead_share=.4)
        mask = combine.hysteresis_mask(dead, data['blocks'])
        home, away = decoder._team_evidence(team, data['X'], data, b.fair.SPEC)
        candidates = []
        for penalty in decoder.LAMBDA_GRID:
            labels = decoder._decode_live_team(home, away, data['blocks'], mask, np.empty(0), penalty)
            labels = decoder._merge_short_runs(labels, data['blocks'], 5)
            candidates.append((abs(decoder.n_possessions(labels, data['blocks']) - 6), penalty, labels))
        expected = min(candidates, key=lambda row: row[:2])[2]
        actual = combine.decode(output, data, stats, 'hysteresis')
        np.testing.assert_array_equal(actual, expected)

    def test_calibration_counts_merged_runs_not_raw_runs(self):
        team = np.r_[np.full(20, .999), np.full(4, .001), np.full(20, .999),
                     np.full(5, .35), np.full(20, .999)]
        data = observations(len(team))
        output = prediction(team, np.full(len(team), .1))
        stats = dict(rate=3 * 300 / len(team), dead_share=.1)
        actual = combine.decode(output, data, stats, 'hysteresis')
        expected = np.zeros(len(team), dtype=np.int8)
        expected[44:49] = 1
        np.testing.assert_array_equal(actual, expected)

    def test_full_lambda_grid_reaches_120_penalty(self):
        team = np.r_[np.full(1000, .999), np.full(600, .001), np.full(1000, .999)]
        output = prediction(team, np.full(len(team), .1))
        data = observations(len(team))
        actual = combine.decode(output, data, dict(rate=0., dead_share=.1), 'hysteresis')
        np.testing.assert_array_equal(actual, np.zeros(len(team), dtype=np.int8))

    def test_dead_interval_and_half_reset_weak_team_memory(self):
        team = np.r_[np.full(20, .999), np.full(5, .5), .4, np.full(20, .999), .4]
        dead = np.full(len(team), .1)
        dead[20:25] = .8
        output = prediction(team, dead)
        data = observations(len(team), [[0, 26], [26, 46], [46, 47]])
        expected = np.zeros(len(team), dtype=np.int8)
        expected[20:25] = 2
        expected[[25, 46]] = 1
        actual = combine.decode(output, data, dict(rate=0., dead_share=.1), 'hysteresis')
        np.testing.assert_array_equal(actual, expected)

    def test_distance_metadata_and_home_away_swap_are_symmetric(self):
        data = observations(40, [[0, 20], [20, 40]])
        data['X'][:, 3] = 1
        data['distances'] = np.r_[np.tile([.1, 3.], (10, 1)), np.tile([3., .1], (10, 1)),
                                  np.tile([.1, 3.], (20, 1))]
        output = prediction(np.r_[np.full(10, .8), np.full(10, .2), np.full(20, .8)], np.full(40, .1))
        reverse_data = copy.deepcopy(data)
        reverse_data['X'][:, :2] = data['X'][:, [1, 0]]
        reverse_data['distances'] = data['distances'][:, [1, 0]].copy()
        reverse_output = b.result(output['probability'][:, [1, 0, 2]])
        saved = copy.deepcopy(data)
        frozen(data)
        for mode in combine.DECODERS:
            direct = combine.decode(output, data, dict(rate=22.5, dead_share=.2), mode)
            reverse = combine.decode(reverse_output, reverse_data, dict(rate=22.5, dead_share=.2), mode)
            np.testing.assert_array_equal(direct, np.r_[np.zeros(10), np.ones(10), np.zeros(20)])
            np.testing.assert_array_equal(reverse, 1 - direct)
        for key, value in saved.items():
            np.testing.assert_array_equal(data[key], value)


if __name__ == '__main__':
    unittest.main()
