import copy
import itertools
import unittest

import numpy as np

import duration


def training_runs(lengths_by_state):
    records = []
    for state, lengths in enumerate(lengths_by_state):
        for index, length in enumerate(lengths):
            labels = np.array([(state + 1) % 3] + [state] * length + [(state + 2) % 3],
                              dtype=np.int8)
            records.append(dict(_name=f'train_{state}_{index}', y=labels,
                                blocks=np.array([[0, len(labels)]])))
    return records


def explicit_prior(hazard, initial=(0.31, 0.47, 0.22), transition=None):
    hazard = np.asarray(hazard, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    if transition is None:
        transition = [[0, 0.7, 0.3], [0.4, 0, 0.6], [0.8, 0.2, 0]]
    transition = np.asarray(transition, dtype=np.float64)
    log_transition = np.zeros((3, 3))
    mask = ~np.eye(3, dtype=bool)
    log_transition[mask] = np.log(transition[mask])
    return dict(cap=hazard.shape[1], hazard=hazard, log_hazard=np.log(hazard),
                log_survival=np.log1p(-hazard), initial=initial,
                log_initial=np.log(initial), transition=transition,
                log_transition=log_transition)


def path_score(path, probability, prior, strength, switch_bias):
    """Independent enumeration score uses uncapped run length and caps only lookup age."""
    probability = np.asarray(probability, dtype=np.float64)
    probability = probability / probability.sum(axis=1, keepdims=True)
    score = np.log(prior['initial'][path[0]]) + np.log(max(probability[0, path[0]], 1e-8))
    length = 1
    for time in range(1, len(path)):
        source, target = path[time - 1], path[time]
        age = min(length, prior['cap']) - 1
        if source == target:
            score += strength * np.log1p(-prior['hazard'][source, age])
            length += 1
        else:
            score += strength * (np.log(prior['hazard'][source, age]) +
                                 np.log(prior['transition'][source, target])) - switch_bias
            length = 1
        score += np.log(max(probability[time, target], 1e-8))
    return score


class DurationPriorTests(unittest.TestCase):

    def test_complete_runs_omit_both_ends_of_each_half_and_known_interval(self):
        labels = np.array([0, 0, 1, 1, 1, 2, 2, 0, -1, 1, 2, 2, 2, 0,
                           0, 0, 1, 1, 0, 0, 2, 2], dtype=np.int8)
        prior = duration.fit([dict(_name='training', y=labels, blocks=[[0, 14], [14, 22]])], cap=4)
        self.assertEqual(prior['complete_durations'], ((2,), (3, 2), (2, 3)))
        np.testing.assert_array_equal(prior['complete_counts'], [1, 2, 2])
        self.assertEqual(prior['known_interval_count'], 3)
        self.assertEqual(prior['censored_run_count'], 6)
        np.testing.assert_array_equal(prior['transition_counts'], [[0, 2, 1], [1, 0, 2], [2, 0, 0]])
        np.testing.assert_array_equal(prior['frame_counts'], [np.sum(labels == state) for state in range(3)])

    def test_adjacent_half_endpoints_and_unknown_gaps_never_create_transitions(self):
        data = dict(_name='training', y=np.array([0, 0, 1, 1, -1, 2, 2]),
                    blocks=[[0, 2], [2, 7]])
        prior = duration.fit([data], cap=3)
        np.testing.assert_array_equal(prior['transition_counts'], 0)
        np.testing.assert_array_equal(prior['complete_counts'], 0)
        np.testing.assert_allclose(prior['initial'], [3 / 9, 3 / 9, 3 / 9])
        self.assertEqual(prior['censored_run_count'], 3)

    def test_empirical_hazards_use_beta_smoothing_and_tail_exposure(self):
        prior = duration.fit(training_runs(([1, 2, 2, 5, 9], [3], [1, 7])), cap=4)
        np.testing.assert_allclose(prior['hazard'][0], [2 / 7, 3 / 6, 1 / 4, 3 / 10])
        np.testing.assert_allclose(prior['hazard'][1], [1 / 3, 1 / 3, 2 / 3, 1 / 2])
        np.testing.assert_allclose(prior['hazard'][2], [2 / 4, 1 / 3, 1 / 3, 2 / 6])
        self.assertEqual(prior['fps'], 5)
        self.assertEqual(prior['tail'], 'age-capped geometric')

    def test_cap_one_uses_geometric_tail_not_exact_duration_events(self):
        prior = duration.fit(training_runs(([2, 8], [3], [5])), cap=1)
        np.testing.assert_allclose(prior['hazard'][:, 0], [3 / 12, 2 / 5, 2 / 7])

    def test_missing_complete_class_is_declared_beta_prior_without_fake_samples(self):
        data = dict(_name='training', y=np.array([0, 0, -1, 1, 2]), blocks=[[0, 5]])
        prior = duration.fit([data], cap=5)
        np.testing.assert_array_equal(prior['complete_counts'], [0, 0, 0])
        self.assertEqual(prior['complete_durations'], ((), (), ()))
        self.assertEqual(prior['missing_complete_states'], (0, 1, 2))
        self.assertEqual(prior['hazard_prior'], 'Beta(1,1)')
        np.testing.assert_allclose(prior['hazard'], 0.5)

    def test_priors_are_finite_normalized_and_log_arrays_match(self):
        prior = duration.fit(training_runs(([1, 5], [2, 8], [3, 11])), cap=4)
        for key in ('hazard', 'log_hazard', 'log_survival', 'transition',
                    'log_transition', 'initial', 'log_initial'):
            self.assertTrue(np.isfinite(prior[key]).all(), key)
        self.assertTrue(np.all((prior['hazard'] > 0) & (prior['hazard'] < 1)))
        np.testing.assert_allclose(prior['transition'].sum(axis=1), 1)
        np.testing.assert_array_equal(np.diag(prior['transition']), 0)
        np.testing.assert_array_equal(np.diag(prior['log_transition']), 0)
        mask = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(np.exp(prior['log_transition'][mask]), prior['transition'][mask])
        self.assertAlmostEqual(prior['initial'].sum(), 1)
        np.testing.assert_allclose(np.exp(prior['log_initial']), prior['initial'])
        np.testing.assert_allclose(np.exp(prior['log_hazard']), prior['hazard'])
        np.testing.assert_allclose(np.exp(prior['log_survival']), 1 - prior['hazard'])

    def test_all_training_sources_contribute_auditable_names_and_counts(self):
        records = training_runs(([2], [5], [9]))
        prior = duration.fit(iter(records), cap=4)
        self.assertEqual(prior['training_names'], tuple(record['_name'] for record in records))
        self.assertEqual(prior['complete_durations'], ((2,), (5,), (9,)))
        counts = np.array([sum(np.count_nonzero(record['y'] == state) for record in records)
                           for state in range(3)])
        np.testing.assert_array_equal(prior['frame_counts'], counts)
        np.testing.assert_allclose(prior['initial'], (counts + 1) / (counts.sum() + 3))
        earlier = duration.fit(records[:-1], cap=4)
        self.assertFalse(np.array_equal(earlier['hazard'][2], prior['hazard'][2]))

    def test_fit_does_not_mutate_training_records(self):
        records = training_runs(([2, 3], [4], [6]))
        snapshot = copy.deepcopy(records)
        duration.fit(records, cap=3)
        for record, before in zip(records, snapshot):
            self.assertEqual(record['_name'], before['_name'])
            for key in ('y', 'blocks'):
                np.testing.assert_array_equal(record[key], before[key])

    def test_empty_or_invalid_training_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            duration.fit([])
        for labels in ([0, 3, 1], [0, np.nan, 1], [[0, 1]]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                duration.fit([dict(_name='bad', y=np.asarray(labels), blocks=[[0, len(labels)]])])
        for cap in (0, -1, 1.5, True, 21846):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                duration.fit(training_runs(([1], [], [])), cap=cap)


class DurationDecoderTests(unittest.TestCase):
    def test_viterbi_matches_exhaustive_three_state_hsmm_path_scores(self):
        probability = np.array([[0.51, 0.29, 0.20], [0.03, 0.93, 0.04],
                                [0.04, 0.09, 0.87], [0.55, 0.40, 0.05],
                                [0, 0.7, 0.3], [0.4, 0.3, 0.3]])
        for cap in (1, 2, 4):
            hazard = np.array([[0.11 + 0.1 * age for age in range(cap)],
                               [0.76 - 0.1 * age for age in range(cap)],
                               [0.23 + 0.13 * age for age in range(cap)]])
            prior = explicit_prior(hazard)
            for length in (1, 2, 5, 6):
                rows = probability[:length]
                for config in duration.CONFIGS:
                    with self.subTest(cap=cap, length=length, config=config['name']):
                        decoded = duration.viterbi(rows, [[0, length]], prior,
                                                   config['strength'], config['switch_bias'])
                        expected = max(path_score(path, rows, prior, config['strength'],
                                                  config['switch_bias'])
                                       for path in itertools.product(range(3), repeat=length))
                        actual = path_score(decoded, rows, prior, config['strength'],
                                            config['switch_bias'])
                        self.assertAlmostEqual(actual, expected, places=10)
                        self.assertEqual(decoded.dtype, np.int8)

    def test_fitted_prior_also_matches_independent_exhaustive_scores(self):
        prior = duration.fit(training_runs(([1, 2, 9], [2, 5], [1, 4, 8])), cap=3)
        rows = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7],
                         [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        decoded = duration.viterbi(rows, [[0, len(rows)]], prior, 0.5, -0.5)
        expected = max(path_score(path, rows, prior, 0.5, -0.5)
                       for path in itertools.product(range(3), repeat=len(rows)))
        self.assertAlmostEqual(path_score(decoded, rows, prior, 0.5, -0.5), expected, places=10)

    def test_long_spell_survives_hundreds_of_tail_self_loops_without_forced_exit(self):
        prior = duration.fit(training_runs(([1, 2], [1], [1])), cap=3)
        probability = np.tile([0.98, 0.01, 0.01], (257, 1))
        for config in duration.CONFIGS:
            with self.subTest(config=config['name']):
                decoded = duration.decode(dict(probability=probability), dict(blocks=[[0, 257]]),
                                          prior, config)
                np.testing.assert_array_equal(decoded, np.zeros(257, dtype=np.int8))

    def test_each_half_restarts_initial_prior_and_age(self):
        prior = explicit_prior(np.full((3, 3), 0.001), initial=(0.4, 0.59, 0.01))
        rows = np.vstack([np.tile([0.999, 0.0009, 0.0001], (5, 1)),
                          np.tile([0.1, 0.89, 0.01], (2, 1))])
        together = duration.viterbi(rows, [[0, 5], [5, 7]], prior, 1, 0)
        separate = np.r_[duration.viterbi(rows[:5], [[0, 5]], prior, 1, 0),
                         duration.viterbi(rows[5:], [[0, 2]], prior, 1, 0)]
        np.testing.assert_array_equal(together, separate)
        np.testing.assert_array_equal(together, [0, 0, 0, 0, 0, 1, 1])
        joined = duration.viterbi(rows, [[0, 7]], prior, 1, 0)
        self.assertFalse(np.array_equal(joined, together))

    def test_right_censored_terminal_omits_exit_factor(self):
        prior = explicit_prior([[0.001, 0.001], [0.999, 0.999], [0.5, 0.5]],
                               initial=(0.5, 0.499, 0.001))
        decoded = duration.viterbi([[0.6, 0.4, 0]], [[0, 1]], prior, 1, 0)
        np.testing.assert_array_equal(decoded, [0])

    def test_dead_is_full_joint_state_without_gate_or_minimum_duration(self):
        prior = explicit_prior(np.full((3, 2), 0.5), initial=(1 / 3, 1 / 3, 1 / 3))
        config = duration.CONFIGS[0]
        observations = dict(blocks=[[0, 1]], distances=np.array([[1e9, 0.0]]))
        home = dict(probability=np.array([[0.98, 0.01, 0.01]]), p_dead=np.ones(1), p_team=np.zeros(1))
        dead = dict(probability=np.array([[0.01, 0.01, 0.98]]), p_dead=np.zeros(1), p_team=np.ones(1))
        np.testing.assert_array_equal(duration.decode(home, observations, prior, config), [0])
        np.testing.assert_array_equal(duration.decode(dead, observations, prior, config), [2])

    def test_ties_choose_lowest_terminal_and_predecessor_indices_deterministically(self):
        prior = explicit_prior(np.full((3, 2), 0.5), initial=(1 / 3, 1 / 3, 1 / 3))
        rows = np.ones((2, 3))
        for _ in range(3):
            np.testing.assert_array_equal(duration.viterbi(rows, [[0, 2]], prior, 0, 0), [1, 0])
        np.testing.assert_array_equal(duration.viterbi(rows[:1], [[0, 1]], prior, 0, 0), [0])

    def test_decode_ignores_heldout_labels_features_and_all_other_observation_keys(self):
        class BoundaryOnly(dict):
            def __getitem__(self, key):
                if key != 'blocks':
                    raise AssertionError(f'decode read held-out field {key}')
                return super().__getitem__(key)

        prior = duration.fit(training_runs(([2], [3], [4])), cap=3)
        rows = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        observation = BoundaryOnly(blocks=[[0, 3]], y='forbidden', X='forbidden',
                                   distances='forbidden', _name='heldout')
        predicted = duration.decode(dict(probability=rows), observation, prior, duration.CONFIGS[2])
        expected = duration.viterbi(rows, [[0, 3]], prior, 0.5, -0.5)
        np.testing.assert_array_equal(predicted, expected)
        self.assertNotIn('heldout', prior['training_names'])

    def test_row_normalization_zero_emissions_and_no_input_mutation(self):
        prior = duration.fit(training_runs(([2], [3], [4])), cap=2)
        rows = np.array([[90.0, 0, 10], [0, 4, 1], [1, 1, 7]], dtype=np.float64)
        prediction = dict(probability=rows, p_dead=np.zeros(3))
        observations = dict(blocks=np.array([[0, 3]]))
        snapshot = copy.deepcopy((prediction, observations, prior))
        actual = duration.decode(prediction, observations, prior, duration.CONFIGS[3])
        expected = duration.viterbi(rows / rows.sum(axis=1, keepdims=True), [[0, 3]], prior, 0.5, 0)
        np.testing.assert_array_equal(actual, expected)
        for current, before in zip((prediction, observations, prior), snapshot):
            for key, value in current.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, before[key])
                else:
                    self.assertEqual(value, before[key])

    def test_invalid_probability_boundaries_and_nonfinite_priors_are_rejected(self):
        prior = duration.fit(training_runs(([1], [2], [3])), cap=2)
        for rows in ([[0, 0, 0]], [[1, -1, 1]], [[np.nan, 1, 1]], [[np.inf, 1, 1]], [[1, 2]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                duration.viterbi(rows, [[0, 1]], prior, 0.5, 0)
        for blocks in ([[1, 2]], [[0, 1]], [[0, 2], [1, 2]]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                duration.viterbi(np.ones((2, 3)), blocks, prior, 0.5, 0)
        for key in ('log_initial', 'log_hazard', 'log_survival', 'log_transition'):
            bad_prior = copy.deepcopy(prior)
            bad_prior[key].flat[0] = np.nan
            with self.subTest(key=key), self.assertRaises(ValueError):
                duration.viterbi(np.ones((2, 3)), [[0, 2]], bad_prior, 0.5, 0)
        with self.assertRaises(ValueError):
            duration.viterbi(np.ones((2, 3)), [[0, 2]], prior, np.nan, 0)
        with self.assertRaises(ValueError):
            duration.viterbi(np.ones((2, 3)), [[0, 2]], prior, 0.5, np.inf)

    def test_empty_timeline_returns_empty_int8_codes(self):
        prior = duration.fit(training_runs(([1], [2], [3])), cap=2)
        actual = duration.decode(dict(probability=np.empty((0, 3))), dict(blocks=[]),
                                 prior, duration.CONFIGS[0])
        self.assertEqual(actual.shape, (0,))
        self.assertEqual(actual.dtype, np.int8)


if __name__ == '__main__':
    unittest.main()
