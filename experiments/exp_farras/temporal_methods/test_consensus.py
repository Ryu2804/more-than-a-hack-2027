import copy
import unittest

import numpy as np

import bridge as b
import consensus


def prediction_from_logits(logits, dead=0.1):
    logits = np.asarray(logits, dtype=np.float64)
    home = 1.0 / (1.0 + np.exp(-logits))
    dead = np.broadcast_to(dead, home.shape)
    return b.result(np.column_stack([(1.0 - dead) * home,
                                     (1.0 - dead) * (1.0 - home), dead]))


def observations_for(n, blocks=None):
    X = np.zeros((n, 16), dtype=np.float64)
    X[:, :2] = 10.0
    return dict(X=X, distances=np.full((n, 2), 10.0),
                blocks=np.asarray([[0, n]] if blocks is None else blocks, dtype=np.int64),
                contact_control=np.zeros((n, 7)), _name='no-context-cache-needed')


def fixture(n=360):
    rng = np.random.default_rng(20261116)
    logits = np.repeat(rng.normal(0.0, 2.0, n // 6), 6)
    dead = np.full(n, 0.1)
    dead[40:45] = 0.8
    dead[95:99] = 0.8
    dead[177:183] = 0.8
    dead[240:250] = 0.8
    prediction = prediction_from_logits(logits, dead)
    observations = observations_for(n, [[0, n // 2], [n // 2, n]])
    observations['X'][::3, 3] = 1.0
    observations['distances'][::3] = rng.uniform(0.0, 0.3, (n // 3, 2))
    bundle = dict(short=prediction_from_logits(np.zeros(n)),
                  setseq=prediction_from_logits(np.full(n, 0.4)))
    return prediction, bundle, observations


def reference_decode(prediction, observations, stats, weights):
    decoder = b.decoder
    spec = b.fair.SPEC
    blocks = decoder._normalize_blocks(observations['blocks'], len(weights))
    p_dead, p_team = decoder._prediction_components(prediction, len(weights))
    dead = decoder.dead_mask(p_dead, blocks, threshold=spec['dead_threshold'],
                             min_s=spec['min_dead_s'], shift=0.0)
    home, away = decoder._team_evidence(p_team, observations['X'], observations, spec)
    target = stats['rate'] * len(weights) / (decoder.FPS * 60.0) * spec['rate_scale']
    candidates = []
    for penalty in decoder.LAMBDA_GRID:
        codes = decoder._decode_live_team(home, away, blocks, dead, penalty * weights, penalty)
        codes = decoder._merge_short_runs(codes, blocks, 5)
        count = decoder.n_possessions(codes, blocks)
        candidates.append((abs(count - target), penalty, codes))
    return min(candidates, key=lambda row: row[:2])[2]


class AgreementTests(unittest.TestCase):

    def test_directional_agreement_uses_smaller_change_and_clips(self):
        bundle = dict(short=prediction_from_logits([0, 1, 0, -3, -3]),
                      setseq=prediction_from_logits([0, 0.5, 1, -2, -3]))
        config = dict(alpha=0.5, contact_beta=0.0)
        weights = consensus.agreement_weights(bundle, observations_for(5), config)
        np.testing.assert_allclose(weights, [1, 0.875, 1, 0.5, 1], atol=1e-12)

    def test_conditional_home_probability_not_joint_home_mass(self):
        bundle = dict(short=prediction_from_logits([0, 0, 0], [0.1, 0.9, 0.2]),
                      setseq=prediction_from_logits([0.4, 0.4, 0.4], [0.8, 0.1, 0.6]))
        weights = consensus.agreement_weights(bundle, observations_for(3), consensus.CONFIGS[4])
        np.testing.assert_allclose(weights, 1.0, atol=1e-12)

    def test_extreme_probabilities_are_clipped_before_differencing(self):
        source = b.result(np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=float))
        bundle = dict(short=source, setseq=copy.deepcopy(source))
        weights = consensus.agreement_weights(bundle, observations_for(3), consensus.CONFIGS[4])
        np.testing.assert_array_equal(weights, [1, 0.25, 0.25])
        self.assertTrue(np.isfinite(weights).all())

    def test_half_start_resets_agreement_and_second_half_is_independent(self):
        bundle = dict(short=prediction_from_logits([-5, -4, 4, 5]),
                      setseq=prediction_from_logits([-5, -4, 4, 5]))
        observations = observations_for(4, [[0, 2], [2, 4]])
        config = dict(alpha=0.5, contact_beta=0.0)
        weights = consensus.agreement_weights(bundle, observations, config)
        np.testing.assert_allclose(weights, [1, 0.75, 1, 0.75])
        second_bundle = {name: b.result(value['probability'][2:]) for name, value in bundle.items()}
        separate = consensus.agreement_weights(second_bundle, observations_for(2), config)
        np.testing.assert_allclose(weights[2:], separate)

    def test_observed_contact_needs_team_and_clips_impulse(self):
        observations = observations_for(5)
        observations['contact_control'][:, 2] = [3, 1.5, 9, -2, 3]
        observations['contact_control'][:, 3] = [1, -1, 0, 1, 0]
        source = prediction_from_logits(np.zeros(5))
        bundle = dict(short=source, setseq=source)
        config = dict(alpha=0.5, contact_beta=0.5)
        weights = consensus.agreement_weights(bundle, observations, config)
        np.testing.assert_allclose(weights, [0.75, 0.875, 1, 1, 1])
        config['contact_beta'] = 0.0
        np.testing.assert_array_equal(consensus.agreement_weights(bundle, observations, config), 1)

    def test_contact_and_agreement_sum_before_influence_clip(self):
        observations = observations_for(3)
        observations['contact_control'][:, 2:4] = [3, 1]
        source = prediction_from_logits([0, 1, 4])
        weights = consensus.agreement_weights(dict(short=source, setseq=source), observations,
                                              dict(alpha=0.75, contact_beta=0.5))
        np.testing.assert_allclose(weights, [0.625, 0.25, 0.25])
        self.assertTrue(np.all(weights > 0))

    def test_gaps_and_nonfinite_consensus_fields_are_rejected(self):
        source = prediction_from_logits(np.zeros(4))
        bundle = dict(short=source, setseq=source)
        config = consensus.CONFIGS[0]
        for blocks in ([[0, 1], [2, 4]], [[0, 3]], [[0, 3], [2, 4]]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                consensus.agreement_weights(bundle, observations_for(4, blocks), config)
        for column in range(7):
            for value in (np.nan, np.inf, -np.inf):
                observations = observations_for(4)
                observations['contact_control'][1, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    consensus.agreement_weights(bundle, observations, config)
        for source_name in ('short', 'setseq'):
            invalid = copy.deepcopy(bundle)
            invalid[source_name]['p_team'][1] = np.nan
            with self.subTest(source=source_name), self.assertRaises(ValueError):
                consensus.agreement_weights(invalid, observations_for(4), config)
        invalid = copy.deepcopy(bundle)
        invalid['short']['probability'][0, 0] = np.inf
        with self.assertRaises(ValueError):
            consensus.agreement_weights(invalid, observations_for(4), config)

    def test_invalid_shapes_config_and_missing_source_are_rejected(self):
        source = prediction_from_logits(np.zeros(4))
        bundle = dict(short=source, setseq=source)
        observations = observations_for(4)
        for shape in ((4,), (2, 4, 7), (4, 6)):
            invalid = dict(observations, contact_control=np.zeros(shape))
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                consensus.agreement_weights(bundle, invalid, consensus.CONFIGS[0])
        for config in (None, {}, dict(alpha=1, contact_beta=0), dict(alpha=np.nan, contact_beta=0),
                       dict(alpha=0.5, contact_beta=np.inf), dict(alpha=0.5, contact_beta=-1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                consensus.agreement_weights(bundle, observations, config)
        with self.assertRaises(ValueError):
            consensus.agreement_weights(dict(short=source), observations, consensus.CONFIGS[0])


class DecodeTests(unittest.TestCase):
    def test_constant_weights_match_actual_dead45_decoder(self):
        prediction, bundle, observations = fixture()
        for rate in (0.0, 8.0, 40.0, 80.0):
            stats = dict(rate=rate, dead_share=0.2)
            expected = b.baseline(prediction, observations, stats)
            for config in consensus.CONFIGS:
                with self.subTest(rate=rate, config=config['name']):
                    np.testing.assert_array_equal(
                        consensus.agreement_weights(bundle, observations, config), 1)
                    actual = consensus.decode(prediction, bundle, observations, stats, config)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, np.int8)

    def test_adaptive_costs_match_full_grid_postmerge_reference(self):
        prediction, bundle, observations = fixture()
        bundle['short'] = prediction
        bundle['setseq'] = prediction_from_logits(np.log(prediction['p_team'] /
                                                        (1.0 - prediction['p_team'])) * 0.7)
        observations['contact_control'][::4, 2:4] = [3, -1]
        for config in consensus.CONFIGS:
            for rate in (4.0, 16.0, 45.0):
                stats = dict(rate=rate, dead_share=0.2)
                weights = consensus.agreement_weights(bundle, observations, config)
                expected = reference_decode(prediction, observations, stats, weights)
                with self.subTest(config=config['name'], rate=rate):
                    actual = consensus.decode(prediction, bundle, observations, stats, config)
                    np.testing.assert_array_equal(actual, expected)

    def test_dead_threshold_minimum_length_and_halves_match_frozen_mask(self):
        prediction, bundle, observations = fixture()
        actual = consensus.decode(prediction, bundle, observations, dict(rate=20, dead_share=0.2),
                                  consensus.CONFIGS[1])
        self.assertTrue(np.all(actual[40:45] == 2))
        self.assertTrue(np.all(actual[95:99] != 2))
        self.assertTrue(np.all(actual[177:183] != 2))
        self.assertTrue(np.all(actual[240:250] == 2))

    def test_live_reset_ignores_entering_cost_at_half_and_after_dead(self):
        decoder = b.decoder
        blocks = np.array([[0, 5], [5, 10]], dtype=np.int64)
        dead = np.zeros(10, dtype=bool)
        dead[2:4] = True
        home = np.array([0, 0, -5, -5, -5, 0, 0, 0, 0, 0], dtype=float)
        away = -5 - home
        weights = np.ones(10)
        expected = decoder._decode_live_team(home, away, blocks, dead, weights, 1.0)
        weights[[0, 4, 5]] = 1e9
        actual = decoder._decode_live_team(home, away, blocks, dead, weights, 1.0)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[4], 1)
        self.assertEqual(actual[5], 0)

    def test_inputs_are_not_mutated(self):
        prediction, bundle, observations = fixture()
        stats = dict(rate=16.0, dead_share=0.2)
        config = copy.deepcopy(consensus.CONFIGS[5])
        inputs = (prediction, bundle, observations, stats, config)
        before = copy.deepcopy(inputs)
        consensus.agreement_weights(bundle, observations, config)
        consensus.decode(prediction, bundle, observations, stats, config)

        def assert_unchanged(actual, expected):
            if isinstance(actual, dict):
                self.assertEqual(actual.keys(), expected.keys())
                for key in actual:
                    assert_unchanged(actual[key], expected[key])
            elif isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                self.assertEqual(actual, expected)

        for actual, expected in zip(inputs, before):
            assert_unchanged(actual, expected)

    def test_bundle_dead_probability_does_not_define_primary_dead_mask(self):
        prediction, bundle, observations = fixture()
        source = prediction_from_logits(np.zeros(len(observations['X'])), 0.99)
        bundle = dict(short=source, setseq=copy.deepcopy(source))
        stats = dict(rate=20.0, dead_share=0.2)
        actual = consensus.decode(prediction, bundle, observations, stats, consensus.CONFIGS[0])
        expected = b.baseline(prediction, observations, stats)
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_training_statistics_and_primary_shape_are_rejected(self):
        prediction, bundle, observations = fixture()
        for stats in (None, {}, dict(rate=-1, dead_share=0.2), dict(rate=np.nan, dead_share=0.2),
                      dict(rate=10, dead_share=-0.1), dict(rate=10, dead_share=np.inf)):
            with self.subTest(stats=stats), self.assertRaises(ValueError):
                consensus.decode(prediction, bundle, observations, stats, consensus.CONFIGS[0])
        invalid = dict(observations, X=np.zeros((len(observations['X']), 15)))
        with self.assertRaises(ValueError):
            consensus.decode(prediction, bundle, invalid, dict(rate=10, dead_share=0.2),
                             consensus.CONFIGS[0])

    def test_observation_only_inputs_need_no_labels_or_context_lookup(self):
        class LabelGuard(dict):
            def get(self, key, default=None):
                if key in ('y', 'y25'):
                    raise AssertionError('ground truth access')
                return super().get(key, default)

            def __getitem__(self, key):
                if key in ('y', 'y25'):
                    raise AssertionError('ground truth access')
                return super().__getitem__(key)

        prediction, bundle, observations = fixture()
        stats = dict(rate=16.0, dead_share=0.2)
        config = consensus.CONFIGS[5]
        expected = consensus.decode(prediction, bundle, observations, stats, config)
        guarded = LabelGuard(observations, y=object(), y25=object())
        actual = consensus.decode(prediction, bundle, guarded, stats, config)
        np.testing.assert_array_equal(actual, expected)
        self.assertNotIn('y', observations)
        self.assertNotIn('y25', observations)


if __name__ == '__main__':
    unittest.main()
