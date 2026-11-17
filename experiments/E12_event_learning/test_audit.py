import unittest

import numpy as np

import audit
import bridge as b
import models

FAMILIES = ('balanced', 'focal', 'boundary', 'ranking', 'ema', 'contact')


def seed_records(steps, parameters):
    return [dict(seed=seed, updates=steps, parameters=parameters) for seed in (0, 1, 2)]


class TrainingAudit(unittest.TestCase):
    def setUp(self):
        self.heldout = b.MATCHES[0]
        self.names = b.fair.split_names(self.heldout)
        self.parameters = 12345

    def validate(self, **changes):
        arguments = dict(heldout=self.heldout, family='balanced', names=self.names,
                         seed_fits=seed_records(500, self.parameters), steps=500,
                         expected_parameters=self.parameters)
        arguments.update(changes)
        return audit.training(**arguments)

    def test_smoke_and_full_budgets_for_every_family_and_fold(self):
        for heldout in b.MATCHES:
            for family in FAMILIES:
                parameters = self.parameters + {'boundary': 65, 'contact': 448}.get(family, 0)
                for steps in (1, 500):
                    with self.subTest(heldout=heldout, family=family, steps=steps):
                        records = seed_records(steps, parameters)
                        before = [dict(record) for record in records]
                        audit.training(heldout, family, list(b.fair.split_names(heldout)),
                                       records, steps, parameters)
                        self.assertEqual(records, before)

    def test_rejects_actual_heldout_contamination(self):
        names = (self.heldout,) + self.names[1:]
        with self.assertRaises(ValueError):
            self.validate(names=names)

    def test_rejects_duplicate_missing_extra_foreign_or_reordered_matches(self):
        invalid = (self.names[:-1], self.names + (self.heldout,),
                   self.names[:-1] + (self.names[0],),
                   self.names[:-1] + ('not-a-match',), self.names[::-1], None)
        for names in invalid:
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.validate(names=names)

    def test_rejects_unknown_family_and_heldout(self):
        for family in ('event_tcn', '', None):
            with self.subTest(family=family), self.assertRaises(ValueError):
                self.validate(family=family)
        with self.assertRaises(ValueError):
            self.validate(heldout='not-a-match')

    def test_steps_must_be_positive_integers(self):
        for steps in (0, -1, 1.5, 500.0, True, np.bool_(True), '500', None):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.validate(steps=steps)

    def test_rejects_missing_extra_reordered_and_duplicate_seeds(self):
        valid = seed_records(500, self.parameters)
        invalid = ([], valid[:2], valid + [dict(valid[0])], valid[::-1],
                   [valid[0], dict(valid[0]), valid[2]], None)
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.validate(seed_fits=records)

    def test_rejects_wrong_updates_or_parameter_counts_on_each_seed(self):
        for index in range(3):
            for key, value in (('seed', 9), ('updates', 499), ('updates', 501),
                               ('parameters', self.parameters + 1),
                               ('parameters', self.parameters - 1)):
                records = seed_records(500, self.parameters)
                records[index][key] = value
                with self.subTest(index=index, key=key, value=value), self.assertRaises(ValueError):
                    self.validate(seed_fits=records)

    def test_rejects_malformed_seed_records_and_noninteger_fields(self):
        for key in ('seed', 'updates', 'parameters'):
            records = seed_records(500, self.parameters)
            del records[1][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.validate(seed_fits=records)
            for value in (True, None, '1', float(seed_records(500, self.parameters)[1][key])):
                records = seed_records(500, self.parameters)
                records[1][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.validate(seed_fits=records)
        with self.assertRaises(ValueError):
            self.validate(seed_fits=[{}, object(), {}])

    def test_expected_parameter_count_must_be_positive_integer(self):
        for count in (0, -1, True, 12345.0, None, {'balanced': self.parameters}):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.validate(expected_parameters=count)


class ObservationAudit(unittest.TestCase):
    def setUp(self):
        self.heldout = b.MATCHES[0]
        self.names = b.fair.split_names(self.heldout)
        self.fitted = dict(training_names=self.names)

    def test_unlabeled_observations_return_unchanged_without_feature_access(self):
        features = np.array([[1., 2.]])
        observed = dict(_name=self.heldout, X=features)
        self.assertIs(audit.observations(self.fitted, observed), observed)
        self.assertEqual(set(observed), {'_name', 'X'})
        self.assertIs(observed['X'], features)
        np.testing.assert_array_equal(features, [[1., 2.]])
        minimal = {'_name': self.heldout}
        self.assertIs(audit.observations(self.fitted, minimal), minimal)

    def test_rejects_every_training_match_even_when_nets_are_absent(self):
        for name in self.names:
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.observations(self.fitted, {'_name': name})

    def test_gt_keys_rejected_without_reading_values(self):
        class GuardedLabels(dict):
            def __getitem__(self, key):
                if key in ('y', 'y25'):
                    raise AssertionError('GT value was read')
                return super().__getitem__(key)

        for key in ('y', 'y25'):
            for labels in (None, np.empty(0), np.array([0, 1, 2])):
                observed = GuardedLabels(_name=self.heldout, **{key: labels})
                with self.subTest(key=key, labels=labels), self.assertRaises(ValueError):
                    audit.observations(self.fitted, observed)
                self.assertIn(key, observed)

    def test_rejects_missing_blank_or_nonstring_observed_name(self):
        for observed in ({}, {'_name': ''}, {'_name': ' '}, {'_name': None},
                         {'_name': 123}, [], None):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                audit.observations(self.fitted, observed)

    def test_rejects_missing_duplicate_or_malformed_training_membership(self):
        invalid = ({}, {'training_names': None}, {'training_names': self.names[:-1]},
                   {'training_names': self.names + (self.heldout,)},
                   {'training_names': self.names[:-1] + (self.names[0],)},
                   {'training_names': self.names[:-1] + (' ',)},
                   {'training_names': self.names[:-1] + ([],)},
                   {'training_names': 'abcdef'}, None)
        for fitted in invalid:
            with self.subTest(fitted=fitted), self.assertRaises(ValueError):
                audit.observations(fitted, {'_name': self.heldout})


class ProbabilityAudit(unittest.TestCase):
    def setUp(self):
        self.values = np.array([[.2, .3, .5], [0., 0., 1.], [1., 0., 0.]])

    def test_valid_array_preserves_identity_dtype_and_mass(self):
        for dtype in (np.float32, np.float64):
            values = self.values.astype(dtype)
            before = values.copy()
            values.flags.writeable = False
            self.assertIs(audit.probability(values, 3), values)
            self.assertEqual(values.dtype, dtype)
            np.testing.assert_array_equal(values, before)
        np.testing.assert_array_equal(audit.probability(self.values.tolist(), 3), self.values)
        self.assertEqual(audit.probability(np.empty((0, 3)), 0).shape, (0, 3))

    def test_accepts_tolerance_without_renormalizing(self):
        values = np.array([[.2, .3, .5000009], [.2, .3, .4999991]])
        before = values.copy()
        self.assertIs(audit.probability(values, 2), values)
        np.testing.assert_array_equal(values, before)

    def test_rejects_nan_infinity_negative_zero_mass_and_bad_normalization(self):
        invalid = ([np.nan, 0., 1.], [np.inf, 0., 1.], [-np.inf, 0., 1.],
                   [-1e-12, .5, .500000000001], [0., 0., 0.],
                   [.2, .3, .5000011], [.2, .3, .4999989], [1., 1., 1.])
        for row in invalid:
            values = self.values.copy()
            values[1] = row
            before = values.copy()
            with self.subTest(row=row), self.assertRaises(ValueError):
                audit.probability(values, 3)
            np.testing.assert_array_equal(values, before)

    def test_rejects_wrong_grid_shape_and_nonnumeric_arrays(self):
        invalid = (np.ones((2, 3)) / 3, np.ones((3, 2)) / 2,
                   np.ones((3, 4)) / 4, self.values.reshape(1, 3, 3),
                   self.values.ravel(), 1., [['x', 'y', 'z']] * 3,
                   self.values.astype(complex), np.array([[.2, .3, .5]], dtype=object),
                   [[.2, .8], [.2, .3, .5]])
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                audit.probability(values, 3)
        for n in (2, 4, -1, True, 3., None):
            with self.subTest(n=n), self.assertRaises(ValueError):
                audit.probability(self.values, n)

    def test_replay_accepts_exact_copy_without_mutation(self):
        actual, expected = self.values.copy(), self.values.copy()
        actual.flags.writeable = False
        expected.flags.writeable = False
        audit.replay(actual, expected)
        np.testing.assert_array_equal(actual, self.values)
        np.testing.assert_array_equal(expected, self.values)

    def test_replay_rejects_one_ulp_even_when_allclose(self):
        for dtype in (np.float32, np.float64):
            expected = self.values.astype(dtype)
            actual = expected.copy()
            actual[0, 0] = np.nextafter(actual[0, 0], dtype(1.))
            self.assertTrue(np.allclose(actual, expected))
            self.assertFalse(np.array_equal(actual, expected))
            for left, right in ((actual, expected), (expected, actual)):
                with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                    audit.replay(left, right)

    def test_replay_rejects_identical_invalid_values_on_either_side(self):
        invalid = (np.array([[np.nan, 0., 1.]]), np.array([[np.inf, 0., 1.]]),
                   np.zeros((1, 3)), np.array([[-.1, .6, .5]]),
                   np.array([[.2, .3, .6]]), np.ones((1, 2)), np.array(1.))
        valid = self.values[:1]
        for values in invalid:
            for actual, expected in ((values, values.copy()), (values, valid), (valid, values)):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    audit.replay(actual, expected)
        with self.assertRaises(ValueError):
            audit.replay(self.values, self.values[:1])


class PredictionConsumerFailures(unittest.TestCase):
    def setUp(self):
        self.heldout = b.MATCHES[0]
        self.names = b.fair.split_names(self.heldout)

    def test_all_families_reject_contamination_before_accessing_missing_models(self):
        for family in FAMILIES:
            fitted = dict(family=family, training_names=self.names)
            for name in self.names:
                with self.subTest(family=family, name=name), self.assertRaises(ValueError):
                    models.predict(fitted, {'_name': name})

    def test_all_families_reject_gt_keys_before_feature_or_model_access(self):
        for family in FAMILIES:
            fitted = dict(family=family, training_names=self.names)
            for key in ('y', 'y25'):
                for value in (None, np.empty(0), np.array([0, 1, 2])):
                    observed = dict(_name=self.heldout, **{key: value})
                    with self.subTest(family=family, key=key, value=value), self.assertRaises(ValueError):
                        models.predict(fitted, observed)
                    self.assertIs(observed[key], value)

    def test_all_families_reject_duplicate_membership_and_unnamed_observations(self):
        duplicate_names = self.names[:-1] + (self.names[0],)
        for family in FAMILIES:
            fitted = dict(family=family, training_names=self.names)
            for observed in ({}, {'_name': ''}, {'_name': None}, []):
                with self.subTest(family=family, observed=observed), self.assertRaises(ValueError):
                    models.predict(fitted, observed)
            with self.subTest(family=family), self.assertRaises(ValueError):
                models.predict(dict(family=family, training_names=duplicate_names), {'_name': self.heldout})


if __name__ == '__main__':
    unittest.main()
