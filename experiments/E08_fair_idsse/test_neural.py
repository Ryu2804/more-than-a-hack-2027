import unittest

import numpy as np

import neural


class NeuralTrainingNames(unittest.TestCase):
    def test_unique_names_preserve_training_order(self):
        datasets = [
            {"_name": "train-two", "y": np.array([0, 2, 1])},
            {"_name": "train-one", "y": np.array([1, -1, 0])},
        ]
        self.assertEqual(neural._training_names(datasets), ("train-two", "train-one"))

    def test_rejects_missing_duplicate_and_single_match_names(self):
        labeled = {"_name": "train-one", "y": np.array([0])}
        cases = [
            [],
            [labeled],
            [labeled, dict(labeled)],
            [labeled, dict(labeled, _source="another-provider")],
            [labeled, {"y": np.array([1])}],
            [labeled, {"_name": "", "y": np.array([1])}],
            [labeled, {"_name": "  ", "y": np.array([1])}],
            [labeled, {"_name": None, "y": np.array([1])}],
            [labeled, {"_name": "train-two"}],
        ]
        for datasets in cases:
            with self.subTest(datasets=datasets), self.assertRaises(ValueError):
                neural._training_names(datasets)
        with self.assertRaises(TypeError):
            neural._training_names([labeled, object()])

    def test_all_families_reject_repeated_match_before_training(self):
        labeled = {"_name": "train-one", "y": np.array([0])}
        for family in neural.FAMILIES:
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "unique"):
                neural.fit([labeled, dict(labeled)], family, steps=1)

    def test_invalid_budget_and_seed_sets_fail_before_training(self):
        for steps in (0, -1, 0.5, 1.5, True, "500"):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "positive integer"):
                neural.fit([], "setseq", steps=steps)
        for seeds in ((0,), (0, 1, 1), (2, 1, 0), (False, 1, 2), (0.0, 1, 2)):
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "seeds"):
                neural.fit([], "setseq", seeds=seeds)
        with self.assertRaisesRegex(ValueError, "unknown neural family"):
            neural.fit([], "unknown")


class NeuralInferenceIsolation(unittest.TestCase):
    def setUp(self):
        self.fitted = {"family": "setseq", "training_names": ("train-one", "train-two")}

    def test_accepts_disjoint_named_observations_without_copy_or_mutation(self):
        observations = {"_name": "held-out"}
        self.assertIs(neural._prediction_observations(self.fitted, observations), observations)
        self.assertEqual(observations, {"_name": "held-out"})

    def test_predict_rejects_each_training_name_before_model_access(self):
        for name in self.fitted["training_names"]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "training names"):
                neural.predict(self.fitted, {"_name": name})

    def test_predict_rejects_labels_even_when_empty_or_none(self):
        for label_key in ("y", "y25"):
            for labels in (None, np.empty(0), np.array([0, 1, 2])):
                observations = {"_name": "held-out", label_key: labels}
                with self.subTest(label_key=label_key, labels=labels):
                    with self.assertRaisesRegex(ValueError, "must not contain labels"):
                        neural.predict(self.fitted, observations)
                    self.assertIs(observations[label_key], labels)

    def test_predict_requires_named_mapping(self):
        for observations in ({}, {"_name": ""}, {"_name": " "}, {"_name": None}):
            with self.subTest(observations=observations), self.assertRaisesRegex(ValueError, "_name"):
                neural.predict(self.fitted, observations)
        with self.assertRaises(TypeError):
            neural.predict(self.fitted, [])


class NeuralProbabilityAveraging(unittest.TestCase):
    def test_averages_joint_probabilities_not_conditional_team_heads(self):
        probabilities = [
            np.array([[0.81, 0.09, 0.10], [0.0, 0.0, 1.0]]),
            np.array([[0.10, 0.40, 0.50], [0.5, 0.5, 0.0]]),
            np.array([[0.02, 0.18, 0.80], [1.0, 0.0, 0.0]]),
        ]
        snapshots = [values.copy() for values in probabilities]
        outputs = [{"probability": values, "boundary": np.full(2, np.nan)} for values in probabilities]
        result = neural._average_probabilities(outputs)
        expected = np.mean(np.stack(probabilities), axis=0)
        np.testing.assert_allclose(result["probability"], expected)
        np.testing.assert_allclose(result["p_dead"], expected[:, 2])
        np.testing.assert_allclose(result["p_team"], expected[:, 0] / expected[:, :2].sum(axis=1))
        self.assertNotAlmostEqual(result["p_team"][0], np.mean([0.9, 0.2, 0.1]))
        self.assertFalse(result["has_boundary"])
        self.assertNotIn("boundary", result)
        for values, snapshot in zip(probabilities, snapshots):
            np.testing.assert_array_equal(values, snapshot)

    def test_normalizes_each_seed_before_equal_weighting(self):
        outputs = [
            {"probability": np.array([[8, 1, 1]])},
            {"probability": np.array([[1, 2, 1]])},
            {"probability": np.array([[0, 0, 9]])},
        ]
        result = neural._average_probabilities(outputs)
        np.testing.assert_allclose(result["probability"], [[0.35, 0.2, 0.45]])
        np.testing.assert_allclose(result["p_team"], [7 / 11])

    def test_all_dead_rows_have_neutral_conditional_team(self):
        outputs = [{"probability": np.array([[0.0, 0.0, 1.0]])} for _ in neural.SEEDS]
        result = neural._average_probabilities(outputs)
        np.testing.assert_array_equal(result["probability"], [[0, 0, 1]])
        np.testing.assert_array_equal(result["p_team"], [0.5])

    def test_empty_predictions_keep_three_class_shape(self):
        outputs = [{"probability": np.empty((0, 3))} for _ in neural.SEEDS]
        result = neural._average_probabilities(outputs)
        self.assertEqual(result["probability"].shape, (0, 3))
        self.assertEqual(result["p_dead"].shape, (0,))
        self.assertEqual(result["p_team"].shape, (0,))

    def test_rejects_missing_seed_mismatched_rows_and_invalid_joint_mass(self):
        valid = {"probability": np.array([[0.2, 0.3, 0.5]])}
        for count in (0, 1, 2, 4):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "three seed"):
                neural._average_probabilities([valid] * count)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            neural._average_probabilities([valid, valid, {"probability": np.ones((2, 3))}])
        invalid = [
            np.zeros((1, 3)),
            np.array([[-0.1, 0.5, 0.6]]),
            np.array([[np.nan, 0.0, 1.0]]),
            np.array([[np.inf, 0.0, 1.0]]),
            np.ones((1, 2)),
        ]
        for probability in invalid:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                neural._average_probabilities([valid, valid, {"probability": probability}])


if __name__ == "__main__":
    unittest.main()
