import unittest

import numpy as np
import torch

import bridge as b
import dropout


CONFIG = dict(
    name="pp_tcn", ch=64, dilations=(1, 2, 4, 8, 16, 1), dropout=0.15,
)


def inputs(batch=8, length=9, device="cpu"):
    generator = torch.Generator().manual_seed(823)
    x = torch.randn(batch, 29, length, generator=generator, dtype=torch.float64)
    pp = torch.randn(batch, length, 48, 10, generator=generator, dtype=torch.float64)
    ppm = torch.rand(batch, length, 48, generator=generator) >= 0.3
    ppm[:, 0] = False
    x[:, 24, 0] = -0.0
    return tuple(value.to(device) for value in (x, pp, ppm))


def assert_unchanged(test, values, snapshots, versions):
    for value, snapshot, version in zip(values, snapshots, versions):
        test.assertTrue(torch.equal(value, snapshot))
        test.assertEqual(value._version, version)


def assert_ball_identical(test, actual, expected):
    test.assertTrue(torch.equal(
        actual[:, 24:].contiguous().view(torch.int64),
        expected[:, 24:].contiguous().view(torch.int64),
    ))


class DropoutMasks(unittest.TestCase):
    def test_pp_mask_matches_numpy_and_is_tied_across_time(self):
        x, pp, ppm = inputs()
        ppm[:, 1] = True
        snapshots = tuple(value.clone() for value in (x, pp, ppm))
        versions = tuple(value._version for value in (x, pp, ppm))
        rng = np.random.default_rng(71)
        reference_rng = np.random.default_rng(71)
        keep = reference_rng.random((x.shape[0], 48)) >= 0.2
        expected = np.logical_and(ppm.numpy(), keep[:, None, :])

        actual_x, actual_pp, actual_mask = dropout.apply(x, pp, ppm, rng, "pp_dropout")

        self.assertIs(actual_x, x)
        self.assertIs(actual_pp, pp)
        self.assertIsNot(actual_mask, ppm)
        np.testing.assert_array_equal(actual_mask.numpy(), expected)
        np.testing.assert_array_equal(actual_mask[:, 1].numpy(), keep)
        self.assertTrue(np.any(~keep))
        self.assertFalse(np.array_equal(keep[0], keep[1]))
        self.assertFalse(bool(actual_mask[:, 0].any()))
        self.assertFalse(bool((actual_mask & ~ppm).any()))
        self.assertEqual(rng.random(), reference_rng.random())
        assert_ball_identical(self, actual_x, snapshots[0])
        assert_unchanged(self, (x, pp, ppm), snapshots, versions)

    def test_modality_exact_one_branch_decision_preserves_ball_and_tokens(self):
        x, pp, ppm = inputs(batch=64)
        ppm[:, 1] = True
        snapshots = tuple(value.clone() for value in (x, pp, ppm))
        versions = tuple(value._version for value in (x, pp, ppm))
        rng = np.random.default_rng(712)
        reference_rng = np.random.default_rng(712)
        draws = reference_rng.random(x.shape[0])
        pooled_drop = draws < 0.2
        token_drop = (draws >= 0.2) & (draws < 0.4)
        untouched = draws >= 0.4
        self.assertTrue(all(np.any(group) for group in (pooled_drop, token_drop, untouched)))
        expected_x = x.numpy().copy()
        expected_x[pooled_drop, :24, :] = 0
        expected_mask = ppm.numpy().copy()
        expected_mask[token_drop] = False

        actual_x, actual_pp, actual_mask = dropout.apply(
            x, pp, ppm, rng, "modality_dropout",
        )

        np.testing.assert_array_equal(actual_x.numpy(), expected_x)
        np.testing.assert_array_equal(actual_mask.numpy(), expected_mask)
        self.assertIs(actual_pp, pp)
        self.assertIsNot(actual_x, x)
        self.assertIsNot(actual_mask, ppm)
        for index in range(x.shape[0]):
            pooled_kept = torch.equal(actual_x[index, :24], x[index, :24])
            tokens_kept = torch.equal(actual_mask[index], ppm[index])
            self.assertTrue(pooled_kept or tokens_kept)
            self.assertEqual(pooled_kept, not pooled_drop[index])
            self.assertEqual(tokens_kept, not token_drop[index])
        self.assertEqual(rng.random(), reference_rng.random())
        assert_ball_identical(self, actual_x, x)
        assert_unchanged(self, (x, pp, ppm), snapshots, versions)

    def test_pp_dropout_allows_empty_mask_without_forced_keep(self):
        x, pp, ppm = inputs(batch=2)
        keep = np.random.default_rng(27).random((2, 48)) >= 0.2
        ppm[:] = torch.from_numpy(~keep)[:, None, :]
        self.assertTrue(bool(ppm.any()))
        actual_x, actual_pp, actual_mask = dropout.apply(
            x, pp, ppm, np.random.default_rng(27), "pp_dropout",
        )
        self.assertFalse(bool(actual_mask.any()))
        self.assertIs(actual_x, x)
        self.assertIs(actual_pp, pp)

    def test_leaf_inputs_remain_immutable_and_gradient_is_branch_local(self):
        for family in ("pp_dropout", "modality_dropout"):
            with self.subTest(family=family):
                x, pp, ppm = inputs(batch=32)
                x.requires_grad_()
                pp.requires_grad_()
                snapshots = tuple(value.detach().clone() for value in (x, pp, ppm))
                versions = tuple(value._version for value in (x, pp, ppm))
                actual_x, actual_pp, actual_mask = dropout.apply(
                    x, pp, ppm, np.random.default_rng(53), family,
                )
                loss = actual_x.sum() + actual_pp.sum()
                loss.backward()
                expected_grad = torch.ones_like(x)
                if family == "modality_dropout":
                    dropped = np.random.default_rng(53).random(32) < 0.2
                    expected_grad[torch.from_numpy(dropped), :24] = 0
                torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)
                torch.testing.assert_close(pp.grad, torch.ones_like(pp), rtol=0, atol=0)
                self.assertEqual(actual_mask.dtype, torch.bool)
                assert_unchanged(self, (x, pp, ppm), snapshots, versions)

    def test_control_is_identity_and_unknown_family_rejected_without_rng_draws(self):
        values = inputs()
        rng = np.random.default_rng(5)
        actual = dropout.apply(*values, rng, "control")
        for result, original in zip(actual, values):
            self.assertIs(result, original)
        with self.assertRaisesRegex(ValueError, "unknown dropout family"):
            dropout.apply(*values, rng, "other")
        self.assertEqual(rng.random(), np.random.default_rng(5).random())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_preserves_device_dtype_and_ball(self):
        for family in ("pp_dropout", "modality_dropout"):
            with self.subTest(family=family):
                values = inputs(batch=16, device="cuda")
                snapshots = tuple(value.clone() for value in values)
                versions = tuple(value._version for value in values)
                result = dropout.apply(*values, np.random.default_rng(1), family)
                for actual, original in zip(result, values):
                    self.assertEqual(actual.device, original.device)
                    self.assertEqual(actual.dtype, original.dtype)
                assert_ball_identical(self, result[0], values[0])
                assert_unchanged(self, values, snapshots, versions)


class ActualModelEffects(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            self.model = b.original._make_model(CONFIG).double().eval()

    def test_pp_mask_changes_logits_and_hidden_token_values_do_not(self):
        x, pp, ppm = inputs(batch=2)
        ppm[:] = True
        actual_x, actual_pp, actual_mask = dropout.apply(
            x, pp, ppm, np.random.default_rng(47), "pp_dropout",
        )
        hidden_values_changed = actual_pp.clone()
        hidden_values_changed[~actual_mask] = 1000
        with torch.no_grad():
            baseline = self.model(x, pp, ppm)
            augmented = self.model(actual_x, actual_pp, actual_mask)
            hidden_changed = self.model(actual_x, hidden_values_changed, actual_mask)
        self.assertTrue(bool(torch.isfinite(augmented).all()))
        self.assertGreater(float((baseline - augmented).abs().max()), 1e-8)
        torch.testing.assert_close(augmented, hidden_changed, rtol=0, atol=0)

    def test_permuting_tokens_and_augmented_mask_preserves_logits(self):
        values = inputs(batch=2)
        actual_x, actual_pp, actual_mask = dropout.apply(
            *values, np.random.default_rng(6), "pp_dropout",
        )
        permutation = torch.from_numpy(np.random.default_rng(8).permutation(48))
        # The sampled keep decisions travel with their player slots.
        with torch.no_grad():
            actual = self.model(actual_x, actual_pp, actual_mask)
            permuted = self.model(
                actual_x, actual_pp[:, :, permutation], actual_mask[:, :, permutation],
            )
        torch.testing.assert_close(actual, permuted, rtol=1e-10, atol=1e-10)

    def test_each_modality_drop_changes_only_selected_examples_logits(self):
        x, pp, ppm = inputs(batch=16)
        ppm[:, 1] = True
        seed = 29
        choices = np.random.default_rng(seed).random(x.shape[0])
        actual = dropout.apply(x, pp, ppm, np.random.default_rng(seed), "modality_dropout")
        with torch.no_grad():
            baseline = self.model(x, pp, ppm)
            augmented = self.model(*actual)
        self.assertTrue(bool(torch.isfinite(augmented).all()))
        for group in (choices < 0.2, (choices >= 0.2) & (choices < 0.4)):
            self.assertTrue(np.any(group))
            delta = (baseline[torch.from_numpy(group)] - augmented[torch.from_numpy(group)])
            self.assertTrue(bool((delta.abs().flatten(1).amax(1) > 1e-8).all()))
        kept = torch.from_numpy(choices >= 0.4)
        self.assertTrue(bool(kept.any()))
        torch.testing.assert_close(baseline[kept], augmented[kept], rtol=0, atol=0)

    def test_empty_pp_is_finite_and_independent_of_all_token_values(self):
        x, pp, ppm = inputs(batch=2)
        keep = np.random.default_rng(27).random((2, 48)) >= 0.2
        ppm[:] = torch.from_numpy(~keep)[:, None, :]
        actual_x, actual_pp, actual_mask = dropout.apply(
            x, pp, ppm, np.random.default_rng(27), "pp_dropout",
        )
        self.assertFalse(bool(actual_mask.any()))
        with torch.no_grad():
            actual = self.model(actual_x, actual_pp, actual_mask)
            other_tokens = self.model(actual_x, -actual_pp * 100, actual_mask)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual, other_tokens, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
