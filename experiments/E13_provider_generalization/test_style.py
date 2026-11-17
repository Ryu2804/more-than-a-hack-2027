import unittest

import numpy as np
import torch

import style


class FixedDraws:
    def __init__(self, decision, permutation=(), coefficients=()):
        self.decision = decision
        self.order = np.asarray(permutation, dtype=np.int64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1, 1, 1)
        self.calls = []

    def random(self):
        self.calls.append(('random',))
        return self.decision

    def permutation(self, count):
        self.calls.append(('permutation', count))
        if self.order.shape != (count,):
            raise AssertionError('Unexpected permutation draw')
        return self.order.copy()

    def beta(self, first, second, size):
        self.calls.append(('beta', first, second, size))
        if (first, second) != (0.5, 0.5) or size != self.coefficients.shape:
            raise AssertionError('MixStyle beta distribution changed')
        return self.coefficients.copy()


def numpy_reference(values, permutation, coefficients):
    result = values.copy()
    means = np.empty((len(values), 23), dtype=values.dtype)
    scales = np.empty_like(means)
    for sample in range(len(values)):
        for channel in range(23):
            series = values[sample, channel]
            means[sample, channel] = sum(series) / len(series)
            centered = series - means[sample, channel]
            scales[sample, channel] = np.sqrt(sum(centered * centered) / len(series) + 1e-6)
    for sample, donor in enumerate(permutation):
        coefficient = coefficients[sample]
        for channel in range(23):
            mean = coefficient * means[sample, channel] + (1 - coefficient) * means[donor, channel]
            scale = coefficient * scales[sample, channel] + (1 - coefficient) * scales[donor, channel]
            result[sample, channel] = (
                (values[sample, channel] - means[sample, channel])
                / scales[sample, channel] * scale + mean
            )
    return result


class MixStyleTests(unittest.TestCase):
    def batch(self, dtype=torch.float64, device='cpu'):
        values = np.random.default_rng(17).normal(size=(4, 29, 11))
        values *= np.array([0.25, 2.0, 0.75, 4.0])[:, None, None]
        values += np.array([-7.0, 3.0, 12.0, -1.0])[:, None, None]
        return torch.tensor(values, dtype=dtype, device=device)

    def test_default_matches_independent_numpy_reference_and_rng_sequence(self):
        x = self.batch()
        before = x.clone()
        rng = np.random.default_rng(2)
        expected_rng = np.random.default_rng(2)
        self.assertLess(expected_rng.random(), 0.5)
        permutation = expected_rng.permutation(len(x))
        coefficients = expected_rng.beta(0.5, 0.5, size=(len(x), 1, 1))
        expected = numpy_reference(before.numpy(), permutation, coefficients[:, 0, 0])
        actual = style.apply(x, rng)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(rng.random(), expected_rng.random())
        torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_applied_draws_match_reference_and_preserve_input(self):
        x = self.batch()
        before = x.clone()
        permutation = [2, 3, 1, 0]
        coefficients = [0.0, 0.25, 1.0, 0.6]
        rng = FixedDraws(0.2, permutation, coefficients)
        actual = style.apply(x, rng)
        expected = numpy_reference(before.numpy(), permutation, coefficients)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
        self.assertNotEqual(actual.data_ptr(), x.data_ptr())
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual[:, :23], before[:, :23]))
        self.assertEqual(rng.calls, [
            ('random',), ('permutation', 4), ('beta', 0.5, 0.5, (4, 1, 1))
        ])

    def test_validity_and_ball_channels_are_bit_identical(self):
        x = self.batch()
        x[:, 23:, 0] = -0.0
        x[:, 23:, 1] = float('nan')
        x[:, 23:, 2] = float('inf')
        original_bits = x[:, 23:].contiguous().view(torch.uint8).clone()
        actual = style.apply(x, FixedDraws(0.0, [1, 2, 3, 0], [0.3] * 4))
        self.assertTrue(torch.equal(actual[:, 23:].contiguous().view(torch.uint8), original_bits))
        self.assertTrue(torch.equal(x[:, 23:].contiguous().view(torch.uint8), original_bits))

    def test_detached_stats_have_only_local_affine_gradient(self):
        x = self.batch().requires_grad_()
        permutation = [1, 2, 3, 0]
        coefficients = np.array([0.25, 0.4, 0.6, 0.8])
        actual = style.apply(x, FixedDraws(0.0, permutation, coefficients))
        upstream = np.random.default_rng(29).normal(size=tuple(x.shape))
        (actual * torch.from_numpy(upstream)).sum().backward()
        values = x.detach().numpy()
        scales = np.sqrt(np.var(values[:, :23], axis=-1, ddof=0) + 1e-6)
        expected = upstream.copy()
        for sample, donor in enumerate(permutation):
            mixed_scale = coefficients[sample] * scales[sample] + (1 - coefficients[sample]) * scales[donor]
            expected[sample, :23] *= (mixed_scale / scales[sample])[:, None]
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-12, atol=1e-12)

    def test_donor_gets_no_gradient_from_recipient_statistics(self):
        x = self.batch().requires_grad_()
        actual = style.apply(x, FixedDraws(0.0, [1, 2, 3, 0], [0.0] * 4))
        actual[0, :23].sum().backward()
        self.assertGreater(x.grad[0, :23].abs().sum().item(), 0)
        self.assertTrue(torch.equal(x.grad[1:], torch.zeros_like(x.grad[1:])))
        self.assertTrue(torch.equal(x.grad[:, 23:], torch.zeros_like(x.grad[:, 23:])))

    def test_constant_channels_and_single_frame_stay_finite(self):
        for length in (1, 9):
            with self.subTest(length=length):
                x = torch.arange(4 * 29, dtype=torch.float64).reshape(4, 29, 1)
                x = x.expand(-1, -1, length).clone().requires_grad_()
                actual = style.apply(x, FixedDraws(0.0, [1, 2, 3, 0], [0.25] * 4))
                self.assertTrue(torch.isfinite(actual).all().item())
                expected = 0.25 * x.detach()[:, :23] + 0.75 * x.detach()[[1, 2, 3, 0], :23]
                torch.testing.assert_close(actual[:, :23], expected, rtol=0, atol=0)
                actual.square().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_probability_boundary_zero_and_singleton_consume_only_decision(self):
        for count, decision, probability in ((4, 0.5, 0.5), (4, 0.0, 0.0), (1, 0.0, 1.0)):
            with self.subTest(count=count, decision=decision, probability=probability):
                x = self.batch()[:count].requires_grad_()
                before = x.detach().clone()
                rng = FixedDraws(decision)
                actual = style.apply(x, rng, probability=probability)
                self.assertIs(actual, x)
                self.assertEqual(rng.calls, [('random',)])
                torch.testing.assert_close(x, before, rtol=0, atol=0)
                actual.sum().backward()
                torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)

    def test_statistics_do_not_persist_between_training_batches(self):
        x = self.batch()
        first = style.apply(x, FixedDraws(0.0, [1, 2, 3, 0], [0.4] * 4))
        unrelated_batch = self.batch() * 1000 + 5000
        style.apply(unrelated_batch, FixedDraws(0.0, [3, 0, 1, 2], [0.1] * 4))
        repeated = style.apply(x, FixedDraws(0.0, [1, 2, 3, 0], [0.4] * 4))
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        changed_other_channels = x.clone()
        changed_other_channels[:, 23:] *= 10000
        actual = style.apply(changed_other_channels, FixedDraws(0.0, [1, 2, 3, 0], [0.4] * 4))
        torch.testing.assert_close(actual[:, :23], first[:, :23], rtol=0, atol=0)

    def test_preserves_dtype_device_and_supports_noncontiguous_batches(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    x = self.batch(dtype=dtype, device=device)[:, :, ::2]
                    self.assertFalse(x.is_contiguous())
                    before = x.clone()
                    actual = style.apply(x, FixedDraws(0.0, [1, 2, 3, 0], [0.3] * 4))
                    self.assertEqual(actual.dtype, x.dtype)
                    self.assertEqual(actual.device, x.device)
                    torch.testing.assert_close(x, before, rtol=0, atol=0)
                    expected = numpy_reference(before.cpu().double().numpy(), [1, 2, 3, 0], [0.3] * 4)
                    np.testing.assert_allclose(actual.cpu().double().numpy(), expected, rtol=2e-6, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
