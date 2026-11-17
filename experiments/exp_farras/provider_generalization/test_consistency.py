import unittest

import numpy as np
import torch

import bridge as b
import consistency


def reference_joint(logits):
    dead = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    team = 1.0 / (1.0 + np.exp(-logits[:, 1]))
    return np.stack(((1.0 - dead) * team, (1.0 - dead) * (1.0 - team), dead), axis=1)


def reference_kl(first, second, y):
    p, q = reference_joint(first), reference_joint(second)
    framewise = 0.5 * (
        p * np.log(p / q) + q * np.log(q / p)
    ).sum(axis=1)
    known = y >= 0
    return framewise[known].sum() / max(int(known.sum()), 1)


def reference_kl_gradient(first, second, y):
    p, q = reference_joint(first), reference_joint(second)
    dead = p[:, 2]
    team = 1.0 / (1.0 + np.exp(-first[:, 1]))
    derivative_dead = np.stack((-p[:, 0] * dead, -p[:, 1] * dead, dead * (1.0 - dead)), axis=1)
    derivative_team = np.stack((p[:, 0] * (1.0 - team), -p[:, 1] * team, np.zeros_like(team)), axis=1)
    factor = 0.5 * (np.log(p / q) + 1.0 - q / p)
    gradient = np.stack(((derivative_dead * factor).sum(axis=1), (derivative_team * factor).sum(axis=1)), axis=1)
    known = y >= 0
    return gradient * known[:, None, :] / max(int(known.sum()), 1)


def reference_state(logits, y, views, weights):
    known = y >= 0
    live = (y == 0) | (y == 1)
    dead_target = (y == 2).astype(np.float64)
    team_target = (y == views[:, None]).astype(np.float64)
    dead_factor = weights * known / max(int(known.sum()), 1)
    team_factor = weights * live / max(int(live.sum()), 1)
    dead = np.logaddexp(0.0, logits[:, 0]) - logits[:, 0] * dead_target
    team = np.logaddexp(0.0, logits[:, 1]) - logits[:, 1] * team_target
    value = (dead * dead_factor + team * team_factor).sum()
    sigmoid = 1.0 / (1.0 + np.exp(-logits))
    gradient = np.stack((
        (sigmoid[:, 0] - dead_target) * dead_factor,
        (sigmoid[:, 1] - team_target) * team_factor,
    ), axis=1)
    return value, gradient


class JointConsistency(unittest.TestCase):
    def setUp(self):
        self.first = torch.tensor([
            [[-0.8, 0.4, 1.2, -0.2, 0.7], [0.1, -0.3, 0.7, 1.4, -0.8]],
            [[0.5, -0.9, -0.1, 0.6, -1.1], [-0.4, 0.8, 1.1, -0.5, 0.2]],
        ], dtype=torch.float64)
        self.second = torch.tensor([
            [[0.3, -0.6, 0.8, -1.3, -0.4], [-0.7, 0.5, -0.2, 0.9, 1.0]],
            [[-0.2, 0.7, 0.4, -0.8, 0.6], [0.9, -0.1, -0.6, 0.3, -0.9]],
        ], dtype=torch.float64)
        self.y = torch.tensor([[0, 1, 2, -1, 0], [1, -1, -1, -1, 2]])
        self.views = torch.tensor([0, 1])
        self.weights = torch.tensor([[0.25, 4, 1, 9, 0.5], [2, 7, 8, 3, 1.5]], dtype=torch.float64)

    def test_joint_matches_independent_probability_math_and_order(self):
        actual = consistency.log_joint(self.first)
        expected = reference_joint(self.first.numpy())
        self.assertEqual(actual.shape, (2, 3, 5))
        np.testing.assert_allclose(actual.exp().numpy(), expected, rtol=1e-13, atol=1e-14)
        torch.testing.assert_close(actual.logsumexp(dim=1), torch.zeros((2, 5), dtype=torch.float64), rtol=0, atol=1e-14)
        zero = consistency.log_joint(torch.zeros((1, 2, 1), dtype=torch.float64)).exp()
        torch.testing.assert_close(zero, torch.tensor([[[0.25], [0.25], [0.5]]], dtype=torch.float64))

    def test_kl_matches_independent_three_state_math_and_analytic_gradients(self):
        first = self.first.clone().requires_grad_()
        second = self.second.clone().requires_grad_()
        actual = consistency.symmetric_kl(first, second, self.y)
        expected = reference_kl(self.first.numpy(), self.second.numpy(), self.y.numpy())
        self.assertAlmostEqual(actual.item(), expected, places=13)
        actual.backward()
        for current, other in ((first, second), (second, first)):
            expected_gradient = reference_kl_gradient(current.detach().numpy(), other.detach().numpy(), self.y.numpy())
            np.testing.assert_allclose(current.grad.numpy(), expected_gradient, rtol=1e-12, atol=1e-14)
            self.assertTrue(torch.equal(current.grad.transpose(1, 2)[self.y < 0], torch.zeros((4, 2), dtype=torch.float64)))
            self.assertGreater(current.grad.abs().sum().item(), 0)

    def test_swap_team_view_and_identical_predictions_are_symmetric(self):
        expected = consistency.symmetric_kl(self.first, self.second, self.y)
        torch.testing.assert_close(consistency.symmetric_kl(self.second, self.first, self.y), expected, rtol=0, atol=0)
        first, second = self.first.clone(), self.second.clone()
        first[:, 1] *= -1
        second[:, 1] *= -1
        torch.testing.assert_close(consistency.symmetric_kl(first, second, self.y), expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(consistency.log_joint(first), consistency.log_joint(self.first)[:, [1, 0, 2]])
        identical_first = self.first.clone().requires_grad_()
        identical_second = self.first.clone().requires_grad_()
        zero = consistency.symmetric_kl(identical_first, identical_second, self.y)
        self.assertEqual(zero.item(), 0)
        zero.backward()
        for logits in (identical_first, identical_second):
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_unknown_values_do_not_change_loss_or_known_gradients(self):
        expected_first = self.first.clone().requires_grad_()
        expected_second = self.second.clone().requires_grad_()
        expected = consistency.loss(expected_first, expected_second, self.y, self.views, self.weights)
        expected.backward()
        first, second = self.first.clone(), self.second.clone()
        first.masked_fill_((self.y < 0)[:, None, :], 10000)
        second.masked_fill_((self.y < 0)[:, None, :], -10000)
        weights = self.weights.clone()
        weights[self.y < 0] = 1000000
        first.requires_grad_()
        second.requires_grad_()
        actual = consistency.loss(first, second, self.y, self.views, weights)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        torch.testing.assert_close(first.grad, expected_first.grad, rtol=0, atol=0)
        torch.testing.assert_close(second.grad, expected_second.grad, rtol=0, atol=0)

    def test_complete_loss_matches_independent_weighted_state_math_and_gradients(self):
        first = self.first.clone().requires_grad_()
        second = self.second.clone().requires_grad_()
        a, ga = reference_state(self.first.numpy(), self.y.numpy(), self.views.numpy(), self.weights.numpy())
        c, gc = reference_state(self.second.numpy(), self.y.numpy(), self.views.numpy(), self.weights.numpy())
        kl = reference_kl(self.first.numpy(), self.second.numpy(), self.y.numpy())
        actual = consistency.loss(first, second, self.y, self.views, self.weights)
        self.assertAlmostEqual(actual.item(), 0.5 * (a + c) + 0.5 * kl, places=13)
        actual.backward()
        for current, other, state_gradient in ((first, second, ga), (second, first, gc)):
            kl_gradient = reference_kl_gradient(current.detach().numpy(), other.detach().numpy(), self.y.numpy())
            np.testing.assert_allclose(current.grad.numpy(), 0.5 * state_gradient + 0.5 * kl_gradient, rtol=1e-12, atol=1e-14)

    def test_complete_loss_preserves_view_and_team_relabeling_symmetry(self):
        expected = consistency.loss(self.first, self.second, self.y, self.views, self.weights)
        swapped_y = torch.where((self.y == 0) | (self.y == 1), 1 - self.y, self.y)
        torch.testing.assert_close(consistency.loss(self.first, self.second, swapped_y, 1 - self.views, self.weights), expected)
        first, second = self.first.clone(), self.second.clone()
        first[:, 1] *= -1
        second[:, 1] *= -1
        torch.testing.assert_close(consistency.loss(first, second, self.y, 1 - self.views, self.weights), expected)

    def test_consistency_includes_dead_frames_but_ignores_label_identities(self):
        all_dead = torch.full_like(self.y, 2)
        all_live = torch.zeros_like(self.y)
        expected = consistency.symmetric_kl(self.first, self.second, all_live)
        self.assertGreater(expected.item(), 0)
        torch.testing.assert_close(consistency.symmetric_kl(self.first, self.second, all_dead), expected, rtol=0, atol=0)

    def test_all_unknown_and_empty_timeline_return_differentiable_zero(self):
        for length in (0, 5):
            for objective in ('kl', 'complete'):
                with self.subTest(length=length, objective=objective):
                    first = self.first[:, :, :length].clone().requires_grad_()
                    second = self.second[:, :, :length].clone().requires_grad_()
                    y = torch.full((2, length), -1)
                    if objective == 'kl':
                        actual = consistency.symmetric_kl(first, second, y)
                    else:
                        actual = consistency.loss(first, second, y, self.views, self.weights[:, :length])
                    self.assertTrue(torch.isfinite(actual))
                    self.assertEqual(actual.item(), 0)
                    actual.backward()
                    for logits in (first, second):
                        self.assertIsNotNone(logits.grad)
                        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_saturated_finite_logits_have_finite_logprob_loss_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                first = torch.tensor([[[10000, -10000, 10000, -10000], [10000, 10000, -10000, -10000]]], dtype=dtype, requires_grad=True)
                second = (-first.detach()).requires_grad_()
                y = torch.tensor([[0, 1, 2, -1]])
                views, weights = torch.tensor([0]), torch.ones((1, 4), dtype=dtype)
                for logits in (first, second):
                    log_prob = consistency.log_joint(logits)
                    self.assertTrue(torch.isfinite(log_prob).all())
                    torch.testing.assert_close(log_prob.exp().sum(dim=1), torch.ones((1, 4), dtype=dtype))
                actual = consistency.loss(first, second, y, views, weights)
                self.assertTrue(torch.isfinite(actual))
                actual.backward()
                for logits in (first, second):
                    self.assertTrue(torch.isfinite(logits.grad).all())
                    self.assertGreater(logits.grad.abs().sum().item(), 0)
                    self.assertTrue(torch.equal(logits.grad[:, :, 3], torch.zeros((1, 2), dtype=dtype)))


class StochasticNetworkGradients(unittest.TestCase):
    def inputs(self):
        generator = torch.Generator().manual_seed(618)
        x = torch.randn((2, 29, 9), generator=generator)
        pp = torch.randn((2, 9, 48, 10), generator=generator)
        ppm = torch.rand((2, 9, 48), generator=generator) > 0.2
        y = torch.tensor([[0, 1, 2, 0, -1, 1, 2, 0, 1], [1, 2, 0, 1, 0, 2, 1, -1, 0]])
        return x, pp, ppm, y

    def model(self):
        return b.original._make_model(dict(name='pp_tcn', ch=8, dilations=(1, 2), dropout=0.5)).train()

    def test_two_stochastic_forwards_share_batch_and_both_backpropagate(self):
        torch.manual_seed(619)
        model = self.model()
        x, pp, ppm, y = self.inputs()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        first = model(x, pp, ppm)
        second = model(x, pp, ppm)
        first.retain_grad()
        second.retain_grad()
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(first.shape, (2, 2, 9))
        actual = consistency.loss(first, second, y, torch.tensor([0, 1]), torch.ones_like(y, dtype=x.dtype))
        actual.backward()
        for logits in (first, second):
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.abs().sum().item(), 0)
            self.assertTrue(torch.equal(logits.grad.transpose(1, 2)[y < 0], torch.zeros((2, 2))))
        for parameter, original in zip(model.parameters(), before):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            torch.testing.assert_close(parameter.detach(), original, rtol=0, atol=0)
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.ppnet[0].weight.grad.abs().sum().item(), 0)

    def test_joint_kl_reaches_parameters_of_both_networks(self):
        torch.manual_seed(620)
        first_model, second_model = self.model(), self.model()
        x, pp, ppm, y = self.inputs()
        actual = consistency.symmetric_kl(first_model(x, pp, ppm), second_model(x, pp, ppm), y)
        actual.backward()
        for model in (first_model, second_model):
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.ppnet[0].weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
