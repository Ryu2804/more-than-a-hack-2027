import unittest

import numpy as np
import torch
from torch.nn import functional as F

import boundary_tasks


def state_reference(logits, labels, views, weights):
    known = (labels >= 0).to(logits.dtype)
    live = ((labels == 0) | (labels == 1)).to(logits.dtype)
    dead_target = (labels == 2).to(logits.dtype)
    team_target = (labels == views[:, None]).to(logits.dtype)
    dead = F.softplus(logits[:, 0]) - logits[:, 0] * dead_target
    team = F.softplus(logits[:, 1]) - logits[:, 1] * team_target
    return ((dead * known * weights).sum() / known.sum().clamp_min(1)
            + (team * live * weights).sum() / live.sum().clamp_min(1))


def pair_reference(logits, labels, views):
    terms = []
    for row in range(len(labels)):
        for frame in range(1, labels.shape[1]):
            before = int(labels[row, frame - 1])
            after = int(labels[row, frame])
            if before in (0, 1) and after in (0, 1) and before != after:
                sign = 1 if after == int(views[row]) else -1
                margin = 1 - sign * (logits[row, 1, frame] - logits[row, 1, frame - 1])
                terms.append(torch.logaddexp(margin, margin.new_zeros(())))
    return torch.stack(terms).mean() if terms else logits.sum() * 0


class FullMatchTargets(unittest.TestCase):
    def test_matches_original_independently_for_each_half(self):
        rng = np.random.default_rng(20261118)
        labels = rng.choice([-1, 0, 1, 2], size=71).astype(np.int8)
        blocks = [(0, 29), (29, 71)]
        before = labels.copy()
        target, valid = boundary_tasks.targets(labels, blocks)
        expected = [boundary_tasks.b.training.boundary_targets(labels[start:end])
                    for start, end in blocks]
        np.testing.assert_array_equal(target, np.concatenate([part[0] for part in expected]))
        np.testing.assert_array_equal(valid, np.concatenate([part[1] for part in expected]))
        self.assertEqual(target.dtype, np.float32)
        self.assertEqual(valid.dtype, np.bool_)
        np.testing.assert_array_equal(labels, before)
        self.assertFalse(np.shares_memory(target, labels))
        self.assertFalse(np.shares_memory(valid, labels))

    def test_half_reset_removes_change_and_dilation_across_halftime(self):
        labels = np.array([0] * 6 + [1] * 6, dtype=np.int8)
        target, valid = boundary_tasks.targets(labels, [(0, 6), (6, 12)])
        np.testing.assert_array_equal(target, np.zeros(12, dtype=np.float32))
        np.testing.assert_array_equal(valid, [False] + [True] * 5 + [False] + [True] * 5)
        concatenated, concatenated_valid = boundary_tasks.b.training.boundary_targets(labels)
        self.assertTrue(concatenated[6])
        self.assertTrue(concatenated_valid[6])

    def test_unknown_censors_adjacency_and_blocks_radius_two_dilation(self):
        labels = np.array([0, 0, 1, -1, 0, 0, 0, 0, 2, 2, -1, 1, 1])
        target, valid = boundary_tasks.targets(labels, [(0, len(labels))])
        np.testing.assert_array_equal(target, [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(
            valid, [False, True, True, False, False, True, True, True, True, True,
                    False, False, True],
        )
        self.assertFalse(target[5])
        self.assertFalse(target[12])

    def test_crop_retains_external_boundary_dilation_and_interior_validity(self):
        labels = np.array([0] * 8 + [1] * 8 + [2] * 8)
        target, valid = boundary_tasks.targets(labels, [(0, len(labels))])
        for start, end, expected in ((6, 8, [1, 1]), (8, 13, [1, 1, 1, 0, 0])):
            with self.subTest(crop=(start, end)):
                np.testing.assert_array_equal(target[start:end], expected)
                self.assertTrue(valid[start:end].all())
                cropped_target, cropped_valid = boundary_tasks.b.training.boundary_targets(
                    labels[start:end],
                )
                self.assertFalse(cropped_valid[0])
                self.assertFalse(cropped_target.any())

    def test_single_sample_halves_unknown_and_empty_match(self):
        target, valid = boundary_tasks.targets(np.array([0, 1, -1]), [(0, 1), (1, 2), (2, 3)])
        np.testing.assert_array_equal(target, [0, 0, 0])
        np.testing.assert_array_equal(valid, [False, False, False])
        target, valid = boundary_tasks.targets(np.array([], dtype=np.int8), [])
        self.assertEqual(target.shape, (0,))
        self.assertEqual(valid.shape, (0,))


class BoundaryLoss(unittest.TestCase):
    def test_full_targets_positive_weight_counts_and_analytic_gradient_mask(self):
        full = np.array([0, 0, 1, -1, 0, 0, 0, 0, 2, 2, -1, 1, 1])
        full_target, full_valid = boundary_tasks.targets(full, [(0, len(full))])
        labels = torch.tensor(np.stack((full[1:10], full[4:13])))
        target = torch.tensor(np.stack((full_target[1:10], full_target[4:13])), dtype=torch.float64)
        valid = torch.tensor(np.stack((full_valid[1:10], full_valid[4:13])))
        views = torch.tensor([0, 1])
        weights = torch.tensor([[2, 2, 9, 2, 2, 2, 2, 1, 1],
                                [2, 2, 2, 2, 1, 1, 9, 2, 2]], dtype=torch.float64)
        logits = torch.linspace(-2, 2, 54, dtype=torch.float64).reshape(2, 3, 9).requires_grad_()
        actual = boundary_tasks.loss(logits, labels, views, weights, 'boundary', (target, valid))
        positive_factor = 1 + 4 * target
        auxiliary = (F.softplus(logits[:, 2]) * (1 - target)
                     + 5 * F.softplus(-logits[:, 2]) * target)
        expected = (state_reference(logits, labels, views, weights)
                    + .2 * (auxiliary * valid).sum() / valid.sum())
        torch.testing.assert_close(actual, expected)
        actual.backward()
        known = (labels >= 0).double()
        live = ((labels == 0) | (labels == 1)).double()
        expected_gradient = torch.stack((
            (logits[:, 0].detach().sigmoid() - (labels == 2).double())
            * known * weights / known.sum(),
            (logits[:, 1].detach().sigmoid() - (labels == views[:, None]).double())
            * live * weights / live.sum(),
            .2 * positive_factor * (logits[:, 2].detach().sigmoid() - target)
            * valid / valid.sum(),
        ), dim=1)
        torch.testing.assert_close(logits.grad, expected_gradient)
        self.assertTrue(torch.equal(logits.grad[:, 2][~valid], torch.zeros_like(logits.grad[:, 2][~valid])))
        self.assertTrue(torch.equal(logits.grad[:, 0][labels == -1],
                                    torch.zeros_like(logits.grad[:, 0][labels == -1])))
        self.assertTrue(torch.equal(logits.grad[:, 1][live == 0],
                                    torch.zeros_like(logits.grad[:, 1][live == 0])))
        self.assertTrue(valid[0, 0])
        self.assertFalse(valid[1, 0])

    def test_external_change_supervises_crop_with_no_label_change_inside(self):
        full = np.array([0] * 8 + [1] * 8)
        target, valid = boundary_tasks.targets(full, [(0, len(full))])
        labels = torch.tensor(full[6:8][None])
        auxiliary = (torch.tensor(target[6:8][None], dtype=torch.float64),
                     torch.tensor(valid[6:8][None]))
        logits = torch.zeros((1, 3, 2), dtype=torch.float64, requires_grad=True)
        views = torch.tensor([0])
        weights = torch.ones((1, 2), dtype=torch.float64)
        actual = boundary_tasks.loss(logits, labels, views, weights, 'boundary', auxiliary)
        expected = state_reference(logits, labels, views, weights) + np.log(2)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(logits.grad[:, 2], torch.full((1, 2), -.25, dtype=torch.float64))

    def test_no_valid_targets_give_exact_base_and_zero_auxiliary_gradient(self):
        for values in ([-1, -1, -1], [0, 1, 2]):
            with self.subTest(labels=values):
                full = np.array(values)
                target, valid = boundary_tasks.targets(full, [(0, 1), (1, 2), (2, 3)])
                labels = torch.tensor(full[None])
                logits = torch.tensor([[[1., -2., 3.], [2., -1., 4.],
                                        [10000., -10000., 0.]]], dtype=torch.float64, requires_grad=True)
                views = torch.tensor([0])
                weights = torch.ones((1, 3), dtype=torch.float64)
                actual = boundary_tasks.loss(
                    logits, labels, views, weights, 'boundary',
                    (torch.tensor(target[None]), torch.tensor(valid[None])),
                )
                torch.testing.assert_close(actual, state_reference(logits, labels, views, weights))
                actual.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                torch.testing.assert_close(logits.grad[:, 2], torch.zeros((1, 3), dtype=torch.float64))
                if values[0] == -1:
                    self.assertEqual(actual.item(), 0)
                    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_extreme_logits_keep_boundary_loss_and_gradient_finite(self):
        full = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, -1])
        target, valid = boundary_tasks.targets(full, [(0, len(full))])
        logits = torch.tensor([[[10000., -10000.] * 5] * 3], dtype=torch.float64, requires_grad=True)
        actual = boundary_tasks.loss(
            logits, torch.tensor(full[None]), torch.tensor([1]), torch.ones((1, 10)),
            'boundary', (torch.tensor(target[None]), torch.tensor(valid[None])),
        )
        self.assertTrue(torch.isfinite(actual))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[:, 2][~torch.tensor(valid[None])],
                                    torch.zeros(2, dtype=torch.float64))

    def test_missing_full_targets_wrong_head_and_shape_rejected(self):
        labels = torch.tensor([[0, 1]])
        logits = torch.zeros((1, 3, 2))
        arguments = (labels, torch.tensor([0]), torch.ones((1, 2)), 'boundary')
        with self.assertRaises(ValueError):
            boundary_tasks.loss(logits, *arguments)
        with self.assertRaises(ValueError):
            boundary_tasks.loss(logits[:, :2], *arguments, (torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool)))
        with self.assertRaises(ValueError):
            boundary_tasks.loss(logits, *arguments, (torch.zeros((2,)), torch.ones((1, 2), dtype=torch.bool)))


class RankingLoss(unittest.TestCase):
    def test_numeric_reference_global_pair_mean_and_gradient(self):
        labels = torch.tensor([[0, 1, 1, 0, 2, 1, -1, 0],
                               [1, 0, 2, 2, -1, 0, 0, 0]])
        views = torch.tensor([0, 1])
        weights = torch.tensor([[2, 2, 2, 2, 1, 2, 9, 2],
                                [2, 2, 1, 1, 9, 2, 2, 2]], dtype=torch.float64)
        logits = torch.linspace(-1.3, 1.7, 32, dtype=torch.float64).reshape(2, 2, 8).requires_grad_()
        actual = boundary_tasks.loss(logits, labels, views, weights, 'ranking')
        expected = (state_reference(logits, labels, views, weights)
                    + .1 * pair_reference(logits, labels, views))
        torch.testing.assert_close(actual, expected)
        actual_gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
        expected_gradient, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertTrue(torch.isfinite(actual_gradient).all())
        self.assertEqual(actual_gradient[0, 1, 6].item(), 0)
        self.assertEqual(actual_gradient[1, 1, 4].item(), 0)
        self.assertEqual(actual_gradient[0, 1, 4].item(), 0)

    def test_signed_gradient_encourages_view_relative_change_in_both_directions(self):
        for values in ([0, 1], [1, 0]):
            for view in (0, 1):
                with self.subTest(labels=values, view=view):
                    labels = torch.tensor([values])
                    views = torch.tensor([view])
                    weights = torch.ones((1, 2), dtype=torch.float64)
                    logits = torch.zeros((1, 2, 2), dtype=torch.float64, requires_grad=True)
                    auxiliary = (boundary_tasks.loss(logits, labels, views, weights, 'ranking')
                                 - state_reference(logits, labels, views, weights))
                    auxiliary.backward()
                    sign = 1 if values[1] == view else -1
                    magnitude = .1 * torch.sigmoid(torch.tensor(1., dtype=torch.float64))
                    expected = torch.zeros_like(logits)
                    expected[0, 1, 0] = sign * magnitude
                    expected[0, 1, 1] = -sign * magnitude
                    torch.testing.assert_close(logits.grad, expected)
                    improved = logits.detach().clone()
                    improved[0, 1, 0] -= sign
                    improved[0, 1, 1] += sign
                    improved_auxiliary = (boundary_tasks.loss(improved, labels, views, weights, 'ranking')
                                          - state_reference(improved, labels, views, weights))
                    self.assertLess(improved_auxiliary.item(), auxiliary.item())

    def test_view_flip_and_logit_negation_preserve_loss_and_flip_team_gradient(self):
        labels = torch.tensor([[0, 1, 0, 2, -1, 1, 0]])
        views = torch.tensor([0])
        weights = torch.tensor([[2, 2, 2, 1, 9, 2, 2]], dtype=torch.float64)
        logits = torch.tensor([[[.1, -.2, .3, .4, -.5, .6, -.7],
                                [.7, -.6, .5, -.4, .3, -.2, .1]]], dtype=torch.float64, requires_grad=True)
        flipped = logits.detach().clone()
        flipped[:, 1].neg_()
        flipped.requires_grad_()
        actual = boundary_tasks.loss(logits, labels, views, weights, 'ranking')
        swapped = boundary_tasks.loss(flipped, labels, 1 - views, weights, 'ranking')
        torch.testing.assert_close(actual, swapped)
        actual.backward()
        swapped.backward()
        torch.testing.assert_close(logits.grad[:, 0], flipped.grad[:, 0])
        torch.testing.assert_close(logits.grad[:, 1], -flipped.grad[:, 1])

    def test_unknown_dead_same_and_separate_crop_edges_have_zero_pair_penalty(self):
        cases = ([0, 0, 0], [1, 1, 1], [0, -1, 1], [0, 2, 1],
                 [-1, -1, -1], [2, 2, 2], [0])
        for values in cases:
            with self.subTest(labels=values):
                labels = torch.tensor([values])
                logits = torch.linspace(-2, 2, 2 * len(values), dtype=torch.float64)
                logits = logits.reshape(1, 2, -1).requires_grad_()
                views = torch.tensor([0])
                weights = torch.full(labels.shape, 2., dtype=torch.float64)
                actual = boundary_tasks.loss(logits, labels, views, weights, 'ranking')
                base = state_reference(logits, labels, views, weights)
                torch.testing.assert_close(actual, base)
                gradient, = torch.autograd.grad(actual - base, logits)
                torch.testing.assert_close(gradient, torch.zeros_like(logits))
        labels = torch.tensor([[0, 0], [1, 1]])
        logits = torch.zeros((2, 2, 2), dtype=torch.float64, requires_grad=True)
        views = torch.tensor([0, 0])
        weights = torch.ones((2, 2), dtype=torch.float64)
        actual = boundary_tasks.loss(logits, labels, views, weights, 'ranking')
        base = state_reference(logits, labels, views, weights)
        torch.testing.assert_close(actual, base)
        gradient, = torch.autograd.grad(actual - base, logits)
        torch.testing.assert_close(gradient, torch.zeros_like(logits))

    def test_extreme_signed_pairs_remain_finite(self):
        labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
        views = torch.tensor([0, 1])
        weights = torch.ones((2, 4), dtype=torch.float64)
        logits = torch.tensor([[[10000., -10000., 10000., -10000.],
                                [-10000., 10000., -10000., 10000.]],
                               [[-10000., 10000., -10000., 10000.],
                                [10000., -10000., 10000., -10000.]]],
                              dtype=torch.float64, requires_grad=True)
        actual = boundary_tasks.loss(logits, labels, views, weights, 'ranking')
        expected = state_reference(logits, labels, views, weights) + .1 * pair_reference(logits, labels, views)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unknown_family_rejected(self):
        with self.assertRaises(ValueError):
            boundary_tasks.loss(torch.zeros((1, 2, 2)), torch.tensor([[0, 1]]),
                                torch.tensor([0]), torch.ones((1, 2)), 'other')


if __name__ == '__main__':
    unittest.main()
