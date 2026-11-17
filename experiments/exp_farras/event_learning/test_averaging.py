from io import BytesIO
import random
import unittest

import numpy as np
import torch
from torch import nn

import averaging


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float64)
        )
        self.bias = nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float64))
        self.norm = nn.BatchNorm1d(2, dtype=torch.float64)
        self.dropout = nn.Dropout(0.4)
        self.register_buffer("enabled", torch.tensor([True, False]))
        self.register_buffer("scratch", torch.tensor([1.0]), persistent=False)

    def forward(self, x):
        return self.dropout(self.norm(x @ self.weight.T + self.bias))


def tensor_snapshot(model):
    return {
        **{f"parameter:{name}": value.detach().clone()
           for name, value in model.named_parameters()},
        **{f"buffer:{name}": value.detach().clone()
           for name, value in model.named_buffers()},
    }


class AveragingTests(unittest.TestCase):
    def assert_snapshot_equal(self, model, expected):
        actual = tensor_snapshot(model)
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            with self.subTest(tensor=name):
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_create_copies_once_without_mutating_training_state_or_gradients(self):
        model = TinyModel().train()
        model.norm.eval()
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, index + 0.25)
        expected = tensor_snapshot(model)
        modes = [module.training for module in model.modules()]
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        shadow = averaging.create(model)
        self.assertIsNot(shadow, model)
        self.assert_snapshot_equal(shadow, expected)
        self.assert_snapshot_equal(model, expected)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertTrue(all(not module.training for module in shadow.modules()))
        for source, copied, gradient in zip(
            model.parameters(), shadow.parameters(), gradients
        ):
            self.assertTrue(source.requires_grad)
            self.assertFalse(copied.requires_grad)
            self.assertIsNone(copied.grad)
            self.assertIsNone(copied.grad_fn)
            self.assertNotEqual(source.data_ptr(), copied.data_ptr())
            torch.testing.assert_close(source.grad, gradient, rtol=0, atol=0)
        for source, copied in zip(model.buffers(), shadow.buffers()):
            self.assertNotEqual(source.data_ptr(), copied.data_ptr())
        with torch.no_grad():
            shadow.weight.add_(17)
            shadow.norm.running_mean.add_(5)
            shadow.enabled.logical_not_()
            shadow.scratch.zero_()
        self.assert_snapshot_equal(model, expected)

    def test_default_decay_matches_independent_closed_form_without_debiasing(self):
        model = TinyModel()
        shadow = averaging.create(model)
        initial = {name: value.detach().clone()
                   for name, value in model.named_parameters()}
        history = []
        for step in range(1, 6):
            with torch.no_grad():
                for index, parameter in enumerate(model.parameters()):
                    parameter.fill_((index + 1) * step - 0.3 * step * step)
            history.append({name: value.detach().clone()
                            for name, value in model.named_parameters()})
            training_state = tensor_snapshot(model)
            averaging.update(shadow, model)
            self.assert_snapshot_equal(model, training_state)
        decay = 0.99
        count = len(history)
        for name, actual in shadow.named_parameters():
            expected = initial[name] * decay ** count
            for step, values in enumerate(history, start=1):
                expected = expected + (1 - decay) * decay ** (count - step) * values[name]
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            self.assertFalse(actual.requires_grad)
            self.assertIsNone(actual.grad)
            self.assertIsNone(actual.grad_fn)

    def test_one_update_per_optimizer_step_and_no_shadow_optimizer_parameters(self):
        model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.fill_(2.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        shadow = averaging.create(model)
        history = []
        for step in range(1, 5):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(torch.tensor([[1.0]], dtype=torch.float64)) - step).square().sum()
            loss.backward()
            shadow_before_step = shadow.weight.detach().clone()
            optimizer.step()
            torch.testing.assert_close(shadow.weight, shadow_before_step, rtol=0, atol=0)
            current = model.weight.detach().clone()
            gradient = model.weight.grad.clone()
            history.append(current)
            averaging.update(shadow, model)
            torch.testing.assert_close(model.weight, current, rtol=0, atol=0)
            torch.testing.assert_close(model.weight.grad, gradient, rtol=0, atol=0)
        expected = torch.full_like(model.weight, 2.0 * 0.99 ** 4)
        for index, value in enumerate(history):
            expected += 0.01 * 0.99 ** (3 - index) * value
        torch.testing.assert_close(shadow.weight, expected, rtol=1e-12, atol=1e-12)
        self.assertIsNone(shadow.weight.grad)
        self.assertFalse(shadow.weight.requires_grad)
        self.assertIs(optimizer.param_groups[0]["params"][0], model.weight)

    def test_initialization_contribution_remains_after_exactly_500_updates(self):
        model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.fill_(2.0)
        shadow = averaging.create(model)
        with torch.no_grad():
            model.weight.fill_(7.0)
        for _ in range(499):
            averaging.update(shadow, model)
        expected_499 = 7.0 + (2.0 - 7.0) * 0.99 ** 499
        self.assertAlmostEqual(shadow.weight.item(), expected_499, places=12)
        averaging.update(shadow, model)
        expected_500 = 7.0 + (2.0 - 7.0) * 0.99 ** 500
        self.assertAlmostEqual(shadow.weight.item(), expected_500, places=12)
        self.assertGreater(abs(expected_500 - expected_499), 1e-5)
        self.assertNotEqual(shadow.weight.item(), model.weight.item())

    def test_all_buffers_copy_latest_value_including_integer_bool_and_nonpersistent(self):
        model = TinyModel().train()
        shadow = averaging.create(model)
        for step in (1, 4):
            with torch.no_grad():
                model.norm.running_mean.fill_(step + 0.5)
                model.norm.running_var.fill_(step * 3.0)
                model.norm.num_batches_tracked.fill_(step)
                model.enabled.logical_not_()
                model.scratch.fill_(step + 11)
            expected = tensor_snapshot(model)
            averaging.update(shadow, model)
            for name, value in shadow.named_buffers():
                torch.testing.assert_close(value, expected[f"buffer:{name}"], rtol=0, atol=0)
            self.assert_snapshot_equal(model, expected)
        self.assertTrue(model.training)
        self.assertTrue(model.norm.training)
        self.assertTrue(all(not module.training for module in shadow.modules()))

    def test_create_and_update_do_not_advance_random_generators(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(0.5))
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        shadow = averaging.create(model)
        averaging.update(shadow, model)
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
        if cuda_before is not None:
            for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_cpu_state_save_load_preserves_predictions_and_next_update(self):
        model = TinyModel()
        shadow = averaging.create(model)
        with torch.no_grad():
            model.weight.add_(3.0)
            model.norm.running_mean.fill_(0.7)
            model.norm.num_batches_tracked.fill_(9)
        averaging.update(shadow, model)
        stream = BytesIO()
        torch.save({name: value.detach().cpu().clone()
                    for name, value in shadow.state_dict().items()}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        restored = averaging.create(TinyModel())
        restored.load_state_dict(saved, strict=True)
        inputs = torch.tensor([[0.1, 0.2], [-0.4, 1.1]], dtype=torch.float64)
        torch.testing.assert_close(restored(inputs), shadow(inputs), rtol=0, atol=0)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, shadow.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(value.device.type, "cpu")
        with torch.no_grad():
            model.weight.sub_(0.4)
        averaging.update(shadow, model)
        averaging.update(restored, model)
        self.assert_snapshot_equal(restored, tensor_snapshot(shadow))
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for p in restored.parameters()))

    def assert_mismatch_does_not_change_shadow(self, mutate, message):
        model = TinyModel()
        shadow = averaging.create(model)
        with torch.no_grad():
            model.weight.add_(10)
        mutate(model)
        before = tensor_snapshot(shadow)
        with self.assertRaisesRegex(ValueError, message):
            averaging.update(shadow, model)
        self.assert_snapshot_equal(shadow, before)

    def test_parameter_name_shape_and_count_mismatches_reject_before_mutation(self):
        def rename(model):
            value = model.bias
            del model.bias
            model.register_parameter("renamed_bias", value)

        mutations = (
            (rename, "parameter names differ"),
            (lambda model: setattr(model, "bias", nn.Parameter(torch.zeros(3))),
             "parameter shape differs"),
            (lambda model: model.register_parameter("extra", nn.Parameter(torch.ones(1))),
             "parameter names differ"),
            (lambda model: delattr(model, "bias"), "parameter names differ"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message, mutate=mutate):
                self.assert_mismatch_does_not_change_shadow(mutate, message)

    def test_buffer_name_shape_and_count_mismatches_reject_before_parameter_update(self):
        def rename(model):
            value = model.scratch
            del model.scratch
            model.register_buffer("renamed_scratch", value)

        mutations = (
            (rename, "buffer names differ"),
            (lambda model: setattr(model, "scratch", torch.zeros(3)), "buffer shape differs"),
            (lambda model: model.register_buffer("extra", torch.ones(1)), "buffer names differ"),
            (lambda model: delattr(model, "scratch"), "buffer names differ"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message, mutate=mutate):
                self.assert_mismatch_does_not_change_shadow(mutate, message)

    def test_shared_training_instance_or_parameter_is_rejected(self):
        model = TinyModel()
        before = tensor_snapshot(model)
        with self.assertRaisesRegex(ValueError, "separate"):
            averaging.update(model, model)
        shadow = averaging.create(model)
        shadow.weight = model.weight
        with self.assertRaisesRegex(ValueError, "shares training tensor"):
            averaging.update(shadow, model)
        self.assert_snapshot_equal(model, before)
        shadow = averaging.create(model)
        shadow.weight = nn.Parameter(model.weight.detach())
        with self.assertRaisesRegex(ValueError, "shares training tensor"):
            averaging.update(shadow, model)
        self.assert_snapshot_equal(model, before)

    def test_decay_endpoints_and_invalid_values(self):
        model = TinyModel()
        shadow = averaging.create(model)
        initial = tensor_snapshot(shadow)
        with torch.no_grad():
            model.weight.add_(5)
            model.norm.num_batches_tracked.fill_(8)
        averaging.update(shadow, model, decay=1.0)
        torch.testing.assert_close(shadow.weight, initial["parameter:weight"], rtol=0, atol=0)
        self.assertEqual(shadow.norm.num_batches_tracked.item(), 8)
        averaging.update(shadow, model, decay=0.0)
        self.assert_snapshot_equal(shadow, tensor_snapshot(model))
        for decay in (-0.01, 1.01, float("nan"), float("inf")):
            before = tensor_snapshot(shadow)
            with self.subTest(decay=decay), self.assertRaisesRegex(ValueError, "decay"):
                averaging.update(shadow, model, decay=decay)
            self.assert_snapshot_equal(shadow, before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_shadow_stays_on_backend_without_rng_draws(self):
        model = TinyModel().cuda()
        before = torch.cuda.get_rng_state_all()
        cpu_before = torch.get_rng_state().clone()
        shadow = averaging.create(model)
        with torch.no_grad():
            model.weight.add_(1)
        averaging.update(shadow, model)
        for value in (*shadow.parameters(), *shadow.buffers()):
            self.assertEqual(value.device, model.weight.device)
        for expected, actual in zip(before, torch.cuda.get_rng_state_all()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), cpu_before, rtol=0, atol=0)
        expected = model.weight.detach() - 0.99
        torch.testing.assert_close(shadow.weight, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
