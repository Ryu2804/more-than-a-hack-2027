"""Synthetic tests for exp_4: bit-exact training loop, aux targets, mirror, data isolation."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import exp4_common as C  # noqa: E402,F401  (sets up exp_1 / exp_2 / exp_3 imports)
import exp1_net as N  # noqa: E402
import exp1_tokens as T  # noqa: E402
import exp2_views as V  # noqa: E402
import exp4_aux as A  # noqa: E402
import exp4_net as X  # noqa: E402
import exp4_protocol as P  # noqa: E402
from exp1_common import Match, sample_indices  # noqa: E402

SMALL = N.TrainConfig(window=40, batch=4, epochs=2, coverage=2.0, width=8, hidden=8, seed=7)


def _arrays(count=2, steps=60, tokens=6, seed=0, aux=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        n = 2 * steps
        item = {
            "tokens": rng.normal(size=(n, tokens, len(T.TOKEN_NAMES))).astype("float16"),
            "mask": rng.random((n, tokens)) < 0.8,
            "glob": rng.normal(size=(n, len(T.GLOBAL_NAMES))).astype("float16"),
            "half": np.r_[np.ones(steps), np.full(steps, 2)].astype("int8"),
            "state": rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.4, 0.15, 0.05]).astype("int8"),
            "boundary": (rng.random(n) < 0.1).astype("int8"),
        }
        item["tokens"][..., 5] = np.where(rng.random((n, tokens)) < 0.5, 1, -1)
        if aux:
            item["aux_xy"] = rng.normal(size=(n, 2)).astype("float32")
            item["aux_ok"] = rng.random(n) < 0.7
            item["aux_team"] = rng.choice([0, 1, 3], size=n).astype("int8")
        out.append(item)
    return out


def _predict(model, item):
    return N.predict(model, item, SMALL.window)


# ---------------------------------------------------------------- bit-exact training loop
def test_aux_off_reproduces_exp1_train_bit_exactly():
    arrays = _arrays()
    reference = N.train(arrays, SMALL)
    candidate = X.train(arrays, SMALL, X.AUX_OFF)
    expected = reference.state_dict()
    observed = candidate.state_dict()
    assert set(expected) < set(observed) and {k for k in observed if k.startswith("aux_head")}
    for key, value in expected.items():
        assert torch.equal(value, observed[key]), key
    for item in arrays:
        for a, b in zip(_predict(reference, item), _predict(candidate, item)):
            assert np.array_equal(a, b)


def test_aux_head_init_leaves_main_parameters_and_rng_untouched():
    cuda = torch.cuda.is_available()
    torch.manual_seed(11)
    base = N.SetSeq(8, 8)
    after_base = torch.rand(3)
    cuda_base = torch.cuda.get_rng_state() if cuda else None
    torch.manual_seed(11)
    extended = X.SetSeqAux(8, 8, head_seed=123)
    after_extended = torch.rand(3)
    for key, value in base.state_dict().items():
        assert torch.equal(value, extended.state_dict()[key]), key
    assert torch.equal(after_base, after_extended)
    if cuda:  # the GRU dropout masks come from the CUDA generator
        assert torch.equal(cuda_base, torch.cuda.get_rng_state())


def test_aux_on_trains_finite_and_changes_the_model():
    arrays = _arrays(aux=True)
    plain = X.train(arrays, SMALL, X.AUX_OFF)
    with_aux = X.train(arrays, SMALL, X.AUX_ON)
    assert any(not torch.equal(v, with_aux.state_dict()[k]) for k, v in plain.state_dict().items())
    probability, boundary = _predict(with_aux, arrays[0])
    assert np.isfinite(probability).all() and np.isfinite(boundary).all()
    assert not with_aux.training


def test_aux_training_refuses_arrays_without_targets():
    with pytest.raises(ValueError):
        X.train(_arrays(aux=False), SMALL, X.AUX_ON)


def test_aux_loss_ignores_missing_targets():
    output = torch.zeros(1, 4, 4)
    xy = torch.tensor([[[5.0, 5.0], [0.0, 0.0], [0.0, 0.0], [9.0, 9.0]]])
    ok = torch.tensor([[False, True, True, False]])
    team = torch.tensor([[3, 3, 3, 3]])
    loss = X.aux_loss(output, xy, ok, team, X.AUX_ON)
    assert float(loss) == 0.0


# ---------------------------------------------------------------- aux targets
def _match(n=400, seed=3, home_left=True):
    rng = np.random.default_rng(seed)
    half = np.r_[np.ones(n // 2), np.full(n // 2, 2)].astype("int8")
    ball = np.cumsum(rng.normal(0, 0.8, size=(n, 2)), axis=0) * 0.3 + np.array([52.0, 34.0])
    ball[50:90] = np.nan
    left = np.concatenate([rng.uniform(5, 50, (n, 5, 1)), rng.uniform(0, 68, (n, 5, 1))], axis=2)
    right = np.concatenate([rng.uniform(55, 100, (n, 5, 1)), rng.uniform(0, 68, (n, 5, 1))], axis=2)
    home, away = (left, right) if home_left else (right, left)
    xy = np.concatenate([home, away], axis=1)
    players = np.concatenate([xy, np.full((n, 10, 1), 3.0)], axis=2)
    players[120:130, 0] = np.nan
    teams = np.array([0] * 5 + [1] * 5, "int8")
    return Match("aux", players, ball, teams, half, np.arange(n), np.zeros(n, "int8"), (105.0, 68.0))


@pytest.mark.parametrize("home_left", [True, False])
def test_aux_targets_swap_under_home_away_mirror(home_left):
    match = _match(home_left=home_left)
    indices = sample_indices(match.half)
    direct = A.aux_targets(match, indices)
    mirrored = A.aux_targets(V.mirror_match(match), indices)
    expected = A.mirror_targets(direct)
    assert np.allclose(mirrored["aux_xy"], expected["aux_xy"], atol=1e-5)
    assert np.array_equal(mirrored["aux_ok"], expected["aux_ok"])
    assert np.array_equal(mirrored["aux_team"], expected["aux_team"])
    assert set(np.unique(direct["aux_team"])) == {0, 1, X.IGNORE}


@pytest.mark.parametrize("home_left", [True, False])
def test_aux_ball_target_is_normalised_like_exp1_ball_globals(home_left):
    match = _match(home_left=home_left)
    tokens = T.build_tokens(match)
    targets = A.aux_targets(match, tokens.indices)
    visible = tokens.glob[:, 0].astype(bool)
    assert visible.any() and (~visible).any()
    assert np.allclose(tokens.glob[visible, 1:3].astype("float32"), targets["aux_xy"][visible], atol=2e-3)
    assert (targets["aux_xy"][~targets["aux_ok"]] == 0).all()


def test_mirror_aux_matches_exp1_batch_mirror_of_ball_globals():
    rng = np.random.default_rng(4)
    glob = torch.from_numpy(rng.normal(size=(3, 5, len(T.GLOBAL_NAMES))).astype("float32")).to(N.DEVICE)
    tokens = torch.zeros(3, 5, 2, len(T.TOKEN_NAMES), device=N.DEVICE)
    state = torch.from_numpy(rng.choice([0, 1, 2, 3], size=(3, 5))).to(N.DEVICE)
    team = state.clone()
    flip = torch.tensor([True, False, True], device=N.DEVICE)
    _, mirrored_glob, mirrored_state = N._mirror_batch(tokens, glob, state, flip)
    xy, mirrored_team = X.mirror_aux(glob[..., 1:3].clone(), team, flip)
    assert torch.equal(mirrored_glob[..., 1:3], xy)
    assert torch.equal(mirrored_state, mirrored_team)


# ---------------------------------------------------------------- data isolation
def test_aux_targets_come_from_training_matches_only(monkeypatch, tmp_path):
    match = _match()
    grid = sample_indices(match.half)
    loaded, trained, predicted = [], [], []

    def fake_tokens(name, kind, round_index, touch):
        item = _arrays(count=1, steps=len(grid) // 2)[0]
        return {**item, "indices": grid, "half": match.half[grid]}

    def fake_clean(name):
        loaded.append(name)
        return match

    def fake_degraded(name, round_index=0):
        loaded.append(name)
        return match

    def fake_train(arrays, config, aux):
        trained.extend(arrays)
        return X.SetSeqAux(4, 4, head_seed=1).to(N.DEVICE)

    def fake_predict(model, item, window):
        predicted.append(item)
        return np.zeros((len(item["half"]), 3), "float32"), np.zeros(len(item["half"]), "float32")

    monkeypatch.setattr(A, "AUX_DIR", tmp_path)
    monkeypatch.setattr(A, "load_clean", fake_clean)
    monkeypatch.setattr(A, "load_degraded", fake_degraded)
    monkeypatch.setattr(P, "train_tokens", fake_tokens)
    monkeypatch.setattr(P, "view_tokens", lambda name, view, touch: fake_tokens(name, view, 0, touch))
    monkeypatch.setattr(X, "train", fake_train)
    monkeypatch.setattr(X, "state_nll", lambda model, arrays, window: (0.0, 1))
    monkeypatch.setattr(N, "predict", fake_predict)

    train_names, held = ["J03WN1", "J03WOH"], ["J03WMX", "J03WQQ"]
    out, _ = P._fit_job("exp4_aux", train_names, held)
    assert sorted(set(loaded)) == sorted(train_names) and not set(loaded) & set(held)
    assert len(trained) == len(train_names) * len(P.RECIPE)
    assert all(set(A.AUX_KEYS) <= set(item) for item in trained)
    assert predicted and all(not set(A.AUX_KEYS) & set(item) for item in predicted)
    assert {f"{v}__{n}_probability" for v in C.VIEWS for n in held} <= set(out)


def test_aux_grid_mismatch_is_refused(monkeypatch, tmp_path):
    match = _match()
    monkeypatch.setattr(A, "AUX_DIR", tmp_path)
    monkeypatch.setattr(A, "load_clean", lambda name: match)
    monkeypatch.setattr(A, "load_degraded", lambda name, round_index=0: match)
    tokens = {"indices": sample_indices(match.half)[:-1]}
    with pytest.raises(RuntimeError):
        A.with_aux(tokens, "J03WMX", 0)


def test_aux_sign_follows_the_observed_round_not_the_clean_match(monkeypatch, tmp_path):
    """Regression: degradation can move a half's mean home x across halfway, which flips that
    half in the observed tokens; the aux target must flip with it."""
    from dataclasses import replace

    clean = _match(home_left=True)
    observed = clean.players.copy()
    first = clean.half == 1
    observed[first, :, 0] = 105.0 - observed[first, :, 0]  # home now on the right in half 1
    degraded = replace(clean, players=observed)
    monkeypatch.setattr(A, "AUX_DIR", tmp_path)
    monkeypatch.setattr(A, "load_clean", lambda name: clean)
    monkeypatch.setattr(A, "load_degraded", lambda name, round_index=0: degraded)
    targets = A.build_aux("J03WMX", 2)
    plain = A.aux_targets(clean, targets["indices"])
    half = clean.half[targets["indices"]]
    assert np.allclose(targets["aux_xy"][half == 1], -plain["aux_xy"][half == 1])
    assert np.allclose(targets["aux_xy"][half == 2], plain["aux_xy"][half == 2])
    assert int(targets["steps_sign_differs_from_clean"]) == int((half == 1).sum())
    tokens = T.build_tokens(replace(degraded, ball=clean.ball))
    visible = tokens.glob[:, 0].astype(bool)
    assert np.allclose(tokens.glob[visible, 1:3].astype("float32"), targets["aux_xy"][visible], atol=2e-3)


def test_fit_false_raises_on_missing_prediction(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "PRED_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="not fitted"):
        P.ensure_predictions("exp4_full", fit=False)
    assert not any(tmp_path.iterdir())


def test_job_lock_is_exclusive_and_stale_locks_are_taken_over(tmp_path):
    lock = tmp_path / "job.lock"
    assert P._claim(lock)
    assert not P._claim(lock)  # held by this live process
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock.write_text(str(dead.pid))
    assert P._claim(lock) and lock.read_text() == str(os.getpid())


def test_arm_table_is_three_families_by_three_seeds():
    assert len(P.ARMS) == 9 and P.FIT_ORDER[:3] == ("exp4_full", "exp4_full_s1", "exp4_full_s2")
    assert P.ARMS["exp4_touch"].aux == X.AUX_OFF and P.ARMS["exp4_aux"].touch is False
    assert {spec.seed for spec in P.ARMS.values()} == {20261108, 20261109, 20261110}
    assert P.ENSEMBLES["ref_ens3"] == ("exp2_aug", "exp2_aug_s1", "exp2_aug_s2")
    assert all(spec.config == N.TrainConfig(seed=spec.seed) for spec in P.ARMS.values())
