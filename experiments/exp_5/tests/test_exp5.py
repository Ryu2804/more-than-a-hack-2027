"""Synthetic tests for exp_5: arm table, isolation from exp_0-exp_4, evaluation never fits,
the two-clause decision rule, the ctrl-file gate and fingerprints."""

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import exp5_common as C  # noqa: E402,I001  (first: env + exp_1-exp_4 imports, before torch)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import exp1_net as N  # noqa: E402
import exp1_protocol as P1  # noqa: E402
import exp1_tokens as T  # noqa: E402
import exp3_touch as S  # noqa: E402
import exp4_net as X  # noqa: E402
import exp4_protocol as P4  # noqa: E402
import exp5_eval as E  # noqa: E402
import exp5_protocol as P  # noqa: E402
import exp5_report as R  # noqa: E402

SMALL = {"window": 40, "batch": 4, "epochs": 1, "coverage": 1.0, "width": 8, "hidden": 8}
JOBS = [("pair_J03WMX_J03WN1", ["J03WOH"], ["J03WMX", "J03WN1"]), ("outer_J03WMX", ["J03WN1", "J03WOH"], ["J03WMX"])]


def _item(seed: int, steps: int = 60, tokens: int = 6) -> dict:
    rng = np.random.default_rng(seed)
    n = 2 * steps
    item = {
        "tokens": rng.normal(size=(n, tokens, len(T.TOKEN_NAMES))).astype("float16"),
        "mask": rng.random((n, tokens)) < 0.8,
        "glob": rng.normal(size=(n, len(T.GLOBAL_NAMES) + len(S.TOUCH_NAMES))).astype("float16"),
        "half": np.r_[np.ones(steps), np.full(steps, 2)].astype("int8"),
        "state": rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.4, 0.15, 0.05]).astype("int8"),
        "boundary": (rng.random(n) < 0.1).astype("int8"),
    }
    item["tokens"][..., 5] = np.where(rng.random((n, tokens)) < 0.5, 1, -1)
    return item


def _small(monkeypatch) -> None:
    """Tiny config, two nested jobs and synthetic touch arrays instead of the IDSSE caches."""
    monkeypatch.setattr(P.ArmSpec, "config", property(lambda self: N.TrainConfig(seed=self.seed, **SMALL)))
    monkeypatch.setattr(P1, "job_keys", lambda: JOBS)
    monkeypatch.setattr(P4, "train_tokens", lambda name, kind, r, touch: _item(hash((name, kind, r)) % 1000))
    monkeypatch.setattr(P4, "view_tokens", lambda name, view, touch: _item(hash((name, view)) % 1000 + 1))


# ---------------------------------------------------------------- arm table
def test_arm_table_is_exp4_touch_with_fresh_seeds():
    assert P.FIT_ORDER == ("exp5_touch", "exp5_touch_s1", "exp5_touch_s2")
    assert [spec.seed for spec in P.ARMS.values()] == [20261111, 20261112, 20261113]
    reference = P4.ARMS["exp4_touch"].as_dict()
    for spec in P.ARMS.values():
        assert spec.touch is True and spec.aux == X.AUX_OFF and not spec.aux.enabled
        assert spec.config == N.TrainConfig(seed=spec.seed)
        assert spec.as_dict() == {**reference, "seed": spec.seed, "config": {**reference["config"], "seed": spec.seed}}
    assert not set(P.SEEDS) & {spec.seed for spec in P4.ARMS.values()}
    assert P.RECIPE == P4.RECIPE
    assert P.ArmSpec(seed=P.REPRO["seed"]).as_dict() == P4.ARMS[P.REPRO["arm"]].as_dict()


def test_ensembles_variants_and_never_adopted():
    assert E.ENSEMBLES["exp5_touch_ens3"] == tuple(P.ARMS)
    assert E.ENSEMBLES["exp4_touch_ens3"] == C.EXP4_TOUCH_ARMS == P4.ENSEMBLES["exp4_touch_ens3"]
    assert E.ENSEMBLES["exp45_touch_ens6"] == C.EXP4_TOUCH_ARMS + tuple(P.ARMS)
    assert E.ENSEMBLES["ref_ens3"] == P4.ENSEMBLES["ref_ens3"]
    assert set(E.NEVER_ADOPTED) == {"exp45_touch_ens6", "exp4_touch_ens3", *P.ARMS} and C.PRIMARY not in E.NEVER_ADOPTED
    assert len(E.tasks()) == 2 * (4 + 4 + 3 * 2)


def test_exp4_folders_are_found_and_other_arms_refused():
    for arm in C.EXP4_TOUCH_ARMS:
        assert C.exp4_touch_folder(arm).name.startswith(f"{arm}_")
    with pytest.raises(ValueError):
        C.exp4_touch_folder("exp4_full")


# ---------------------------------------------------------------- isolation
def test_tiny_fit_and_evaluation_write_only_exp5_paths(monkeypatch, tmp_path):
    before = {name: C.tree_digest(path) for name, path in C.watched_folders().items()}
    monkeypatch.setattr(C, "PRED_DIR", tmp_path / "predictions")
    monkeypatch.setattr(C, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(E, "_WORKER_CACHE", {})
    _small(monkeypatch)

    def stub_evaluate(predictions, variant, view, label_set):
        p = predictions["outer_J03WMX"][view]["J03WMX"][0]
        return {"rows": [{"match": "J03WMX", "variant": variant, "spell_f1": float(p.mean())}],
                "selections": {}, "final_config": {}}

    monkeypatch.setattr(E.P3, "evaluate_predictions", stub_evaluate)
    log = []
    predictions = P.ensure_predictions("exp5_touch", log=log.append)
    assert len(log) == 2 and P.proxy_fit_count() == 2
    assert P.prediction_folder("exp5_touch").parent == tmp_path / "predictions"
    assert set(predictions["pair_J03WMX_J03WN1"]["target_like"]) == {"J03WMX", "J03WN1"}
    assert E.evaluate_and_save(("exp5_touch", "target_like", "provider"))[-1] == "computed"
    assert (tmp_path / "results" / "variant_exp5_touch__target_like__provider.json").exists()
    assert E.evaluate_and_save(("exp5_touch", "target_like", "provider"))[-1] == "cached"
    rows = P.train_nll_rows("exp5_touch")
    assert len(rows) == 2 and all(np.isfinite(r["train_nll"]) for r in rows)
    after = {name: C.tree_digest(path) for name, path in C.watched_folders().items()}
    assert after == before, "exp_5 wrote under exp_0-exp_4 or data/"


def test_fit_path_is_exp4_train_with_touch_globals(monkeypatch):
    _small(monkeypatch)
    spec = P.ARMS["exp5_touch"]
    model, arrays = P.fit_spec(spec, ["J03WN1"])
    assert len(arrays) == len(P.RECIPE)
    with P4.globals_context(True):
        direct = X.train(arrays, spec.config, X.AUX_OFF)
    for key, value in direct.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    assert N.GLOBAL_NAMES == tuple(T.GLOBAL_NAMES)  # context restored


def test_fit_false_raises_and_creates_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "PRED_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="not fitted"):
        P.ensure_predictions("exp5_touch", fit=False)
    with pytest.raises(RuntimeError, match="not fitted"):
        E.predictions_for("exp5_touch_ens3", {})
    assert not any(tmp_path.iterdir())


def test_freeze_refuses_once_fits_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "PRED_DIR", tmp_path / "predictions")
    monkeypatch.setattr(C, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(C, "PROTOCOL", tmp_path / "results" / "protocol.json")
    (tmp_path / "predictions" / "exp5_touch_x").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="precede every fit"):
        C.freeze_prereg()
    with pytest.raises(RuntimeError, match="no longer allowed"):
        C.refreeze_prereg()


# ---------------------------------------------------------------- decision rule
@pytest.mark.parametrize(("mean", "low", "spread", "ci", "seed", "passed"), [
    (0.008, 0.002, 0.011, True, False, False),  # CI lower bound > 0 but Δ ≤ spread
    (0.011, 0.002, 0.011, True, False, False),  # Δ equal to the spread does not exceed it
    (0.020, 0.000, 0.011, False, True, False),  # Δ > spread but CI lower bound ≤ 0
    (0.020, -0.004, 0.011, False, True, False),
    (0.015, 0.004, 0.011, True, True, True),  # both clauses
])
def test_h1_needs_both_clauses(mean, low, spread, ci, seed, passed):
    rule = R.h1_rule(mean, low, spread)
    assert (rule["ci_clause"]["passed"], rule["seed_clause"]["passed"], rule["passed"]) == (ci, seed, passed)


def test_h1_fails_without_a_seed_spread():
    assert R.h1_rule(0.05, 0.02, None)["passed"] is False


def test_ctrl_file_is_gated_on_non_inferiority():
    failed = R.adoption_plan(False, 0.05)
    assert failed["write_files"] is False and failed["files"] == {}
    assert set(R.adoption_plan(True, -0.0099)["files"]) == {"provider", "ctrl"}
    assert set(R.adoption_plan(True, -0.01)["files"]) == {"provider"}  # strictly greater than -0.01
    assert set(R.adoption_plan(True, -0.03)["files"]) == {"provider"}


def _rows(delta: float, seed_offsets: tuple[float, float, float]) -> pd.DataFrame:
    """Synthetic per-match rows: primary = reference + delta; single seeds around the reference."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.6, 0.7, size=len(C.MATCH_IDS))
    variants = {C.REFERENCE: base, C.PRIMARY: base + delta, "exp4_touch_ens3": base + 0.01,
                "exp45_touch_ens6": base + 0.012, **{arm: base + o for arm, o in zip(P.ARMS, seed_offsets)}}
    out = []
    for label_set in C.S1_LABELS:
        for view in C.VIEWS:
            for variant, values in variants.items():
                if variant in P.ARMS and label_set not in C.DECISION_LABELS:
                    continue
                out += [{"match": m, "variant": variant, "view": view, "label_set": label_set, "spell_f1": v,
                         "macro_f1": 0.5, "count_error_ratio": 0.0} for m, v in zip(C.MATCH_IDS, values)]
    return pd.DataFrame(out)


def test_decide_reads_the_primary_cell_and_the_exp5_seed_spread(monkeypatch):
    monkeypatch.setattr(R, "final_config", lambda variant, view, label_set: {"label_set": label_set})
    noisy = R.decide(_rows(0.015, (-0.01, 0.0, 0.01)))  # spread 0.02 > Δ 0.015
    assert noisy["H1"]["ci_clause"]["passed"] and not noisy["H1"]["seed_clause"]["passed"]
    assert noisy["H1"]["passed"] is False and noisy["adoption"]["write_files"] is False
    assert noisy["H1"]["seed_clause"]["seed_spread"] == pytest.approx(0.02)
    clean = R.decide(_rows(0.015, (-0.002, 0.0, 0.002)))
    assert clean["H1"]["passed"] is True
    assert set(clean["adoption"]["files"]) == {"provider", "ctrl"}  # ctrl Δ = +0.015 here
    assert clean["adoption"]["configs"]["provider"] == {"label_set": "provider"}
    assert len(clean["report_only"]["replication"]) == 8


# ---------------------------------------------------------------- fingerprints
def test_fingerprint_changes_when_any_fit_module_changes(monkeypatch, tmp_path):
    assert {"exp4_net.py", "exp4_aux.py", "exp4_protocol.py", "exp5_protocol.py"} <= set(P.FIT_CODE)
    real = P.fingerprint("exp5_touch")
    for name, folder in P.FIT_CODE.items():
        shutil.copy(folder / name, tmp_path / name)
    monkeypatch.setattr(P, "FIT_CODE", {name: tmp_path for name in P.FIT_CODE})
    assert P.fingerprint("exp5_touch") == real
    for name in P.FIT_CODE:
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n# changed\n")
        assert P.fingerprint("exp5_touch") != real, name
        path.write_bytes(original)
    assert P.fingerprint("exp5_touch") == real
    assert len({P.fingerprint(arm) for arm in P.ARMS}) == 3
