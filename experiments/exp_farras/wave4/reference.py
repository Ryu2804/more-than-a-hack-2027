from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import types
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

WAVE4 = Path(__file__).resolve().parent
EXPERIMENTS = WAVE4.parent
WAVE3 = EXPERIMENTS / "wave3"
REPO = WAVE4.parents[2]
WEIGHTS = WAVE4 / "cache" / "ryu_weights"
WAVE3_DATA = WAVE3 / "cache" / "data"
SOURCE_COMMIT = "5eaf3e595e91fa0502aac48037d0cae8c356ec6f"
_ALLOWED = {"target", "metrica3_clean", "metrica3_r0"}
_MISSING = object()
_IMPORT_LOCK = threading.RLock()
_RUNTIME: dict[str, Any] | None = None
_MODEL_CACHE: dict[str, tuple[Any, ...]] = {}


def _load_source_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load source module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_runtime() -> dict[str, Any]:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _IMPORT_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        feature_path = WAVE4 / "reference" / "features.py"
        model_path = WAVE4 / "reference" / "train_model.py"
        aliases = {key: sys.modules.get(key, _MISSING) for key in ("features", "other_games")}
        old_path = sys.path[:]
        env_names = ("USE_PP", "TOUCH_FEATS", "CHANGE_HEAD")
        old_env = {key: os.environ.get(key) for key in env_names}
        model_name = f"_wave4_ryu_train_{uuid.uuid4().hex}"
        dummy = types.ModuleType("other_games")

        def reject_game_loading(*args, **kwargs):
            raise RuntimeError("The shipped inference adapter must not load training games")

        dummy.load = reject_game_loading
        try:
            os.environ["USE_PP"] = "1"
            os.environ["TOUCH_FEATS"] = "0"
            os.environ["CHANGE_HEAD"] = "0"
            ft = _load_source_module(feature_path, "features")
            sys.modules["other_games"] = dummy
            tm = _load_source_module(model_path, model_name)
        finally:
            for key, value in aliases.items():
                if value is _MISSING:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
            sys.modules.pop(model_name, None)
            sys.path[:] = old_path
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        if not ft.USE_PP or ft.TOUCH_FEATS or tm.CHANGE_HEAD:
            raise RuntimeError("Pinned Ryu inference configuration differs from shipped weights")
        if tm.NCH != len(ft.PLAYER_FEATS) + 1 + len(ft.BALL_FEATS):
            raise RuntimeError("Pinned Ryu input channel count is inconsistent")

        with np.load(WEIGHTS / "meta.npz", allow_pickle=False) as metadata:
            if not bool(metadata["use_pp"]):
                raise ValueError("Shipped Ryu weights do not enable player-set inputs")
            mu = np.asarray(metadata["mu"], dtype=np.float32).copy()
            sd = np.asarray(metadata["sd"], dtype=np.float32).copy()
            dil = tuple(int(value) for value in metadata["dil"])
            crop = int(metadata["crop"])
        if mu.shape != (len(ft.PLAYER_FEATS),) or sd.shape != mu.shape:
            raise ValueError("Shipped Ryu normalization metadata has unexpected dimensions")
        if not np.isfinite(mu).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
            raise ValueError("Shipped Ryu normalization metadata is invalid")
        if crop != tm.CROP or not dil:
            raise ValueError("Shipped Ryu architecture metadata differs from pinned source")

        report = json.loads((WEIGHTS / "training_report.json").read_text(encoding="utf-8"))
        stats = _balanced_training_stats(report)
        _RUNTIME = {"ft": ft, "tm": tm, "mu": mu, "sd": sd, "dil": dil,
                    "crop": crop, "stats": stats}
        return _RUNTIME


def _balanced_training_stats(report: dict[str, Any]) -> dict[str, Any]:
    providers: dict[str, dict[str, list[float]]] = {}
    rate_rows = report.get("poss_per_min")
    dead_rows = report.get("dead_share")
    if not isinstance(rate_rows, dict) or not isinstance(dead_rows, dict) or set(rate_rows) != set(dead_rows):
        raise ValueError("Shipped Ryu training report lacks paired provider statistics")
    for name, rate in rate_rows.items():
        if not isinstance(name, str) or "_" not in name:
            raise ValueError("Shipped Ryu training report has an unrecognized game name")
        provider = name.split("_", 1)[0]
        dead = float(dead_rows[name])
        rate = float(rate)
        if not np.isfinite(rate) or rate < 0 or not np.isfinite(dead) or not 0 <= dead <= 1:
            raise ValueError("Shipped Ryu training report contains invalid statistics")
        item = providers.setdefault(provider, {"rate": [], "dead_share": []})
        item["rate"].append(rate)
        item["dead_share"].append(dead)
    if set(providers) != {"idsse", "skillcorner"}:
        raise ValueError("Shipped Ryu training report does not contain expected providers")
    per_provider = {
        provider: {
            "matches": len(values["rate"]),
            "rate": float(np.mean(values["rate"])),
            "dead_share": float(np.mean(values["dead_share"])),
        }
        for provider, values in sorted(providers.items())
    }
    return {
        "rate": float(np.mean([value["rate"] for value in per_provider.values()])),
        "dead_share": float(np.mean([value["dead_share"] for value in per_provider.values()])),
        "balance": "equal provider means; provider is training-report name prefix before underscore",
        "providers": per_provider,
    }


def _load_models(device, runtime: dict[str, Any]):
    torch = __import__("torch")
    target_device = torch.device(device)
    key = str(target_device)
    with _IMPORT_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached, target_device

        models = []
        channels = None
        for seed in (0, 1, 2):
            path = WEIGHTS / f"tcn_seed{seed}.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or "inp.weight" not in state:
                raise ValueError(f"Shipped Ryu seed {seed} has invalid state dictionary")
            seed_channels = int(state["inp.weight"].shape[0])
            if channels is None:
                channels = seed_channels
                if channels != runtime["tm"].CH:
                    raise ValueError("Shipped Ryu model width differs from pinned source")
            elif seed_channels != channels:
                raise ValueError("Shipped Ryu seeds disagree on model width")
            if channels % 8:
                raise ValueError("Shipped Ryu width is incompatible with GroupNorm")
            net = runtime["tm"].TCN2(runtime["tm"].NCH, ch=channels, dil=runtime["dil"])
            net.load_state_dict(state, strict=True)
            net.to(target_device).eval()
            models.append(net)
        result = tuple(models)
        _MODEL_CACHE[key] = result
        return result, target_device


def _load_wave3_helpers():
    old_bytecode = sys.dont_write_bytecode
    module_name = f"_wave4_prepare_local_{uuid.uuid4().hex}"
    try:
        sys.dont_write_bytecode = True
        helper = _load_source_module(WAVE3 / "prepare_local.py", module_name)
    finally:
        sys.dont_write_bytecode = old_bytecode
    for child in (getattr(helper, "feature_data", None), getattr(helper, "reference", None)):
        if child is not None:
            sys.modules.pop(child.__name__, None)
    sys.modules.pop(module_name, None)
    return helper


def _load_pack(name: str) -> dict[str, np.ndarray]:
    path = WAVE3_DATA / f"{name}.npz"
    required = ("teams", "sampled", "blocks", "blocks25", "players", "ball", "n_frames", "fps")
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Wave3 pack {name} lacks fields: {', '.join(missing)}")
        return {key: np.asarray(archive[key]).copy() for key in required}


def _compare_pack_array(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True):
        raise ValueError(f"Wave3 {name} pack does not align with full-match source")


def _check_pack_alignment(pack: dict[str, np.ndarray], players: np.ndarray, ball: np.ndarray,
                          teams: np.ndarray, half: np.ndarray, helper) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pack["fps"].ndim != 0 or int(pack["fps"]) != 5:
        raise ValueError("Wave3 pack is not on the 5 fps grid")
    if pack["n_frames"].ndim != 0 or int(pack["n_frames"]) != len(half):
        raise ValueError("Wave3 pack frame count differs from full-match source")
    if not np.array_equal(pack["teams"], teams):
        raise ValueError("Wave3 team vector differs from full-match source")
    sampled, blocks, blocks25 = helper._sample_indices(half)
    if not (np.array_equal(pack["sampled"], sampled)
            and np.array_equal(pack["blocks"], blocks)
            and np.array_equal(pack["blocks25"], blocks25)):
        raise ValueError("Wave3 sampled grid differs from full-match source")
    if pack["players"].shape != (len(sampled), players.shape[1], 2):
        raise ValueError("Wave3 player samples have unexpected shape")
    if pack["ball"].shape != (len(sampled), 2):
        raise ValueError("Wave3 ball samples have unexpected shape")
    _compare_pack_array(pack["players"], players[sampled], "player")
    _compare_pack_array(pack["ball"], ball[sampled], "ball")
    return sampled, blocks, blocks25


def _load_target(pack: dict[str, np.ndarray], helper) -> dict[str, Any]:
    cache = helper.feature_data.RYU / "_cache"
    flat = np.load(cache / "atd_full.npy", mmap_mode="r", allow_pickle=False)
    if flat.ndim != 2 or flat.shape[1] != 141:
        raise ValueError(f"Cached target ATD has unexpected shape: {flat.shape}")
    teams = np.asarray(pack["teams"], dtype=np.int8)
    if teams.shape != (46,) or np.any(~np.isin(teams, (-1, 0, 1))):
        raise ValueError("Wave3 target pack has invalid team-slot assignments")
    unknown = np.flatnonzero(teams == -1)
    if not np.array_equal(unknown, np.asarray((21, 22, 43))) or [int(np.count_nonzero(teams == team)) for team in (0, 1)] != [22, 21]:
        raise ValueError("Wave3 target pack does not carry the expected unknown team slots")

    n_frames = len(flat)
    players = np.asarray(flat[:, :-3].reshape(n_frames, 46, 3)[:, :, :2], dtype=np.float32).copy()
    players[:, unknown, :] = np.nan
    ball = np.asarray(flat[:, -3:-1], dtype=np.float32).copy()
    del flat

    games = _load_source_module(REPO / "games.py", f"_wave4_games_{uuid.uuid4().hex}")
    half = np.zeros(n_frames, dtype=np.int8)
    target_blocks = [(int(period), int(start), int(stop) + 1)
                     for period, start, stop in games.half_video_blocks("BAR-ATH")]
    for period, start, stop in target_blocks:
        if start < 0 or stop > n_frames or np.any(half[start:stop]):
            sys.modules.pop(games.__name__, None)
            raise ValueError("Pinned target half boundaries overlap or exceed cached ATD")
        half[start:stop] = period
    sys.modules.pop(games.__name__, None)

    sampled, blocks, blocks25 = _check_pack_alignment(pack, players, ball, teams, half, helper)
    return {"players": players, "ball": ball, "teams": teams, "half": half,
            "sampled": sampled, "blocks": blocks, "blocks25": blocks25,
            "unknown_slots": unknown.tolist(), "source": "cached Ryu ATD; W3 target_clean pack team vector"}


def _load_metrica3(name: str, pack: dict[str, np.ndarray], helper) -> dict[str, Any]:
    match, _ = helper._load_metrica_game3()
    match = replace(match, labels=None)
    if name == "metrica3_r0":
        match = helper.reference.degrade(match, seed=20262111)
    teams = np.asarray(pack["teams"], dtype=np.int8)
    if np.any(~np.isin(teams, (0, 1))) or teams.shape != (match.players.shape[1],):
        raise ValueError("Wave3 Metrica3 pack has invalid team assignments")
    if not np.array_equal(teams, match.teams):
        raise ValueError("Wave3 Metrica3 team vector differs from corrected full-match source")
    pitch = np.asarray(match.pitch, dtype=np.float32)
    players = np.asarray(match.players[:, :, :2], dtype=np.float32) / pitch
    ball = np.asarray(match.ball, dtype=np.float32) / pitch
    half = np.asarray(match.half, dtype=np.int8).copy()
    sampled, blocks, blocks25 = _check_pack_alignment(pack, players, ball, teams, half, helper)
    source = "wave3 corrected Metrica3 full-match loader"
    if name == "metrica3_r0":
        source += "; wave3 reference.degrade(seed=20262111)"
    return {"players": players, "ball": ball, "teams": teams, "half": half,
            "sampled": sampled, "blocks": blocks, "blocks25": blocks25,
            "unknown_slots": [], "source": source}


def _load_match(name: str) -> dict[str, Any]:
    helper = _load_wave3_helpers()
    pack = _load_pack("target_clean" if name == "target" else name)
    if name == "target":
        return _load_target(pack, helper)
    return _load_metrica3(name, pack, helper)


def predict(name: str, device: str = "cpu") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Predict supported target and Metrica3 inputs with the shipped three-seed Ryu ensemble."""
    if not isinstance(name, str) or name not in _ALLOWED:
        raise ValueError("Supported inputs are target, metrica3_clean, and metrica3_r0; IDSSE and SkillCorner are rejected")

    runtime = _load_runtime()
    models, target_device = _load_models(device, runtime)
    match = _load_match(name)
    ft, tm = runtime["ft"], runtime["tm"]
    features = ft.build(match["players"], match["ball"], match["teams"], match["half"])
    pp, ppm = ft.player_set(match["players"], match["ball"], match["teams"], match["half"], match["sampled"])
    inputs = tm.make_inputs(runtime["mu"], runtime["sd"])
    sampled = match["sampled"]
    blocks = match["blocks"]
    blocks25 = match["blocks25"]
    full_frames = len(match["half"])
    unknown_slots = match["unknown_slots"]
    source = match["source"]
    player_features = features["player"]
    ball_features = features["ball"]
    valid_features = features["pvalid"]
    del features, match

    p_dead = np.empty(len(sampled), dtype=np.float32)
    p_team = np.empty_like(p_dead)
    for start, stop in blocks:
        start, stop = int(start), int(stop)
        if stop <= start:
            raise ValueError("Wave3 pack contains an empty half block")
        ix = sampled[start:stop]
        player = player_features[:, ix]
        ball = ball_features[:, ix]
        valid = valid_features[:, ix]
        dead_sum = np.zeros(stop - start, dtype=np.float32)
        team_sum = np.zeros_like(dead_sum)
        for net in models:
            dead, team = tm.predict(
                net, inputs, player, ball, valid, False, target_device,
                pp=pp[:, start:stop], ppm=ppm[:, start:stop],
            )
            dead_sum += np.asarray(dead, dtype=np.float32)
            team_sum += np.asarray(team, dtype=np.float32)
        p_dead[start:stop] = dead_sum / len(models)
        p_team[start:stop] = team_sum / len(models)

    live = 1.0 - p_dead
    probability = np.column_stack((live * p_team, live * (1.0 - p_team), p_dead)).astype(np.float32)
    prediction = {
        "probability": probability,
        "p_dead": p_dead,
        "p_team": p_team,
        "boundary": np.full(len(p_dead), 0.5, dtype=np.float32),
    }
    training_stats = runtime["stats"]
    audit = {
        "input": name,
        "source_commit": SOURCE_COMMIT,
        "weights": [f"cache/ryu_weights/tcn_seed{seed}.pt" for seed in (0, 1, 2)],
        "architecture": {
            "model": "pinned Ryu TCN2",
            "channels": int(models[0].inp.out_channels),
            "dilations": list(runtime["dil"]),
            "player_set": True,
            "crop": runtime["crop"],
        },
        "training_stats": training_stats,
        "source": source,
        "full_frames": full_frames,
        "sampled_frames": len(sampled),
        "blocks": len(blocks),
        "blocks25": blocks25.tolist(),
        "unknown_team_slots": unknown_slots,
        "network_labels_used": False,
        "boundary_probability": "fixed 0.5; shipped Ryu has no boundary head",
        "adaptations": [
            "Target positions come from cached Ryu ATD; team assignments come from the W3 target_clean pack, and unknown slots are NaN across the full match.",
            "Pinned Ryu feature and player-set code runs at 25 fps; predictions use the exact W3 5 fps sample grid, which differs from the original 25 fps decoder input.",
            "The shipped weights were trained on IDSSE and SkillCorner; adapter rejects those provider names.",
        ],
        "bit_exact_ryu_csv_claim": False,
    }
    return prediction, audit
