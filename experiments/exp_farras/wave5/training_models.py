from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import support


_WAVE4_MODULE = "wave5_pinned_wave4_models"
_WAVE4_PATH = support.PREVIOUS / "models.py"
_WAVE4 = sys.modules.get(_WAVE4_MODULE)
if _WAVE4 is None:
    _WAVE4 = support.import_path(_WAVE4_MODULE, _WAVE4_PATH)

_BASE_CONFIG = dict(_WAVE4._CONFIG_BY_NAME["pp_tcn"])
CONFIGS = [
    {**_BASE_CONFIG, "name": "boundary_tcn"},
    {**_BASE_CONFIG, "name": "short_tcn"},
    {**_BASE_CONFIG, "name": "masked_tcn", "mask_full": 0.6},
]
_CONFIG_BY_NAME = {config["name"]: config for config in CONFIGS}
_SHORT_LIMIT = 20
_SHORT_WEIGHT = 2.0
_BOUNDARY_RADIUS = 2
_BOUNDARY_POSITIVE_WEIGHT = 5.0
_BOUNDARY_LOSS_WEIGHT = 0.2


def _labels(labels: Any, dimensions: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim not in dimensions:
        raise ValueError(f"labels must have {dimensions} dimensions")
    if not np.isin(values, (-1, 0, 1, 2)).all():
        raise ValueError("labels must use -1, 0, 1, or 2")
    return values


def boundary_targets(labels: Any) -> tuple[np.ndarray, np.ndarray]:
    values = _labels(labels, (1, 2))
    known = values >= 0
    changes = np.zeros(values.shape, dtype=np.bool_)
    if values.shape[-1] > 1:
        changes[..., 1:] = (
            known[..., 1:]
            & known[..., :-1]
            & (values[..., 1:] != values[..., :-1])
        )
    targets = changes.copy()
    length = values.shape[-1]
    for distance in range(1, _BOUNDARY_RADIUS + 1):
        if distance >= length:
            break
        path_known = known[..., : length - distance].copy()
        for step in range(1, distance + 1):
            path_known &= known[..., step : length - distance + step]
        targets[..., : length - distance] |= changes[..., distance:] & path_known
        targets[..., distance:] |= changes[..., : length - distance] & path_known
    valid = np.zeros(values.shape, dtype=np.bool_)
    if length > 1:
        valid[..., 1:] = known[..., 1:] & known[..., :-1]
    targets &= valid
    return targets.astype(np.float32), valid


def short_spell_weights(labels: Any, blocks: Any) -> np.ndarray:
    values = _labels(labels, (1,))
    normalized_blocks = _WAVE4._normalize_blocks(blocks, len(values))
    weights = np.ones(len(values), dtype=np.float32)
    for block_start, block_end in normalized_blocks:
        part = values[block_start:block_end]
        live = (part == 0) | (part == 1)
        if not live.any():
            continue
        starts = np.flatnonzero(live & np.r_[True, part[1:] != part[:-1]])
        ends = np.flatnonzero(live & np.r_[part[:-1] != part[1:], True]) + 1
        for run_start, run_end in zip(starts, ends):
            left_unknown = run_start > 0 and part[run_start - 1] == -1
            right_unknown = run_end < len(part) and part[run_end] == -1
            if run_end - run_start < _SHORT_LIMIT and not (left_unknown or right_unknown):
                weights[block_start + run_start : block_start + run_end] = _SHORT_WEIGHT
    return weights


def _resolve_config(config: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(config, str):
        name = config
        supplied: Mapping[str, Any] = {}
    elif isinstance(config, Mapping):
        supplied = config
        name = supplied.get("name")
    else:
        raise TypeError("config must be a model name or config mapping")
    if name not in _CONFIG_BY_NAME:
        raise ValueError(f"unknown wave5 TCN config: {name!r}")
    resolved = dict(_CONFIG_BY_NAME[name])
    resolved.update(supplied)
    resolved["name"] = "pp_tcn"
    resolved = _WAVE4._resolve_config(resolved)
    resolved["name"] = name
    return resolved


def _wave4_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result["name"] = "pp_tcn"
    return result


def _prepare_records(
    datasets: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    helper_config = _wave4_config(config)
    records = _WAVE4._prepare_records(datasets, helper_config)
    if config["name"] == "short_tcn":
        for dataset, record in zip(datasets, records):
            full_blocks = _WAVE4._normalize_blocks(dataset.get("blocks"), len(record["y"]))
            record["short_weights"] = short_spell_weights(record["y"], full_blocks)
    return records


def _make_batch(
    records: list[dict[str, Any]],
    grouped: dict[str, dict[str, list[int]]],
    mean: np.ndarray,
    scale: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    tuple[np.ndarray, np.ndarray] | None,
]:
    batch = int(config["batch"])
    crop = int(config["crop"])
    x_batch = np.zeros((batch, 29, crop), dtype=np.float32)
    pp_batch = np.empty((batch, crop, 48, 10), dtype=np.float16)
    ppm_batch = np.empty((batch, crop, 48), dtype=np.bool_)
    y_batch = np.empty((batch, crop), dtype=np.int64)
    view_batch = np.empty(batch, dtype=np.int64)
    short_weight_batch = (
        np.ones((batch, crop), dtype=np.float32)
        if config["name"] == "short_tcn"
        else None
    )
    for batch_index in range(batch):
        record, start, view = _WAVE4._sample_record(records, grouped, rng, crop)
        end = start + crop
        mask = _WAVE4._random_mask(crop, float(config["mask_full"]), rng)
        player = record["player"][view, start:end].astype(np.float32)
        player = np.nan_to_num((player - mean) / scale, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if float(config["noise"]):
            player += float(config["noise"]) * rng.standard_normal(player.shape).astype(np.float32)
        x_batch[batch_index, :23] = player.T
        x_batch[batch_index, 23] = record["valid"][view, start:end]
        ball = record["ball"][view, start:end].astype(np.float32)
        ball = np.nan_to_num(ball, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        ball[mask] = 0.0
        x_batch[batch_index, 24:29] = ball.T
        per_player = record["pp"][view, start:end].copy()
        per_player[mask, :, 6:9] = 0.0
        pp_batch[batch_index] = per_player
        ppm_batch[batch_index] = record["ppm"][view, start:end]
        y_batch[batch_index] = record["y"][start:end]
        view_batch[batch_index] = view
        if short_weight_batch is not None:
            short_weight_batch[batch_index] = record["short_weights"][start:end]
    if config["name"] == "boundary_tcn":
        boundary_target, boundary_valid = boundary_targets(y_batch)
    else:
        boundary_target = boundary_valid = None
    return (
        x_batch,
        pp_batch,
        ppm_batch,
        y_batch,
        view_batch,
        short_weight_batch,
        (boundary_target, boundary_valid) if boundary_target is not None else None,
    )


def fit(datasets: list[Mapping[str, Any]], config: Mapping[str, Any] | str) -> dict[str, Any]:
    resolved = _resolve_config(config)
    records = _prepare_records(list(datasets), resolved)
    grouped = _WAVE4._balanced_index(records)
    mean, scale = _WAVE4._player_standardisation(records)
    device = _WAVE4._device()
    nets = []
    helper_config = _wave4_config(resolved)
    for seed in resolved["seeds"]:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = _WAVE4._make_model(helper_config).to(device)
        if resolved["name"] == "boundary_tcn":
            net.head = torch.nn.Conv1d(int(resolved["ch"]), 3, 1).to(device)
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=float(resolved["lr"]),
            weight_decay=float(resolved["weight_decay"]),
        )
        scheduler = (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(resolved["lr"]),
                total_steps=int(resolved["steps"]),
            )
            if int(resolved["steps"]) > 1
            else None
        )
        positive_weight = (
            torch.tensor(_BOUNDARY_POSITIVE_WEIGHT, dtype=torch.float32, device=device)
            if resolved["name"] == "boundary_tcn"
            else None
        )
        for _ in range(int(resolved["steps"])):
            x_np, pp_np, ppm_np, y_np, view_np, short_np, boundary = _make_batch(
                records, grouped, mean, scale, resolved, rng
            )
            x = torch.from_numpy(x_np).to(device)
            pp = torch.from_numpy(pp_np).to(device=device, dtype=torch.float32)
            ppm = torch.from_numpy(ppm_np).to(device)
            y = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)
            view = torch.from_numpy(view_np).to(device=device)
            target_dead = (y == 2).to(torch.float32)
            mask_dead = (y >= 0).to(torch.float32)
            mask_team = ((y == 0) | (y == 1)).to(torch.float32)
            target_team = (y == view[:, None]).to(torch.float32)
            logits = net(x, pp, ppm)
            dead_loss = F.binary_cross_entropy_with_logits(
                logits[:, 0], target_dead, reduction="none"
            )
            team_loss = F.binary_cross_entropy_with_logits(
                logits[:, 1], target_team, reduction="none"
            )
            if short_np is None:
                state_weight = 1.0
            else:
                state_weight = torch.from_numpy(short_np).to(device=device)
            loss = (dead_loss * mask_dead * state_weight).sum() / mask_dead.sum().clamp_min(1.0)
            loss = loss + (team_loss * mask_team * state_weight).sum() / mask_team.sum().clamp_min(1.0)
            if boundary is not None:
                target_np, valid_np = boundary
                target_boundary = torch.from_numpy(target_np).to(device=device)
                mask_boundary = torch.from_numpy(valid_np).to(device=device, dtype=torch.float32)
                boundary_loss = F.binary_cross_entropy_with_logits(
                    logits[:, 2],
                    target_boundary,
                    pos_weight=positive_weight,
                    reduction="none",
                )
                loss = loss + _BOUNDARY_LOSS_WEIGHT * (
                    boundary_loss * mask_boundary
                ).sum() / mask_boundary.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        net.eval().to("cpu")
        nets.append(net)
        del optimizer, scheduler
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        by_source[record["source"]][record["name"]] += 1
    match_counts = {
        source: len(matches) for source, matches in sorted(by_source.items())
    }
    round_counts = {
        source: sum(matches.values()) for source, matches in sorted(by_source.items())
    }
    match_audit = [
        {"source": source, "match": name, "rounds": rounds}
        for source, matches in sorted(by_source.items())
        for name, rounds in sorted(matches.items())
    ]
    training = {
        "matches": sum(match_counts.values()),
        "rounds": len(records),
        "source_matches": match_counts,
        "source_rounds": round_counts,
        "match_audit": match_audit,
        "device": str(device),
    }
    return {"nets": nets, "mean": mean, "scale": scale, "config": resolved, "training": training}


@torch.no_grad()
def predict(
    fitted: Mapping[str, Any],
    test: Mapping[str, Any],
    config: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    if config is None:
        config = fitted["config"]
    resolved = _resolve_config(config)
    if resolved["name"] != fitted["config"]["name"]:
        raise ValueError("prediction config does not match fitted model")
    nets = fitted["nets"]
    if len(nets) != 3:
        raise ValueError("fitted TCN ensemble must contain three seeds")
    arrays = _WAVE4._feature_arrays(test, need_pp=True, need_control=False)
    n_frames = arrays["player"].shape[1]
    mean = np.asarray(fitted["mean"], dtype=np.float32)
    scale = np.asarray(fitted["scale"], dtype=np.float32)
    if mean.shape != (23,) or scale.shape != (23,):
        raise ValueError("fitted player standardization must contain 23 values")
    device = _WAVE4._device()
    dead = np.zeros(n_frames, dtype=np.float64)
    home = np.zeros(n_frames, dtype=np.float64)
    boundary_score = np.zeros(n_frames, dtype=np.float64)
    predicted = np.zeros(n_frames, dtype=np.bool_)
    has_boundary = resolved["name"] == "boundary_tcn"
    helper_config = _wave4_config(resolved)
    for net in nets:
        net.to(device).eval()
        try:
            for start, end in arrays["blocks"]:
                predicted[start:end] = True
                for view in (0, 1):
                    x_np, pp_np, ppm_np = _WAVE4._inference_input(
                        arrays, start, end, view, mean, scale, helper_config
                    )
                    x = torch.from_numpy(x_np[None]).to(device)
                    pp = torch.from_numpy(pp_np[None]).to(device=device, dtype=torch.float32)
                    ppm = torch.from_numpy(ppm_np[None]).to(device)
                    logits = net(x, pp, ppm)[0]
                    p_dead = torch.sigmoid(logits[0]).cpu().numpy()
                    p_view = torch.sigmoid(logits[1]).cpu().numpy()
                    weight = 1.0 / (2.0 * len(nets))
                    dead[start:end] += weight * p_dead
                    home[start:end] += weight * (p_view if view == 0 else 1.0 - p_view)
                    if has_boundary:
                        p_boundary = torch.sigmoid(logits[2]).cpu().numpy()
                        boundary_score[start:end] += weight * p_boundary
        finally:
            net.to("cpu")
    if not predicted.all():
        raise ValueError("prediction blocks did not cover every observation")
    dead = np.clip(dead, 0.0, 1.0).astype(np.float32)
    home = np.clip(home, 0.0, 1.0).astype(np.float32)
    probability = np.empty((n_frames, 3), dtype=np.float32)
    live = 1.0 - dead
    probability[:, 0] = live * home
    probability[:, 1] = live * (1.0 - home)
    probability[:, 2] = dead
    totals = probability.sum(axis=1, keepdims=True)
    np.divide(probability, totals, out=probability, where=totals > 0.0)
    p_dead = probability[:, 2].copy()
    live_total = probability[:, 0] + probability[:, 1]
    p_team = np.divide(
        probability[:, 0],
        live_total,
        out=np.full(n_frames, 0.5, dtype=np.float32),
        where=live_total > 1e-8,
    )
    if has_boundary:
        boundary = np.clip(boundary_score, 0.0, 1.0).astype(np.float32)
    else:
        boundary = np.full(n_frames, 0.5, dtype=np.float32)
    return {
        "probability": probability,
        "p_dead": p_dead,
        "p_team": p_team,
        "boundary": boundary,
        "has_boundary": has_boundary,
    }
