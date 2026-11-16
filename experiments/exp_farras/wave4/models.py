from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


_BASE_CONFIG = {
    "steps": 1500,
    "batch": 16,
    "crop": 384,
    "ch": 64,
    "dilations": (1, 2, 4, 8, 16, 1),
    "seeds": (0, 1, 2),
    "mask_full": 0.3,
    "noise": 0.5,
    "lr": 0.002,
    "weight_decay": 0.01,
    "dropout": 0.15,
}
CONFIGS = [
    {**_BASE_CONFIG, "name": "plain_tcn"},
    {**_BASE_CONFIG, "name": "pp_tcn"},
    {**_BASE_CONFIG, "name": "control_tcn"},
]
_CONFIG_BY_NAME = {config["name"]: config for config in CONFIGS}


class _RyuTCN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        dilations: tuple[int, ...],
        dropout: float,
        use_pp: bool,
    ) -> None:
        super().__init__()
        self.use_pp = use_pp
        self.inp = nn.Conv1d(input_channels, channels, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=5,
                        padding=dilation * 2,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for dilation in dilations
            ]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(8, channels) for _ in dilations])
        if use_pp:
            self.ppnet = nn.Sequential(nn.Linear(10, 48), nn.GELU(), nn.Linear(48, 48))
            self.ppatt = nn.Linear(48, 1)
            self.ppproj = nn.Conv1d(96, channels, 1)
        self.head = nn.Conv1d(channels, 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        pp: torch.Tensor | None = None,
        ppm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.inp(x)
        if self.use_pp:
            if pp is None or ppm is None:
                raise ValueError("per-player inputs required by this model")
            encoded = self.ppnet(pp)
            encoded = encoded.masked_fill(~ppm.unsqueeze(-1), 0.0)
            scores = self.ppatt(encoded).squeeze(-1).masked_fill(~ppm, -1e4)
            attention = torch.softmax(scores, dim=-1)
            pooled = (attention.unsqueeze(-1) * encoded).sum(dim=2)
            maximum = encoded.masked_fill(~ppm.unsqueeze(-1), -1e4).max(dim=2).values
            maximum = torch.where(ppm.any(dim=-1, keepdim=True), maximum, 0.0)
            h = h + self.ppproj(torch.cat((pooled, maximum), dim=-1).transpose(1, 2))
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        return self.head(h)


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
        raise ValueError(f"unknown TCN config: {name!r}")
    resolved = dict(_CONFIG_BY_NAME[name])
    resolved.update(supplied)
    for key in ("steps", "batch", "crop", "ch"):
        if int(resolved[key]) < 1:
            raise ValueError(f"{key} must be positive")
        resolved[key] = int(resolved[key])
    if resolved["ch"] % 8:
        raise ValueError("ch must be divisible by 8 for GroupNorm")
    resolved["dilations"] = tuple(int(value) for value in resolved["dilations"])
    if not resolved["dilations"] or any(value < 1 for value in resolved["dilations"]):
        raise ValueError("dilations must contain positive integers")
    resolved["seeds"] = tuple(int(seed) for seed in resolved["seeds"])
    if resolved["seeds"] != (0, 1, 2):
        raise ValueError("wave4 TCNs use seeds (0, 1, 2)")
    for key in ("mask_full", "dropout"):
        value = float(resolved[key])
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{key} must be in [0, 1)")
        resolved[key] = value
    for key in ("noise", "weight_decay"):
        value = float(resolved[key])
        if value < 0.0:
            raise ValueError(f"{key} must be nonnegative")
        resolved[key] = value
    resolved["lr"] = float(resolved["lr"])
    if resolved["lr"] <= 0.0:
        raise ValueError("lr must be positive")
    return resolved


def _device() -> torch.device:
    requested = os.environ.get("FARRAS_TRAIN_DEVICE", "").strip()
    if requested:
        device = torch.device(requested)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("FARRAS_TRAIN_DEVICE must select CPU or CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FARRAS_TRAIN_DEVICE requests CUDA, but CUDA is unavailable")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _feature_dict(data: Mapping[str, Any]) -> Mapping[str, Any]:
    cached = data.get("_features")
    if cached is not None:
        if not isinstance(cached, Mapping):
            raise TypeError("_features must be a feature mapping")
        return cached
    import features

    return features.build(data)


def _normalize_blocks(blocks: Any, length: int) -> list[tuple[int, int]]:
    if blocks is None:
        if length == 0:
            return []
        raise ValueError("dataset is missing contiguous blocks")
    normalized = sorted((int(block[0]), int(block[1])) for block in blocks)
    result: list[tuple[int, int]] = []
    previous_end = 0
    for start, end in normalized:
        if start != previous_end or end <= start or end > length:
            raise ValueError("blocks must partition observations in order")
        result.append((start, end))
        previous_end = end
    if previous_end != length:
        raise ValueError("blocks must cover every observation")
    return result


def _feature_arrays(
    data: Mapping[str, Any], need_pp: bool, need_control: bool
) -> dict[str, Any]:
    features = _feature_dict(data)
    player = np.asarray(features["player"], dtype=np.float16)
    ball = np.asarray(features["ball"], dtype=np.float16)
    valid = np.asarray(features["valid"], dtype=np.bool_)
    if player.ndim != 3 or player.shape[0] != 2 or player.shape[2] != 23:
        raise ValueError("player features must have shape (2, N, 23)")
    n_frames = player.shape[1]
    if ball.shape != (2, n_frames, 5):
        raise ValueError("ball features must have shape (2, N, 5)")
    if valid.shape != (2, n_frames):
        raise ValueError("valid mask must have shape (2, N)")
    result: dict[str, Any] = {"player": player, "ball": ball, "valid": valid}
    if need_pp:
        pp = np.asarray(features["pp"], dtype=np.float16)
        ppm = np.asarray(features["ppm"], dtype=np.bool_)
        if pp.ndim != 4 or pp.shape[:2] != (2, n_frames) or pp.shape[2:] != (48, 10):
            raise ValueError("pp features must have shape (2, N, 48, 10)")
        if ppm.shape != (2, n_frames, 48):
            raise ValueError("ppm mask must have shape (2, N, 48)")
        result["pp"] = pp
        result["ppm"] = ppm
    if need_control:
        control = np.asarray(features["control"], dtype=np.float16)
        if control.shape != (2, n_frames, 7):
            raise ValueError("control features must have shape (2, N, 7)")
        result["control"] = control
    result["blocks"] = _normalize_blocks(features.get("blocks", data.get("blocks")), n_frames)
    return result


def _prepare_records(
    datasets: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    name = config["name"]
    need_pp = name in ("pp_tcn", "control_tcn")
    need_control = name == "control_tcn"
    crop = int(config["crop"])
    records: list[dict[str, Any]] = []
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            raise TypeError("each dataset must be a mapping")
        source_value = dataset.get("_source")
        match_value = dataset.get("_name")
        if not isinstance(source_value, str) or not source_value.strip():
            raise ValueError("each training dataset needs _source")
        if match_value is None or not str(match_value).strip():
            raise ValueError("each training dataset needs _name")
        source = source_value.strip().lower()
        if source not in ("idsse", "skillcorner"):
            raise ValueError(f"unsupported training source: {source_value!r}")
        arrays = _feature_arrays(dataset, need_pp, need_control)
        labels = np.asarray(dataset["y"])
        if labels.ndim != 1 or labels.shape[0] != arrays["player"].shape[1]:
            raise ValueError("training labels must have one value per sampled frame")
        if not np.isin(labels, (-1, 0, 1, 2)).all():
            raise ValueError("training labels must use -1, 0, 1, or 2")
        usable_blocks = [
            (start, end) for start, end in arrays["blocks"] if end - start >= crop
        ]
        if not usable_blocks:
            raise ValueError(f"{source}:{match_value} has no block at least crop frames long")
        arrays["blocks"] = usable_blocks
        records.append(
            {
                **arrays,
                "y": labels,
                "source": source,
                "name": str(match_value),
            }
        )
    if not records:
        raise ValueError("fit requires at least one training dataset")
    if not any(np.isin(record["y"], (0, 1, 2)).any() for record in records):
        raise ValueError("training datasets contain no known labels")
    return records


def _player_standardisation(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    totals = np.zeros(23, dtype=np.float64)
    squared_totals = np.zeros(23, dtype=np.float64)
    counts = np.zeros(23, dtype=np.int64)
    for record in records:
        player = record["player"][0]
        valid = record["valid"][0]
        for start in range(0, player.shape[0], 65536):
            end = min(start + 65536, player.shape[0])
            values = player[start:end].astype(np.float64)
            finite = np.isfinite(values) & valid[start:end, None]
            clean = np.where(finite, values, 0.0)
            totals += clean.sum(axis=0, dtype=np.float64)
            squared_totals += (clean * clean).sum(axis=0, dtype=np.float64)
            counts += finite.sum(axis=0, dtype=np.int64)
    mean = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    variance = np.divide(squared_totals, counts, out=np.zeros_like(totals), where=counts > 0)
    variance = np.maximum(variance - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(counts > 0, np.maximum(scale, 1e-6), 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def _balanced_index(records: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        grouped[record["source"]][record["name"]].append(index)
    return {source: dict(matches) for source, matches in grouped.items()}


def _sample_record(
    records: list[dict[str, Any]],
    grouped: dict[str, dict[str, list[int]]],
    rng: np.random.Generator,
    crop: int,
) -> tuple[dict[str, Any], int, int]:
    sources = sorted(grouped)
    source = sources[int(rng.integers(len(sources)))]
    matches = grouped[source]
    names = sorted(matches)
    name = names[int(rng.integers(len(names)))]
    variants = matches[name]
    record = records[variants[int(rng.integers(len(variants)))]]
    blocks = record["blocks"]
    lengths = np.asarray([end - start for start, end in blocks], dtype=np.float64)
    block_index = int(rng.choice(len(blocks), p=lengths / lengths.sum()))
    start, end = blocks[block_index]
    window_start = int(rng.integers(start, end - crop + 1))
    view = int(rng.integers(2))
    return record, window_start, view


def _random_mask(length: int, probability: float, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < probability:
        return np.ones(length, dtype=np.bool_)
    mask = np.zeros(length, dtype=np.bool_)
    start = 0
    while start < length:
        span = int(rng.integers(8, 160))
        if rng.random() < 0.5:
            mask[start : min(start + span, length)] = True
        start += span
    return mask


def _make_batch(
    records: list[dict[str, Any]],
    grouped: dict[str, dict[str, list[int]]],
    mean: np.ndarray,
    scale: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    batch = int(config["batch"])
    crop = int(config["crop"])
    use_pp = config["name"] in ("pp_tcn", "control_tcn")
    use_control = config["name"] == "control_tcn"
    channels = 29 + (7 if use_control else 0)
    x_batch = np.zeros((batch, channels, crop), dtype=np.float32)
    pp_batch = np.empty((batch, crop, 48, 10), dtype=np.float16) if use_pp else None
    ppm_batch = np.empty((batch, crop, 48), dtype=np.bool_) if use_pp else None
    y_batch = np.empty((batch, crop), dtype=np.int64)
    view_batch = np.empty(batch, dtype=np.int64)
    for batch_index in range(batch):
        record, start, view = _sample_record(records, grouped, rng, crop)
        end = start + crop
        mask = _random_mask(crop, float(config["mask_full"]), rng)
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
        if use_control:
            control = record["control"][view, start:end].astype(np.float32)
            control = np.nan_to_num(control, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            control[mask] = 0.0
            x_batch[batch_index, 29:36] = control.T
        if use_pp:
            per_player = record["pp"][view, start:end].copy()
            per_player[mask, :, 6:9] = 0.0
            pp_batch[batch_index] = per_player
            ppm_batch[batch_index] = record["ppm"][view, start:end]
        y_batch[batch_index] = record["y"][start:end]
        view_batch[batch_index] = view
    return x_batch, pp_batch, ppm_batch, y_batch, view_batch


def _make_model(config: Mapping[str, Any]) -> _RyuTCN:
    use_pp = config["name"] in ("pp_tcn", "control_tcn")
    input_channels = 29 + (7 if config["name"] == "control_tcn" else 0)
    return _RyuTCN(
        input_channels,
        int(config["ch"]),
        tuple(config["dilations"]),
        float(config["dropout"]),
        use_pp,
    )


def fit(datasets: list[Mapping[str, Any]], config: Mapping[str, Any] | str) -> dict[str, Any]:
    resolved = _resolve_config(config)
    records = _prepare_records(list(datasets), resolved)
    grouped = _balanced_index(records)
    mean, scale = _player_standardisation(records)
    device = _device()
    nets: list[_RyuTCN] = []
    for seed in resolved["seeds"]:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = _make_model(resolved).to(device)
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
        for _ in range(int(resolved["steps"])):
            x_np, pp_np, ppm_np, y_np, view_np = _make_batch(
                records, grouped, mean, scale, resolved, rng
            )
            x = torch.from_numpy(x_np).to(device)
            pp = torch.from_numpy(pp_np).to(device=device, dtype=torch.float32) if pp_np is not None else None
            ppm = torch.from_numpy(ppm_np).to(device) if ppm_np is not None else None
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
            loss = (dead_loss * mask_dead).sum() / mask_dead.sum().clamp_min(1.0)
            loss = loss + (team_loss * mask_team).sum() / mask_team.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        net.eval().to("cpu")
        nets.append(net)
        del optimizer, scheduler
    match_counts = {
        source: len(matches) for source, matches in sorted(grouped.items())
    }
    round_counts = {
        source: sum(len(variants) for variants in matches.values())
        for source, matches in sorted(grouped.items())
    }
    training = {
        "matches": sum(match_counts.values()),
        "rounds": len(records),
        "source_matches": match_counts,
        "source_rounds": round_counts,
        "device": str(device),
    }
    return {"nets": nets, "mean": mean, "scale": scale, "config": resolved, "training": training}


def _inference_input(
    arrays: Mapping[str, Any],
    start: int,
    end: int,
    view: int,
    mean: np.ndarray,
    scale: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    length = end - start
    use_pp = config["name"] in ("pp_tcn", "control_tcn")
    use_control = config["name"] == "control_tcn"
    channels = 29 + (7 if use_control else 0)
    x = np.zeros((channels, length), dtype=np.float32)
    player = arrays["player"][view, start:end].astype(np.float32)
    player = np.nan_to_num((player - mean) / scale, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    x[:23] = player.T
    x[23] = arrays["valid"][view, start:end]
    ball = arrays["ball"][view, start:end].astype(np.float32)
    x[24:29] = np.nan_to_num(ball, copy=False, nan=0.0, posinf=0.0, neginf=0.0).T
    if use_control:
        control = arrays["control"][view, start:end].astype(np.float32)
        x[29:36] = np.nan_to_num(control, copy=False, nan=0.0, posinf=0.0, neginf=0.0).T
    pp = arrays["pp"][view, start:end].astype(np.float32) if use_pp else None
    ppm = arrays["ppm"][view, start:end] if use_pp else None
    return x, pp, ppm


@torch.no_grad()
def predict(
    fitted: Mapping[str, Any],
    test: Mapping[str, Any],
    config: Mapping[str, Any] | str | None = None,
) -> dict[str, np.ndarray]:
    if config is None:
        config = fitted["config"]
    resolved = _resolve_config(config)
    if resolved["name"] != fitted["config"]["name"]:
        raise ValueError("prediction config does not match fitted model")
    nets = fitted["nets"]
    if len(nets) != 3:
        raise ValueError("fitted TCN ensemble must contain three seeds")
    use_pp = resolved["name"] in ("pp_tcn", "control_tcn")
    use_control = resolved["name"] == "control_tcn"
    arrays = _feature_arrays(test, use_pp, use_control)
    n_frames = arrays["player"].shape[1]
    mean = np.asarray(fitted["mean"], dtype=np.float32)
    scale = np.asarray(fitted["scale"], dtype=np.float32)
    if mean.shape != (23,) or scale.shape != (23,):
        raise ValueError("fitted player standardization must contain 23 values")
    device = _device()
    dead = np.zeros(n_frames, dtype=np.float64)
    home = np.zeros(n_frames, dtype=np.float64)
    predicted = np.zeros(n_frames, dtype=np.bool_)
    for net in nets:
        net.to(device).eval()
        try:
            for start, end in arrays["blocks"]:
                predicted[start:end] = True
                for view in (0, 1):
                    x_np, pp_np, ppm_np = _inference_input(
                        arrays, start, end, view, mean, scale, resolved
                    )
                    x = torch.from_numpy(x_np[None]).to(device)
                    pp = (
                        torch.from_numpy(pp_np[None]).to(device)
                        if pp_np is not None
                        else None
                    )
                    ppm = (
                        torch.from_numpy(ppm_np[None]).to(device)
                        if ppm_np is not None
                        else None
                    )
                    logits = net(x, pp, ppm)[0]
                    p_dead = torch.sigmoid(logits[0]).cpu().numpy()
                    p_view = torch.sigmoid(logits[1]).cpu().numpy()
                    weight = 1.0 / (2.0 * len(nets))
                    dead[start:end] += weight * p_dead
                    home[start:end] += weight * (p_view if view == 0 else 1.0 - p_view)
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
    return {
        "probability": probability,
        "p_dead": p_dead,
        "p_team": p_team,
        "boundary": np.full(n_frames, 0.5, dtype=np.float32),
    }
