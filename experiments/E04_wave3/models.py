from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    "window": 300,
    "batch": 16,
    "epochs": 8,
    "coverage": 1.0,
    "seed": 2027,
    "lr": 0.002,
    "weight_decay": 0.0001,
    "width": 32,
    "hidden": 32,
    "boundary_weight": 0.5,
    "boundary_pos_weight": 3.0,
}
CONFIGS = [
    {**_BASE_CONFIG, "name": "setseq", "global_features": 12, "mirror": True},
    {**_BASE_CONFIG, "name": "hybrid", "global_features": 19, "mirror": True},
    {**_BASE_CONFIG, "name": "feature_gru", "global_features": 0, "mirror": False},
]
_CONFIG_BY_NAME = {config["name"]: config for config in CONFIGS}
_IGNORE = -1


class _SetSequenceModel(nn.Module):
    def __init__(self, global_features: int, width: int, hidden: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(10, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.attention = nn.Linear(width, 1)
        self.frame = nn.Sequential(
            nn.Linear(5 * width + global_features, 2 * width),
            nn.GELU(),
            nn.LayerNorm(2 * width),
        )
        self.gru = nn.GRU(
            input_size=2 * width,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.state_head = nn.Linear(2 * hidden, 3)
        self.boundary_head = nn.Linear(2 * hidden, 1)

    @staticmethod
    def _pool(encoded: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = weight.sum(dim=2)
        mean = (encoded * weight).sum(dim=2) / count.clamp_min(1.0)
        peak = torch.where(weight > 0, encoded, torch.full_like(encoded, -1e4)).amax(dim=2)
        peak = torch.where(count > 0, peak, torch.zeros_like(peak))
        return mean, peak

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, glob: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.phi(tokens)
        present = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        team = tokens[..., 5:6]
        home = present * (team > 0).to(dtype=encoded.dtype)
        away = present * (team < 0).to(dtype=encoded.dtype)
        score = self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e4)
        attention = torch.softmax(score, dim=2).unsqueeze(-1) * present
        pooled = [*self._pool(encoded, home), *self._pool(encoded, away)]
        pooled.append((attention * encoded).sum(dim=2))
        frame = self.frame(torch.cat([*pooled, glob], dim=-1))
        output, _ = self.gru(frame)
        return self.state_head(output), self.boundary_head(output).squeeze(-1)


class _FeatureGRU(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=16,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.state_head = nn.Linear(2 * hidden, 3)
        self.boundary_head = nn.Linear(2 * hidden, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.gru(features)
        return self.state_head(output), self.boundary_head(output).squeeze(-1)


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
        raise ValueError(f"unknown sequence model config: {name!r}")
    resolved = dict(_CONFIG_BY_NAME[name])
    resolved.update(supplied)
    return resolved


def _normalize_blocks(blocks: Any, length: int) -> list[tuple[int, int]]:
    if length == 0 and blocks is None:
        return []
    if blocks is None:
        raise ValueError("dataset blocks are required")
    values = np.asarray(blocks)
    if values.size == 0:
        if length == 0:
            return []
        raise ValueError("dataset blocks do not cover observations")
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("blocks must contain [start, end) pairs")
    result: list[tuple[int, int]] = []
    cursor = 0
    for raw_start, raw_end in values:
        start, end = int(raw_start), int(raw_end)
        if start != raw_start or end != raw_end or start != cursor or end <= start or end > length:
            raise ValueError("blocks must partition observations in order")
        result.append((start, end))
        cursor = end
    if cursor != length:
        raise ValueError("dataset blocks do not cover observations")
    return result


def _token_api():
    import tokens

    return tokens


def _encoded_tokens(data: Mapping[str, Any], token_api: Any) -> Mapping[str, Any]:
    cached = data.get("_tokens")
    if isinstance(cached, Mapping) and all(
        key in cached for key in ("tokens", "mask", "glob", "blocks")
    ):
        return cached
    return token_api.build(data)


def _boundary_targets(state: np.ndarray, blocks: list[tuple[int, int]]) -> np.ndarray:
    boundary = np.zeros(len(state), dtype=np.float32)
    for start, end in blocks:
        current = state[start:end]
        changed = np.zeros(len(current), dtype=bool)
        if len(current) > 1:
            known = (current[1:] >= 0) & (current[:-1] >= 0)
            changed[1:] = known & (current[1:] != current[:-1])
        dilated = changed.copy()
        if len(current) > 1:
            dilated[:-1] |= changed[1:]
            dilated[1:] |= changed[:-1]
        boundary[start:end] = dilated
    return boundary


def _valid_starts(state: np.ndarray, start: int, end: int, window: int) -> np.ndarray:
    labels = state[start:end]
    if len(labels) < window:
        return np.empty(0, dtype=np.int64)
    known_prefix = np.empty(len(labels) + 1, dtype=np.int64)
    known_prefix[0] = 0
    np.cumsum(labels >= 0, dtype=np.int64, out=known_prefix[1:])
    counts = known_prefix[window:] - known_prefix[:-window]
    return np.flatnonzero(counts > 0).astype(np.int64, copy=False)


def _feature_stats(feature_arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    count = 0
    total = np.zeros(16, dtype=np.float64)
    square_total = np.zeros(16, dtype=np.float64)
    for features in feature_arrays:
        values = np.asarray(features, dtype=np.float64)
        count += len(values)
        total += values.sum(axis=0, dtype=np.float64)
        square_total += np.square(values).sum(axis=0, dtype=np.float64)
    if count == 0:
        raise ValueError("feature_gru needs training observations")
    mean = total / count
    variance = np.maximum(square_total / count - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def _training_blocks(
    datasets: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None, Any | None]:
    name = config["name"]
    token_api = _token_api() if name != "feature_gru" else None
    raw_feature_arrays: list[np.ndarray] = []
    blocks_for_fit: list[dict[str, Any]] = []
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            raise TypeError("each training dataset must be a mapping")
        if "y" not in dataset:
            raise ValueError("training datasets require y labels")
        state = np.asarray(dataset["y"], dtype=np.int64)
        if state.ndim != 1 or not np.isin(state, [-1, 0, 1, 2]).all():
            raise ValueError("training y must contain only -1, 0, 1, or 2")
        if name == "feature_gru":
            raw_features = np.asarray(dataset["X"], dtype=np.float32)
            if raw_features.shape != (len(state), 16):
                raise ValueError("training X must have shape (N, 16)")
            raw_feature_arrays.append(raw_features)
            features = np.ascontiguousarray(raw_features, dtype=np.float16)
            observations: Mapping[str, Any] = {"blocks": dataset.get("blocks")}
        else:
            observations = _encoded_tokens(dataset, token_api)
            token_array = np.asarray(observations["tokens"])
            mask_array = np.asarray(observations["mask"])
            glob_array = np.asarray(observations["glob"])
            if token_array.ndim != 3 or token_array.shape[0] != len(state) or token_array.shape[2] != 10:
                raise ValueError("tokens must have shape (N, K, 10)")
            if mask_array.shape != token_array.shape[:2]:
                raise ValueError("token mask shape must match token rows")
            if glob_array.ndim != 2 or glob_array.shape[0] != len(state) or glob_array.shape[1] != 19:
                raise ValueError("glob must have shape (N, 19)")
            observations = {
                "tokens": np.ascontiguousarray(token_array, dtype=np.float16),
                "mask": np.ascontiguousarray(mask_array, dtype=np.bool_),
                "glob": np.ascontiguousarray(glob_array, dtype=np.float16),
                "blocks": observations["blocks"],
            }
        blocks = _normalize_blocks(observations.get("blocks"), len(state))
        boundary = _boundary_targets(state, blocks)
        for start, end in blocks:
            starts = _valid_starts(state, start, end, int(config["window"]))
            if not len(starts):
                continue
            block = {
                "start": start,
                "end": end,
                "length": end - start,
                "starts": starts,
                "state": state,
                "boundary": boundary,
            }
            if name == "feature_gru":
                block["X"] = features
            else:
                block["tokens"] = observations["tokens"]
                block["mask"] = observations["mask"]
                block["glob"] = observations["glob"]
            blocks_for_fit.append(block)
    normalization = _feature_stats(raw_feature_arrays) if name == "feature_gru" else None
    if not blocks_for_fit:
        raise ValueError("no labeled training window is at least 300 frames long")
    return blocks_for_fit, normalization, token_api


def _batch_arrays(
    picks: list[tuple[dict[str, Any], int]],
    config: Mapping[str, Any],
    normalization: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    window = int(config["window"])
    state = np.stack([block["state"][start : start + window] for block, start in picks])
    boundary = np.stack([block["boundary"][start : start + window] for block, start in picks])
    if config["name"] == "feature_gru":
        features = np.stack([block["X"][start : start + window] for block, start in picks])
        features = (features - normalization["mean"]) / normalization["scale"]
        return None, None, None, features.astype(np.float32, copy=False), state, boundary

    tokens = np.stack([block["tokens"][start : start + window] for block, start in picks])
    mask = np.stack([block["mask"][start : start + window] for block, start in picks])
    glob = np.stack([block["glob"][start : start + window] for block, start in picks])
    return tokens, mask, glob, None, state, boundary


def _sample_batch(
    blocks: list[dict[str, Any]], rng: np.random.Generator, batch_size: int
) -> list[tuple[dict[str, Any], int]]:
    counts = np.asarray([len(block["starts"]) for block in blocks], dtype=np.float64)
    probabilities = counts / counts.sum()
    chosen = rng.choice(len(blocks), size=batch_size, p=probabilities)
    result = []
    for block_index in chosen:
        block = blocks[int(block_index)]
        start = block["start"] + int(block["starts"][rng.integers(len(block["starts"]))])
        result.append((block, start))
    return result


def _torch_batch(
    arrays: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    tokens, mask, glob, features, state, boundary = arrays
    if config["name"] == "feature_gru":
        return (
            None,
            None,
            None,
            torch.from_numpy(features).to(device=device, dtype=torch.float32),
            torch.from_numpy(state).to(device=device, dtype=torch.long),
            torch.from_numpy(boundary).to(device=device, dtype=torch.float32),
        )
    return (
        torch.from_numpy(tokens).to(device=device, dtype=torch.float32),
        torch.from_numpy(mask).to(device=device, dtype=torch.bool),
        torch.from_numpy(glob[..., : int(config["global_features"])]).to(device=device, dtype=torch.float32),
        None,
        torch.from_numpy(state).to(device=device, dtype=torch.long),
        torch.from_numpy(boundary).to(device=device, dtype=torch.float32),
    )


def _model_loss(
    model: nn.Module,
    batch: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor],
    config: Mapping[str, Any],
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    tokens, mask, glob, features, state, boundary = batch
    if config["name"] == "feature_gru":
        state_logits, boundary_logits = model(features)
    else:
        state_logits, boundary_logits = model(tokens, mask, glob)
    state_loss = F.cross_entropy(
        state_logits.reshape(-1, 3), state.reshape(-1), ignore_index=_IGNORE
    )
    valid = (state != _IGNORE).to(dtype=boundary_logits.dtype)
    boundary_loss = F.binary_cross_entropy_with_logits(
        boundary_logits,
        boundary,
        pos_weight=positive_weight,
        reduction="none",
    )
    boundary_loss = (boundary_loss * valid).sum() / valid.sum().clamp_min(1.0)
    return state_loss + float(config["boundary_weight"]) * boundary_loss


def fit(datasets: list[Mapping[str, Any]], config: Mapping[str, Any] | str) -> dict[str, Any]:
    resolved = _resolve_config(config)
    if not datasets:
        raise ValueError("fit requires at least one training dataset")
    blocks, normalization, token_api = _training_blocks(datasets, resolved)
    device = torch.device(os.environ.get("FARRAS_TRAIN_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(resolved["seed"])
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if resolved["name"] == "feature_gru":
        model: nn.Module = _FeatureGRU(int(resolved["hidden"]))
    else:
        model = _SetSequenceModel(
            int(resolved["global_features"]), int(resolved["width"]), int(resolved["hidden"])
        )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(resolved["lr"]), weight_decay=float(resolved["weight_decay"])
    )
    total_frames = sum(block["length"] for block in blocks)
    windows_per_epoch = max(
        int(resolved["batch"]),
        int(float(resolved["coverage"]) * total_frames / int(resolved["window"])),
    )
    steps_per_epoch = max(1, windows_per_epoch // int(resolved["batch"]))
    positive_weight = torch.tensor(float(resolved["boundary_pos_weight"]), device=device)
    model.train()
    for _ in range(int(resolved["epochs"]) * steps_per_epoch):
        picks = _sample_batch(blocks, rng, int(resolved["batch"]))
        arrays = _batch_arrays(picks, resolved, normalization)
        if resolved["mirror"] and resolved["name"] != "feature_gru":
            tokens, mask, glob, features, state, boundary = arrays
            flips = rng.random(len(picks)) < 0.5
            for index in np.flatnonzero(flips):
                mirrored_tokens, mirrored_glob, mirrored_state = token_api.mirror(
                    tokens[index], glob[index], state[index]
                )
                tokens[index] = mirrored_tokens
                glob[index] = mirrored_glob
                state[index] = mirrored_state
            arrays = tokens, mask, glob, features, state, boundary
        batch = _torch_batch(arrays, resolved, device)
        loss = _model_loss(model, batch, resolved, positive_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return {"model": model, "config": resolved, "normalization": normalization}


def _window_starts(length: int, window: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, 100))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def _predict_blocks(
    test: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[tuple[int, int]]]:
    if config["name"] == "feature_gru":
        features = np.asarray(test["X"], dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 16:
            raise ValueError("test X must have shape (N, 16)")
        blocks = _normalize_blocks(test.get("blocks"), len(features))
        return None, None, None, features, blocks
    token_api = _token_api()
    observations = _encoded_tokens(test, token_api)
    tokens = np.asarray(observations["tokens"])
    mask = np.asarray(observations["mask"])
    glob = np.asarray(observations["glob"])
    if tokens.ndim != 3 or tokens.shape[2] != 10:
        raise ValueError("tokens must have shape (N, K, 10)")
    if mask.shape != tokens.shape[:2]:
        raise ValueError("token mask shape must match token rows")
    if glob.ndim != 2 or glob.shape != (len(tokens), 19):
        raise ValueError("glob must have shape (N, 19)")
    blocks = _normalize_blocks(observations["blocks"], len(tokens))
    return tokens, mask, glob, None, blocks


def predict(
    fitted: Mapping[str, Any], test: Mapping[str, Any], config: Mapping[str, Any] | str | None
) -> dict[str, np.ndarray]:
    stored_config = _resolve_config(fitted["config"])
    resolved = stored_config if config is None else _resolve_config(config)
    if resolved["name"] != stored_config["name"]:
        raise ValueError("predict config does not match fitted model")
    model = fitted["model"]
    device = next(model.parameters()).device
    tokens, mask, glob, features, blocks = _predict_blocks(test, resolved)
    length = len(features) if features is not None else len(tokens)
    probability = np.zeros((length, 3), dtype=np.float64)
    boundary = np.zeros(length, dtype=np.float64)
    weight = np.zeros(length, dtype=np.float64)
    window = int(resolved["window"])
    normalization = fitted["normalization"]
    model.eval()
    with torch.no_grad():
        for block_start, block_end in blocks:
            block_length = block_end - block_start
            local_window = min(window, block_length)
            tent = np.minimum(
                np.arange(local_window, dtype=np.float64) + 1,
                local_window - np.arange(local_window, dtype=np.float64),
            )
            for local_start in _window_starts(block_length, window):
                local_end = local_start + local_window
                start, end = block_start + local_start, block_start + local_end
                if features is not None:
                    values = (features[start:end] - normalization["mean"]) / normalization["scale"]
                    inputs = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
                    inputs = inputs.to(device=device, dtype=torch.float32).unsqueeze(0)
                    state_logits, boundary_logits = model(inputs)
                else:
                    token_input = torch.from_numpy(np.array(tokens[start:end], copy=True, order="C"))
                    mask_input = torch.from_numpy(np.array(mask[start:end], dtype=np.bool_, copy=True, order="C"))
                    glob_input = torch.from_numpy(np.array(glob[start:end, : int(resolved["global_features"])], copy=True, order="C"))
                    token_input = token_input.to(device=device, dtype=torch.float32).unsqueeze(0)
                    mask_input = mask_input.to(device=device, dtype=torch.bool).unsqueeze(0)
                    glob_input = glob_input.to(device=device, dtype=torch.float32).unsqueeze(0)
                    state_logits, boundary_logits = model(token_input, mask_input, glob_input)
                probs = torch.softmax(state_logits[0], dim=-1).cpu().numpy().astype(np.float64, copy=False)
                edge = torch.sigmoid(boundary_logits[0]).cpu().numpy().astype(np.float64, copy=False)
                probability[start:end] += tent[:, None] * probs
                boundary[start:end] += tent * edge
                weight[start:end] += tent
    if length and (weight <= 0).any():
        raise RuntimeError("prediction windows did not cover every observation")
    if length:
        probability /= weight[:, None]
        boundary /= weight
        row_sum = probability.sum(axis=1, keepdims=True)
        if not np.isfinite(probability).all() or not np.isfinite(boundary).all() or (row_sum <= 0).any():
            raise RuntimeError("model produced non-finite predictions")
        probability /= row_sum
    return {
        "probability": probability.astype(np.float32),
        "boundary": np.clip(boundary, 0.0, 1.0).astype(np.float32),
    }
