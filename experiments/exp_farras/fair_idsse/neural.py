from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import torch

from bridge import original, probability_result, sequence, training


FAMILIES = ("pp_tcn", "short_tcn", "setseq")
SEEDS = (0, 1, 2)


def _training_names(datasets: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if len(datasets) < 2:
        raise ValueError("neural training requires at least two unique matches")
    names = []
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            raise TypeError("each training dataset must be a mapping")
        name = dataset.get("_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each training dataset requires a nonempty _name")
        if "y" not in dataset:
            raise ValueError("each training dataset requires y labels")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("training match names must be unique")
    return tuple(names)


def _prediction_observations(
    fitted: Mapping[str, Any], observations: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(observations, Mapping):
        raise TypeError("prediction observations must be a mapping")
    if "y" in observations or "y25" in observations:
        raise ValueError("prediction observations must not contain labels")
    name = observations.get("_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("prediction observations require a nonempty _name")
    if name in fitted["training_names"]:
        raise ValueError("prediction match must not occur in training names")
    return observations


def _average_probabilities(outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(outputs) != len(SEEDS):
        raise ValueError("neural prediction requires three seed outputs")
    average = None
    for output in outputs:
        joint = probability_result(output["probability"])["probability"]
        if average is None:
            average = joint
        else:
            if joint.shape != average.shape:
                raise ValueError("seed predictions must have identical shapes")
            average += joint
    average /= len(outputs)
    return probability_result(average)


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit_setseq_seed(
    blocks: list[dict[str, Any]],
    normalization: Any,
    token_api: Any,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = sequence._SetSequenceModel(
        int(config["global_features"]), int(config["width"]), int(config["hidden"])
    )
    optimizer = batch = loss = positive_weight = None
    try:
        model.to(device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
        )
        positive_weight = torch.tensor(float(config["boundary_pos_weight"]), device=device)
        for _ in range(config["steps"]):
            picks = sequence._sample_batch(blocks, rng, int(config["batch"]))
            arrays = sequence._batch_arrays(picks, config, normalization)
            if config["mirror"]:
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
            batch = sequence._torch_batch(arrays, config, device)
            loss = sequence._model_loss(model, batch, config, positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        return {"model": model, "config": dict(config, seed=seed), "normalization": normalization}
    finally:
        model.to("cpu")
        optimizer = batch = loss = positive_weight = None
        _empty_cuda_cache()


def _fit_setseq(datasets: list[Mapping[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    blocks, normalization, token_api = sequence._training_blocks(datasets, config)
    device = original._device()
    members = []
    for seed in config["seeds"]:
        members.append(_fit_setseq_seed(blocks, normalization, token_api, config, seed, device))
    return {
        "members": members,
        "config": config,
        "training": {"matches": len(datasets), "device": str(device)},
    }


def fit(
    datasets: Sequence[Mapping[str, Any]],
    family: str,
    steps: int = 500,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"unknown neural family: {family!r}")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    seeds = tuple(seeds)
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds) or seeds != SEEDS:
        raise ValueError("neural recipes require seeds (0, 1, 2)")
    datasets = list(datasets)
    names = _training_names(datasets)
    if family == "setseq":
        config = dict(sequence._CONFIG_BY_NAME[family])
        config.pop("epochs", None)
        config.pop("coverage", None)
        config.pop("seed", None)
        config.update(
            steps=int(steps), seeds=SEEDS, window=384, batch=16, lr=0.002,
            width=32, hidden=32, boundary_weight=0.5, boundary_pos_weight=3.0,
            weight_decay=0.0001, mirror=True,
        )
        fitted = _fit_setseq(datasets, config)
    else:
        module = original if family == "pp_tcn" else training
        config = dict(module._CONFIG_BY_NAME[family])
        config.update(steps=int(steps), seeds=SEEDS, batch=16, crop=384, lr=0.002)
        try:
            fitted = module.fit(datasets, config)
        finally:
            _empty_cuda_cache()
    fitted.update(family=family, training_names=names, seeds=SEEDS, steps=int(steps))
    fitted["training"].update(names=names, seeds=SEEDS, steps=int(steps))
    return fitted


def predict(fitted: Mapping[str, Any], observations: Mapping[str, Any]) -> dict[str, Any]:
    observations = _prediction_observations(fitted, observations)
    family = fitted["family"]
    if family not in FAMILIES:
        raise ValueError(f"unknown neural family: {family!r}")
    try:
        if family != "setseq":
            module = original if family == "pp_tcn" else training
            return probability_result(module.predict(fitted, observations)["probability"])
        members = fitted["members"]
        if len(members) != len(SEEDS):
            raise ValueError("fitted SetSeq ensemble must contain three seeds")
        device = original._device()
        outputs = []
        for member in members:
            model = member["model"]
            try:
                model.to(device).eval()
                outputs.append(sequence.predict(member, observations, None))
            finally:
                model.to("cpu")
                _empty_cuda_cache()
        return _average_probabilities(outputs)
    finally:
        _empty_cuda_cache()
