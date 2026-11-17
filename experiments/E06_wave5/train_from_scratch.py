"""Train the final method from scratch and build every input the E06 notebook needs.

Two independent parts, because the match data may not leave the licensed machine:

  public  (no challenge data; can run on a GPU sandbox)
      1. IDSSE packs        E04 prepare_idsse.prepare()        downloads 7 matches (PySport mirror, CC BY 4.0)
      2. SkillCorner packs  E05 prepare_skillcorner.prepare()  downloads 20 matches (SkillCorner open data, MIT)
      3. features           E05 prepare_features.prepare()     training packs only
      4. BiGRU              E04 models.fit(.., 'hybrid')       5 IDSSE dev matches x 3 rounds, CPU, 1 thread
      5. short-loss PP-TCN  E06 training_models.fit(.., 'short_tcn')  19 matches, 3 seeds x 1,500 updates

  target  (needs data/ from the licensed bundle; run locally)
      6. tracking caches    experiments/_cache/atd_full.npy, team_dist.npy   (common.load_atd_arrays/team_distances)
      7. target pack        E04 prepare_local._load_target() + _write_pack()
      8. target features    E05 prepare_features.prepare()     target only
      9. BiGRU prior        E04 models.predict on the target   -> E05 cache/prior/target_clean.npz

Every step calls the campaigns' own functions; nothing here re-implements the method.

    python experiments/E06_wave5/train_from_scratch.py public  [--device cuda]
    python experiments/E06_wave5/train_from_scratch.py target
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
E02, E04, E05 = EXP / "E02_wave1", EXP / "E04_wave3", EXP / "E05_wave4"
IDSSE_DEV = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY")


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _support():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import support  # E06 helpers: paths, loaders, training(), training_stats()
    return support


def _log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# ------------------------------------------------------------------ public part
def prepare_public_data(force: bool = False) -> None:
    """Steps 1-3: download IDSSE + SkillCorner and build packs/features for training."""
    c = _support()
    idsse_packs = [c.base.pack_path(n, f"r{i}") for n in (*c.IDSSE_DEV, *c.IDSSE_HOLD) for i in range(3)]
    if force or not all(p.exists() for p in idsse_packs):
        _module("e06_prepare_idsse", E04 / "prepare_idsse.py").prepare()
    _log("IDSSE packs ready:", sum(p.exists() for p in idsse_packs), "/", len(idsse_packs))
    sc_packs = [c.base.pack_path(n, "clean") for n in (*c.SC_DEV, *c.SC_HOLD)]
    if force or not all(p.exists() for p in sc_packs):
        _module("e06_prepare_skillcorner", E05 / "prepare_skillcorner.py").prepare()
    _log("SkillCorner packs ready:", sum(p.exists() for p in sc_packs), "/", len(sc_packs))
    pf = _module("e06_prepare_features", E05 / "prepare_features.py")
    pf.cases = lambda: [(n, f"r{i}") for n in (*c.IDSSE_DEV, *c.IDSSE_HOLD) for i in range(3)] + [
        (n, "clean") for n in (*c.SC_DEV, *c.SC_HOLD)]
    pf.prepare()


def train_gru() -> Path:
    """Step 4: the wave3 hybrid set-sequence BiGRU, exactly as its campaign's final fit (CPU, 1 thread)."""
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    with _env(FARRAS_TRAIN_DEVICE="cpu"):
        models = _module("e04_gru_models", E04 / "models.py")
        if str(E04) not in sys.path:
            sys.path.insert(0, str(E04))  # models imports tokens lazily
        data = []
        for name in IDSSE_DEV:
            for i in range(3):
                with np.load(E04 / "cache/data" / f"{name}_r{i}.npz", allow_pickle=False) as z:
                    data.append({k: z[k] for k in z.files})
        config = dict(next(x for x in models.CONFIGS if x["name"] == "hybrid"))
        t0 = time.monotonic()
        with threadpool_limits(limits=1):
            fitted = models.fit(data, config)
    out = E04 / "out/models/hybrid.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, out)
    _log(f"BiGRU trained in {time.monotonic() - t0:.0f} s -> {out.relative_to(REPO)} sha256 {_sha(out)[:16]}")
    return out


def train_tcn(device: str = "cuda") -> Path:
    """Step 5: the short-loss PP-TCN final fit on the 19 training matches (3 seeds x 1,500 updates)."""
    import torch
    from threadpoolctl import threadpool_limits
    with _env(FARRAS_TRAIN_DEVICE=device):
        c = _support()
        import training_models as models
        config = dict(next(x for x in models.CONFIGS if x["name"] == "short_tcn"))
        t0 = time.monotonic()
        datasets = c.training(c.DEV)
        with threadpool_limits(limits=1):
            fitted = models.fit(datasets, config)
    out = c.OUT / "models/short_tcn.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, out)
    _log(f"PP-TCN trained in {time.monotonic() - t0:.0f} s -> {out.relative_to(REPO)} sha256 {_sha(out)[:16]};",
         "training", fitted["training"]["source_matches"])
    return out


# ------------------------------------------------------------------ target part (licensed data)
def build_target_inputs() -> None:
    """Steps 6-9: target pack, features and BiGRU prior from data/ (run on the licensed machine)."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    sys.path.insert(0, str(EXP))
    import common
    common.load_atd_arrays()          # experiments/_cache/atd_full.npy
    common.team_distances()           # experiments/_cache/team_dist.npy
    _log("tracking caches ready in", Path(common.CACHE).relative_to(REPO))
    local = _module("e06_prepare_local", E04 / "prepare_local.py")
    target, *_ = local._load_target()
    local._write_pack("target", "clean", target,
                      "native cached ATD tracking; team slots inferred only from cached nearest distances")
    _log("target pack written")
    c = _support()
    pf = _module("e06_prepare_features", E05 / "prepare_features.py")
    pf.cases = lambda: [("target", "clean")]
    pf.prepare()
    import torch
    with _env(FARRAS_TRAIN_DEVICE="cpu"):
        models = _module("e04_gru_models", E04 / "models.py")
        if str(E04) not in sys.path:
            sys.path.insert(0, str(E04))
        fitted = torch.load(E04 / "out/models/hybrid.pt", weights_only=False, map_location="cpu")
        config = dict(next(x for x in models.CONFIGS if x["name"] == "hybrid"))
        with threadpool_limits(limits=1):
            neural = models.predict(fitted, c.observations(c.load("target", "clean")), config)
    out = c.base.CACHE / "prior/target_clean.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # the final method uses only the BiGRU's own probabilities (hybrid_*), not the Metrica tree anchor
    np.savez_compressed(out, hybrid_probability=neural["probability"], hybrid_boundary=neural["boundary"])
    _log("BiGRU prior written:", out.relative_to(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=("public", "target"))
    parser.add_argument("--device", default="cuda", help="device for the PP-TCN (cuda or cpu)")
    parser.add_argument("--force", action="store_true", help="rebuild packs even if they exist")
    args = parser.parse_args()
    if args.part == "public":
        prepare_public_data(force=args.force)
        train_gru()
        train_tcn(args.device)
    else:
        build_target_inputs()


if __name__ == "__main__":
    main()
