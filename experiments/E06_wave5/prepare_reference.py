import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile

os.environ['FARRAS_TRAIN_DEVICE'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False
    return True


def _load_support():
    wave4 = HERE.parent / 'E05_wave4'
    campaign = sys.modules.get('campaign')
    if campaign is not None and Path(getattr(campaign, '__file__', '')).resolve() != (wave4 / 'campaign.py').resolve():
        raise RuntimeError('Reference inference requires wave4 campaign support in a fresh process')
    if str(wave4) not in sys.path:
        sys.path.insert(0, str(wave4))
    support_path = HERE / 'support.py'
    support = sys.modules.get('support')
    if support is not None:
        if Path(getattr(support, '__file__', '')).resolve() != support_path.resolve():
            raise RuntimeError('Reference inference requires wave5 support in a fresh process')
        return support
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import support
    if Path(support.__file__).resolve() != support_path.resolve():
        raise RuntimeError('Imported support module is not wave5 support')
    return support


def _load_previous(c):
    wave3 = c.HERE.parent / 'E04_wave3'
    for name in ('decode', 'ensemble', 'models', 'tokens', 'trees'):
        module = sys.modules.get(name)
        module_path = getattr(module, '__file__', None) if module is not None else None
        if module_path is not None and not _inside(module_path, wave3):
            raise RuntimeError(f'{name} is already imported outside wave3; run reference inference in a fresh process')
    sys.path.insert(0, str(wave3))
    previous = c.import_path('wave3_reference_runner', wave3 / 'run.py')
    for name in ('decode', 'ensemble', 'models', 'trees'):
        module = sys.modules.get(name)
        if module is None or not _inside(getattr(module, '__file__', ''), wave3):
            raise RuntimeError(f'Unqualified {name} did not resolve from wave3')
    if Path(previous.__file__).resolve() != (wave3 / 'run.py').resolve():
        raise RuntimeError('Reference runner did not load from wave3')
    return previous


def _validate_stress_inputs(c):
    selection_path = c.OUT / 'selection.json'
    if not selection_path.is_file():
        raise FileNotFoundError('Freeze wave5 selection before reference inference')
    selection_sha256 = c.digest(selection_path)
    json.loads(selection_path.read_text(encoding='utf-8'))

    manifest_path = c.CACHE / 'stress_audit.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Prepare the frozen SC_HOLD stress packs before reference inference')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported wave5 stress audit schema')
    if manifest.get('selection_sha256') != selection_sha256:
        raise ValueError('Stress packs were not generated from the current frozen selection')

    expected = {f'{name}_stress' for name in c.SC_HOLD}
    packs = manifest.get('packs')
    if not isinstance(packs, dict) or set(packs) != expected or len(c.SC_HOLD) != 6:
        raise ValueError('Stress audit must contain exactly the six SC_HOLD packs')

    inputs = {}
    for name in c.SC_HOLD:
        key = f'{name}_stress'
        record = packs[key]
        if record.get('labels_preserved') is not True:
            raise ValueError(f'Stress labels are not certified as preserved: {key}')
        source_path = c.base.pack_path(name, 'clean')
        pack_path = c.pack_path(name, 'stress')
        feature_path = c.CACHE / 'features' / f'{name}_stress.npz'
        hashes = {
            'source_sha256': c.digest(source_path),
            'pack_sha256': c.digest(pack_path),
            'feature_sha256': c.digest(feature_path),
        }
        if any(record.get(field) != digest for field, digest in hashes.items()):
            raise ValueError(f'Stress input provenance mismatch: {key}')
        inputs[name] = dict(key=key, source_path=source_path, pack_path=pack_path,
            feature_path=feature_path, **hashes)
    return selection_sha256, manifest_path, manifest, inputs


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix='.npz', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _validate_prediction(result, sample_count):
    if sample_count <= 0:
        raise ValueError('Wave3 hybrid input must contain at least one sample')
    probability = np.asarray(result['probability'], dtype=np.float32)
    boundary = np.asarray(result['boundary'], dtype=np.float32)
    if probability.shape != (sample_count, 3) or boundary.shape != (sample_count,):
        raise ValueError('Wave3 hybrid returned an unexpected prediction shape')
    if not np.isfinite(probability).all() or not np.isfinite(boundary).all():
        raise ValueError('Wave3 hybrid returned non-finite predictions')
    if (probability < 0).any() or (probability > 1).any() or (boundary < 0).any() or (boundary > 1).any():
        raise ValueError('Wave3 hybrid returned values outside probability bounds')
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError('Wave3 hybrid state probabilities are not normalized')
    return probability, boundary


def _prepare():
    if 'torch' in sys.modules:
        raise RuntimeError('Reference inference must start before torch is imported; invoke this script in an isolated CPU process')
    c = _load_support()
    selection_sha256, stress_manifest_path, stress_manifest, inputs = _validate_stress_inputs(c)
    previous = _load_previous(c)
    previous_selection = previous.frozen_selection()
    source_selection_path = previous.OUT / 'selection.json'
    source_selection_sha256 = c.digest(source_selection_path)
    checkpoint_path = previous.checkpoint_path('hybrid')
    checkpoint_sha256 = c.digest(checkpoint_path)
    if previous_selection.get('checkpoint_sha256', {}).get('hybrid') != checkpoint_sha256:
        raise ValueError('Wave3 hybrid checkpoint does not match its frozen selection')

    torch = sys.modules['torch']
    torch.set_num_threads(1)
    original_torch_load = torch.load

    def _load_checkpoint_on_cpu(*args, **kwargs):
        kwargs['map_location'] = 'cpu'
        return original_torch_load(*args, **kwargs)

    torch.load = _load_checkpoint_on_cpu
    try:
        fitted = previous.load_fitted('hybrid')
    finally:
        torch.load = original_torch_load
    model = fitted['model']
    model.to('cpu')
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise RuntimeError('Wave3 hybrid checkpoint could not be placed on CPU')
    directory = c.CACHE / 'prior'
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = c.CACHE / 'reference_audit.json'
    audit = dict(schema_version=1, device='cpu', model='wave3.hybrid',
        source_checkpoint=checkpoint_path.relative_to(c.REPO).as_posix(),
        source_checkpoint_sha256=checkpoint_sha256,
        source_selection_sha256=source_selection_sha256,
        selection_sha256=selection_sha256,
        stress_manifest_sha256=c.digest(stress_manifest_path),
        stress_seed=stress_manifest['seed'], packs={})
    c.write_json(audit_path, audit)

    for name in c.SC_HOLD:
        data = c.load(name, 'stress')
        prediction = previous.predict_model('hybrid', fitted, data)
        probability, boundary = _validate_prediction(prediction, len(data['ball']))
        output = directory / f'{name}_stress.npz'
        _atomic_npz(output, hybrid_probability=probability, hybrid_boundary=boundary)
        source = inputs[name]
        audit['packs'][source['key']] = dict(sha256=c.digest(output),
            source='Wave3 frozen hybrid checkpoint; zero-shot SC_HOLD stress inference',
            source_path=source['source_path'].relative_to(c.REPO).as_posix(),
            input_pack_path=source['pack_path'].relative_to(c.REPO).as_posix(),
            input_feature_path=source['feature_path'].relative_to(c.REPO).as_posix(),
            source_sha256=source['source_sha256'], input_pack_sha256=source['pack_sha256'],
            input_feature_sha256=source['feature_sha256'], samples=len(probability),
            probability_sha256=_array_sha256(probability), boundary_sha256=_array_sha256(boundary))
        c.write_json(audit_path, audit)
        print('REFERENCE', name, 'stress', flush=True)

    if set(audit['packs']) != {f'{name}_stress' for name in c.SC_HOLD}:
        raise RuntimeError('Reference inference did not produce all six SC_HOLD predictions')
    print('REFERENCE READY', len(audit['packs']), flush=True)
    return audit

def prepare():
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
        cwd=HERE, check=True)
    return json.loads((HERE / 'cache' / 'reference_audit.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare frozen Wave3 hybrid predictions for SC_HOLD stress cases.')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        _prepare()
    else:
        prepare()
