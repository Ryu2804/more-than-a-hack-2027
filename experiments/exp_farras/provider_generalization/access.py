import os
from pathlib import Path

from guard import ReadGuard

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_GUARD = None


def initialize():
    global _GUARD
    if _GUARD is not None:
        return _GUARD
    parent = HERE.parent
    roots = [REPO / 'evaluation/gt', REPO / 'evaluation/results', REPO / 'data',
             REPO / 'annotation_harness', REPO / 'experiments/_cache', parent / 'results']
    files = []
    for folder in parent.iterdir():
        if folder.is_dir() and folder != HERE:
            roots.append(folder / 'out')
            for relative in ('cache/data', 'cache/features', 'cache/prior'):
                files.extend((folder / relative).glob('target*.npz'))
    configured = os.environ.get('MTAH_DATA_DIR')
    if configured:
        roots.append(Path(configured))
    _GUARD = ReadGuard(denied_roots=roots, denied_files=files).install()
    _GUARD.probe(REPO / 'evaluation/gt/BAR-ATH_possession_gt_public.csv')
    return _GUARD


def restrict(role, source=None):
    current = initialize()
    import bridge as b
    if role in ('fit', 'predict', 'replay', 'smoke-fit', 'smoke-predict', 'smoke-replay'):
        current.add(roots=[b.CACHE / 'labels'])
    if role in ('fit', 'smoke-fit'):
        denied = b.test_names(source)
        current.add(roots=[b.CACHE / 'observations', b.CACHE / 'features'])
        current.add(files=[path for name in denied for path in (b.original_pack(name), b.original_features(name))])
        current.probe(b.original_pack(denied[0]))
        current.probe(b.CACHE / 'labels' / f'{denied[0]}.npz')
    elif role in ('predict', 'replay', 'smoke-predict', 'smoke-replay'):
        current.add(files=[path for names in b.PROVIDERS.values() for name in names
                           for path in (b.original_pack(name), b.original_features(name))])
        current.probe(b.original_pack(b.IDSSE[0]))
        current.probe(b.CACHE / 'labels' / f'{b.SC[0]}.npz')
    return current


def report():
    value = initialize().report()
    if value['nonprobe_blocked_attempts']:
        raise RuntimeError('Unexpected forbidden read attempt; campaign must not continue')
    return value
