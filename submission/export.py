"""Export the submitted CSV (final method, retrained from scratch) without training or reading ground truth.

The selection evidence is the frozen wave5 receipt; the submitted file is tied to it through the
retrain receipt written from the E06 notebook run. The frozen wave5 CSV stays in artifacts/ as reference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def export_selected(output):
    method = json.loads((HERE / 'method.json').read_text(encoding='utf-8'))
    artifact = method['artifact']
    evidence = method['verification']
    receipt_bytes = (REPO / evidence['path']).read_bytes()
    if sha256(receipt_bytes) != evidence['sha256']:
        raise ValueError('Wave5 verification receipt changed')
    receipt = json.loads(receipt_bytes)
    if (method['status'] != 'primary'
            or not receipt['promoted']
            or receipt['selected']['family'] != method['family']
            or receipt['selected']['decoder']['name'] != method['decoder']
            or receipt['selection_sha256'] != method['selection_sha256']):
        raise ValueError('Primary method differs from the frozen wave5 selection')
    frozen = method['frozen_artifact']
    if receipt['selected_csv_sha256'] != frozen['sha256'] or receipt['target_rows'] != frozen['rows']:
        raise ValueError('Frozen wave5 reference differs from the selection receipt')
    retrain_bytes = (REPO / method['retrain']['path']).read_bytes()
    if sha256(retrain_bytes) != method['retrain']['sha256']:
        raise ValueError('Retrain verification receipt changed')
    retrain = json.loads(retrain_bytes)
    if (retrain['selection_sha256'] != method['selection_sha256']
            or retrain['family'] != method['family'] or retrain['decoder'] != method['decoder']
            or retrain['selected_csv_sha256'] != artifact['sha256']
            or retrain['target_rows'] != artifact['rows']):
        raise ValueError('Submitted CSV is not the one recorded by the retrain receipt')

    source = REPO / artifact['path']
    if not source.is_file():
        raise FileNotFoundError(
            'Published submission CSV is missing; see submission/README.md. '
            'This command exports an existing artifact, not raw-data inference.')
    payload = source.read_bytes()
    if sha256(payload) != artifact['sha256']:
        raise ValueError('Submitted CSV changed; refusing to export another candidate')

    output = output.resolve()
    exists = output.exists()
    if exists and sha256(output.read_bytes()) != artifact['sha256']:
        raise FileExistsError(f'Refusing to overwrite a different file: {output}')

    subprocess.run(
        [sys.executable, '-B', str(REPO / 'evaluation/validate_submission.py'),
         str(source), '--rows', str(artifact['rows'])], check=True)
    if not exists:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as handle:
            handle.write(payload)
    actual = sha256(output.read_bytes())
    if actual != artifact['sha256']:
        raise ValueError('Exported CSV does not match the submitted artifact')
    print(f'METHOD  {method["id"]}')
    print(f'{"UNCHANGED" if exists else "EXPORTED"}  {output}')
    print(f'SHA256  {actual}')
    print('STATUS  Final method retrained from scratch (E06 notebook); frozen wave5 CSV kept as reference.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'out/farras.csv',
                        help='destination CSV; an existing different file is never overwritten')
    args = parser.parse_args()
    try:
        export_selected(args.output)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'ERROR  {error}\n')


if __name__ == '__main__':
    main()
