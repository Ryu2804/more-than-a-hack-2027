"""Export the frozen primary CSV without training or reading ground truth."""

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
            or receipt['selection_sha256'] != method['selection_sha256']
            or receipt['selected_csv_sha256'] != artifact['sha256']
            or receipt['target_rows'] != artifact['rows']):
        raise ValueError('Primary method differs from the frozen wave5 selection')

    source = REPO / artifact['path']
    if not source.is_file():
        raise FileNotFoundError(
            'Published wave5 CSV is missing; see submission/README.md. '
            'This command exports an existing artifact, not raw-data inference.')
    payload = source.read_bytes()
    if sha256(payload) != artifact['sha256']:
        raise ValueError('Frozen wave5 CSV changed; refusing to export another candidate')

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
        raise ValueError('Exported CSV does not match the frozen artifact')
    print(f'METHOD  {method["id"]}')
    print(f'{"UNCHANGED" if exists else "EXPORTED"}  {output}')
    print(f'SHA256  {actual}')
    print('STATUS  Primary method; frozen CSV export only, not submission-ready.')
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
