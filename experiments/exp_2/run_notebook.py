"""Execute one exp_2 notebook in place with the exp2 kernel.

Failures are summarised in one line; full details stay in a local ``.error.txt`` so a
failing BAR-ATH cell can never echo target values into a transcript.
"""

from pathlib import Path
import os
import sys

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
ALLOWED = (
    "00_protocol.ipynb",
    "01_reference_arms.ipynb",
    "02_fits.ipynb",
    "03_evaluation.ipynb",
    "04_comparison.ipynb",
    "05_target.ipynb",
)


def main(name: str) -> int:
    if name not in ALLOWED:
        raise ValueError("Unknown exp_2 notebook")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    notebook = nbformat.read(ROOT / name, as_version=4)
    manager = KernelManager(
        kernel_name="exp2",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(ROOT / "kernels")]),
    )
    client = NotebookClient(notebook, timeout=None, km=manager,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    except Exception as error:  # noqa: BLE001 - details deliberately kept local
        nbformat.write(notebook, ROOT / name)
        (ROOT / (name + ".error.txt")).write_text(str(error))
        print("NOTEBOOK FAILED", name, type(error).__name__, "details saved locally", flush=True)
        return 1
    nbformat.write(notebook, ROOT / name)
    stale = ROOT / (name + ".error.txt")
    if stale.exists():
        stale.unlink()
    cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print("NOTEBOOK EXECUTED", name, "code_cells", cells, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
