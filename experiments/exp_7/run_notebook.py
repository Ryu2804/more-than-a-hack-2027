"""Execute one exp_7 notebook in place with the exp7 kernel (exp_6's runner pattern, not imported).

Failures are summarised in one line; full details stay in a local ``.error.txt`` so a failing BAR-ATH cell can
never echo target values into a transcript. The kernel is always shut down (nbclient does not when ``km`` is
passed).
"""

from pathlib import Path
import os
import sys

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
ALLOWED = ("00_protocol.ipynb", "01_fits.ipynb", "02_evaluation.ipynb", "03_decision.ipynb", "04_target.ipynb")


def main(name: str) -> int:
    if name not in ALLOWED:
        raise ValueError("Unknown exp_7 notebook")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    notebook = nbformat.read(ROOT / name, as_version=4)
    manager = KernelManager(kernel_name="exp7",
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(ROOT / "kernels")]))
    client = NotebookClient(notebook, timeout=None, km=manager, resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    except Exception as error:  # noqa: BLE001 - details deliberately kept local
        nbformat.write(notebook, ROOT / name)
        (ROOT / (name + ".error.txt")).write_text(str(error))
        print("NOTEBOOK FAILED", name, type(error).__name__, "details saved locally", flush=True)
        return 1
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(notebook, ROOT / name)
    stale = ROOT / (name + ".error.txt")
    if stale.exists():
        stale.unlink()
    print("NOTEBOOK EXECUTED", name, "code_cells", sum(c.cell_type == "code" for c in notebook.cells), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
