"""Execute the local notebook and retain every cell's output."""

from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "Notebook/notebooks/unit2_local.ipynb"
OUTPUT = ROOT / "Notebook/notebooks/unit2_executed.ipynb"


def main():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
