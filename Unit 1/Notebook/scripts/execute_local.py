"""
Execute unit1_executed.ipynb in-place, preserving cell outputs in the .ipynb file.
This is the local equivalent of "Run All" in Jupyter.
"""
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
NB = NOTEBOOK_ROOT / "notebooks" / "unit1_executed.ipynb"
assert NB.exists(), f"Notebook not found: {NB}"

# Optional: cell 45 (training) takes ~7 min on CPU. We allow it but cap execution timeout
# per cell so a stuck cell doesn't hang the whole notebook indefinitely.
TIMEOUT_PER_CELL = 1800  # 30 min cap per cell, well above the ~7 min training step

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(
    nb,
    kernel_name="python3",
    timeout=TIMEOUT_PER_CELL,
    resources={"metadata": {"path": str(NOTEBOOK_ROOT)}},
)

t0 = time.time()
print(f"Executing notebook {NB.name} (timeout per cell: {TIMEOUT_PER_CELL}s)")
client.execute()

nbformat.write(nb, NB)
dt = time.time() - t0
print(f"\nDone in {dt/60:.1f} min. Notebook saved with outputs: {NB}")
