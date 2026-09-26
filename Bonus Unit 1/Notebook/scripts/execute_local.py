"""Execute the adapted Bonus Unit 1 notebook with its registered kernel."""

from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient


NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = NOTEBOOK_DIR / "notebooks" / "bonus_unit1.ipynb"
# PPO training can exceed a fixed timeout on CPU; allow the training cell to finish.
TIMEOUT_PER_CELL = None

if not NOTEBOOK_PATH.is_file():
    raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
client = NotebookClient(
    notebook,
    kernel_name="bonus-unit1",
    timeout=TIMEOUT_PER_CELL,
    resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
)

started_at = time.time()
timeout_label = "disabled" if TIMEOUT_PER_CELL is None else f"{TIMEOUT_PER_CELL // 60} minutes"
print(f"Executing {NOTEBOOK_PATH.name}; per-cell timeout is {timeout_label}.")
try:
    client.execute()
finally:
    nbformat.write(notebook, NOTEBOOK_PATH)
print(f"Execution complete in {(time.time() - started_at) / 60:.1f} minutes.")
