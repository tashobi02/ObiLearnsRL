# Unit 1 — Reinforcement Learning Fundamentals

This unit contains the study notes, the LunarLander PPO lesson and training runs, and the papers cited in the notes.

## Contents

- [`Notes/README.md`](Notes/README.md) — notes index and links to related material.
- [`Notes/Unit1.md`](Notes/Unit1.md) — lesson notes.
- [`Notebook/README.md`](Notebook/README.md) — notebook and training instructions.
- [`Notebook/notebooks/`](Notebook/notebooks/) — original and locally executed course notebooks.
- [`Notebook/scripts/`](Notebook/scripts/) — training, evaluation, upload, and notebook helpers.
- [`Notebook/runs/`](Notebook/runs/) — retained models and demo videos for the 200k and 1m runs.
- [`Papers/README.md`](Papers/README.md) — reading list and downloaded papers.

## Set up the environment

From the workspace root, enter this directory and install the listed dependencies:

```bash
cd "Unit 1"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For headless video recording, install `xvfb` and `ffmpeg` with your system package manager.
