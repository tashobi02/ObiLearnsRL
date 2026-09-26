# Huggy PPO Notebook

This folder contains the local notebook, its original Colab source, the PPO configuration, and a helper for executing the notebook from the environment.

## Create the environment

From the `Bonus Unit 1` directory:

```bash
conda env create -f environment.yml
conda activate huggingface-bonus-unit1
python -m ipykernel install --user --name bonus-unit1 --display-name "Python 3.10 (Huggy)"
```

The environment pins Python 3.10.12 and ML-Agents 1.1.0, matching the [official ML-Agents installation guidance](https://github.com/Unity-Technologies/ml-agents/blob/develop/com.unity.ml-agents/Documentation~/Installation.md). It uses CPU-only PyTorch and a compatible setuptools release. The bundled Huggy build requires Linux x86_64.

After registering the kernel, open the notebook and select **Kernel → Change Kernel → Python 3.10 (Huggy)**, then restart the kernel and run the cells from the top. If the environment already exists, update it from `Bonus Unit 1` with `conda env update -n huggingface-bonus-unit1 -f environment.yml --prune`, then repeat the `ipykernel install` command above. The kernel version appears in the notebook toolbar; it must show Python 3.10, not the system Python.

## Open and run

From `Bonus Unit 1/Notebook`:

```bash
jupyter lab notebooks/bonus_unit1.ipynb
```

Run cells from top to bottom. The notebook downloads and extracts Huggy under `runs/environment/`, uses [`config/Huggy.yaml`](config/Huggy.yaml), and writes model checkpoints and TensorBoard results under `runs/results/<run-id>/`. The default run trains for 2,000,000 steps; CPU training may take substantially longer than the course's 30–45 minute estimate.

To execute the whole notebook from the terminal instead:

```bash
python scripts/execute_local.py
```

## Optional Hub upload

The upload cell is skipped by default. To enable it, set these variables in the shell before launching Jupyter:

```bash
export HF_TOKEN=hf_...
export HF_REPO_ID=your-username/your-huggy-repo
```

`HUGGY_RUN_ID` selects another run name, and `HUGGY_RESUME=1` resumes that run if checkpoints exist.

The original [`bonus_unit1_colab.ipynb`](notebooks/bonus_unit1_colab.ipynb) is retained as course source. The adapted [`bonus_unit1.ipynb`](notebooks/bonus_unit1.ipynb) removes Colab-only setup and uses project-relative paths.
