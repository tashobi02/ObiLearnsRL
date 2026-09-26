# FrozenLake and Taxi Q-learning

The original course exercise is [`notebooks/unit2.ipynb`](notebooks/unit2.ipynb). [`notebooks/unit2_local.ipynb`](notebooks/unit2_local.ipynb) fills every code exercise and replaces Colab setup with local setup. [`notebooks/unit2_executed.ipynb`](notebooks/unit2_executed.ipynb) contains saved outputs for every code cell.

## Reproduce the run

From `Unit 2`, activate a Python environment with [`requirements.txt`](../requirements.txt), then run:

```bash
python Notebook/scripts/prepare_local.py
python Notebook/scripts/execute_local.py
```

Each agent trains for **1,000,000 environment steps**. The run creates Q-table files in `runs/1m/models/`, Hub-ready local packages in `runs/1m/hub/`, individual replay videos in `runs/1m/videos/`, and [`all-agents-1m.mp4`](runs/1m/videos/all-agents-1m.mp4), which displays both trained agents side by side. The notebook evaluates each model over 100 episodes and checks that the saved model reloads successfully.

| Agent | Mean reward | Reward standard deviation |
|---|---:|---:|
| FrozenLake-v1, 4×4 non-slippery | 1.00 | 0.00 |
| Taxi-v3 | 7.56 | 2.71 |

## Publish the agents

With a Hugging Face account authenticated through `huggingface_hub` or `HF_TOKEN`, run:

```bash
python Notebook/scripts/upload_to_hub.py
```

The script checks that the authenticated account is `tashobi02`, creates one model repository per agent, and uploads only `q-learning.pkl`. Model cards, evaluation results, and replay videos remain in the local packages. The local Git repository is not committed by this workflow.

Published models: [FrozenLake](https://huggingface.co/tashobi02/q-FrozenLake-v1-4x4-noSlippery-1m) and [Taxi](https://huggingface.co/tashobi02/q-Taxi-v3-1m).
