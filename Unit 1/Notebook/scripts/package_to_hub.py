"""
Upload trained PPO agent to Hugging Face Hub.
Replaces Colab Cell 9.

The Colab notebook hardcodes repo_id = "maremidev/PPO-LunarLander-v3".
We do the same here, but allow override via env var.

IMPORTANT: requires a write token. Either:
  - Set HF_TOKEN env var, or
  - Run `huggingface-cli login` interactively before running this script.
"""
import os
from pathlib import Path
from huggingface_sb3 import package_to_hub
from huggingface_hub import whoami
from stable_baselines3 import PPO

# --- User-configurable ---------------------------------------------------------
REPO_ID = os.environ.get("HF_REPO_ID", "tashobi02/PPO-LunarLander-v3")
ENV_ID = "LunarLander-v3"
NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = NOTEBOOK_DIR / "runs" / "200k"
MODEL_PATH = RUN_DIR / "models" / "ppo_lunarlander_final.zip"
# --- Auth check ----------------------------------------------------------------
try:
    user = whoami()
    print(f"Logged in to HuggingFace as: {user.get('name', '<unknown>')}")
except Exception as e:
    raise SystemExit(
        "HuggingFace login required.\n"
        "Either `huggingface-cli login` or `export HF_TOKEN=hf_...`\n"
        f"Original error: {e}"
    )

# --- Pre-flight ----------------------------------------------------------------
assert MODEL_PATH.exists(), f"Missing model at {MODEL_PATH}"

# --- Push to Hub (writes README, model card, model weights) --------------------
# package_to_hub is a one-shot helper that handles evaluation recording,
# model card generation, and pushing.
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

eval_env = Monitor(gym.make(ENV_ID, render_mode="rgb_array"))
eval_env = DummyVecEnv([lambda: eval_env])

package_to_hub(
    model=PPO.load(MODEL_PATH, device="cpu"),
    model_name=MODEL_PATH.stem,
    model_architecture="PPO",
    env_id=ENV_ID,
    eval_env=eval_env,
    repo_id=REPO_ID,
    commit_message="Upload PPO-LunarLander-v3 trained agent (re-run on local CPU)",
    logs=str(RUN_DIR / "logs" / "PPO_1"),
)
print(f"\nUploaded to https://huggingface.co/{REPO_ID}")
