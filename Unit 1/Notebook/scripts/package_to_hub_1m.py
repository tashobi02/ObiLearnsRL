"""
Upload 1M PPO model to Hugging Face Hub, replacing the existing 200k model.
"""
import os
from pathlib import Path
from huggingface_sb3 import package_to_hub
from huggingface_hub import whoami
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

REPO_ID = "tashobi02/PPO-LunarLander-v3"
ENV_ID = "LunarLander-v3"
NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = NOTEBOOK_DIR / "runs" / "1m"
MODEL_PATH = RUN_DIR / "models" / "ppo_lunarlander_1m_final.zip"
LOGS_DIR = RUN_DIR / "logs" / "PPO_1"

# --- Auth check ---------------------------------------------------------------
try:
    user = whoami()
    print(f"Logged in to HuggingFace as: {user.get('name', '<unknown>')}")
except Exception as e:
    raise SystemExit(
        "HuggingFace login required.\n"
        "Either `huggingface-cli login` or `export HF_TOKEN=hf_...`\n"
        f"Original error: {e}"
    )

# --- Pre-flight ---------------------------------------------------------------
assert MODEL_PATH.exists(), f"Missing model at {MODEL_PATH}"

# --- Eval env with rgb_array rendering (needed for video) ---------------------
eval_env = Monitor(gym.make(ENV_ID, render_mode="rgb_array"))
eval_env = DummyVecEnv([lambda: eval_env])

# --- Push to Hub --------------------------------------------------------------
print("\nUploading 1M PPO model to Hub (replacing the 200k model)...")
package_to_hub(
    model=PPO.load(MODEL_PATH, device="cpu"),
    model_name=MODEL_PATH.stem,                       # ppo_lunarlander_1m_final
    model_architecture="PPO",
    env_id=ENV_ID,
    eval_env=eval_env,
    repo_id=REPO_ID,
    commit_message=(
        "Replace 200k model with 1M-step PPO-LunarLander-v3 "
        "(solves env: mean reward +239.85 +/- 17.87 over 20 deterministic episodes)"
    ),
    logs=str(LOGS_DIR),
    n_eval_episodes=10,
)
print(f"\nUploaded to https://huggingface.co/{REPO_ID}")
