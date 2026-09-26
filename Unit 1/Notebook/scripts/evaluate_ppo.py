"""
Evaluate the trained PPO agent — replaces Colab Cells 6-8.
- Loads final model
- Runs 10 deterministic episodes
- Reports mean / std reward
- Saves a short MP4 demo if a video recorder is available
"""
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = NOTEBOOK_DIR / "runs" / "200k"
MODEL_PATH = RUN_DIR / "models" / "ppo_lunarlander_final.zip"
VIDEO_DIR = RUN_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load model
assert os.path.exists(MODEL_PATH), f"Missing model: {MODEL_PATH}"
model = PPO.load(MODEL_PATH, device="cpu")
print(f"Loaded model from {MODEL_PATH}")

# 2. Eval env (no vec normalize needed since Colab didn't use VecNormalize)
env = gym.make("LunarLander-v3")
env = DummyVecEnv([lambda: env])

# 3. Record one deterministic rollout to mp4 (best effort)
try:
    rec_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    rec_env = DummyVecEnv([lambda: rec_env])
    rec_env = VecVideoRecorder(
        rec_env,
        VIDEO_DIR,
        record_video_trigger=lambda x: x == 0,
        video_length=1000,
        name_prefix="ppo-lunarlander-demo",
    )
    obs = rec_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = rec_env.step(action)
        if done[0]:
            break
    rec_env.close()
    print(f"Video saved to {VIDEO_DIR}")
except Exception as e:
    print(f"Video recording skipped: {e}")

# 4. Numerical eval — 10 episodes, deterministic
env = gym.make("LunarLander-v3")
rewards = []
for ep in range(10):
    obs, info = env.reset()
    total = 0.0
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = env.step(action)
        total += float(r)
    rewards.append(total)
    print(f"Episode {ep+1:2d}: reward = {total:8.2f}")

rewards = np.array(rewards)
print("\n=== Eval results (10 episodes) ===")
print(f"mean_reward: {rewards.mean():.2f} +/- {rewards.std():.2f}")
print(f"min: {rewards.min():.2f}   max: {rewards.max():.2f}")
env.close()
