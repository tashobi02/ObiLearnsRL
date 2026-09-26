"""
Fresh 1M-step PPO training on LunarLander-v3.
Saves model and TensorBoard logs under runs/1m/.
"""
from pathlib import Path
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = NOTEBOOK_DIR / "runs" / "1m"
MODEL_DIR = RUN_DIR / "models"
LOG_DIR = RUN_DIR / "logs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Build envs
env = gym.make("LunarLander-v3")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

eval_env = gym.make("LunarLander-v3")
eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])

# 2. Eval callback (every 25k steps, save best)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(MODEL_DIR / "best"),
    log_path=str(LOG_DIR),
    n_eval_episodes=10,
    eval_freq=25_000,
    deterministic=True,
    render=False,
    verbose=1,
)

# 3. PPO with Colab hyperparams
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.98,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=str(LOG_DIR),
    device="cpu",
    seed=0,
)

# 4. Train 1,000,000 timesteps
print("=== Training PPO on LunarLander-v3 for 1,000,000 timesteps ===")
t0 = time.time()
model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=False)
dt = time.time() - t0
print(f"\n=== Training done in {dt/60:.1f} min ({dt:.0f} s) ===")
print(f"Throughput: {1_000_000/dt:.0f} steps/sec")

# 5. Save final model
final_path = MODEL_DIR / "ppo_lunarlander_1m_final"
model.save(final_path)
print(f"Saved final model to {final_path}.zip")

env.close()
eval_env.close()
