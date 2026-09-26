"""
Pre-process unit1_executed.ipynb for local execution.

Important: source MUST be a list of lines each ending with '\n' (or last line no \n).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "unit1_executed.ipynb"
nb = json.loads(NB.read_text())


def set_source(idx, source):
    if isinstance(source, str):
        source = source.split("\n")
        # nbformat convention: each element ends with '\n' except the last.
        out = []
        for i, line in enumerate(source):
            if i < len(source) - 1 and not line.endswith("\n"):
                out.append(line + "\n")
            else:
                out.append(line)
        source = out
    nb["cells"][idx]["source"] = source
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["metadata"] = {}
    if "execution_count" in nb["cells"][idx]:
        nb["cells"][idx]["execution_count"] = None


# Skip the install cells (already installed)
set_source(14, """\
# Original Colab cell: `!apt install swig cmake`
# Already installed system-wide in this environment.
import subprocess
for cmd in [['swig', '--version'], ['cmake', '--version']]:
    print(cmd[0], '->', subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0])
""")

set_source(15, """\
# Original Colab cell: `!pip install -r https://...requirements-unit1.txt`
# Equivalent dependencies already installed in .venv.
import stable_baselines3, gymnasium, huggingface_sb3, huggingface_hub, pyvirtualdisplay
print('sb3', stable_baselines3.__version__, '| gym', gymnasium.__version__)
""")

set_source(16, """\
# Original Colab cell: `pip install stable-baselines3 huggingface-sb3 gymnasium[box2d]`
# Already installed; no-op.
print('Dependencies already installed.')
""")

set_source(18, """\
# Original Colab cell installs ffmpeg/xvfb/pyvirtualdisplay.
# Already installed system-wide; no-op.
import shutil
for b in ['ffmpeg', 'xvfb-run']:
    print(b, '->', shutil.which(b))
""")

# Don't crash the kernel with os.kill
set_source(20, """\
# Original Colab cell restarts the runtime so freshly-installed libs are picked up.
# Locally we don't need to restart -- environment was prepared before launching Jupyter.
print('Skipped runtime restart (running locally).')
""")

# Set model_name so cell 45's `model.save(model_name)` works
set_source(43, """\
# TODO cell filled in for local run.
model_name = 'runs/1m/models/notebook/ppo-LunarLander-v2'
import os
os.makedirs(os.path.dirname(model_name), exist_ok=True)
print(f'Will save trained model to {model_name}.zip')
""")

# Replace `notebook_login` with env-token-based login
set_source(54, """\
# Local replacement for notebook_login(): use HF_TOKEN env var.
import os
from huggingface_hub import login
token = os.environ.get('HF_TOKEN')
assert token, 'Set HF_TOKEN before running this notebook.'
login(token=token, add_to_git_credential=True)
print('Logged in via HF_TOKEN env var.')
""")

# Fill in cell 58 with our actual repo
set_source(58, """\
# TODO cell filled in for local run: upload to tashobi02 namespace.
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from huggingface_sb3 import package_to_hub

repo_id = 'tashobi02/PPO-LunarLander-v3'
env_id = 'LunarLander-v3'
eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode='rgb_array'))])
model_architecture = 'PPO'
commit_message = 'Upload PPO LunarLander-v3 trained agent (unit1.ipynb executed locally)'

package_to_hub(
    model=model,
    model_name=model_name,
    model_architecture=model_architecture,
    env_id=env_id,
    eval_env=eval_env,
    repo_id=repo_id,
    commit_message=commit_message,
)
""")

# Skip duplicate SOLUTION (cell 60 already covered by cell 58)
set_source(60, """\
# Cell 60 was the SOLUTION duplicate of cell 58 -- already executed above.
print('Skipped duplicate SOLUTION cell.')
""")

# shimmy already installed
set_source(65, """\
# Original Colab cell: `!pip install shimmy`
# Already installed.
import shimmy
print('shimmy', shimmy.__version__)
""")

# Point cell 66 at our own uploaded model
set_source(66, """\
# Original cell uses placeholder repo 'Classroom-workshop/assignment2-omar'.
# Local adaptation: load the model we just pushed to our Hub namespace.
from huggingface_sb3 import load_from_hub
repo_id = 'tashobi02/PPO-LunarLander-v3'
filename = 'ppo_lunarlander_1m_final.zip'

checkpoint = load_from_hub(repo_id, filename)
model = PPO.load(checkpoint, print_system_info=True)
""")

NB.write_text(json.dumps(nb, indent=1))
print("Notebook preprocessed.")

# Quick sanity check: dump first lines of each patched cell
nb2 = json.loads(NB.read_text())
for idx in [43, 54, 58, 65, 66]:
    c = nb2["cells"][idx]
    print(f"\n--- cell {idx} (first 2 source lines) ---")
    for line in c["source"][:2]:
        print(" ", repr(line))
