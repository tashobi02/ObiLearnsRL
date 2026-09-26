"""Build a runnable local version of the original Unit 2 course notebook."""

from pathlib import Path
import nbformat


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Notebook/notebooks/unit2.ipynb"
DEST = ROOT / "Notebook/notebooks/unit2_local.ipynb"


def main():
    notebook = nbformat.read(SOURCE, as_version=4)
    code = {
        12: "print('Dependencies are listed in Unit 2/requirements.txt and installed before notebook execution.')",
        13: "print('Gymnasium rgb_array rendering works headlessly; no apt packages or virtual display are required.')",
        15: "print('Local kernel ready; no Colab runtime restart is needed.')",
        16: "import os\nos.environ.setdefault('SDL_VIDEODRIVER', 'dummy')\nprint('SDL_VIDEODRIVER =', os.environ['SDL_VIDEODRIVER'])",
        18: "from pathlib import Path\nimport json, pickle, random, shutil\nimport numpy as np\nimport gymnasium as gym\nimport imageio.v2 as imageio\nfrom PIL import Image, ImageDraw\nfrom IPython.display import display, Video\nrandom.seed(42)\nnp.random.seed(42)\nRUNS = Path('../runs/1m').resolve()\nfor name in ('models', 'videos', 'hub'):\n    (RUNS / name).mkdir(parents=True, exist_ok=True)\nprint('Gymnasium', gym.__version__, '| outputs:', RUNS)",
        23: "env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False, render_mode='rgb_array')\nenv.reset(seed=42)\nenv.action_space.seed(42)\nprint('FrozenLake environment:', env.spec.id, env.spec.kwargs)",
        25: "print('Solution: the 4x4 non-slippery FrozenLake environment is ready.')",
        28: "print('Observation space:', env.observation_space, '| sample:', env.observation_space.sample())",
        30: "print('Action space:', env.action_space, '| sample:', env.action_space.sample())",
        33: "state_space = env.observation_space.n\naction_space = env.action_space.n\nprint('States:', state_space, '| actions:', action_space)",
        34: "def initialize_q_table(state_space, action_space):\n    return np.zeros((state_space, action_space), dtype=np.float64)\nprint('Initialized Q-table shape:', initialize_q_table(state_space, action_space).shape)",
        35: "Qtable_frozenlake = initialize_q_table(state_space, action_space)\nprint('FrozenLake Q-table initialized:', Qtable_frozenlake.shape)",
        37: "state_space, action_space = env.observation_space.n, env.action_space.n\nprint('Solution: states =', state_space, ', actions =', action_space)",
        38: "print('Solution: np.zeros((state_space, action_space)) gives', initialize_q_table(state_space, action_space).shape)",
        39: "print('FrozenLake initial nonzero Q-values:', np.count_nonzero(Qtable_frozenlake))",
        41: "def greedy_policy(Qtable, state):\n    return int(np.argmax(Qtable[state]))\nprint('Greedy action for initial state:', greedy_policy(Qtable_frozenlake, 0))",
        43: "print('Solution: choose argmax of the current state row:', greedy_policy(Qtable_frozenlake, 0))",
        45: "def epsilon_greedy_policy(Qtable, state, epsilon, action_space):\n    return int(action_space.sample()) if random.random() < epsilon else greedy_policy(Qtable, state)\nprint('Exploratory action:', epsilon_greedy_policy(Qtable_frozenlake, 0, 1.0, env.action_space))",
        47: "print('Solution: epsilon-greedy policy accepts epsilon and the environment action space.')",
        49: "training_steps = 1_000_000\nlearning_rate = 0.7\ngamma = 0.95\nn_eval_episodes = 100\nenv_id = 'FrozenLake-v1'\nmax_steps = 99\neval_seed = list(range(100))\nmax_epsilon, min_epsilon, decay_rate = 1.0, 0.05, 0.0005\nprint('FrozenLake training budget:', training_steps, 'environment steps')",
        51: "def train(training_steps, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable, learning_rate, gamma):\n    steps = episodes = 0\n    next_report = 200_000\n    episode_returns = []\n    while steps < training_steps:\n        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episodes)\n        state, _ = env.reset()\n        episode_reward = 0.0\n        for _ in range(min(max_steps, training_steps - steps)):\n            action = epsilon_greedy_policy(Qtable, state, epsilon, env.action_space)\n            next_state, reward, terminated, truncated, _ = env.step(action)\n            target = reward if (terminated or truncated) else reward + gamma * np.max(Qtable[next_state])\n            Qtable[state, action] += learning_rate * (target - Qtable[state, action])\n            state = next_state\n            episode_reward += reward\n            steps += 1\n            if terminated or truncated:\n                break\n        episodes += 1\n        episode_returns.append(episode_reward)\n        if steps >= next_report:\n            print(f'{env.spec.id}: {steps:,} steps, {episodes:,} episodes, recent mean reward {np.mean(episode_returns[-100:]):.2f}')\n            next_report += 200_000\n    return Qtable, {'training_steps': steps, 'n_training_episodes': episodes, 'last_100_train_reward': float(np.mean(episode_returns[-100:]))}\nprint('Training function ready.')",
        53: "print('Solution: the training loop updates Q(s,a) after each environment step and stops at the exact step budget.')",
        55: "Qtable_frozenlake, frozen_stats = train(training_steps, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake, learning_rate, gamma)\nprint('FrozenLake training:', frozen_stats)",
        57: "print('FrozenLake Q-table:\\n', Qtable_frozenlake)",
        59: "def evaluate_agent(env, max_steps, n_eval_episodes, Q, seeds):\n    rewards = []\n    for episode in range(n_eval_episodes):\n        state, _ = env.reset(seed=seeds[episode] if seeds else episode)\n        total_reward = 0.0\n        for _ in range(max_steps):\n            action = greedy_policy(Q, state)\n            state, reward, terminated, truncated, _ = env.step(action)\n            total_reward += reward\n            if terminated or truncated:\n                break\n        rewards.append(total_reward)\n    return float(np.mean(rewards)), float(np.std(rewards))\nprint('Evaluation function ready.')",
        61: "frozen_mean, frozen_std = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)\nprint(f'FrozenLake: {frozen_mean:.2f} +/- {frozen_std:.2f} over {n_eval_episodes} episodes')",
        65: "from datetime import datetime, timezone\nprint('Local package dependencies loaded.')",
        66: "def _episode_frames(env, Qtable, seeds, max_steps):\n    frames, rewards = [], []\n    for seed in seeds:\n        state, _ = env.reset(seed=seed)\n        frames.append(np.asarray(env.render()))\n        total = 0.0\n        for _ in range(max_steps):\n            state, reward, terminated, truncated, _ = env.step(greedy_policy(Qtable, state))\n            frames.append(np.asarray(env.render()))\n            total += reward\n            if terminated or truncated:\n                break\n        rewards.append(total)\n    return frames, rewards\n\ndef record_video(env, Qtable, path, seeds=(16, 54, 165), max_steps=99, fps=4):\n    frames, rewards = _episode_frames(env, Qtable, seeds, max_steps)\n    with imageio.get_writer(path, fps=fps, codec='libx264', macro_block_size=2) as writer:\n        for frame in frames:\n            writer.append_data(frame)\n    return {'path': str(path), 'frames': len(frames), 'rewards': rewards}\nprint('Video recorder ready.')",
        67: "def package_model(model, env, repo_name, mean_reward, std_reward, video_path):\n    package = RUNS / 'hub' / repo_name\n    package.mkdir(parents=True, exist_ok=True)\n    with (package / 'q-learning.pkl').open('wb') as f:\n        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)\n    results = {'env_id': model['env_id'], 'mean_reward': mean_reward, 'std_reward': std_reward, 'n_eval_episodes': model['n_eval_episodes'], 'eval_datetime': datetime.now(timezone.utc).isoformat()}\n    (package / 'results.json').write_text(json.dumps(results, indent=2) + '\\n')\n    env_kwargs = {k: model[k] for k in ('map_name', 'is_slippery') if k in model}\n    card = f\"\"\"---\\ntags:\\n- reinforcement-learning\\n- q-learning\\n- {model['env_id']}\\n---\\n# Q-learning agent for {model['env_id']}\\n\\nTrained for {model['training_steps']:,} environment steps with tabular Q-learning.\\nMean reward over {model['n_eval_episodes']} evaluation episodes: {mean_reward:.2f} ± {std_reward:.2f}.\\n\\nLoad `q-learning.pkl` with Python pickle from a trusted source. Create the environment with `gym.make({model['env_id']!r}, **{env_kwargs!r})` and act with `np.argmax(model['qtable'][state])`.\\n\"\"\"\n    (package / 'README.md').write_text(card)\n    shutil.copy2(video_path, package / 'replay.mp4')\n    return package\nprint('Model packaging function ready.')",
        70: "print('Hugging Face upload runs after local training via Notebook/scripts/upload_to_hub.py.')",
        73: "frozen_model = {'env_id': env_id, 'map_name': '4x4', 'is_slippery': False, 'max_steps': max_steps, 'training_steps': frozen_stats['training_steps'], 'n_training_episodes': frozen_stats['n_training_episodes'], 'n_eval_episodes': n_eval_episodes, 'eval_seed': eval_seed, 'learning_rate': learning_rate, 'gamma': gamma, 'max_epsilon': max_epsilon, 'min_epsilon': min_epsilon, 'decay_rate': decay_rate, 'qtable': Qtable_frozenlake}\nprint('FrozenLake model keys:', sorted(frozen_model))",
        75: "print('FrozenLake model Q-table shape:', frozen_model['qtable'].shape)",
        76: "frozen_video = RUNS / 'videos/frozenlake-1m.mp4'\nfrozen_video_info = record_video(env, Qtable_frozenlake, frozen_video, seeds=(16, 54, 165), max_steps=max_steps)\nfrozen_package = package_model(frozen_model, env, 'q-FrozenLake-v1-4x4-noSlippery-1m', frozen_mean, frozen_std, frozen_video)\nshutil.copy2(frozen_package / 'q-learning.pkl', RUNS / 'models/frozenlake-1m.pkl')\nprint('FrozenLake replay:', frozen_video_info)\nprint('FrozenLake Hub package:', frozen_package)",
        79: "env.close()\nenv = gym.make('Taxi-v3', render_mode='rgb_array')\nenv.reset(seed=42)\nenv.action_space.seed(42)\nprint('Taxi environment:', env.spec.id)",
        81: "state_space = env.observation_space.n\nprint('Taxi states:', state_space)",
        82: "action_space = env.action_space.n\nprint('Taxi actions:', action_space)",
        84: "Qtable_taxi = initialize_q_table(state_space, action_space)\nprint('Taxi Q-table shape:', Qtable_taxi.shape)",
        86: "training_steps = 1_000_000\nlearning_rate = 0.7\nn_eval_episodes = 100\neval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148]\nenv_id = 'Taxi-v3'\nmax_steps = 99\ngamma = 0.95\nmax_epsilon, min_epsilon, decay_rate = 1.0, 0.05, 0.005\nprint('Taxi training budget:', training_steps, 'environment steps')",
        88: "Qtable_taxi, taxi_stats = train(training_steps, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_taxi, learning_rate, gamma)\nprint('Taxi training:', taxi_stats)\ntaxi_mean, taxi_std = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_taxi, eval_seed)\nprint(f'Taxi: {taxi_mean:.2f} +/- {taxi_std:.2f} over {n_eval_episodes} episodes')",
        90: "taxi_model = {'env_id': env_id, 'max_steps': max_steps, 'training_steps': taxi_stats['training_steps'], 'n_training_episodes': taxi_stats['n_training_episodes'], 'n_eval_episodes': n_eval_episodes, 'eval_seed': eval_seed, 'learning_rate': learning_rate, 'gamma': gamma, 'max_epsilon': max_epsilon, 'min_epsilon': min_epsilon, 'decay_rate': decay_rate, 'qtable': Qtable_taxi}\nprint('Taxi model Q-table shape:', taxi_model['qtable'].shape)",
        91: "taxi_video = RUNS / 'videos/taxi-1m.mp4'\ntaxi_video_info = record_video(env, Qtable_taxi, taxi_video, seeds=(16, 54, 165), max_steps=max_steps)\ntaxi_package = package_model(taxi_model, env, 'q-Taxi-v3-1m', taxi_mean, taxi_std, taxi_video)\nshutil.copy2(taxi_package / 'q-learning.pkl', RUNS / 'models/taxi-1m.pkl')\nprint('Taxi replay:', taxi_video_info)\nprint('Taxi Hub package:', taxi_package)",
        96: "def load_local_model(package):\n    with (package / 'q-learning.pkl').open('rb') as f:\n        return pickle.load(f)\nprint('Local model loader ready.')",
        98: "loaded_taxi = load_local_model(taxi_package)\nloaded_taxi_mean, loaded_taxi_std = evaluate_agent(env, loaded_taxi['max_steps'], loaded_taxi['n_eval_episodes'], loaded_taxi['qtable'], loaded_taxi['eval_seed'])\nprint(f'Reloaded Taxi: {loaded_taxi_mean:.2f} +/- {loaded_taxi_std:.2f}')",
        99: "loaded_frozen = load_local_model(frozen_package)\nfrozen_env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False, render_mode='rgb_array')\nloaded_frozen_mean, loaded_frozen_std = evaluate_agent(frozen_env, loaded_frozen['max_steps'], loaded_frozen['n_eval_episodes'], loaded_frozen['qtable'], loaded_frozen['eval_seed'])\nprint(f'Reloaded FrozenLake: {loaded_frozen_mean:.2f} +/- {loaded_frozen_std:.2f}')\nfrom PIL import ImageOps\nleft_frames, _ = _episode_frames(frozen_env, loaded_frozen['qtable'], (16,54,165), 99)\nright_frames, _ = _episode_frames(env, loaded_taxi['qtable'], (16,54,165), 99)\ncombined_path = RUNS / 'videos/all-agents-1m.mp4'\ncount = max(len(left_frames), len(right_frames))\nwith imageio.get_writer(combined_path, fps=4, codec='libx264', macro_block_size=2) as writer:\n    for index in range(count):\n        canvas = Image.new('RGB', (1040, 600), 'white')\n        left = ImageOps.contain(Image.fromarray(left_frames[min(index, len(left_frames)-1)]), (500, 530))\n        right = ImageOps.contain(Image.fromarray(right_frames[min(index, len(right_frames)-1)]), (500, 530))\n        canvas.paste(left, ((520-left.width)//2, 50+(530-left.height)//2))\n        canvas.paste(right, (520+(520-right.width)//2, 50+(530-right.height)//2))\n        draw = ImageDraw.Draw(canvas)\n        draw.text((20, 20), 'FrozenLake Q-learning agent', fill='black')\n        draw.text((540, 20), 'Taxi Q-learning agent', fill='black')\n        writer.append_data(np.asarray(canvas))\nprint('Combined demonstration:', combined_path, '| frames:', count)\ndisplay(Video(str(combined_path), embed=True))\nfrozen_env.close()\nenv.close()",
    }

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            cell.source = code[index]
            cell.outputs = []
            cell.execution_count = None
        elif index == 3:
            cell.source = '## Objectives\n\nUse Gymnasium, implement tabular Q-learning, train FrozenLake and Taxi agents, save local replay videos, and publish the trained Q-tables to Hugging Face.'
        elif index == 11:
            cell.source = '## Local setup\n\nInstall `requirements.txt` once, then run this notebook with Python from the `Notebook/notebooks` directory. RGB array rendering works headlessly.'
        elif index == 14:
            cell.source = 'The local kernel remains active throughout training.'
        elif index == 62:
            cell.source = '## Save the trained agents\n\nThe following cells save local model packages and replay videos. The upload script publishes the packages after the notebook finishes.'
        elif index == 64:
            cell.source = 'Define the local model packaging helpers.'
        elif index == 68:
            cell.source = 'The package helper saves the Q-table, evaluation result, model card, and replay video locally. The upload script publishes only the Q-table model file.'
        elif index == 69:
            cell.source = 'To publish after training, use `python Notebook/scripts/upload_to_hub.py` with a Hugging Face login or `HF_TOKEN`.'
        elif index == 71:
            cell.source = 'The upload script checks that the authenticated account matches the Unit 1 account, `tashobi02`.'
        elif index == 72:
            cell.source = 'Store the FrozenLake hyperparameters and Q-table in a model dictionary.'
        elif index == 74:
            cell.source = 'Save a local FrozenLake model package and replay video.'
        elif index == 77:
            cell.source = 'FrozenLake is saved. Next, train a second Q-learning agent in Taxi.'
        elif index == 89:
            cell.source = '## Save the Taxi model\n\nStore the Taxi hyperparameters and Q-table before packaging.'
        elif index == 92:
            cell.source = 'The Taxi model and replay are saved locally. The upload script publishes both agents after the notebook run.'
        elif index == 93:
            cell.source = '# Part 3: Reload saved local models\n\nCheck that both saved Q-tables evaluate correctly and render a joint demonstration.'
        elif index == 94:
            cell.source = 'The local package uses the same `q-learning.pkl` filename as the Hub model.'
        elif index == 95:
            cell.source = 'Define a local loader and verify the saved models.'
        elif index == 97:
            cell.source = 'Reload the Taxi model, then the FrozenLake model, and create the combined demonstration video.'
    notebook.metadata.pop('colab', None)
    notebook.metadata.pop('gpuClass', None)
    notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    DEST.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, DEST)
    print(DEST)


if __name__ == '__main__':
    main()
