---
tags:
- reinforcement-learning
- q-learning
- FrozenLake-v1
---
# Q-learning agent for FrozenLake-v1

Trained for 1,000,000 environment steps with tabular Q-learning.
Mean reward over 100 evaluation episodes: 1.00 ± 0.00.

Load `q-learning.pkl` with Python pickle from a trusted source. Create the environment with `gym.make('FrozenLake-v1', **{'map_name': '4x4', 'is_slippery': False})` and act with `np.argmax(model['qtable'][state])`.
