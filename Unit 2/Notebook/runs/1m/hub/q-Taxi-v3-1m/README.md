---
tags:
- reinforcement-learning
- q-learning
- Taxi-v3
---
# Q-learning agent for Taxi-v3

Trained for 1,000,000 environment steps with tabular Q-learning.
Mean reward over 100 evaluation episodes: 7.56 ± 2.71.

Load `q-learning.pkl` with Python pickle from a trusted source. Create the environment with `gym.make('Taxi-v3', **{})` and act with `np.argmax(model['qtable'][state])`.
