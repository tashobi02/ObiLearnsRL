# Unit 2 — Q-learning

Tabular Q-learning estimates the value of each action in each discrete state. After observing a transition \((s,a,r,s')\), it updates

\[
Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\right].
\]

For a terminal transition, the target is just \(r\). The greedy policy chooses \(\arg\max_a Q(s,a)\). During training, epsilon-greedy exploration chooses a random action with probability \(\epsilon\) and the greedy action otherwise. This is an off-policy method because the update uses a greedy target while the behavior policy continues to explore. The original algorithm and its convergence result are in [Watkins and Dayan (1992)](../Papers/watkins-dayan-q-learning-1992.pdf); Watkins's [1989 thesis](../Papers/watkins-learning-from-delayed-rewards-1989.pdf) develops the delayed-reward setting. Sutton's [temporal-difference paper](../Papers/sutton-temporal-differences-1988.pdf) provides the broader bootstrapping context.

## Environments

- **FrozenLake-v1:** The 4×4 non-slippery map has 16 states and four actions. Reaching the goal gives +1. Its deterministic dynamics make a small Q-table sufficient.
- **Taxi-v3:** The passenger pickup and delivery task has 500 states and six actions. A legal delivery gives +20, each ordinary step costs −1, and illegal pickup or drop-off costs −10.

The [executed notebook](../Notebook/notebooks/unit2_executed.ipynb) trains each agent for exactly 1,000,000 environment steps, evaluates on 100 episodes, and saves the Q-tables and demonstration videos. The saved model is a Python pickle; load only from a trusted source.

## Limits

A Q-table grows with the number of state-action pairs. It works here because both spaces are small and discrete. It does not scale directly to large image observations. Fixed learning rate and epsilon floor also mean the practical run does not itself establish the convergence conditions from the theory paper.
