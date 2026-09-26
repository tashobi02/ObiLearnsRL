# Bonus Unit 1 Papers

These papers support the [Huggy training notes](../Notes/Huggy.md). Read them in this order:

| Order | Paper | Why it matters for Huggy | Source |
|---:|---|---|---|
| 1 | [Schulman et al., *Proximal Policy Optimization Algorithms* (2017)](schulman-ppo-2017.pdf) | Explains the PPO trainer and clipped policy updates used in `Huggy.yaml`. | [arXiv](https://arxiv.org/abs/1707.06347) · [PDF](https://arxiv.org/pdf/1707.06347) |
| 2 | [Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015 preprint; ICLR 2016)](schulman-gae-2015.pdf) | Explains advantage estimation for continuous control and the `lambd` parameter in the PPO config. | [arXiv](https://arxiv.org/abs/1506.02438) · [PDF](https://arxiv.org/pdf/1506.02438) |
| 3 | [Juliani et al., *Unity: A General Platform for Intelligent Agents* (2018 preprint)](juliani-unity-ml-agents-2018.pdf) | Describes the Unity ML-Agents platform that runs Huggy's simulation and training interface. | [arXiv](https://arxiv.org/abs/1809.02627) · [PDF](https://arxiv.org/pdf/1809.02627) |
| 4 | [Ng, Harada & Russell, *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping* (1999)](ng-reward-shaping-1999.pdf) | Gives the theory behind adding guidance rewards, useful when evaluating Huggy's orientation bonus and penalties. | [Author PDF](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) |

The reward-shaping paper describes conditions for preserving an optimal policy; it does not establish that Huggy's specific reward function satisfies them. The PPO paper is also available in [Unit 1's paper folder](../../Unit%201/Papers/README.md); a copy is kept here so this unit's reading set is self-contained.
