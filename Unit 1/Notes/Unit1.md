# Unit 1 — Fundamentals of RL and Deep RL

> My personal study notes for the Hugging Face Deep RL Course. Cross-references papers collected in [`../Papers/`](../Papers/).

---

## 1. What is Reinforcement Learning?

Reinforcement Learning (RL) is a computational approach to **learning from interaction**.
An **agent** takes **actions** in an **environment** to maximize a cumulative **reward** signal. It is fundamentally *trial-and-error* — there is no supervisor telling the agent the correct action, only a scalar reward that says how well it is doing.

The **reward hypothesis** (Sutton & Barto, 2018):

> *"All goals can be described as the maximization of the expected cumulative reward."*

So whatever the objective (win a chess game, walk, drive, land a spaceship), we frame it as maximizing a scalar signal over time.

**RL vs. the other ML paradigms**

| Paradigm | Feedback signal | Goal |
|---|---|---|
| Supervised learning | label (correct answer) per example | generalize input → output mapping |
| Unsupervised learning | none | find structure in data |
| **Reinforcement learning** | **reward (delayed, possibly sparse)** | **learn a policy that maximizes expected return** |

The "delayed, possibly sparse" part is what makes RL qualitatively different.

---

## 2. The RL process / loop

At every timestep *t*:

```
                    action A_t
   Agent  ─────────────────────────►  Environment
      ▲                                 │
      │       state S_{t+1}, reward R_{t+1}
      └─────────────────────────────────┘
```

Formally we have a **Markov Decision Process** (MDP):
- $S_t$ — current state of the environment
- $A_t$ — action chosen by the agent
- $R_{t+1}$ — scalar reward received after taking action $A_t$
- $S_{t+1}$ — next state
- $\gamma \in [0, 1]$ — discount factor

An **episode** is one full trajectory: $S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_T$.

The **return** is the discounted sum of future rewards from time *t*:

$$
G_t \;=\; R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \;=\; \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

**Why discount?** The future is uncertain; immediate rewards are usually more predictable. Mathematically, discounting guarantees finite returns for infinite-horizon problems and stabilizes value estimation.

---

## 3. The MDP formalism

An MDP is the tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$:

- $\mathcal{S}$ — set of states
- $\mathcal{A}$ — set of actions
- $P(s'|s, a)$ — transition probability of going to $s'$ after taking $a$ in $s$
- $R(s, a)$ — expected reward for taking $a$ in $s$
- $\gamma$ — discount factor

The **Markov property** says the next state depends only on the current (state, action), not the full history.

---

## 4. Two families of RL methods

There are two broad ways to find a good policy $\pi(a|s)$:

1. **Policy-based methods** — directly parameterize the policy $\pi_\theta(a|s)$ and optimize its parameters (usually via gradient ascent on expected return). Examples: REINFORCE (Williams, 1992), PPO (Schulman et al., 2017), A3C (Mnih et al., 2016).
2. **Value-based methods** — learn a value function $V_\pi(s)$ or action-value $Q_\pi(s,a)$, and derive a policy from it (e.g. greedy in Q). Example: DQN (Mnih et al., 2013).

When the policy or value function is represented by a **neural network**, we call it **Deep RL** — this is the modern paradigm since ~2013.

Most modern methods are **actor-critic**: they combine a learned policy (actor) with a learned value function (critic). PPO and A3C are actor-critic algorithms.

---

## 5. Exploration vs. exploitation

To learn, the agent has to **try actions whose outcomes it doesn't fully know**. This is the **exploration–exploitation tradeoff**:

- *Exploit*: take the best-known action so far.
- *Explore*: try a novel action to discover a possibly better strategy.

Common mechanisms:
- **ε-greedy** (value-based): with probability ε take a random action.
- **Entropy bonus** (policy-based, used by PPO via the `ent_coef` hyperparameter): reward the policy for keeping its action distribution spread out, encouraging exploration.

---

## 6. PPO — Proximal Policy Optimization

PPO (Schulman et al., 2017) is an **on-policy actor-critic** algorithm that became one of the default workhorses of modern RL.

The key idea: improve the policy a little bit at a time, so the new policy doesn't stray too far from the old one. This is implemented with a **clipped surrogate objective**:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big[\, \min\big( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, A_t \big) \,\Big]
$$

where

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio of the new vs. old policy,
- $A_t$ is an **advantage estimate** (how much better was the action vs. average at that state), and
- $\epsilon \approx 0.1\text{–}0.3$ is the clipping range.

**Intuition**: the `min(...)` and `clip(...)` together ensure that if the new policy tries to move $r_t(\theta)$ outside $[1-\epsilon, 1+\epsilon]$, the gradient stops pushing it further. This keeps policy updates safe.

PPO is popular because it is:
- **Simple** to implement (only a clipped loss, no second-order methods).
- **Stable** across many hyperparameter settings.
- **Sample-efficient enough** for most tasks people actually want to try.

**Why on-policy?** PPO learns from rollouts collected by the *current* policy, then discards them after the update. Off-policy methods (DQN, SAC) reuse old data, which can be more efficient but trickier.

---

## 7. Stable-Baselines3 (SB3)

SB3 ([paper](../Papers/Stable-Baseline3.pdf)) is a set of reliable PyTorch implementations of major RL algorithms (PPO, DQN, SAC, A2C, TD3, …) maintained by the same group as the original Stable Baselines.

Key abstractions we'll touch in the notebook:

- **`gymnasium.Env`** — unified env interface (`reset()`, `step(action)` returns `(obs, reward, terminated, truncated, info)`).
- **`DummyVecEnv` / `SubprocVecEnv`** — run multiple envs in parallel. PPO uses these for batched rollouts.
- **`Monitor`** — wraps an env to log episode returns/rewards.
- **`evaluate_policy`** — run a greedy evaluation over N episodes and report mean ± std reward.

That last one is what we use to measure whether the agent has actually learned anything.

---

## 8. The LunarLander environment

[`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/) is a 2D physics simulation where the agent pilots a lander to a small landing pad.

- **Observation** (8-dim continuous vector):
  - $x, y$ — horizontal / vertical position
  - $\dot x, \dot y$ — horizontal / vertical velocity
  - $\theta, \dot\theta$ — angle and angular velocity
  - left leg contact, right leg contact (booleans)
- **Action** (discrete, 4 options):
  - 0: do nothing
  - 1: fire left orientation engine
  - 2: fire main engine
  - 3: fire right orientation engine
- **Reward shaping** per step:
  - closer/further to pad → ±
  - slower/faster → ±
  - more tilted → penalty
  - each leg on ground → +10
  - side engine firing → -0.03 / frame
  - main engine firing → -0.3 / frame
  - +100 bonus for safe landing, -100 for crashing
- **Success** = mean episode reward ≥ **200 points** over 10 evaluation episodes.

This is a classic control problem with a shaped reward — well-suited to PPO.

---

## 9. Hyperparameters we used (and why)

| Hyperparameter | Value | Why |
|---|---|---|
| `n_steps` | 1024 | Rollout length per env before each update (16 envs × 1024 ≈ 16k samples per update) |
| `batch_size` | 64 | Mini-batch size for the 4 inner epochs |
| `n_epochs` | 4 | Number of passes over the collected rollout per update |
| `gamma` (γ) | 0.999 | Long-horizon discount — matches the slow dynamics of a lander approach |
| `gae_lambda` | 0.98 | Generalized Advantage Estimation smoothing |
| `ent_coef` | 0.01 | Small entropy bonus to keep exploration alive |

These are tuned for LunarLander; for harder envs you'll typically want more samples / longer training.

---

## 10. The bigger picture — what's next

Unit 1 introduced:
- The RL loop and MDP formalism.
- The policy-based vs. value-based split.
- A first practical agent (PPO on LunarLander-v3).

Subsequent units will dive into:
- **Value-based methods** — DQN, experience replay, target networks.
- **Improvements to PPO** — GAE, normalization, multi-step returns.
- **Continuous control** — SAC, TD3, DDPG.
- **Multi-agent** and **environment design**.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Policy** $\pi(a\|s)$ | Distribution over actions given the current state. The agent's "brain". |
| **Value function** $V_\pi(s)$ | Expected return starting from $s$ and following $\pi$. |
| **Action-value** $Q_\pi(s,a)$ | Expected return after taking $a$ in $s$ and then following $\pi$. |
| **Advantage** $A(s,a) = Q(s,a) - V(s)$ | How much better than average is this action at this state? |
| **Rollout** | A trajectory collected by running a policy in the env. |
| **On-policy** | Learns from data collected by the *current* policy (PPO, A3C). |
| **Off-policy** | Can learn from data collected by *any* policy (DQN, SAC). |
| **Sample efficiency** | How many environment samples are needed to reach a given performance. |
| **Discount γ** | Factor reducing the weight of future rewards. |
| **Entropy bonus** | An objective term rewarding uncertain policies to keep exploration alive. |

---

## 12. Pointers — papers in [`../Papers/`](../Papers/)

- [Mnih et al., DQN (2013)](../Papers/mnih-dqn-2013.pdf) — deep value learning.
- [Mnih et al., A3C (2016)](../Papers/mnih-a3c-2016.pdf) — asynchronous actor-critic learning.
- [Williams, REINFORCE (1992)](../Papers/williams-reinforce-1992.pdf) — a foundational policy-gradient method.
- [Schulman et al., TRPO (2015)](../Papers/schulman-trpo-2015.pdf) — trust-region policy updates.
- [Schulman et al., PPO (2017)](../Papers/schulman-ppo-2017.pdf) — the algorithm trained in the notebook.
- [Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.](../Papers/sutton-barto-RL-2nd-ed.pdf) — a broad reference for RL.
- [Raffin et al., Stable-Baselines3 (2020 preprint)](../Papers/Stable-Baseline3.pdf) — the library used in the notebook.

See the [Papers reading list](../Papers/README.md) for source links and the suggested order.

> After reading these notes once, follow the order in the [Papers index](../Papers/README.md): Sutton & Barto → DQN → REINFORCE → A3C → TRPO → PPO, then the Stable-Baselines3 paper.
