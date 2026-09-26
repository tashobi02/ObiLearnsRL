# Huggy: fetch the stick

## Task

Huggy is a Unity ML-Agents environment. The agent learns to move toward and fetch a stick by interacting with the simulation and receiving rewards.

## Observations and actions

The environment provides the agent with state information about Huggy and the target. The agent controls Huggy's leg joints through continuous actions. The observation and action spaces are defined by the Unity environment.

## Reward design

The reward encourages reaching the target while penalizing excessive rotation. This gives the policy a goal (fetch the stick) and discourages unproductive spinning.

## Training

The local notebook uses PPO through Unity ML-Agents. Its configuration is in [`Notebook/config/Huggy.yaml`](../Notebook/config/Huggy.yaml). It sets a 2,000,000-step limit, saves checkpoints every 200,000 steps, and uses a 50,000-step summary interval. Run instructions and the pinned Python environment are in [`Notebook/README.md`](../Notebook/README.md).

For the original environment and lesson walkthrough, see the [Huggy repository](https://github.com/huggingface/Huggy) and the [Hugging Face bonus lesson](https://huggingface.co/learn/deep-rl-course/unitbonus1/train).
