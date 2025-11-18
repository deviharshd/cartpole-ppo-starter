# Reinforcement Learning Basics

This note is a compact reference for core reinforcement learning concepts used in this project and in typical locomotion research.

---

## 1. Markov Decision Process (MDP)

An RL problem is often modeled as a **Markov Decision Process (MDP)**, defined by:

- **State space** \\(\\mathcal{S}\\): all possible states of the environment.
- **Action space** \\(\\mathcal{A}\\): all possible actions the agent can take.
- **Transition function** \\(P(s' \\mid s, a)\\): probability of reaching state \\(s'\\) from state \\(s\\) after action \\(a\\).
- **Reward function** \\(r(s, a, s')\\): scalar feedback for transitioning from \\(s\\) to \\(s'\\) via action \\(a\\).
- **Discount factor** \\(\\gamma \\in [0,1]\\): how much future rewards are discounted.

The goal is to find a **policy** \\(\\pi(a \\mid s)\\) that maximizes expected **return**:

\\[
G_t = \\sum_{k=0}^{\\infty} \\gamma^k r_{t+k+1}
\\]

---

## 2. Policies, Value Functions, and Q-Functions

- **Policy** \\(\\pi(a \\mid s)\\):
  - Stochastic: gives a probability distribution over actions.
  - Deterministic: selects a specific action for each state.
- **State value function** \\(V^\\pi(s)\\):

  \\[
  V^\\pi(s) = \\mathbb{E}_\\pi [G_t \\mid s_t = s]
  \\]

- **Action value function (Q-function)** \\(Q^\\pi(s, a)\\):

  \\[
  Q^\\pi(s, a) = \\mathbb{E}_\\pi [G_t \\mid s_t = s, a_t = a]
  \\]

These functions measure how good it is to be in a state (or state-action pair) under a given policy.

---

## 3. On-Policy vs Off-Policy

- **On-policy** methods:
  - Learn about the policy that is **currently being executed**.
  - Example: PPO (used in this repo).
- **Off-policy** methods:
  - Learn about a target policy while following a different behavior policy.
  - Examples: DQN, DDPG, TD3, SAC.

For locomotion tasks:

- On-policy methods often work well but can be sample-inefficient.
- Off-policy methods can be more data-efficient but may require careful tuning.

---

## 4. Episodic Tasks and Returns

In this project, environments are **episodic**:

- An episode starts from an initial state.
- The agent interacts with the environment until a terminal condition is met.

Example: **CartPole-v1**

- The pole must stay balanced.
- Episode terminates when:
  - The pole angle exceeds a threshold.
  - The cart moves too far from the center.
  - A maximum number of steps is reached.

The **episode return** is the sum of rewards during the episode.

---

## 5. Function Approximation and Deep RL

For complex environments (like humanoid locomotion), tabular methods are infeasible.
Instead, we use **function approximation**:

- Neural networks approximate:
  - Policies (mapping states to actions).
  - Value functions (mapping states to expected returns).

Trade-offs:

- Expressive models (deep networks) can approximate complex mappings.
- But they can be unstable to train, especially with non-stationary targets.

PPO (and related algorithms) add stability via tricks such as:

- Clipped objective functions.
- Trust-region style constraints.
- Advantage normalization and baselines.

---

## 6. Exploration vs Exploitation

The agent must balance:

- **Exploration**: trying new actions to discover better rewards.
- **Exploitation**: choosing the best-known action to maximize reward.

Strategies include:

- Stochastic policies (as in PPO).
- Adding noise to actions.
- Intrinsic motivation or curiosity-based rewards.

For locomotion:

- Too little exploration leads to suboptimal gaits.
- Too much exploration destabilizes learning and may prevent convergence.

---

## 7. Locomotion-Specific Considerations

Locomotion tasks (e.g., humanoid walking) add extra complexity:

- High-dimensional continuous action spaces.
- Strong coupling between joints (e.g., balance and step placement).
- Contacts with the environment (foot-ground interaction, friction).
- Long time horizons and delayed rewards.

Important design choices:

- Observation space design (joint positions, velocities, forces, etc.).
- Action space (joint torques vs target angles vs high-level commands).
- Reward shaping (forward velocity, energy use, stability, style).

---

## 8. Connecting RL and Diffusion Models

Although this repository currently focuses on **RL**, future work might:

- Use **diffusion models** to generate motion trajectories.
- Use RL to refine or correct diffusion-generated motions.
- Compare:
  - Sample efficiency.
  - Stability and robustness.
  - Expressiveness of gaits / motion styles.

The core RL concepts in this document are the foundation for evaluating such comparisons rigorously.

