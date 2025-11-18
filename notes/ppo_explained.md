# Proximal Policy Optimization (PPO) Explained

This document provides an intuitive explanation of Proximal Policy Optimization (PPO), the algorithm used for the CartPole example in this repository and commonly used for locomotion tasks.

---

## 1. Motivation

Policy gradient methods directly optimize a stochastic policy \\(\\pi_\\theta(a \\mid s)\\) by adjusting parameters \\(\\theta\\) to maximize expected return.

However:

- Naive gradient ascent can be unstable.
- Large policy updates can **collapse performance**.
- Trust Region Policy Optimization (TRPO) proposes restricting updates using a KL-divergence constraint, but it is complex to implement.

PPO aims to be:

- Easier to implement than TRPO.
- More stable than naive policy gradients.

---

## 2. Core Idea: Clipped Objective

Let:

- \\(\\pi_\\theta\\): new policy.
- \\(\\pi_{\\theta_{old}}\\): old policy (before the update).
- \\(A_t\\): estimated advantage at time step \\(t\\).

Define the probability ratio:

\\[
r_t(\\theta) = \\frac{\\pi_\\theta(a_t \\mid s_t)}{\\pi_{\\theta_{old}}(a_t \\mid s_t)}.
\\]

The vanilla policy gradient objective is roughly:

\\[
L^{PG}(\\theta) = \\mathbb{E}_t [ r_t(\\theta) A_t ].
\\]

PPO introduces a **clipped** surrogate objective:

\\[
L^{CLIP}(\\theta) =
\\mathbb{E}_t \\left[
\\min\\left(
r_t(\\theta) A_t,\n
\\text{clip}(r_t(\\theta), 1 - \\epsilon, 1 + \\epsilon) A_t
\\right)
\\right]
\\]

Where:

- \\(\\epsilon\\) is a small hyperparameter (e.g., 0.1 or 0.2).
- The `min` ensures that the objective does not grow too large when the policy update is overly aggressive.

Intuition:

- If the new policy improves too much relative to the old policy, the clipping prevents large steps.
- This keeps policy updates **proximal** (close) to the previous policy.

---

## 3. Advantage Estimation (GAE)

PPO commonly uses **Generalized Advantage Estimation (GAE)**:

- Provides a trade-off between bias and variance in advantage estimates.
- Uses a parameter \\(\\lambda \\in [0, 1]\\) to control this trade-off.

Roughly:

\\[
A_t = \\delta_t + (\\gamma \\lambda) \\delta_{t+1} + (\\gamma \\lambda)^2 \\delta_{t+2} + \\dots
\\]

where:

\\[
\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t).
\\]

Benefits:

- Lower variance advantages.
- Smoother learning.

---

## 4. Value Function and Entropy Bonus

The full PPO loss usually includes:

1. **Policy loss** (clipped objective).
2. **Value function loss**:

   \\[
   L^{VF}(\\theta) = \\mathbb{E}_t \\left[ (V_\\theta(s_t) - V_t^{target})^2 \\right].
   \\]

3. **Entropy bonus**:

   Encourages exploration by maximizing policy entropy:

   \\[
   L^{ENT}(\\theta) = \\mathbb{E}_t [H(\\pi_\\theta(\\cdot \\mid s_t))].
   \\]

The total objective becomes something like:

\\[
L^{PPO}(\\theta) = \\mathbb{E}_t \\left[
L^{CLIP}(\\theta)
- c_1 L^{VF}(\\theta)
 + c_2 L^{ENT}(\\theta)
\\right].
\\]

Where \\(c_1\\) and \\(c_2\\) are weights.

---

## 5. Mini-batches and Epochs

PPO:

- Collects a batch of trajectories using the current policy.
- Splits the batch into mini-batches.
- For several epochs:
  - Performs gradient updates on those mini-batches.

This effectively reuses data more than pure on-policy methods,
while still being considered **on-policy** due to the clipping mechanism.

---

## 6. PPO in Locomotion

PPO is widely used in locomotion tasks (e.g., Mujoco, PyBullet, DeepMind Control Suite) because:

- It handles **continuous action spaces** naturally.
- The clipped objective and GAE provide stable learning in high-dimensional settings.
- It works well with neural-network policies that parameterize joint torques or target angles.

However:

- It can require **large amounts of data**.
- Hyperparameters (learning rate, batch size, clip range, etc.) matter a lot.

---

## 7. PPO in this Repository

In this project:

- `cartpole/cartpole_ppo_train.py` uses PPO from **Stable Baselines3**.
- Stable Baselines3 handles:
  - Advantage estimation (GAE).
  - PPO loss computation.
  - Optimization details.

You only need to:

- Configure the environment.
- Specify the policy type (e.g., `MlpPolicy`).
- Choose hyperparameters, such as:
  - `total_timesteps`
  - `learning_rate`
  - `n_steps`, `batch_size`, `gamma`, etc. (via the PPO constructor).

---

## 8. Connection to Diffusion Models for Locomotion

For your thesis comparing **RL locomotion** vs **diffusion-based locomotion**:

- PPO learns a **policy function** mapping states to actions.
- Diffusion models typically learn a **trajectory distribution** in state or pose space.

Comparisons may focus on:

- Data requirements for achieving a given gait quality.
- Robustness to perturbations (e.g., pushes, terrain changes).
- Diversity and style of motions:
  - RL may converge to a single robust gait.
  - Diffusion models can sample many variations of motions conditioned on high-level goals.

PPO thus serves as a strong, well-understood baseline for learned locomotion.

