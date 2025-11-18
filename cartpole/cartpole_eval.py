"""
Evaluate a trained PPO policy on CartPole-v1 and report statistics.

Usage (from project root):

    python cartpole/cartpole_eval.py --episodes 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""

    model_name: str = "cartpole_ppo_model"
    episodes: int = 20
    render_mode: str | None = None  # use "human" if you want to watch


def get_model_path(model_name: str) -> Path:
    """
    Build the path to a saved model in cartpole/models.

    Parameters
    ----------
    model_name : str
        Base name of the saved model (without extension).
    """
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find model at {model_path}. "
            "Train it first with cartpole_ppo_train.py.",
        )
    return model_path


def evaluate_policy(config: EvalConfig) -> List[float]:
    """
    Evaluate a trained PPO policy on CartPole-v1.

    Parameters
    ----------
    config : EvalConfig
        Evaluation configuration.

    Returns
    -------
    list of float
        Episode returns for each evaluation episode.
    """
    model_path = get_model_path(config.model_name)
    print(f"[INFO] Loading model from {model_path}")
    model = PPO.load(model_path)

    env = gym.make(
        "CartPole-v1",
        render_mode=config.render_mode,
    )

    episode_returns: List[float] = []
    for episode_index in range(1, config.episodes + 1):
        observation, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        episode_returns.append(total_reward)
        print(f"[INFO] Episode {episode_index}: return = {total_reward:.2f}")

    env.close()
    return episode_returns


def summarize_returns(returns: List[float]) -> None:
    """
    Print summary statistics for a list of episode returns.

    Parameters
    ----------
    returns : list of float
        Episode returns.
    """
    rewards = np.array(returns, dtype=float)
    print("\n[INFO] Evaluation summary")
    print(f"  Episodes          : {len(rewards)}")
    print(f"  Mean return       : {rewards.mean():.2f}")
    print(f"  Std of returns    : {rewards.std():.2f}")
    print(f"  Min / max returns : {rewards.min():.2f} / {rewards.max():.2f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy on CartPole-v1.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cartpole_ppo_model",
        help="Base name of the trained model (default: cartpole_ppo_model).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment while evaluating.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for command-line usage."""
    args = parse_args()
    config = EvalConfig(
        model_name=args.model_name,
        episodes=args.episodes,
        render_mode="human" if args.render else None,
    )
    returns = evaluate_policy(config)
    summarize_returns(returns)


if __name__ == "__main__":
    main()

