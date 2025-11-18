"""
Visualize a trained PPO policy on the CartPole-v1 environment.

This script:
- Loads the model saved by ``cartpole_ppo_train.py``.
- Creates a renderable Gymnasium CartPole-v1 environment.
- Runs a configurable number of evaluation episodes.
- Prints episode rewards and renders the environment.

Usage (from the project root):

    python cartpole/cartpole_ppo_infer.py --episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


def get_paths() -> dict:
    """
    Compute and return important filesystem paths.

    Returns
    -------
    dict
        A dictionary with keys:
        - "script_dir": Directory containing this script.
        - "models_dir": Directory where models are stored.
    """
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return {"script_dir": script_dir, "models_dir": models_dir}


def run_inference(
    model_name: str = "cartpole_ppo_model",
    episodes: int = 5,
    render_mode: str = "human",
) -> None:
    """
    Run inference using a trained PPO model on CartPole-v1.

    Parameters
    ----------
    model_name : str, optional
        Base name of the model (without extension), by default "cartpole_ppo_model".
    episodes : int, optional
        Number of evaluation episodes, by default 5.
    render_mode : str, optional
        Gymnasium render mode, by default "human".
    """
    paths = get_paths()
    models_dir: Path = paths["models_dir"]

    model_path = models_dir / f"{model_name}.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Train the model first using cartpole_ppo_train.py.",
        )

    print(f"[INFO] Loading model from {model_path}")
    model = PPO.load(model_path)

    env = gym.make("CartPole-v1", render_mode=render_mode)

    for episode_index in range(1, episodes + 1):
        observation, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated

        print(f"[INFO] Episode {episode_index}: total reward = {episode_reward:.2f}")

    env.close()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with 'episodes' and 'model_name'.
    """
    parser = argparse.ArgumentParser(
        description="Visualize a trained PPO policy on CartPole-v1.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cartpole_ppo_model",
        help="Base name of the trained model (default: cartpole_ppo_model).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the script.

    Parses arguments and runs inference.
    """
    args = parse_args()
    run_inference(
        model_name=args.model_name,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()

