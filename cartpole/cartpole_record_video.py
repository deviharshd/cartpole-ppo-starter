"""
Record a video of a trained CartPole PPO policy using Gymnasium's RecordVideo.

Usage (from project root):

    python cartpole/cartpole_record_video.py --episodes 3

The videos will be stored under ``videos/cartpole/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO


def get_paths() -> dict:
    """Return paths used by this script."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    models_dir = script_dir / "models"
    videos_dir = project_root / "videos" / "cartpole"
    models_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return {
        "script_dir": script_dir,
        "project_root": project_root,
        "models_dir": models_dir,
        "videos_dir": videos_dir,
    }


def record_cartpole_video(
    model_name: str = "cartpole_ppo_model",
    episodes: int = 3,
) -> None:
    """
    Record one or more episodes of a trained PPO policy.

    Parameters
    ----------
    model_name : str
        Base name of the trained model (without extension).
    episodes : int
        Number of episodes to record.
    """
    paths = get_paths()
    models_dir: Path = paths["models_dir"]
    videos_dir: Path = paths["videos_dir"]

    model_path = models_dir / f"{model_name}.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it with cartpole_ppo_train.py.",
        )

    print(f"[INFO] Loading model from {model_path}")
    model = PPO.load(model_path)

    # Use RecordVideo wrapper to capture episodes.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(videos_dir),
        name_prefix="cartpole_ppo",
    )

    for episode_index in range(1, episodes + 1):
        observation, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        print(f"[INFO] Recorded episode {episode_index}: return = {total_reward:.2f}")

    env.close()
    print(f"[INFO] Videos saved under {videos_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Record CartPole PPO rollouts to video files.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to record (default: 3).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cartpole_ppo_model",
        help="Base name of the trained model (default: cartpole_ppo_model).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for recording videos."""
    args = parse_args()
    record_cartpole_video(
        model_name=args.model_name,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()

