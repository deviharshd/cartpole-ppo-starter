"""
Train a PPO agent on the CartPole-v1 environment using Stable Baselines3.

This script:
- Creates a Gymnasium CartPole-v1 environment.
- Wraps it in a Monitor and DummyVecEnv for Stable Baselines3.
- Trains a PPO policy for a configurable number of timesteps.
- Saves the trained model to ``cartpole/models/``.
- Reads the Monitor log file to generate a reward curve plot.
- Saves the plot both to:
  - ``cartpole/training_plot.png``
  - ``images/cartpole_graph.png``

Usage (from the project root):

    python cartpole/cartpole_ppo_train.py --timesteps 200000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


@dataclass
class PathConfig:
    """
    Simple container for frequently used paths.

    Attributes
    ----------
    script_dir : Path
        Directory containing this script.
    project_root : Path
        Root directory of the project.
    logs_dir : Path
        Directory where Monitor logs will be written.
    models_dir : Path
        Directory where trained models are stored.
    images_dir : Path
        Directory where plots are stored.
    """

    script_dir: Path
    project_root: Path
    logs_dir: Path
    models_dir: Path
    images_dir: Path


def create_directories() -> PathConfig:
    """
    Create and return important directories used by the script.

    Returns
    -------
    PathConfig
        Dataclass containing common paths.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    logs_dir = script_dir / "logs"
    models_dir = script_dir / "models"
    images_dir = project_root / "images"

    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    return PathConfig(
        script_dir=script_dir,
        project_root=project_root,
        logs_dir=logs_dir,
        models_dir=models_dir,
        images_dir=images_dir,
    )


def make_env(logs_dir: Path):
    """
    Build an environment creation function suitable for DummyVecEnv.

    Parameters
    ----------
    logs_dir : Path
        Directory where Monitor will write episode statistics.

    Returns
    -------
    callable
        A callable that creates a Monitor-wrapped Gymnasium environment.
    """

    def _init():
        """
        Inner function that actually constructs the environment instance.
        """
        env = gym.make("CartPole-v1")
        # Wrap in Monitor to log episode statistics to a CSV file.
        env = Monitor(env, filename=str(logs_dir / "monitor.csv"))
        return env

    return _init


def load_monitor_csv(logs_dir: Path) -> pd.DataFrame:
    """
    Load the Monitor CSV file into a pandas DataFrame.

    The Monitor wrapper writes a CSV file with metadata header lines
    starting with '#', followed by rows containing episode statistics.

    Parameters
    ----------
    logs_dir : Path
        Directory containing the Monitor CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with episode rewards and lengths.
        Columns: 'episode_reward', 'episode_length', 'time'.

    Raises
    ------
    FileNotFoundError
        If the monitor file does not exist.
    """
    monitor_path = logs_dir / "monitor.csv"
    if not monitor_path.exists():
        raise FileNotFoundError(
            f"Monitor log not found at {monitor_path}. "
            "Ensure training completed and the environment is wrapped with Monitor.",
        )

    # Monitor files include comment lines with metadata; skip them.
    with monitor_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    data_lines = [line for line in lines if not line.startswith("#")]

    csv_buffer = StringIO("".join(data_lines))
    df = pd.read_csv(csv_buffer)

    # Stable Baselines3's Monitor usually uses 'r', 'l', 't'.
    rename_map: Dict[str, str] = {}
    if "r" in df.columns:
        rename_map["r"] = "episode_reward"
    if "l" in df.columns:
        rename_map["l"] = "episode_length"
    if "t" in df.columns:
        rename_map["t"] = "time"
    df = df.rename(columns=rename_map)

    return df


def plot_training_rewards(
    monitor_df: pd.DataFrame,
    output_paths: List[Path],
    rolling_window: int = 10,
) -> None:
    """
    Plot episode rewards over time and save the figure.

    Parameters
    ----------
    monitor_df : pandas.DataFrame
        DataFrame containing episode statistics (with 'episode_reward' column).
    output_paths : list of Path
        List of locations where the PNG plot will be saved.
    rolling_window : int, optional
        Window size for computing the rolling mean reward, by default 10.
    """
    if "episode_reward" not in monitor_df.columns:
        raise ValueError("Monitor DataFrame must contain 'episode_reward' column.")

    episode_rewards = monitor_df["episode_reward"].to_numpy()
    episodes = np.arange(1, len(episode_rewards) + 1)

    rolling_rewards = pd.Series(episode_rewards).rolling(rolling_window).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_rewards, label="Episode reward", alpha=0.4)
    plt.plot(episodes, rolling_rewards, label=f"Rolling mean ({rolling_window} eps)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole PPO Training Rewards")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        print(f"[INFO] Saved training curve to {path}")

    plt.close()


@dataclass
class TrainConfig:
    """
    Configuration for CartPole PPO training.

    Making this explicit keeps experiments easy to log and repeat.
    """

    total_timesteps: int = 200_000
    model_name: str = "cartpole_ppo_model"
    seed: int | None = 0
    learning_rate: float = 3e-4
    batch_size: int = 64


def train_cartpole(config: TrainConfig) -> None:
    """
    Train a PPO agent on the CartPole-v1 environment.

    Parameters
    ----------
    config : TrainConfig
        Training configuration.
    """
    paths = create_directories()

    if config.seed is not None:
        print(f"[INFO] Using random seed {config.seed}")

    # Vectorized environment with a single Monitor-wrapped CartPole.
    env_fn = make_env(paths.logs_dir)
    vec_env = DummyVecEnv([env_fn])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=config.seed,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
    )

    print(f"[INFO] Starting training for {config.total_timesteps} timesteps...")
    model.learn(total_timesteps=config.total_timesteps)

    model_path = paths.models_dir / f"{config.model_name}.zip"
    model.save(model_path)
    print(f"[INFO] Saved trained model to {model_path}")

    # Close environment to flush Monitor logs.
    vec_env.close()

    # Generate training plot.
    try:
        monitor_df = load_monitor_csv(paths.logs_dir)
        output_paths = [
            paths.script_dir / "training_plot.png",
            paths.images_dir / "cartpole_graph.png",
        ]
        plot_training_rewards(monitor_df, output_paths)
    except (FileNotFoundError, ValueError) as error:
        print(f"[WARN] Could not generate training plot: {error}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with 'timesteps', 'seed', and 'model_name'.
    """
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on CartPole-v1 and save reward plots.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total timesteps for training (default: 200000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training (default: 0).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cartpole_ppo_model",
        help="Base name for the saved model (default: cartpole_ppo_model).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for PPO (default: 3e-4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for PPO updates (default: 64).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the script.

    Parses arguments and starts training.
    """
    args = parse_args()
    config = TrainConfig(
        total_timesteps=args.timesteps,
        model_name=args.model_name,
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    train_cartpole(config)


if __name__ == "__main__":
    main()
