"""Evaluation helpers for HalfCheetah-v5 PPO models.

No training happens here: the model is loaded, actions are deterministic,
and only rewards are recorded.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

PROJECT_DIR = Path(__file__).resolve().parent


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def evaluate_model(
    model_path: str | Path,
    env_id: str = "HalfCheetah-v5",
    episodes: int = 10,
    eval_seed: int = 1000,
    video_path: str | Path | None = None,
    screenshot_path: str | Path | None = None,
    max_video_steps: int = 300,
) -> dict:
    """Run deterministic evaluation and optionally save a video/screenshot."""

    model_path = _resolve(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(str(model_path), device="auto")
    need_render = video_path is not None or screenshot_path is not None
    env = gym.make(env_id, render_mode="rgb_array" if need_render else None)

    returns: list[float] = []
    lengths: list[int] = []
    frames: list[np.ndarray] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=eval_seed + episode)
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1

            if need_render and len(frames) < max_video_steps:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            done = terminated or truncated

        returns.append(episode_return)
        lengths.append(episode_length)

    env.close()

    saved_video = None
    if video_path is not None and frames:
        video_path = _resolve(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30)
        saved_video = str(video_path)

    saved_screenshot = None
    if screenshot_path is not None and frames:
        screenshot_path = _resolve(screenshot_path)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(screenshot_path, frames[-1])
        saved_screenshot = str(screenshot_path)

    return {
        "env": env_id,
        "model": str(model_path),
        "episodes": returns,
        "lengths": lengths,
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "video": saved_video,
        "screenshot": saved_screenshot,
    }
