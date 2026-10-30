"""PPO training entry point for Stage 0.

Run this file directly for a single default run, or use
``run_cartpole_3seeds.py`` from PyCharm to train three seeds.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

PROJECT_DIR = Path(__file__).resolve().parent


def make_env(env_id: str, seed: int):
    """Create one Gymnasium environment with a fixed reset seed."""

    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def build_vec_env(env_id: str, seed: int, monitor_path: Path):
    venv = DummyVecEnv([make_env(env_id, seed)])
    return VecMonitor(venv, filename=str(monitor_path))


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def train(
    env_id: str = "CartPole-v1",
    seed: int = 0,
    steps: int = 50_000,
    logdir: str | Path = "logs/cartpole/seed_0",
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    eval_seed: int = 12_345,
    device: str = "auto",
) -> Path:
    """Train one PPO model and return its log directory."""

    logdir = resolve_path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = build_vec_env(env_id, seed, logdir / "monitor.csv")
    eval_env = build_vec_env(env_id, eval_seed, logdir / "eval_monitor.csv")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(logdir / "best_model"),
        log_path=str(logdir),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, steps // 10),
        save_path=str(logdir / "checkpoints"),
        name_prefix="ppo",
        verbose=0,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        tensorboard_log=str(logdir / "tb"),
        seed=seed,
        device=device,
        verbose=1,
    )

    config = {
        "env_id": env_id,
        "seed": seed,
        "steps": steps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "lr": lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "eval_freq": eval_freq,
        "eval_episodes": eval_episodes,
        "eval_seed": eval_seed,
        "device": str(model.device),
        "gymnasium": gym.__version__,
        "torch": torch.__version__,
    }
    (logdir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"env={env_id} seed={seed} steps={steps}")
    print(f"gymnasium={gym.__version__} torch={torch.__version__} device={model.device}")
    print(f"logdir={logdir}")
    print("=" * 70)

    model.learn(total_timesteps=steps, callback=[eval_cb, ckpt_cb])
    model.save(logdir / "final_model")

    train_env.close()
    eval_env.close()

    print(f"saved: {logdir / 'final_model.zip'}")
    return logdir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--logdir", type=str, default="logs/cartpole/seed_0")
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--eval-freq", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-seed", type=int, default=12_345)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        env_id=args.env,
        seed=args.seed,
        steps=args.steps,
        logdir=args.logdir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        device=args.device,
    )


if __name__ == "__main__":
    print("start")
    main()
