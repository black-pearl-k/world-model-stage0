# World Model Stage 0 — PPO Warmup

This repository contains the Stage 0 reinforcement-learning warmup for a
world-model learning path.

## Experiments

| Experiment | Environment | Status |
|---|---|---|
| `ppo_cartpole/` | CartPole-v1 | Complete: 3 seeds, 500 ± 0 |
| `ppo_halfcheetah/` | HalfCheetah-v4/v5 | Planned |

## Current result

CartPole-v1, PPO, 3 seeds, 50k steps per seed:

```text
seed_0: 500
seed_1: 500
seed_2: 500
mean ± std: 500.0 ± 0.0
```

This is a ceiling effect: CartPole caps at 500, so it is a smoke test rather
than a sensitive benchmark.

See [`ppo_cartpole/README.md`](ppo_cartpole/README.md) for commands, results,
evaluation protocol, and notes.

## Environment

- Windows 11
- Python 3.10.21
- torch 2.14.1+cpu
- gymnasium 1.4.0
- stable-baselines3 2.9.2
- mujoco 3.17.0

## Repository policy

Training artifacts are intentionally not committed:

- `logs/`
- model checkpoints (`*.zip`, `*.pt`, `*.pth`, `*.onnx`, `*.ckpt`)
- videos (`*.mp4`, `*.avi`, `*.mov`)

Small result files (JSON/CSV/PNG) and screenshots are kept for reproducibility.
