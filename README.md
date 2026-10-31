# 世界模型 Stage 0 — PPO 热身

本仓库包含世界模型学习路径中 Stage 0 的强化学习热身内容。

## 实验

| 实验 | 环境 | 状态 |
|---|---|---|
| `ppo_cartpole/` | CartPole-v1 | 已完成：3 个 seed，500 ± 0 |
| `ppo_halfcheetah/` | HalfCheetah-v5 | 脚本已就绪，待训练 |

## 当前结果

CartPole-v1，PPO，3 个 seed，每个 seed 训练 5 万步：

```text
seed_0: 500
seed_1: 500
seed_2: 500
mean ± std: 500.0 ± 0.0
```

这是封顶效应：CartPole 最高只能到 500，所以它更像一个 smoke test，而不是敏感的基准任务。

运行命令、结果、评测协议和注意事项见 [`ppo_cartpole/README.md`](ppo_cartpole/README.md)。

## 环境

- Windows 11
- Python 3.10.21
- torch 2.14.1+cpu
- gymnasium 1.4.0
- stable-baselines3 2.9.2
- mujoco 3.17.0

## 仓库策略

训练产物不提交到仓库：

- `logs/`
- 模型 checkpoint（`*.zip`、`*.pt`、`*.pth`、`*.onnx`、`*.ckpt`）
- 视频（`*.mp4`、`*.avi`、`*.mov`）

小型结果文件（JSON/CSV/PNG）和截图会保留，便于复现。
