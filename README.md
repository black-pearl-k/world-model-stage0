# 世界模型 Stage 0 — PPO 热身

本仓库包含世界模型学习路径中 Stage 0 的强化学习热身内容。

## 实验

| 实验 | 环境 | 状态 |
|---|---|---|
| `ppo_cartpole/` | CartPole-v1 | 已完成：3 个 seed，500 ± 0 |
| `ppo_halfcheetah/` | HalfCheetah-v5 | 已完成 300K 3 seeds；步态/站立/0.3/0.8/1.5/1.95/2.0 m/s 实验代码已就绪 |

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

HalfCheetah-v5，PPO，3 个 seed，每个 seed 训练 30 万步，最终评测 10 局：

```text
seed_0: 976.48
seed_1: 1228.87
seed_2: 879.20
mean ± std: 1028.18 ± 180.48
```

HalfCheetah 的 300K 结果和“倒着滑”的奖励漏洞分析见 [`ppo_halfcheetah/README.md`](ppo_halfcheetah/README.md)。

原始 HalfCheetah 奖励训练的模型会倒着滑，说明继续增加训练步数不能解决奖励漏洞。下一步用奖励塑形让模型保持身体水平、避免躯干/头碰地，再重新训练。相关代码：

```text
ppo_halfcheetah/reward_wrappers.py
ppo_halfcheetah/run_halfcheetah_gait.py
ppo_halfcheetah/run_halfcheetah_stand.py
ppo_halfcheetah/run_halfcheetah_speed.py
ppo_halfcheetah/run_halfcheetah_fast.py
ppo_halfcheetah/run_halfcheetah_faster.py
ppo_halfcheetah/run_halfcheetah_native.py
ppo_halfcheetah/run_halfcheetah_final.py
```

`run_halfcheetah_stand.py` 使用 `StandRewardWrapper`，完全去掉原始前进奖励，先让模型学会身体水平、腿在下面、不用头/背滑行。

站立后再用 `run_halfcheetah_speed.py` 从站立模型继续，加入很小的目标速度 `0.3 m/s`。

速度稳定后用 `run_halfcheetah_fast.py` 从 0.3 m/s 模型继续，把目标提高到 `0.8 m/s`。

再用 `run_halfcheetah_faster.py` 从 0.8 m/s 模型继续，把目标提高到 `1.5 m/s`。

最后用 `run_halfcheetah_native.py` 从 1.5 m/s 模型继续，把目标提高到约 `1.95 m/s`。

最终用 `run_halfcheetah_final.py` 从 1.95 m/s 模型继续，把目标提高到 `2.0 m/s`。

运行方式、奖励公式和调参规则见 [`ppo_halfcheetah/README.md`](ppo_halfcheetah/README.md)。

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
