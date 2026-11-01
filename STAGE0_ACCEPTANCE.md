# Stage 0 验收说明

这份文档按阶段 0 的验收清单逐项对照，并记录本仓库和清单之间的少量差异。

## 结论

**实验和交付物已经完成，可以作为阶段 0 的成果交付。**

严格按字面逐条对照，有两处偏差：

```text
1. HalfCheetah 实际使用 v5，不是清单里的 v4。
2. 目录名是 stage_0/ppo_cartpole + ppo_halfcheetah，
   不是清单里的单个 rl-warmup/ 目录。
```

这两点不影响本阶段要练习的内容，但已经在下面明确记录。

## 验收对照表

| 验收项 | 状态 | 证据 |
|---|---|---|
| MDP、return、value、advantage | 已完成 | 本文第 1 节 |
| PPO、GAE、clip、advantage normalization | 已完成 | 本文第 2 节 |
| on-policy / off-policy | 已完成 | 本文第 3 节 |
| reward scale、seed | 已完成 | 本文第 4 节 |
| CartPole 3 个 seed | 已完成 | 500.0 ± 0.0 |
| HalfCheetah 3 个 seed | 已完成 | 300K：1028.18 ± 180.48 |
| 回报曲线 | 已完成 | `results/*_curves.png`、TensorBoard |
| TensorBoard 记录 | 已完成 | `logs/.../tb/` |
| train.py / eval.py / README.md | 已完成 | 两个子项目都有 |
| PPO clip 的解释 | 已完成 | 本文第 2 节 |
| GAE 的解释 | 已完成 | 本文第 2 节 |
| advantage normalization 的解释 | 已完成 | 本文第 2 节 |
| reward scale 的解释 | 已完成 | 本文第 4 节 |
| 3 个 seed 的均值和方差 | 已完成 | CartPole 和 HalfCheetah 都有 |
| 视频或 render 截图 | 已完成 | `videos/` |
| Dreamer 训练循环 | 已完成阅读说明 | 本文第 5 节，未实现 Dreamer |
| `rl-warmup/` 目录名 | 等价完成 | 使用 `stage_0/` 作为总目录 |
| HalfCheetah-v4 | 有偏差 | 实际使用 v5 |

## 1. MDP、return、value、advantage

一个 MDP 通常写成：

```text
(S, A, P, R, gamma)
```

含义是：

```text
S     状态集合
A     动作集合
P     状态转移概率
R     奖励函数
gamma 折扣因子，越接近 1 越重视长期回报
```

一次交互是：

```text
s_t --a_t--> r_t, s_{t+1}
```

Return 是从当前时刻开始的折扣奖励和：

```text
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
```

Value 是“从某个状态出发，按当前策略走下去，预期能拿到的 return”：

```text
V(s) = E[G_t | s_t = s]
```

Q 值则额外指定第一个动作：

```text
Q(s, a) = E[G_t | s_t = s, a_t = a]
```

Advantage 表示“这个动作比当前策略的平均水平好多少”：

```text
A(s, a) = Q(s, a) - V(s)
```

直觉：

```text
A > 0：这个动作比平均水平好，应该增加概率
A < 0：这个动作比平均水平差，应该降低概率
```

## 2. PPO、GAE、clip、advantage normalization

### PPO 的 clip

PPO 是 on-policy 算法。它用当前策略采样，然后做多次梯度更新。

新旧策略的概率比是：

```text
ratio = pi_new(a|s) / pi_old(a|s)
```

PPO 的 clipped objective 是：

```text
L = min(
    ratio * A,
    clip(ratio, 1 - epsilon, 1 + epsilon) * A
)
```

本项目使用：

```text
clip_range = 0.2
```

含义：

```text
如果 ratio 在 [0.8, 1.2] 内，正常使用
如果超出这个范围，梯度被截断
```

这样做的目的不是“让 ratio 永远不超过 1.2”，而是**避免一次更新把策略推得太远**。PPO 仍然可以做多次 epoch，但每次更新都有保守约束。

### GAE

GAE 用 TD 误差的加权和估计 advantage：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

A_t = delta_t
    + (gamma * lambda) * delta_{t+1}
    + (gamma * lambda)^2 * delta_{t+2}
    + ...
```

本项目使用：

```text
gamma = 0.99
gae_lambda = 0.95
```

`lambda` 控制偏差和方差的权衡：

```text
lambda 接近 0：更像一步 TD，偏差大、方差小
lambda 接近 1：更像 Monte Carlo return，偏差小、方差大
```

### Advantage normalization

PPO 在计算策略损失前，通常会把一个 minibatch 的 advantage 标准化：

```text
A_norm = (A - mean(A)) / (std(A) + 1e-8)
```

目的：

```text
让不同 batch 的 advantage 尺度接近
避免某次 batch 的数值过大导致梯度爆炸
让策略损失和 value 损失的尺度更稳定
```

Stable-Baselines3 的 PPO 默认开启：

```text
normalize_advantage = True
```

注意：advantage normalization 不是 reward normalization。它只改变 advantage 的数值尺度，不改变环境奖励本身。

## 3. On-policy 和 off-policy

```text
On-policy：
  只用当前策略刚采样的数据训练
  数据用完就丢掉
  代表：PPO、A2C、REINFORCE

Off-policy：
  可以使用旧策略、探索策略或 replay buffer 里的数据
  代表：DQN、SAC、TD3
```

PPO 虽然会对同一批数据做多个 epoch，但它仍然是 on-policy，因为这些数据来自更新前的当前策略，而且用 `ratio` 修正新旧策略差异。

## 4. Reward scale、normalization、seed

### Reward scale 为什么重要

PPO 同时优化策略损失和 value 损失。奖励尺度会直接影响：

```text
value loss 的大小
advantage 的大小
梯度的大小
clip 的触发频率
```

奖励太大时：

```text
value loss 可能主导训练
advantage 很大，策略更新过猛
clip fraction 升高，学习不稳定
```

奖励太小时：

```text
信号容易被噪声淹没
value 很难学准
策略几乎不更新
```

常见处理方式：

```text
对 reward 做缩放
使用 reward normalization
调整 value loss 系数
调整学习率
```

本项目的 HalfCheetah 最终阶段使用自定义奖励塑形，并对速度误差、目标速度、姿态等项设置了权重，这本身就是在控制 reward scale。

### Seed 的作用

seed 会影响：

```text
网络初始化
环境 reset 的初始状态
动作采样
部分环境随机性
```

所以只看一个 seed 不可靠。阶段 0 使用：

```text
seed 0、1、2
每个 seed 评测 10 局
报告 seed 均值和标准差
```

CartPole 的结果是：

```text
seed 0: 500
seed 1: 500
seed 2: 500
mean ± std: 500.0 ± 0.0
```

这是封顶效应，说明 CartPole 在这里更像 smoke test，不是敏感基准。

HalfCheetah 300K 原始奖励的结果是：

```text
seed 0: 976.48
seed 1: 1228.87
seed 2: 879.20
mean ± std: 1028.18 ± 180.48
```

HalfCheetah 最终 2.0 m/s 自定义奖励的结果是：

```text
seed 0: 1074.75 ± 989.86
seed 1: 1508.22 ± 491.94
seed 2: 1546.45 ± 262.04
3-seed mean: 1376.47 ± 262.00
```

## 5. Dreamer 训练循环

Dreamer 和 PPO 的核心区别：

```text
PPO：
  在真实环境里采样
  用真实 transition 更新 actor-critic

Dreamer：
  先用真实数据学习一个世界模型
  再在世界模型的 latent space 里“想象”轨迹
  用想象出来的 return 训练 actor-critic
```

一个简化的 Dreamer 循环：

```text
1. 用当前策略在真实环境收集数据，放进 replay buffer
2. 用 replay buffer 训练 world model
   - 编码观测到 latent state
   - 预测下一 latent state、reward、done
3. 从 replay buffer 里取起始 latent state
4. 在 world model 里展开多步 imagination
5. 用 imagined reward 和 value 训练 actor-critic
6. 回到第 1 步
```

需要能解释的关键点：

```text
world model 学的是 latent dynamics，不是直接复制像素
imagination 不调用真实环境
actor 在想象轨迹上优化
critic 估计想象轨迹的 value
```

阶段 0 只要求“能看懂”，没有要求实现 Dreamer。本仓库实现的是 PPO 热身。

## 6. 交付目录

清单里的结构是：

```text
rl-warmup/
├── train.py
├── eval.py
├── README.md
├── logs/
└── videos/
```

本仓库用两个子项目替代：

```text
stage_0/
├── README.md
├── ppo_cartpole/
│   ├── train.py
│   ├── eval.py
│   ├── README.md
│   ├── logs/
│   ├── videos/
│   └── results/
└── ppo_halfcheetah/
    ├── train.py
    ├── eval.py
    ├── README.md
    ├── logs/
    ├── videos/
    └── results/
```

内容和清单等价，只是按任务拆成了两个目录。

## 7. 已知差异和注意事项

### HalfCheetah-v5 而不是 v4

清单写的是 `HalfCheetah-v4`，实际使用 `HalfCheetah-v5`。

v5 是较新的版本，本项目的观测、动作和 PPO 训练流程与 v4 基本一致。  
如果验收方严格要求 v4，需要额外补一次 v4 的 3-seed 实验，或者把环境 id 改成 v4 后重跑。

### 最终 2.0 m/s 是自定义奖励

最终 2.0 m/s 使用的是目标速度奖励和姿态门控，不是原始 HalfCheetah 奖励。  
因此它的回报数字不能和 300K 原始奖励的 1028.18 直接比较。

### seed 1/2 共享初始化

最终 2.0 m/s 的 seed 1/2 从 seed 0 的 1.95 m/s 模型继续训练，因此三个 seed 不是完全独立的训练过程。  
报告时应该注明这一点。

### 中间模型已清理

只保留每个训练的：

```text
final_model.zip
best_model/best_model.zip
```

过程数据（CSV、JSON、TensorBoard、视频、截图、曲线）保留。
