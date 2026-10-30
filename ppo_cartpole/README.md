# PPO CartPole — Stage 0 Practice

本项目是阶段 0 的第一组实验：用 PPO 在 `CartPole-v1` 上跑 3 个 seed，记录评测曲线、视频和结果表。

## 环境

- Python: 3.10.21
- torch: 2.14.1+cpu
- gymnasium: 1.4.0
- stable-baselines3: 2.9.2
- mujoco: 3.17.0
- 设备: CPU

PyCharm 解释器：

```text
D:\Anaconda\Anaconda_envs\envs\RL\python.exe
```

## 文件说明

```text
ppo_cartpole/
├── train.py                  # PPO 训练函数，也可单独运行
├── run_cartpole_3seeds.py    # 一键训练 3 个 seed
├── eval.py                   # 评测函数，不做训练
├── run_eval_cartpole.py      # 评测 3 个模型并生成结果表、视频
├── plot_curves.py            # 画 3 个 seed 的评测曲线
├── run_tensorboard.py        # 在 PyCharm 内启动 TensorBoard
├── logs/                     # 训练日志、TensorBoard、evaluations.npz
├── models/                   # 预留
├── videos/                   # 评测视频/截图
├── results/                  # JSON/CSV/曲线图
└── notes/                    # 实验笔记
```

## 在 PyCharm 里怎么运行

不需要终端命令，全部用右键 Run。

### 1. 训练 3 个 seed

右键：

```text
run_cartpole_3seeds.py → Run
```

它会依次训练：

```text
seed 0 → 50000 步
seed 1 → 50000 步
seed 2 → 50000 步
```

CPU 上大约每个 seed 1～3 分钟。

输出目录：

```text
logs/cartpole/seed_0/
logs/cartpole/seed_1/
logs/cartpole/seed_2/
```

每个目录里有：

```text
final_model.zip
config.json
evaluations.npz
monitor.csv
tb/PPO_1/...
```

### 2. 看 TensorBoard

右键：

```text
run_tensorboard.py → Run
```

浏览器会自动打开：

```text
http://localhost:6006
```

重点看：

```text
eval/mean_reward
train/approx_kl
train/clip_fraction
train/explained_variance
train/value_loss
```

看完后，在 PyCharm 的 Run 控制台按 Enter，停止 TensorBoard。

### 3. 评测 3 个模型

右键：

```text
run_eval_cartpole.py → Run
```

它会：

- 用固定 eval seed `1000～1009` 跑 10 个 episode；
- 使用确定性动作；
- 生成：

```text
results/eval_cartpole_seed_0.json
results/eval_cartpole_seed_1.json
results/eval_cartpole_seed_2.json
results/cartpole_summary.json
results/cartpole_results.csv
```

还会给 seed 0 录一个视频和截图：

```text
videos/cartpole_seed_0.mp4
videos/cartpole_seed_0.png
```

### 4. 画曲线

右键：

```text
plot_curves.py → Run
```

生成：

```text
results/cartpole_curves.png
```

## 评测协议

- 每个 seed 评测 10 个 episode。
- eval seed 固定为 `1000 + episode_index`。
- 动作使用 `deterministic=True`。
- 不更新模型权重。
- 最终报告 3 个 seed 的 `mean ± std`。

## 结果记录模板

```text
Env: CartPole-v1
Seeds: 0, 1, 2
Steps: 50000
Eval episodes per seed: 10

seed 0: ...
seed 1: ...
seed 2: ...

Mean ± Std: ...
```

## 失败分析模板

```text
现象：
原因：
下一步：
```

## 注意

- `MUJOCO_LOG.TXT` 是 MuJoCo 自动生成的警告日志，可以删除。
- CartPole 训练不需要 MuJoCo，HalfCheetah 才需要。
- 现在 torch 是 CPU 版，`device="auto"` 会使用 CPU。
