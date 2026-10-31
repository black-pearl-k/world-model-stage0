# PPO HalfCheetah-v5 — Stage 0 Practice

本项目是阶段 0 的第二组实验：用 PPO 在 `HalfCheetah-v5` 上跑 3 个 seed，记录评测曲线、结果表、视频/截图。

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

`HalfCheetah-v5` 已经包含在当前 `gymnasium[mujoco]` 中，不需要额外安装。

如果想自己验证环境：

```powershell
D:\Anaconda\Anaconda_envs\envs\RL\python.exe -c "import gymnasium as gym; e=gym.make('HalfCheetah-v5'); print(e); e.close()"
```

## 文件说明

```text
ppo_halfcheetah/
├── train.py                       # PPO 训练函数，也可单独运行
├── run_halfcheetah_smoke.py       # 2 万步 smoke test
├── run_halfcheetah_3seeds.py      # 一键训练 3 个 seed
├── eval.py                        # 评测函数，不做训练
├── run_eval_halfcheetah.py        # 评测 3 个模型并生成结果表、视频
├── plot_halfcheetah_curves.py     # 画 3 个 seed 的评测曲线
├── run_tensorboard.py             # 在 PyCharm 内启动 TensorBoard
├── logs/                          # 训练日志、TensorBoard、evaluations.npz
├── models/                        # 预留
├── videos/                        # 评测视频/截图
├── results/                       # JSON/CSV/曲线图
└── notes/                         # 实验笔记
```

## 在 PyCharm 里怎么运行

不需要终端命令，全部用右键 Run。

### 1. Smoke test

右键：

```text
run_halfcheetah_smoke.py → Run
```

它会跑：

```text
HalfCheetah-v5
seed 0
20000 步
每 10000 步评测 3 局
```

输出目录：

```text
logs/halfcheetah/smoke_seed_0/
```

smoke test 只用于确认环境、日志、模型保存正常，不代表最终成绩。

### 2. 训练 3 个 seed

右键：

```text
run_halfcheetah_3seeds.py → Run
```

默认：

```text
seed 0 → 300000 步
seed 1 → 300000 步
seed 2 → 300000 步
```

输出目录：

```text
logs/halfcheetah/seed_0/
logs/halfcheetah/seed_1/
logs/halfcheetah/seed_2/
```

每个目录里有：

```text
final_model.zip
config.json
evaluations.npz
monitor.csv
tb/PPO_1/...
```

如果 30 万步后曲线还在明显上升，可以把 `run_halfcheetah_3seeds.py` 里的：

```python
STEPS = 300_000
```

改成：

```python
STEPS = 500_000
```

然后 3 个 seed 一起重跑。不要不同 seed 用不同步数。

### 3. 看 TensorBoard

右键：

```text
run_tensorboard.py → Run
```

浏览器会自动打开：

```text
http://localhost:6007
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

### 4. 评测 3 个模型

右键：

```text
run_eval_halfcheetah.py → Run
```

它会：

- 用固定 eval seed `1000～1009` 跑 10 个 episode；
- 使用确定性动作；
- 生成：

```text
results/eval_halfcheetah_seed_0.json
results/eval_halfcheetah_seed_1.json
results/eval_halfcheetah_seed_2.json
results/halfcheetah_summary.json
results/halfcheetah_results.csv
```

还会给 seed 0 录一个视频和截图：

```text
videos/halfcheetah_seed_0.mp4
videos/halfcheetah_seed_0.png
```

视频只录前 300 帧，避免文件太大。

### 5. 画曲线

右键：

```text
plot_halfcheetah_curves.py → Run
```

生成：

```text
results/halfcheetah_curves.png
```

曲线来自训练中 `EvalCallback` 的 `evaluations.npz`，不是最终评测。

## 训练超参

```text
env_id: HalfCheetah-v5
n_steps: 2048
batch_size: 64
n_epochs: 10
learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
eval_freq: 25000
eval_episodes: 5
```

## 评测协议

- 每个 seed 评测 10 个 episode。
- eval seed 固定为 `1000 + episode_index`。
- 动作使用 `deterministic=True`。
- 不更新模型权重。
- 最终报告 3 个 seed 的 `mean ± std`。
- 训练中的 eval 和最终 eval 是两套不同 seed，不要混用。

## 结果记录模板

```text
Env: HalfCheetah-v5
Seeds: 0, 1, 2
Steps: 300000
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
- HalfCheetah 一局是 1000 步，所以评测比 CartPole 贵很多。
- 训练日志、模型 `.zip`、视频 `.mp4` 不要提交到 Git。
- 结果 JSON/CSV、曲线 PNG、截图 PNG 可以提交。
- 现在 torch 是 CPU 版，`device="auto"` 会使用 CPU。
