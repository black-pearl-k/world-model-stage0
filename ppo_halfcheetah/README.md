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
├── reward_wrappers.py             # 奖励塑形：步态版 + 只站立版
├── run_halfcheetah_gait.py        # seed 0 步态奖励训练，保存过程视频和步态统计
├── run_halfcheetah_final.py       # 最终速度阶段，目标提高到 2.0 m/s
├── run_halfcheetah_final_3seeds.py # 补跑最终 2.0 m/s 的 seed 1/2
├── summarize_halfcheetah_final_3seeds.py # 汇总最终 3 个 seed 并画图
├── plot_halfcheetah_gait_curves.py # 画步态奖励实验曲线
├── watch_halfcheetah.py           # 实时打开 MuJoCo 窗口观看模型
├── eval.py                        # 评测函数，不做训练
├── run_eval_halfcheetah.py        # 评测 3 个模型并生成结果表、视频
├── plot_halfcheetah_curves.py     # 画 3 个 seed 的评测曲线
├── run_tensorboard.py             # 在 PyCharm 内启动 300K TensorBoard
├── run_tensorboard_gait.py        # 在 PyCharm 内启动步态奖励 TensorBoard
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

### 6. 奖励塑形实验（解决头朝地）

原生奖励训练出来的模型会头朝地，原因是原始 HalfCheetah 奖励只奖励前进速度，
不奖励“身体正、用腿跑”。继续增加训练步数不会解决这个问题，需要改奖励。

右键：

```text
run_halfcheetah_gait.py → Run
```

默认设置：

```text
只训练 seed 0
300000 步
每 25000 步验证一次
每 50000 步保存视频、截图、模型和步态统计
最后用 seed 1000–1009 跑 10 局原始奖励评测
```

输出目录：

```text
logs/halfcheetah_gait/seed_0/
videos/halfcheetah_gait/
results/halfcheetah_gait_seed_0_snapshots.csv
results/halfcheetah_gait_seed_0.json
```

奖励塑形代码在：

```text
reward_wrappers.py
```

第一版奖励：

```text
新奖励 = 原始奖励
       + 0.5 * cos(rooty)
       - 1.0 * 躯干/头碰地
       - 0.3 * 高度偏离
```

解释：

- `rooty=0` 表示身体水平，`cos(rooty)=1`；
- 身体翻过来时 `cos(rooty)` 接近 `-1`；
- 躯干或头碰地每步扣分，脚碰地不扣；
- 高度只在超过 `0.15` 的自然波动范围后才扣分；
- 第一阶段 `TARGET_SPEED=None`，先只解决姿势，不限制速度。

这些参数在 `run_halfcheetah_gait.py` 顶部：

```python
W_UPRIGHT = 0.5
W_BODY_CONTACT = 1.0
W_HEIGHT = 0.3
HEIGHT_TOLERANCE = 0.15
TARGET_SPEED = None
```

调参规则：

```text
300K 后仍然头朝地
→ W_UPRIGHT 调到 1.0，W_BODY_CONTACT 调到 2.0

模型站着不动、几乎不前进
→ 降低 W_BODY_CONTACT / W_HEIGHT，先保留前进奖励

姿势正常但速度太快
→ 第二阶段把 TARGET_SPEED 改成 1.5
```

`run_halfcheetah_gait.py` 会同时记录：

```text
mean_rooty         平均躯干俯仰角
mean_upright       平均 cos(rooty)
body_contact_rate  躯干/头碰地的步数比例
mean_x_velocity    平均前进速度
original_mean      原始 HalfCheetah 奖励
shaped_mean        塑形奖励
```

注意：`shaped_mean` 和原来的 976 不是同一个尺度，不能直接比较。
最终 JSON 里的 `mean` 是用**没有 wrapper 的环境**重新评测的原始回报。
训练中的 `best_model` 是按塑形验证奖励选出的，不是按原始奖励选出的；这样更符合“先让姿势正常”的目标。

训练完成后，右键：

```text
plot_halfcheetah_gait_curves.py → Run
```

生成：

```text
results/halfcheetah_gait_curve.png
```

左图是塑形奖励曲线，右图是各快照节点的原始回报。

奖励塑形实验的 TensorBoard：

```text
run_tensorboard_gait.py → Run
http://localhost:6009
```

先只跑 seed 0 的 300K，确认视频里不再用头/背滑行，再考虑 3 个 seed。

### 7. 最终速度 2.0 m/s

最终阶段已完成，目标速度 `2.0 m/s`。历史上是从 1.95 m/s 模型继续训练；中间课程产物现在已经清理。

右键：

```text
run_halfcheetah_final.py → Run
```

历史训练使用过的初始化模型：

```text
logs/halfcheetah_native/seed_0/final_model.zip
```

这个中间模型已经清理；如果重新运行最终脚本，初始模型不存在时会从零开始，因此不适合用来复现已有结果。

目标速度：

```text
2.0 m/s
```

输出：

```text
logs/halfcheetah_final/seed_0/
videos/halfcheetah_final/
results/halfcheetah_final_seed_0_snapshots.csv
results/halfcheetah_final_seed_0.json
```

参数在 `run_halfcheetah_final.py` 顶部：

```python
TARGET_SPEED = 2.0
SPEED_TOLERANCE = 0.2
W_SPEED_ERROR = 4.0
W_SPEED_TARGET = 2.0
RESET_LOG_STD = -2.0
```

我测试过 100K 步：

```text
mean_upright 约 0.997
头/躯干接触 0
mean x velocity 约 2.062 m/s
目标速度 2.0 m/s
speed error 约 0.068
```

训练完成后用：

```text
watch_halfcheetah.py → Run
```

选择：

```text
logs/halfcheetah_final/seed_0/final_model.zip
```

### 8. 补跑最终 2.0 m/s 的 seed 1 和 seed 2

seed 0 已经训练过，剩下两个 seed 用：

```text
run_halfcheetah_final_3seeds.py → Run
```

输出：

```text
logs/halfcheetah_final/seed_1/
logs/halfcheetah_final/seed_2/
results/halfcheetah_final_seed_1.json
results/halfcheetah_final_seed_1_snapshots.csv
results/halfcheetah_final_seed_2.json
results/halfcheetah_final_seed_2_snapshots.csv
```

注意：当时只有 seed 0 有完整的 1.95 m/s 前一阶段模型，因此 seed 1/2 都从：

```text
logs/halfcheetah_native/seed_0/final_model.zip
```

继续训练。训练采样和动作噪声分别使用 seed 1 / seed 2，但初始化模型是共享的，不是完全独立的三个 seed。

这个共享初始化模型现在已经作为中间产物清理，所以该脚本仅保留作记录，重新运行前需要先恢复它。

### 9. 最终三个 seed 的总结

右键：

```text
summarize_halfcheetah_final_3seeds.py → Run
```

输出：

```text
results/halfcheetah_final_3seeds_summary.csv
results/halfcheetah_final_3seeds_summary.json
results/halfcheetah_final_3seeds_summary.png
```

当前三个 final 模型的 10 局测试结果：

```text
seed 0: 1074.75 ± 989.86
seed 1: 1508.22 ± 491.94
seed 2: 1546.45 ± 262.04
```

三个 seed 的 final mean：`1376.47 ± 262.00`。

注意：seed 0 的均值被 3 个负回报 episode 拉低；如果只看视频里的正常 episode，会感觉它表现不错，但统计上它的方差很大。

图中包含：

```text
3 个 seed 的训练评测曲线
3 个 seed 的快照 x velocity
3 个 seed 的最终 10 局回报分布
最终快照的速度和 upright
```

## 实时观看模型

右键：

```text
watch_halfcheetah.py → Run
```

它会打开一个 MuJoCo 实时窗口，加载模型跑 3 局。

默认开启跟随镜头，并每步暂停 `0.05` 秒，方便观察。  
如果还是觉得太快，把 `watch_halfcheetah.py` 里的：

```python
SLOWDOWN_SECONDS = 0.05
```

改成：

```python
SLOWDOWN_SECONDS = 0.1
```

如果想关闭跟随镜头：

```python
FOLLOW_CAMERA = False
```

运行后，它会自动扫描：

```text
logs/
models/
```

下面的所有 `.zip` 模型，并按修改时间从新到旧列出，例如：

```text
[ 1] 2026-11-01 00:21  165.4 KB  logs\halfcheetah_gait\seed_0\final_model.zip
[ 2] 2026-11-01 00:16  165.4 KB  logs\halfcheetah_gait\seed_0\best_model\best_model.zip
[ 3] 2026-10-31 21:41  165.4 KB  logs\halfcheetah\seed_0\final_model.zip
...
```

然后输入编号选择：

```text
输入模型编号（直接回车=1，q=退出）:
```

- 输入数字：加载对应模型；
- 直接回车：加载第一个，也就是最新的模型；
- 输入 `q`：退出。

列表会包含：

```text
300K 的 final_model.zip
300K 的 best_model.zip
300K 的 checkpoints
步态奖励实验的 final_model.zip
步态奖励实验的 best_model.zip
步态奖励实验的 snapshots
步态奖励实验的 checkpoints
只站立实验的 final_model.zip
只站立实验的 best_model.zip
只站立实验的 snapshots
只站立实验的 checkpoints
小目标速度实验的 final_model.zip
小目标速度实验的 best_model.zip
小目标速度实验的 snapshots
小目标速度实验的 checkpoints
较快速度实验的 final_model.zip
较快速度实验的 best_model.zip
较快速度实验的 snapshots
较快速度实验的 checkpoints
最终 2.0 m/s seed 0 的 final_model.zip
最终 2.0 m/s seed 0 的 best_model.zip
最终 2.0 m/s seed 0 的 snapshots
最终 2.0 m/s seed 0 的 checkpoints
最终 2.0 m/s seed 1/2 的模型
smoke test 模型
```

`best_model.zip` 是训练过程中 `EvalCallback` 按验证集平均回报选出的模型；`final_model.zip` 是训练结束时的模型。最终对比时，三个 seed 最好使用同一种选择规则，不要只挑某个 seed 的 best。

注意：正式训练 3 个 seed 时不要开实时窗口，渲染会拖慢训练。实时观看只用于检查模型和演示。

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

## 当前结果

300K，3 个 seed，最终评测 10 局：

```text
seed 0: 976.48
seed 1: 1228.87
seed 2: 879.20
Mean ± Std: 1028.18 ± 180.48
```

## 结果记录模板

```text
Env: HalfCheetah-v5
Seeds: ...
Steps: ...
Eval episodes per seed: 10

seed ...: ...

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
