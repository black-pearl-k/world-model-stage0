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
├── run_halfcheetah_1m.py          # seed 0 训练 1M 步，并保存过程视频
├── reward_wrappers.py             # 奖励塑形：步态版 + 只站立版
├── run_halfcheetah_gait.py        # seed 0 步态奖励训练，保存过程视频和步态统计
├── run_halfcheetah_stand.py       # seed 0 只站立训练，先去掉前进奖励
├── plot_halfcheetah_gait_curves.py # 画步态奖励实验曲线
├── plot_halfcheetah_stand_curves.py # 画站立实验的奖励和姿态指标
├── watch_halfcheetah.py           # 实时打开 MuJoCo 窗口观看模型
├── eval.py                        # 评测函数，不做训练
├── run_eval_halfcheetah.py        # 评测 3 个模型并生成结果表、视频
├── plot_halfcheetah_curves.py     # 画 3 个 seed 的评测曲线
├── plot_halfcheetah_1m_curves.py  # 画 1M seed 0 的评测曲线
├── run_tensorboard.py             # 在 PyCharm 内启动 300K TensorBoard
├── run_tensorboard_1m.py          # 在 PyCharm 内启动 1M TensorBoard
├── run_tensorboard_gait.py        # 在 PyCharm 内启动步态奖励 TensorBoard
├── run_tensorboard_stand.py       # 在 PyCharm 内启动站立实验 TensorBoard
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

### 3. 单独跑 seed 0 到 1M 步

右键：

```text
run_halfcheetah_1m.py → Run
```

它会：

```text
只训练 seed 0
训练 1,000,000 步
PPO 超参和 300K baseline 完全一致
不覆盖原来的 logs/halfcheetah/seed_0/
```

输出目录：

```text
logs/halfcheetah_1m/seed_0/
videos/halfcheetah_1m/
results/halfcheetah_1m_seed_0.json
results/halfcheetah_1m_snapshots.csv
```

在以下步数各保存一次视频、截图、模型和回报：

```text
0
250K
500K
750K
1M
```

每个视频只录前 300 帧，约 10 秒。  
每个节点跑 3 局，CSV 里记录平均回报和标准差。  
最后再用 `seed 1000–1009` 跑 10 局最终评测，和原来的 seed 0 结果对比。

预计时间：20～40 分钟，取决于 CPU 和视频编码速度。

训练完成后，右键：

```text
plot_halfcheetah_1m_curves.py → Run
```

会生成：

```text
results/halfcheetah_1m_curve.png
```

图里会同时画：

- 300K seed 0 的验证曲线；
- 1M seed 0 的验证曲线；
- 1M 验证最高点；
- 最终 10 局测试均值参考线。

数据来源：

```text
logs/halfcheetah_1m/seed_0/evaluations.npz  # 每 25K 步的验证曲线
results/halfcheetah_1m_snapshots.csv        # 0/250K/500K/750K/1M 五个视频节点
logs/halfcheetah_1m/seed_0/monitor.csv      # 每个训练 episode 的回报
```

### 4. 看 TensorBoard

右键：

```text
run_tensorboard.py → Run
```

浏览器会自动打开：

```text
http://localhost:6007
```

1M 实验用另一个端口：

```text
run_tensorboard_1m.py → Run
http://localhost:6008
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

### 5. 评测 3 个模型

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

### 6. 画曲线

右键：

```text
plot_halfcheetah_curves.py → Run
```

生成：

```text
results/halfcheetah_curves.png
```

曲线来自训练中 `EvalCallback` 的 `evaluations.npz`，不是最终评测。

### 7. 奖励塑形实验（解决头朝地）

1M 实验仍然会头朝地，原因是原始 HalfCheetah 奖励只奖励前进速度，
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

注意：`shaped_mean` 和原来的 976 / 1616 不是同一个尺度，不能直接比较。
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

先只跑 seed 0 的 300K，确认视频里不再用头/背滑行，再考虑 3 个 seed 或 1M。

如果以后把 `STEPS` 改成 `1_000_000`，建议同时把 `SNAPSHOT_INTERVAL` 改成 `100_000`，避免产生 19 个中间视频。

### 8. 先练习站立（完全去掉前进奖励）

上一版步态奖励仍然保留了原始前进奖励，所以模型还是会为了速度往前跑，并且可能翻跟头。

如果现在的目标是：

```text
先不管跑
先让身体水平
先让腿在下面
先不要用头/背滑
先不要翻滚和猛蹬
```

就运行：

```text
run_halfcheetah_stand.py → Run
```

它使用 `reward_wrappers.py` 里的新类：

```python
StandRewardWrapper
```

这个类**完全不使用原始 HalfCheetah 奖励**。  
原始奖励只写进 `info` 做诊断，不参与训练。

站立奖励：

```text
站立奖励 =
    + 1.0 * cos(rooty)
    - 2.0 * 躯干碰地
    - 5.0 * 头碰地
    - 0.5 * 高度偏离
    - 0.5 * x_velocity^2
    - 0.2 * 根节点速度平方
    - 0.02 * 关节速度平方
    - 0.05 * 动作幅度平方
    - 0.05 * 动作变化平方
```

它和步态版的区别：

```text
步态版：原始前进奖励还在
站立版：原始前进奖励完全去掉
```

所以站立版的模型不会为了速度跑，反而会被惩罚：

```text
x_velocity
根节点速度
关节速度
动作变化
```

输出：

```text
logs/halfcheetah_stand/seed_0/
videos/halfcheetah_stand/
results/halfcheetah_stand_seed_0_snapshots.csv
results/halfcheetah_stand_seed_0.json
```

默认每 50K 保存一次视频：

```text
0
50K
100K
150K
200K
250K
300K
```

重点看：

```text
mean_upright       是否接近 1
mean_rooty         是否接近 0
torso_contact_rate 是否接近 0
head_contact_rate  是否接近 0
mean_x_velocity    是否接近 0
```

最终 JSON 里的原始 HalfCheetah 回报会接近 0 或为负，这是正常的，因为现在根本没有奖励前进。不要拿它和 976 / 1616 比较。

训练完成后，运行 `watch_halfcheetah.py`，选择：

```text
logs/halfcheetah_stand/seed_0/final_model.zip
```

看它能不能稳定站住。

如果站立版仍然翻倒：

```text
提高 W_HEAD_CONTACT
提高 W_ROOT_VELOCITY
提高 W_ACTION_RATE
```

如果它站着但一直抖：

```text
提高 W_JOINT_VELOCITY
提高 W_ACTION
```

如果它完全不站、直接趴下：

```text
提高 W_UPRIGHT
提高 W_HEIGHT
```

参数都在 `run_halfcheetah_stand.py` 顶部。

训练完成后，右键：

```text
plot_halfcheetah_stand_curves.py → Run
```

生成：

```text
results/halfcheetah_stand_curve.png
```

图里有：

```text
站立奖励
mean_upright
躯干/头碰地比例
平均 x 速度
```

站立实验的 TensorBoard：

```text
run_tensorboard_stand.py → Run
http://localhost:6010
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
[ 1] 2026-10-31 23:17  165.4 KB  logs\halfcheetah_1m\seed_0\snapshots\model_1000000.zip
[ 2] 2026-10-31 23:17  165.4 KB  logs\halfcheetah_1m\seed_0\final_model.zip
[ 3] 2026-10-31 23:15  165.4 KB  logs\halfcheetah_1m\seed_0\best_model\best_model.zip
...
```

然后输入编号选择：

```text
输入模型编号 1-65（直接回车=1，q=退出）:
```

- 输入数字：加载对应模型；
- 直接回车：加载第一个，也就是最新的模型；
- 输入 `q`：退出。

列表会包含：

```text
300K 的 final_model.zip
300K 的 best_model.zip
300K 的 checkpoints
1M 的 final_model.zip
1M 的 best_model.zip
1M 的 snapshots
1M 的 checkpoints
步态奖励实验的 final_model.zip
步态奖励实验的 best_model.zip
步态奖励实验的 snapshots
步态奖励实验的 checkpoints
只站立实验的 final_model.zip
只站立实验的 best_model.zip
只站立实验的 snapshots
只站立实验的 checkpoints
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

1M，seed 0，最终评测 10 局（同样的 seed 1000–1009）：

```text
mean ± std: 1616.10 ± 37.86
best validation eval: 1625.20 @ 900K
```

1M 曲线：

```text
results/halfcheetah_1m_curve.png
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
