"""用奖励塑形训练 HalfCheetah-v5，让模型尽量正常用腿跑。

在 PyCharm 里右键这个文件 -> Run，不需要终端参数。

这个脚本不会覆盖原来的 300K baseline 或 1M 实验，输出到：
    logs/halfcheetah_gait/seed_0/
    videos/halfcheetah_gait/
    results/halfcheetah_gait_seed_0.json
    results/halfcheetah_gait_seed_0_snapshots.csv

第一阶段建议先跑 300K，只看 seed 0。每 50K 保存一次视频、截图、模型和
步态统计。确认姿势正常后，再考虑跑 3 个 seed 或 1M。

重要：
    - 训练时用带奖励塑形的环境；
    - 最终评测用没有 wrapper 的环境，报告原始 HalfCheetah 回报；
    - 塑形奖励和原始奖励尺度不同，不能直接把训练 reward 和 baseline 比较。
"""

from __future__ import annotations  # 让类型注解延迟求值

import csv  # 保存每个视频节点的统计
import json  # 保存配置和最终评测
import random  # Python 随机种子
from pathlib import Path  # 处理路径

import gymnasium as gym  # 创建环境
import imageio.v2 as imageio  # 保存视频和截图
import numpy as np  # 计算均值和标准差
import torch  # 设置 PyTorch 随机种子
from stable_baselines3 import PPO  # PPO 算法
from stable_baselines3.common.callbacks import (  # 训练回调
    BaseCallback,  # 自定义回调基类
    CheckpointCallback,  # 定期保存模型
    EvalCallback,  # 定期评测并保存 best_model
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor  # 向量化环境和日志

from eval import evaluate_model  # 最终用原始环境评测
from reward_wrappers import GaitRewardWrapper  # 奖励塑形包装器
from train import PROJECT_DIR  # 项目目录

ENV_ID = "HalfCheetah-v5"  # 环境名
SEED = 0  # 先只跑 seed 0
STEPS = 300_000  # 第一轮先跑 300K，不要直接上 1M

# 这些 PPO 超参和原 baseline 保持一致，方便单独观察奖励塑形的影响。
N_STEPS = 2048  # 每次采样 2048 步
BATCH_SIZE = 64  # 每次梯度更新 64 条样本
N_EPOCHS = 10  # 每批数据训练 10 轮
LR = 3e-4  # 学习率
GAMMA = 0.99  # 折扣因子
GAE_LAMBDA = 0.95  # GAE lambda
CLIP_RANGE = 0.2  # PPO 裁剪范围

EVAL_FREQ = 25_000  # 每 25K 步做一次验证
EVAL_EPISODES = 5  # 每次验证跑 5 局
EVAL_SEED = 12_345  # 训练中验证集 seed
DEVICE = "cpu"  # 当前环境使用 CPU PyTorch

# ------------------------- 奖励塑形参数 -------------------------
# 第一轮建议先用这一组；如果仍然头朝地，就提高 W_UPRIGHT 和 W_BODY_CONTACT；
# 如果模型站着不动，就降低惩罚，或者先关掉高度项。
W_UPRIGHT = 0.5  # 身体水平奖励权重
W_BODY_CONTACT = 1.0  # 躯干/头碰地惩罚权重
W_HEIGHT = 0.3  # 高度偏离惩罚权重
HEIGHT_TOLERANCE = 0.15  # 允许身体自然上下波动的范围

# 第二阶段再打开目标速度。例如改成 1.5，表示希望大约 1.5 m/s。
# 第一阶段保持 None，先解决“头朝地”和“用腿”的问题。
TARGET_SPEED: float | None = None
W_SPEED_OVER = 1.0  # 超过目标速度后的惩罚强度

# ------------------------- 视频和评测设置 -------------------------
SNAPSHOT_INTERVAL = 50_000  # 每 50K 保存一次中间过程
SNAPSHOT_CHECKPOINTS = tuple(  # 中间视频节点，不包含 0 和最终步
    range(SNAPSHOT_INTERVAL, STEPS, SNAPSHOT_INTERVAL)
)
SNAPSHOT_EPISODES = 3  # 每个节点跑 3 局，视频只录第一局
SNAPSHOT_EVAL_SEED = 1000  # 视频和快照使用固定 seed
MAX_VIDEO_STEPS = 300  # 每个视频最多 300 帧，约 10 秒
VIDEO_FPS = 30  # 视频帧率

FINAL_EVAL_EPISODES = 10  # 最终评测 10 局，和 baseline 一致
FINAL_EVAL_SEED = 1000  # 最终评测使用 seed 1000–1009

# ------------------------- 输出路径 -------------------------
LOG_DIR = PROJECT_DIR / "logs" / "halfcheetah_gait" / "seed_0"  # 训练日志和模型
VIDEO_DIR = PROJECT_DIR / "videos" / "halfcheetah_gait"  # 视频和截图
RESULTS_DIR = PROJECT_DIR / "results"  # 结果文件
SNAPSHOT_CSV = RESULTS_DIR / "halfcheetah_gait_seed_0_snapshots.csv"  # 过程统计表
FINAL_JSON = RESULTS_DIR / "halfcheetah_gait_seed_0.json"  # 最终评测结果
SNAPSHOT_MODEL_DIR = LOG_DIR / "snapshots"  # 中间模型目录

SNAPSHOT_FIELDS = (  # 过程 CSV 的列
    "step",  # 训练步数
    "shaped_mean",  # 塑形奖励的 3 局平均
    "shaped_std",  # 塑形奖励的标准差
    "original_mean",  # 原始 HalfCheetah 奖励的 3 局平均
    "original_std",  # 原始奖励的标准差
    "mean_length",  # 平均局长
    "mean_rooty",  # 平均躯干俯仰角
    "mean_upright",  # 平均 cos(rooty)
    "mean_rootz",  # 平均躯干高度
    "body_contact_rate",  # 躯干/头碰地的步数比例
    "mean_x_velocity",  # 平均前进速度
    "video",  # 视频路径
    "screenshot",  # 截图路径
    "model",  # 模型路径
)


def build_gait_env(render_mode: str | None = None) -> gym.Env:
    """创建 HalfCheetah 环境，并在外面套上奖励塑形。"""
    env = gym.make(ENV_ID, render_mode=render_mode)
    return GaitRewardWrapper(
        env,
        w_upright=W_UPRIGHT,
        w_body_contact=W_BODY_CONTACT,
        w_height=W_HEIGHT,
        height_tolerance=HEIGHT_TOLERANCE,
        target_speed=TARGET_SPEED,
        w_speed_over=W_SPEED_OVER,
    )


def make_env(seed: int):
    """创建 DummyVecEnv 需要的无参数环境工厂。"""

    def _init():
        env = build_gait_env()
        env.reset(seed=seed)  # 固定初始状态
        env.action_space.seed(seed)  # 固定动作空间随机种子
        return env

    return _init


def build_vec_env(seed: int, monitor_path: Path) -> VecMonitor:
    """创建带日志的单个训练/验证环境。"""
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    venv = DummyVecEnv([make_env(seed)])  # Windows 下 DummyVecEnv 最稳
    return VecMonitor(venv, filename=str(monitor_path))


def sample_std(values: list[float]) -> float:
    """样本标准差；只有一个值时返回 0。"""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def init_snapshot_csv(path: Path) -> None:
    """创建过程 CSV，并写表头。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SNAPSHOT_FIELDS)
        writer.writeheader()


def append_snapshot(path: Path, row: dict) -> None:
    """向过程 CSV 追加一行。"""
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SNAPSHOT_FIELDS)
        writer.writerow(row)


def record_snapshot(
    model: PPO,
    step: int,
    video_dir: Path,
    model_dir: Path,
    episodes: int,
    eval_seed: int,
    max_video_steps: int,
    fps: int,
) -> dict:
    """保存模型、视频、截图，并统计姿态和接触情况。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"gait_{step:07d}.zip"
    model.save(str(model_path))

    env = build_gait_env(render_mode="rgb_array")
    shaped_returns: list[float] = []
    original_returns: list[float] = []
    lengths: list[int] = []
    frames: list[np.ndarray] = []

    rootys: list[float] = []
    uprights: list[float] = []
    rootzs: list[float] = []
    body_contacts: list[float] = []
    velocities: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=eval_seed + episode)
        done = False
        shaped_return = 0.0
        original_return = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            shaped_return += float(reward)
            original_return += float(info.get("reward_original", reward))
            episode_length += 1

            rootys.append(float(info["rooty"]))
            uprights.append(float(info["upright"]))
            rootzs.append(float(info["rootz"]))
            body_contacts.append(float(info["body_contact"]))
            velocities.append(float(info["x_velocity"]))

            if episode == 0 and len(frames) < max_video_steps:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            done = terminated or truncated

        shaped_returns.append(shaped_return)
        original_returns.append(original_return)
        lengths.append(episode_length)

    env.close()

    video_path = video_dir / f"halfcheetah_gait_{step:07d}.mp4"
    screenshot_path = video_dir / f"halfcheetah_gait_{step:07d}.png"
    if frames:
        imageio.mimsave(video_path, frames, fps=fps)
        imageio.imwrite(screenshot_path, frames[-1])

    result = {
        "step": int(step),
        "shaped_mean": round(float(np.mean(shaped_returns)), 2),
        "shaped_std": round(sample_std(shaped_returns), 2),
        "original_mean": round(float(np.mean(original_returns)), 2),
        "original_std": round(sample_std(original_returns), 2),
        "mean_length": round(float(np.mean(lengths)), 2),
        "mean_rooty": round(float(np.mean(rootys)), 4),
        "mean_upright": round(float(np.mean(uprights)), 4),
        "mean_rootz": round(float(np.mean(rootzs)), 4),
        "body_contact_rate": round(float(np.mean(body_contacts)), 4),
        "mean_x_velocity": round(float(np.mean(velocities)), 4),
        "video": str(video_path) if frames else "",
        "screenshot": str(screenshot_path) if frames else "",
        "model": str(model_path),
    }

    print(
        f"[snapshot] step={step:,} "
        f"shaped={result['shaped_mean']:.2f} "
        f"original={result['original_mean']:.2f} "
        f"rooty={result['mean_rooty']:.3f} "
        f"upright={result['mean_upright']:.3f} "
        f"contact={result['body_contact_rate']:.3f} "
        f"x_vel={result['mean_x_velocity']:.3f}"
    )
    return result


class VideoCheckpointCallback(BaseCallback):
    """在指定训练步数保存视频、模型和姿态统计。"""

    def __init__(
        self,
        checkpoints: tuple[int, ...],
        video_dir: Path,
        model_dir: Path,
        csv_path: Path,
        episodes: int,
        eval_seed: int,
        max_video_steps: int,
        fps: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.checkpoints = tuple(sorted(checkpoints))
        self.next_index = 0
        self.video_dir = video_dir
        self.model_dir = model_dir
        self.csv_path = csv_path
        self.episodes = episodes
        self.eval_seed = eval_seed
        self.max_video_steps = max_video_steps
        self.fps = fps

    def _on_step(self) -> bool:
        if self.next_index >= len(self.checkpoints):
            return True

        target = self.checkpoints[self.next_index]
        if self.model.num_timesteps >= target:
            actual_step = int(self.model.num_timesteps)
            if self.verbose > 0:
                print(f"\n[video] target={target:,} actual={actual_step:,}")

            row = record_snapshot(
                model=self.model,
                step=actual_step,
                video_dir=self.video_dir,
                model_dir=self.model_dir,
                episodes=self.episodes,
                eval_seed=self.eval_seed,
                max_video_steps=self.max_video_steps,
                fps=self.fps,
            )
            append_snapshot(self.csv_path, row)
            self.next_index += 1

        return True


def main() -> None:
    """训练、录视频、保存模型，最后用原始奖励做 10 局评测。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    init_snapshot_csv(SNAPSHOT_CSV)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = build_vec_env(SEED, LOG_DIR / "monitor.csv")
    eval_env = build_vec_env(EVAL_SEED, LOG_DIR / "eval_monitor.csv")

    # 注意：这个 eval_env 带奖励塑形，所以 best_model 是按塑形奖励选的。
    # 最终 JSON 会另外用没有 wrapper 的环境报告原始 HalfCheetah 分数。
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(LOG_DIR / "best_model"),
        log_path=str(LOG_DIR),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, STEPS // 10),
        save_path=str(LOG_DIR / "checkpoints"),
        name_prefix="ppo",
        verbose=0,
    )

    video_cb = VideoCheckpointCallback(
        checkpoints=SNAPSHOT_CHECKPOINTS,
        video_dir=VIDEO_DIR,
        model_dir=SNAPSHOT_MODEL_DIR,
        csv_path=SNAPSHOT_CSV,
        episodes=SNAPSHOT_EPISODES,
        eval_seed=SNAPSHOT_EVAL_SEED,
        max_video_steps=MAX_VIDEO_STEPS,
        fps=VIDEO_FPS,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        tensorboard_log=str(LOG_DIR / "tb"),
        seed=SEED,
        device=DEVICE,
        verbose=1,
    )

    reward_config = {
        "w_upright": W_UPRIGHT,
        "w_body_contact": W_BODY_CONTACT,
        "w_height": W_HEIGHT,
        "height_tolerance": HEIGHT_TOLERANCE,
        "target_speed": TARGET_SPEED,
        "w_speed_over": W_SPEED_OVER,
    }

    snapshot_steps = (0,) + tuple(SNAPSHOT_CHECKPOINTS) + (STEPS,)
    config = {
        "env_id": ENV_ID,
        "seed": SEED,
        "steps": STEPS,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": CLIP_RANGE,
        "eval_freq": EVAL_FREQ,
        "eval_episodes": EVAL_EPISODES,
        "eval_seed": EVAL_SEED,
        "final_eval_episodes": FINAL_EVAL_EPISODES,
        "final_eval_seed": FINAL_EVAL_SEED,
        "device": DEVICE,
        "reward": reward_config,
        "snapshot_steps": snapshot_steps,
        "snapshot_episodes": SNAPSHOT_EPISODES,
        "snapshot_eval_seed": SNAPSHOT_EVAL_SEED,
        "max_video_steps": MAX_VIDEO_STEPS,
        "video_fps": VIDEO_FPS,
    }
    (LOG_DIR / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print(f"env={ENV_ID} seed={SEED} steps={STEPS:,}")
    print(f"reward={reward_config}")
    print(f"logdir={LOG_DIR}")
    print(f"videos={VIDEO_DIR}")
    print("=" * 78)

    initial_row = record_snapshot(
        model=model,
        step=0,
        video_dir=VIDEO_DIR,
        model_dir=SNAPSHOT_MODEL_DIR,
        episodes=SNAPSHOT_EPISODES,
        eval_seed=SNAPSHOT_EVAL_SEED,
        max_video_steps=MAX_VIDEO_STEPS,
        fps=VIDEO_FPS,
    )
    append_snapshot(SNAPSHOT_CSV, initial_row)

    model.learn(
        total_timesteps=STEPS,
        callback=[eval_cb, checkpoint_cb, video_cb],
    )
    model.save(str(LOG_DIR / "final_model"))

    final_row = record_snapshot(
        model=model,
        step=STEPS,
        video_dir=VIDEO_DIR,
        model_dir=SNAPSHOT_MODEL_DIR,
        episodes=SNAPSHOT_EPISODES,
        eval_seed=SNAPSHOT_EVAL_SEED,
        max_video_steps=MAX_VIDEO_STEPS,
        fps=VIDEO_FPS,
    )
    append_snapshot(SNAPSHOT_CSV, final_row)

    train_env.close()
    eval_env.close()

    print(f"\nRunning final {FINAL_EVAL_EPISODES}-episode original-reward evaluation ...")
    final_result = evaluate_model(
        model_path=LOG_DIR / "final_model.zip",
        env_id=ENV_ID,
        episodes=FINAL_EVAL_EPISODES,
        eval_seed=FINAL_EVAL_SEED,
        video_path=VIDEO_DIR / "halfcheetah_gait_final_eval.mp4",
        screenshot_path=VIDEO_DIR / "halfcheetah_gait_final_eval.png",
        max_video_steps=MAX_VIDEO_STEPS,
    )
    final_result["train_seed"] = SEED
    final_result["steps"] = STEPS
    final_result["reward_config"] = reward_config
    final_result["final_snapshot"] = final_row
    final_result["snapshot_csv"] = str(SNAPSHOT_CSV)
    final_result["note"] = (
        "final mean/std is evaluated on the original HalfCheetah reward; "
        "final_snapshot contains shaped and original snapshot metrics."
    )
    FINAL_JSON.write_text(
        json.dumps(final_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        f"Final original eval: mean={final_result['mean']:.2f} "
        f"± {final_result['std']:.2f}"
    )
    print(f"final snapshot: rooty={final_row['mean_rooty']:.3f} "
          f"contact={final_row['body_contact_rate']:.3f} "
          f"x_vel={final_row['mean_x_velocity']:.3f}")
    print(f"Saved: {FINAL_JSON}")
    print(f"Videos: {VIDEO_DIR}")
    print(f"Snapshots: {SNAPSHOT_CSV}")
    print("=" * 78)


if __name__ == "__main__":
    main()
