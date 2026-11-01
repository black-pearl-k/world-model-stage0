"""第一阶段：让 HalfCheetah 先学会站稳，不奖励前进。

在 PyCharm 里右键这个文件 -> Run。

和 run_halfcheetah_gait.py 的区别：

    - 完全去掉原始前进奖励；
    - 不奖励 x 方向速度；
    - 反而惩罚 x 方向速度；
    - 惩罚翻滚、关节乱动、动作突变；
    - 头部碰地比躯干碰地罚得更重；
    - 先只训练“身体水平、脚在下面、站住”。

输出：
    logs/halfcheetah_stand/seed_0/
    videos/halfcheetah_stand/
    results/halfcheetah_stand_seed_0_snapshots.csv
    results/halfcheetah_stand_seed_0.json

注意：
    - 这个阶段不是跑步实验；
    - 最终原始 HalfCheetah 回报会很低，这是正常的，因为现在不奖励跑；
    - 最终 JSON 里的 mean 只是诊断值，不要和 976 比较。
"""

from __future__ import annotations  # 让类型注解延迟求值

import csv  # 保存每个快照的统计
import json  # 保存配置和最终诊断结果
import random  # Python 随机种子
from pathlib import Path  # 处理路径

import gymnasium as gym  # 创建环境
import imageio.v2 as imageio  # 保存视频和截图
import numpy as np  # 计算统计量
import torch  # PyTorch 随机种子
from stable_baselines3 import PPO  # PPO 算法
from stable_baselines3.common.callbacks import (  # 训练回调
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from eval import evaluate_model  # 最终原始奖励诊断
from reward_wrappers import StandRewardWrapper  # 站立奖励包装器
from train import PROJECT_DIR  # 项目目录

ENV_ID = "HalfCheetah-v5"
SEED = 0
STEPS = 300_000  # 先跑 300K；站立通常比跑步容易

# PPO 超参保持和之前一致，先只改变奖励函数。
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

EVAL_FREQ = 25_000
EVAL_EPISODES = 5
EVAL_SEED = 12_345
DEVICE = "cpu"

# ------------------------- 站立奖励参数 -------------------------
# 这些参数的目标：站直、少碰地、不要跑、不要翻滚、不要猛蹬。
W_UPRIGHT = 1.0  # cos(rooty) 奖励：身体水平得 +1，翻过来约 -1
W_TORSO_CONTACT = 2.0  # 躯干碰地惩罚
W_HEAD_CONTACT = 5.0  # 头碰地惩罚，故意比躯干重
W_HEIGHT = 0.5  # 高度偏离惩罚
HEIGHT_TOLERANCE = 0.15  # 允许身体自然上下浮动
W_X_VELOCITY = 0.5  # 惩罚前后移动，先不让它跑
W_ROOT_VELOCITY = 0.2  # 惩罚根节点平移/俯仰速度，减少翻滚
W_JOINT_VELOCITY = 0.02  # 惩罚关节乱动
W_ACTION = 0.05  # 惩罚大幅度动作
W_ACTION_RATE = 0.05  # 惩罚动作突变，减少猛蹬

# ------------------------- 视频和评测 -------------------------
SNAPSHOT_INTERVAL = 50_000
SNAPSHOT_CHECKPOINTS = tuple(
    range(SNAPSHOT_INTERVAL, STEPS, SNAPSHOT_INTERVAL)
)
SNAPSHOT_EPISODES = 3
SNAPSHOT_EVAL_SEED = 1000
MAX_VIDEO_STEPS = 300
VIDEO_FPS = 30

FINAL_EVAL_EPISODES = 10
FINAL_EVAL_SEED = 1000

# ------------------------- 输出路径 -------------------------
LOG_DIR = PROJECT_DIR / "logs" / "halfcheetah_stand" / "seed_0"
VIDEO_DIR = PROJECT_DIR / "videos" / "halfcheetah_stand"
RESULTS_DIR = PROJECT_DIR / "results"
SNAPSHOT_CSV = RESULTS_DIR / "halfcheetah_stand_seed_0_snapshots.csv"
FINAL_JSON = RESULTS_DIR / "halfcheetah_stand_seed_0.json"
SNAPSHOT_MODEL_DIR = LOG_DIR / "snapshots"

SNAPSHOT_FIELDS = (
    "step",
    "stand_mean",
    "stand_std",
    "original_mean",
    "original_std",
    "mean_length",
    "mean_rooty",
    "mean_upright",
    "mean_rootz",
    "torso_contact_rate",
    "head_contact_rate",
    "body_contact_rate",
    "mean_x_velocity",
    "mean_root_velocity",
    "mean_action_rate",
    "video",
    "screenshot",
    "model",
)


def build_stand_env(render_mode: str | None = None) -> gym.Env:
    """创建 HalfCheetah，并套上“只站立”的奖励。"""
    env = gym.make(ENV_ID, render_mode=render_mode)
    return StandRewardWrapper(
        env,
        w_upright=W_UPRIGHT,
        w_torso_contact=W_TORSO_CONTACT,
        w_head_contact=W_HEAD_CONTACT,
        w_height=W_HEIGHT,
        height_tolerance=HEIGHT_TOLERANCE,
        w_x_velocity=W_X_VELOCITY,
        w_root_velocity=W_ROOT_VELOCITY,
        w_joint_velocity=W_JOINT_VELOCITY,
        w_action=W_ACTION,
        w_action_rate=W_ACTION_RATE,
    )


def make_env(seed: int):
    """DummyVecEnv 需要的无参数环境工厂。"""

    def _init():
        env = build_stand_env()
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def build_vec_env(seed: int, monitor_path: Path) -> VecMonitor:
    """创建带日志的单个训练/验证环境。"""
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    venv = DummyVecEnv([make_env(seed)])
    return VecMonitor(venv, filename=str(monitor_path))


def sample_std(values: list[float]) -> float:
    """样本标准差；只有一个值时返回 0。"""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def init_snapshot_csv(path: Path) -> None:
    """创建过程 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=SNAPSHOT_FIELDS).writeheader()


def append_snapshot(path: Path, row: dict) -> None:
    """向过程 CSV 追加一行。"""
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=SNAPSHOT_FIELDS).writerow(row)


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
    """保存模型、视频、截图，并统计站立指标。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"stand_{step:07d}.zip"
    model.save(str(model_path))

    env = build_stand_env(render_mode="rgb_array")
    stand_returns: list[float] = []
    original_returns: list[float] = []
    lengths: list[int] = []
    frames: list[np.ndarray] = []

    rootys: list[float] = []
    uprights: list[float] = []
    rootzs: list[float] = []
    torso_contacts: list[float] = []
    head_contacts: list[float] = []
    body_contacts: list[float] = []
    velocities: list[float] = []
    root_velocities: list[float] = []
    action_rates: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=eval_seed + episode)
        done = False
        stand_return = 0.0
        original_return = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            stand_return += float(reward)
            original_return += float(info.get("reward_original", 0.0))
            episode_length += 1

            rootys.append(float(info["rooty"]))
            uprights.append(float(info["upright"]))
            rootzs.append(float(info["rootz"]))
            torso_contacts.append(float(info["torso_contact"]))
            head_contacts.append(float(info["head_contact"]))
            body_contacts.append(float(info["body_contact"]))
            velocities.append(float(info["x_velocity"]))
            root_velocities.append(float(info["root_velocity"]))
            action_rates.append(float(info["action_rate"]))

            if episode == 0 and len(frames) < max_video_steps:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            done = terminated or truncated

        stand_returns.append(stand_return)
        original_returns.append(original_return)
        lengths.append(episode_length)

    env.close()

    video_path = video_dir / f"halfcheetah_stand_{step:07d}.mp4"
    screenshot_path = video_dir / f"halfcheetah_stand_{step:07d}.png"
    if frames:
        imageio.mimsave(video_path, frames, fps=fps)
        imageio.imwrite(screenshot_path, frames[-1])

    result = {
        "step": int(step),
        "stand_mean": round(float(np.mean(stand_returns)), 2),
        "stand_std": round(sample_std(stand_returns), 2),
        "original_mean": round(float(np.mean(original_returns)), 2),
        "original_std": round(sample_std(original_returns), 2),
        "mean_length": round(float(np.mean(lengths)), 2),
        "mean_rooty": round(float(np.mean(rootys)), 4),
        "mean_upright": round(float(np.mean(uprights)), 4),
        "mean_rootz": round(float(np.mean(rootzs)), 4),
        "torso_contact_rate": round(float(np.mean(torso_contacts)), 4),
        "head_contact_rate": round(float(np.mean(head_contacts)), 4),
        "body_contact_rate": round(float(np.mean(body_contacts)), 4),
        "mean_x_velocity": round(float(np.mean(velocities)), 4),
        "mean_root_velocity": round(float(np.mean(root_velocities)), 4),
        "mean_action_rate": round(float(np.mean(action_rates)), 4),
        "video": str(video_path) if frames else "",
        "screenshot": str(screenshot_path) if frames else "",
        "model": str(model_path),
    }

    print(
        f"[stand] step={step:,} "
        f"stand={result['stand_mean']:.2f} "
        f"upright={result['mean_upright']:.3f} "
        f"rooty={result['mean_rooty']:.3f} "
        f"torso={result['torso_contact_rate']:.3f} "
        f"head={result['head_contact_rate']:.3f} "
        f"x_vel={result['mean_x_velocity']:.3f} "
        f"root_vel={result['mean_root_velocity']:.3f}"
    )
    return result


class VideoCheckpointCallback(BaseCallback):
    """在指定步数保存站立模型的视频和统计。"""

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
                print(f"\n[stand-video] target={target:,} actual={actual_step:,}")

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
    """训练站立阶段，并保存视频和统计。"""
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

    # 这个 eval_env 也是站立奖励，所以 best_model 按站立表现选。
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
        "w_torso_contact": W_TORSO_CONTACT,
        "w_head_contact": W_HEAD_CONTACT,
        "w_height": W_HEIGHT,
        "height_tolerance": HEIGHT_TOLERANCE,
        "w_x_velocity": W_X_VELOCITY,
        "w_root_velocity": W_ROOT_VELOCITY,
        "w_joint_velocity": W_JOINT_VELOCITY,
        "w_action": W_ACTION,
        "w_action_rate": W_ACTION_RATE,
        "use_original_reward": False,
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
    print(f"env={ENV_ID} seed={SEED} steps={STEPS:,} reward=stand-only")
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

    # 这个原始奖励评测只用于诊断，不应该拿来和 976 比较。
    print(f"\nRunning final {FINAL_EVAL_EPISODES}-episode original-reward diagnostic ...")
    final_result = evaluate_model(
        model_path=LOG_DIR / "final_model.zip",
        env_id=ENV_ID,
        episodes=FINAL_EVAL_EPISODES,
        eval_seed=FINAL_EVAL_SEED,
        video_path=VIDEO_DIR / "halfcheetah_stand_final_eval.mp4",
        screenshot_path=VIDEO_DIR / "halfcheetah_stand_final_eval.png",
        max_video_steps=MAX_VIDEO_STEPS,
    )
    final_result["train_seed"] = SEED
    final_result["steps"] = STEPS
    final_result["reward_config"] = reward_config
    final_result["final_snapshot"] = final_row
    final_result["snapshot_csv"] = str(SNAPSHOT_CSV)
    final_result["note"] = (
        "This is the stand-only reward stage. The original return is only "
        "a diagnostic and is expected to be low because forward reward was removed."
    )
    FINAL_JSON.write_text(
        json.dumps(final_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        f"Final stand evaluation: stand={final_row['stand_mean']:.2f} "
        f"upright={final_row['mean_upright']:.3f} "
        f"torso={final_row['torso_contact_rate']:.3f} "
        f"head={final_row['head_contact_rate']:.3f} "
        f"x_vel={final_row['mean_x_velocity']:.3f}"
    )
    print(
        f"Original-return diagnostic: mean={final_result['mean']:.2f} "
        f"± {final_result['std']:.2f} (not comparable to running baselines)"
    )
    print(f"Saved: {FINAL_JSON}")
    print(f"Videos: {VIDEO_DIR}")
    print(f"Snapshots: {SNAPSHOT_CSV}")
    print("=" * 78)


if __name__ == "__main__":
    main()
