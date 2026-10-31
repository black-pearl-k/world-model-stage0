"""Train HalfCheetah-v5 seed 0 for 1,000,000 steps and save progress videos.

In PyCharm: right-click this file -> Run.

This script does not modify the existing 300K baseline. It writes to:
  logs/halfcheetah_1m/seed_0/
  videos/halfcheetah_1m/
  results/halfcheetah_1m_seed_0.json
  results/halfcheetah_1m_snapshots.csv

At steps 0, 250K, 500K, 750K and 1M it saves:
  - one short deterministic video
  - one screenshot
  - one model checkpoint
  - one row in snapshots.csv
"""

from __future__ import annotations  # 让类型注解延迟求值

import csv  # 记录每个 checkpoint 的回报
import json  # 保存训练配置和最终评测结果
from pathlib import Path  # 处理文件路径

import gymnasium as gym  # 创建 HalfCheetah 环境
import imageio.v2 as imageio  # 保存视频和截图
import numpy as np  # 计算平均回报和标准差
from stable_baselines3 import PPO  # 使用 PPO 算法
from stable_baselines3.common.callbacks import (  # 训练中使用的回调
    BaseCallback,  # 自定义回调基类
    CheckpointCallback,  # 定期保存模型
    EvalCallback,  # 定期评测并保存 best_model
)

from eval import evaluate_model  # 最终 10 局评测
from train import PROJECT_DIR, build_vec_env  # 复用环境创建函数和项目目录

ENV_ID = "HalfCheetah-v5"  # 使用 HalfCheetah-v5
SEED = 0  # 只训练 seed 0
STEPS = 1_000_000  # 总训练步数：100 万步

# 这些超参和 300K baseline 完全一致，方便单独观察“训练更久”的影响。
N_STEPS = 2048  # 每次采样 2048 步后更新一次
BATCH_SIZE = 64  # 每次梯度更新使用 64 条样本
N_EPOCHS = 10  # 每批数据训练 10 轮
LR = 3e-4  # 学习率
GAMMA = 0.99  # 折扣因子
GAE_LAMBDA = 0.95  # GAE 的 lambda
CLIP_RANGE = 0.2  # PPO 裁剪范围
EVAL_FREQ = 25_000  # 每 25K 步做一次验证集评测
EVAL_EPISODES = 5  # 每次验证评测跑 5 局
EVAL_SEED = 12_345  # 训练中验证集使用的 seed
DEVICE = "cpu"  # 当前环境使用 CPU 版 PyTorch

SNAPSHOT_CHECKPOINTS = (250_000, 500_000, 750_000)  # 中间视频 checkpoint；0 和 1M 单独处理
SNAPSHOT_EPISODES = 3  # 每个 checkpoint 跑 3 局，视频只保存第一局
SNAPSHOT_EVAL_SEED = 1000  # 视频和 checkpoint 回报使用固定 seed
MAX_VIDEO_STEPS = 300  # 每个视频最多 300 帧，约 10 秒
VIDEO_FPS = 30  # 视频帧率
FINAL_EVAL_EPISODES = 10  # 最终评测 10 局，和 baseline 一致
FINAL_EVAL_SEED = 1000  # 最终评测 seed 1000–1009

LOG_DIR = PROJECT_DIR / "logs" / "halfcheetah_1m" / "seed_0"  # 1M 实验日志目录
VIDEO_DIR = PROJECT_DIR / "videos" / "halfcheetah_1m"  # 1M 实验视频目录
RESULTS_DIR = PROJECT_DIR / "results"  # 结果目录
SNAPSHOT_CSV = RESULTS_DIR / "halfcheetah_1m_snapshots.csv"  # checkpoint 汇总表
SNAPSHOT_MODEL_DIR = LOG_DIR / "snapshots"  # checkpoint 模型目录
SNAPSHOT_FIELDS = (  # checkpoint 汇总表的列名
    "step",  # 训练步数
    "mean_return",  # 3 局平均回报
    "std_return",  # 3 局回报标准差
    "mean_length",  # 平均局长
    "video",  # 视频路径
    "model",  # 模型路径
)


def init_snapshot_csv(path: Path) -> None:  # 创建 checkpoint 汇总表并写表头
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保 results 目录存在
    with path.open("w", newline="", encoding="utf-8") as file:  # 以覆盖模式打开 CSV
        writer = csv.DictWriter(file, fieldnames=SNAPSHOT_FIELDS)  # 创建 CSV 写入器
        writer.writeheader()  # 写入表头


def append_snapshot(path: Path, row: dict) -> None:  # 追加一行 checkpoint 结果
    with path.open("a", newline="", encoding="utf-8") as file:  # 以追加模式打开 CSV
        writer = csv.DictWriter(file, fieldnames=SNAPSHOT_FIELDS)  # 使用固定列名
        writer.writerow(row)  # 写入这一行


def record_snapshot(  # 记录某个训练阶段的模型效果
    model: PPO,  # 当前 PPO 模型
    step: int,  # 当前训练步数
    video_dir: Path,  # 视频保存目录
    model_dir: Path,  # checkpoint 保存目录
    episodes: int,  # 评测局数
    eval_seed: int,  # 评测起始 seed
    max_video_steps: int,  # 视频最多帧数
    fps: int,  # 视频帧率
) -> dict:  # 返回这一阶段的汇总结果
    model_dir.mkdir(parents=True, exist_ok=True)  # 创建 checkpoint 目录
    video_dir.mkdir(parents=True, exist_ok=True)  # 创建视频目录

    model_path = model_dir / f"model_{step:07d}.zip"  # 保存当前模型快照
    model.save(str(model_path))  # 写入模型文件

    env = gym.make(ENV_ID, render_mode="rgb_array")  # 用离屏渲染录制视频
    returns: list[float] = []  # 保存每一局回报
    lengths: list[int] = []  # 保存每一局长度
    frames: list[np.ndarray] = []  # 只保存第一局的画面帧

    for episode in range(episodes):  # 跑多局让回报更稳定
        obs, _ = env.reset(seed=eval_seed + episode)  # 固定这一局的初始状态
        done = False  # 当前局是否结束
        episode_return = 0.0  # 当前局累计回报
        episode_length = 0  # 当前局步数

        while not done:  # 一直跑到这一局结束
            action, _ = model.predict(obs, deterministic=True)  # 用确定性策略选择动作
            obs, reward, terminated, truncated, _ = env.step(action)  # 执行动作
            episode_return += float(reward)  # 累加奖励
            episode_length += 1  # 步数加一

            if episode == 0 and len(frames) < max_video_steps:  # 只录第一局的前 300 帧
                frame = env.render()  # 离屏渲染当前画面
                if frame is not None:  # 防止某些渲染模式返回 None
                    frames.append(frame)  # 保存这一帧

            done = terminated or truncated  # 判断这一局是否结束

        returns.append(episode_return)  # 记录这一局回报
        lengths.append(episode_length)  # 记录这一局长度

    env.close()  # 关闭离屏环境

    video_path = video_dir / f"halfcheetah_1m_{step:07d}.mp4"  # 视频文件名带步数
    screenshot_path = video_dir / f"halfcheetah_1m_{step:07d}.png"  # 截图文件名带步数
    if frames:  # 如果确实录到了画面
        imageio.mimsave(video_path, frames, fps=fps)  # 保存 mp4 视频
        imageio.imwrite(screenshot_path, frames[-1])  # 保存最后一帧截图

    mean_return = float(np.mean(returns))  # 多局平均回报
    std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0  # 样本标准差
    mean_length = float(np.mean(lengths))  # 平均局长

    result = {  # 收集这一阶段的汇总信息
        "step": step,  # 训练步数
        "mean_return": round(mean_return, 2),  # 保留两位小数
        "std_return": round(std_return, 2),  # 保留两位小数
        "mean_length": round(mean_length, 2),  # 保留两位小数
        "video": str(video_path) if frames else "",  # 视频路径
        "model": str(model_path),  # 模型路径
    }
    print(  # 打印这一阶段的结果
        f"[snapshot] step={step:,} "  # 当前步数
        f"mean={mean_return:.2f} std={std_return:.2f} "  # 平均回报和标准差
        f"video={video_path.name}"  # 视频文件名
    )
    return result  # 返回汇总结果


class VideoCheckpointCallback(BaseCallback):  # 在指定训练步数保存视频和模型
    def __init__(  # 初始化回调
        self,
        checkpoints: tuple[int, ...],  # 需要录视频的步数
        video_dir: Path,  # 视频目录
        model_dir: Path,  # checkpoint 目录
        csv_path: Path,  # 汇总表路径
        episodes: int,  # 每次录视频跑几局
        eval_seed: int,  # 视频评测 seed
        max_video_steps: int,  # 视频最多帧数
        fps: int,  # 视频帧率
        verbose: int = 1,  # 是否打印信息
    ) -> None:
        super().__init__(verbose=verbose)  # 调用父类构造函数
        self.checkpoints = tuple(sorted(checkpoints))  # 按从小到大排序
        self.next_index = 0  # 下一个要触发的 checkpoint 下标
        self.video_dir = video_dir  # 保存视频目录
        self.model_dir = model_dir  # 保存模型目录
        self.csv_path = csv_path  # 保存 CSV 路径
        self.episodes = episodes  # 每个 checkpoint 跑几局
        self.eval_seed = eval_seed  # 评测 seed
        self.max_video_steps = max_video_steps  # 视频帧数上限
        self.fps = fps  # 视频帧率

    def _on_step(self) -> bool:  # 每个环境步都会调用
        if self.next_index >= len(self.checkpoints):  # 所有 checkpoint 都处理完了
            return True  # 继续训练

        target = self.checkpoints[self.next_index]  # 当前等待的 checkpoint
        if self.model.num_timesteps >= target:  # 到达或超过目标步数
            actual_step = int(self.model.num_timesteps)  # 记录实际训练步数
            if self.verbose > 0:  # 如果需要打印
                print(  # 打印触发信息
                    f"\n[video] target={target:,} actual={actual_step:,}"
                )
            row = record_snapshot(  # 录制这一阶段的视频和模型
                model=self.model,  # 当前 PPO 模型
                step=actual_step,  # 实际步数
                video_dir=self.video_dir,  # 视频目录
                model_dir=self.model_dir,  # 模型目录
                episodes=self.episodes,  # 评测局数
                eval_seed=self.eval_seed,  # 评测 seed
                max_video_steps=self.max_video_steps,  # 视频帧数上限
                fps=self.fps,  # 视频帧率
            )
            append_snapshot(self.csv_path, row)  # 把结果追加到 CSV
            self.next_index += 1  # 处理下一个 checkpoint

        return True  # 继续训练


def main() -> None:  # 主函数
    LOG_DIR.mkdir(parents=True, exist_ok=True)  # 创建 1M 实验日志目录
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)  # 创建视频目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # 创建结果目录
    SNAPSHOT_MODEL_DIR.mkdir(parents=True, exist_ok=True)  # 创建 checkpoint 目录
    init_snapshot_csv(SNAPSHOT_CSV)  # 初始化 checkpoint 汇总表

    train_env = build_vec_env(ENV_ID, SEED, LOG_DIR / "monitor.csv")  # 创建训练环境
    eval_env = build_vec_env(ENV_ID, EVAL_SEED, LOG_DIR / "eval_monitor.csv")  # 创建验证环境

    eval_cb = EvalCallback(  # 定期验证并保存 best_model
        eval_env,  # 使用单独的验证环境
        best_model_save_path=str(LOG_DIR / "best_model"),  # best_model 保存目录
        log_path=str(LOG_DIR),  # 保存 evaluations.npz
        eval_freq=EVAL_FREQ,  # 每 25K 步评测一次
        n_eval_episodes=EVAL_EPISODES,  # 每次评测 5 局
        deterministic=True,  # 使用确定性动作
        verbose=1,  # 打印评测结果
    )

    checkpoint_cb = CheckpointCallback(  # 每 100K 步保存一个模型
        save_freq=max(1, STEPS // 10),  # 1M / 10 = 100K
        save_path=str(LOG_DIR / "checkpoints"),  # checkpoint 保存目录
        name_prefix="ppo",  # 文件名前缀
        verbose=0,  # 不打印太多信息
    )

    video_cb = VideoCheckpointCallback(  # 在 250K / 500K / 750K 录视频
        checkpoints=SNAPSHOT_CHECKPOINTS,  # 中间 checkpoint
        video_dir=VIDEO_DIR,  # 视频目录
        model_dir=SNAPSHOT_MODEL_DIR,  # 模型快照目录
        csv_path=SNAPSHOT_CSV,  # 汇总表
        episodes=SNAPSHOT_EPISODES,  # 每个 checkpoint 跑 3 局
        eval_seed=SNAPSHOT_EVAL_SEED,  # 视频评测 seed
        max_video_steps=MAX_VIDEO_STEPS,  # 每个视频最多 300 帧
        fps=VIDEO_FPS,  # 视频帧率
        verbose=1,  # 打印 checkpoint 信息
    )

    model = PPO(  # 创建 PPO 模型
        "MlpPolicy",  # 使用 MLP 策略
        train_env,  # 训练环境
        learning_rate=LR,  # 学习率
        n_steps=N_STEPS,  # 每次采样步数
        batch_size=BATCH_SIZE,  # 小批量大小
        n_epochs=N_EPOCHS,  # 每批数据训练轮数
        gamma=GAMMA,  # 折扣因子
        gae_lambda=GAE_LAMBDA,  # GAE lambda
        clip_range=CLIP_RANGE,  # PPO 裁剪范围
        tensorboard_log=str(LOG_DIR / "tb"),  # TensorBoard 日志
        seed=SEED,  # 训练 seed
        device=DEVICE,  # 使用 CPU
        verbose=1,  # 打印训练进度
    )

    snapshot_steps = (0,) + tuple(SNAPSHOT_CHECKPOINTS) + (STEPS,)  # 所有视频节点
    config = {  # 保存这次实验的完整配置
        "env_id": ENV_ID,  # 环境名
        "seed": SEED,  # 训练 seed
        "steps": STEPS,  # 总训练步数
        "n_steps": N_STEPS,  # 采样步数
        "batch_size": BATCH_SIZE,  # 小批量大小
        "n_epochs": N_EPOCHS,  # 训练轮数
        "lr": LR,  # 学习率
        "gamma": GAMMA,  # 折扣因子
        "gae_lambda": GAE_LAMBDA,  # GAE lambda
        "clip_range": CLIP_RANGE,  # 裁剪范围
        "eval_freq": EVAL_FREQ,  # 验证频率
        "eval_episodes": EVAL_EPISODES,  # 验证局数
        "eval_seed": EVAL_SEED,  # 验证 seed
        "device": DEVICE,  # 设备
        "snapshot_steps": snapshot_steps,  # 视频节点
        "snapshot_episodes": SNAPSHOT_EPISODES,  # 每个节点评测局数
        "max_video_steps": MAX_VIDEO_STEPS,  # 视频帧数上限
        "video_fps": VIDEO_FPS,  # 视频帧率
    }
    (LOG_DIR / "config.json").write_text(  # 写入配置文件
        json.dumps(config, indent=2, ensure_ascii=False),  # 带缩进的 JSON
        encoding="utf-8",  # 使用 UTF-8
    )

    print("=" * 70)  # 打印分隔线
    print(f"env={ENV_ID} seed={SEED} steps={STEPS:,}")  # 打印训练信息
    print(f"logdir={LOG_DIR}")  # 打印日志目录
    print(f"videos={VIDEO_DIR}")  # 打印视频目录
    print("=" * 70)  # 再打印分隔线

    initial_row = record_snapshot(  # 训练开始前录一次初始策略
        model=model,  # 刚创建的 PPO 模型
        step=0,  # 第 0 步
        video_dir=VIDEO_DIR,  # 视频目录
        model_dir=SNAPSHOT_MODEL_DIR,  # 模型快照目录
        episodes=SNAPSHOT_EPISODES,  # 评测局数
        eval_seed=SNAPSHOT_EVAL_SEED,  # 评测 seed
        max_video_steps=MAX_VIDEO_STEPS,  # 视频帧数上限
        fps=VIDEO_FPS,  # 视频帧率
    )
    append_snapshot(SNAPSHOT_CSV, initial_row)  # 把第 0 步结果写入 CSV

    model.learn(  # 开始 1M 步训练
        total_timesteps=STEPS,  # 总步数
        callback=[eval_cb, checkpoint_cb, video_cb],  # 三个回调同时工作
    )
    model.save(str(LOG_DIR / "final_model"))  # 保存最终模型

    final_row = record_snapshot(  # 训练结束后录 1M 步的模型
        model=model,  # 最终模型
        step=STEPS,  # 1M 步
        video_dir=VIDEO_DIR,  # 视频目录
        model_dir=SNAPSHOT_MODEL_DIR,  # 模型快照目录
        episodes=SNAPSHOT_EPISODES,  # 评测局数
        eval_seed=SNAPSHOT_EVAL_SEED,  # 评测 seed
        max_video_steps=MAX_VIDEO_STEPS,  # 视频帧数上限
        fps=VIDEO_FPS,  # 视频帧率
    )
    append_snapshot(SNAPSHOT_CSV, final_row)  # 把 1M 步结果写入 CSV

    train_env.close()  # 关闭训练环境
    eval_env.close()  # 关闭验证环境

    print(f"\nRunning final {FINAL_EVAL_EPISODES}-episode evaluation ...")  # 提示开始最终评测
    final_result = evaluate_model(  # 用最终模型跑 10 局
        model_path=LOG_DIR / "final_model.zip",  # 最终模型路径
        env_id=ENV_ID,  # 环境名
        episodes=FINAL_EVAL_EPISODES,  # 10 局
        eval_seed=FINAL_EVAL_SEED,  # seed 1000–1009
    )
    final_result["train_seed"] = SEED  # 记录训练 seed
    final_result["steps"] = STEPS  # 记录总步数
    final_result_path = RESULTS_DIR / "halfcheetah_1m_seed_0.json"  # 最终结果路径
    final_result_path.write_text(  # 写入最终评测 JSON
        json.dumps(final_result, indent=2, ensure_ascii=False),  # 带缩进的 JSON
        encoding="utf-8",  # 使用 UTF-8
    )

    print("=" * 70)  # 打印分隔线
    print(f"Final eval: mean={final_result['mean']:.2f} ± {final_result['std']:.2f}")  # 打印最终结果
    print(f"Saved: {final_result_path}")  # 打印结果文件
    print(f"Videos: {VIDEO_DIR}")  # 打印视频目录
    print(f"Snapshots: {SNAPSHOT_CSV}")  # 打印 checkpoint 汇总表
    print("=" * 70)  # 再打印分隔线


if __name__ == "__main__":  # 只有直接运行这个文件时才执行
    main()  # 调用主函数
