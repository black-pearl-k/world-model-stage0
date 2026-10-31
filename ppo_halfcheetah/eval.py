"""Evaluation helpers for HalfCheetah-v5 PPO models.

No training happens here: the model is loaded, actions are deterministic,
and only rewards are recorded.
"""

from __future__ import annotations  # 让类型注解延迟求值

from pathlib import Path  # 处理文件路径

import gymnasium as gym  # 创建 HalfCheetah 环境
import imageio.v2 as imageio  # 保存视频和图片
import numpy as np  # 计算均值和标准差
from stable_baselines3 import PPO  # 加载 PPO 模型

PROJECT_DIR = Path(__file__).resolve().parent  # 当前文件所在目录


def _resolve(path: str | Path) -> Path:  # 把相对路径转成项目目录下的绝对路径
    path = Path(path)  # 统一转成 Path 对象
    return path if path.is_absolute() else PROJECT_DIR / path  # 绝对路径原样返回，相对路径拼到项目目录


def evaluate_model(  # 评测一个模型的主函数
    model_path: str | Path,  # 模型文件路径
    env_id: str = "HalfCheetah-v5",  # 环境名称
    episodes: int = 10,  # 评测多少局
    eval_seed: int = 1000,  # 评测起始 seed
    video_path: str | Path | None = None,  # 视频保存路径，None 表示不录视频
    screenshot_path: str | Path | None = None,  # 截图保存路径，None 表示不截图
    max_video_steps: int = 300,  # 最多录多少帧，避免视频太大
) -> dict:  # 返回一个包含评测结果的字典
    """Run deterministic evaluation and optionally save a video/screenshot."""

    model_path = _resolve(model_path)  # 把模型路径转成绝对路径
    if not model_path.exists():  # 检查模型文件是否存在
        raise FileNotFoundError(f"Model not found: {model_path}")  # 不存在就报错

    model = PPO.load(str(model_path), device="auto")  # 加载模型，自动选择 CPU 或 GPU
    need_render = video_path is not None or screenshot_path is not None  # 只要需要视频或截图，就需要渲染
    env = gym.make(env_id, render_mode="rgb_array" if need_render else None)  # 需要渲染时用 rgb_array，否则不渲染

    returns: list[float] = []  # 保存每一局的 return
    lengths: list[int] = []  # 保存每一局的步数
    frames: list[np.ndarray] = []  # 保存录制的画面帧

    for episode in range(episodes):  # 循环评测每一局
        obs, _ = env.reset(seed=eval_seed + episode)  # 用固定 seed 重置环境
        done = False  # 当前局是否结束
        episode_return = 0.0  # 当前局累计回报
        episode_length = 0  # 当前局累计步数

        while not done:  # 只要这一局没结束就继续
            action, _ = model.predict(obs, deterministic=True)  # 用确定性策略选择动作
            obs, reward, terminated, truncated, _ = env.step(action)  # 执行动作，得到下一步结果
            episode_return += float(reward)  # 累加这一步的奖励
            episode_length += 1  # 步数加一

            if need_render and len(frames) < max_video_steps:  # 如果需要渲染，并且还没达到最大帧数
                frame = env.render()  # 渲染当前画面
                if frame is not None:  # 有些渲染模式可能返回 None
                    frames.append(frame)  # 把画面加入帧列表

            done = terminated or truncated  # 真正结束或时间到，都算这一局结束

        returns.append(episode_return)  # 保存这一局的回报
        lengths.append(episode_length)  # 保存这一局的长度

    env.close()  # 关闭环境，释放资源

    saved_video = None  # 默认没有保存视频
    if video_path is not None and frames:  # 如果要求视频，并且确实有画面
        video_path = _resolve(video_path)  # 转成绝对路径
        video_path.parent.mkdir(parents=True, exist_ok=True)  # 创建视频目录
        imageio.mimsave(video_path, frames, fps=30)  # 按 30 帧每秒保存视频
        saved_video = str(video_path)  # 记录视频路径

    saved_screenshot = None  # 默认没有保存截图
    if screenshot_path is not None and frames:  # 如果要求截图，并且确实有画面
        screenshot_path = _resolve(screenshot_path)  # 转成绝对路径
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)  # 创建截图目录
        imageio.imwrite(screenshot_path, frames[-1])  # 保存最后一帧作为截图
        saved_screenshot = str(screenshot_path)  # 记录截图路径

    return {  # 返回评测结果
        "env": env_id,  # 环境名
        "model": str(model_path),  # 模型路径
        "episodes": returns,  # 每一局的回报列表
        "lengths": lengths,  # 每一局的长度列表
        "mean": float(np.mean(returns)),  # 所有局的平均回报
        "std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,  # 样本标准差
        "video": saved_video,  # 视频路径
        "screenshot": saved_screenshot,  # 截图路径
    }
