"""Watch a trained HalfCheetah-v5 model in a real-time MuJoCo window.

In PyCharm: right-click this file -> Run.

It prefers logs/halfcheetah/seed_0/final_model.zip.
If that does not exist, it falls back to the smoke-test model.
"""

from __future__ import annotations  # 让类型注解延迟求值

from pathlib import Path  # 处理文件路径

import gymnasium as gym  # 创建 HalfCheetah 环境
from stable_baselines3 import PPO  # 加载 PPO 模型

PROJECT_DIR = Path(__file__).resolve().parent  # 当前文件所在目录
MODEL_CANDIDATES = (  # 按优先级列出可能的模型路径
    PROJECT_DIR / "logs" / "halfcheetah" / "seed_0" / "final_model.zip",  # 优先使用正式训练的 seed 0
    PROJECT_DIR / "logs" / "halfcheetah" / "smoke_seed_0" / "final_model.zip",  # 没有正式模型就用 smoke 模型
)
EPISODES = 3  # 实时观看跑 3 局
MAX_STEPS = 1000  # 每局最多 1000 步
EVAL_SEED = 1000  # 观看时使用的起始 seed


def find_model() -> Path | None:  # 查找第一个存在的模型
    for path in MODEL_CANDIDATES:  # 依次检查候选路径
        if path.exists():  # 如果文件存在
            return path  # 返回这个路径
    return None  # 一个都不存在就返回 None


def main() -> None:  # 定义主函数
    model_path = find_model()  # 查找模型
    if model_path is None:  # 如果没有找到模型
        print("No model found.")  # 打印提示
        print("Run run_halfcheetah_smoke.py or run_halfcheetah_3seeds.py first.")  # 提示先训练
        for path in MODEL_CANDIDATES:  # 打印所有预期路径
            print(f"  expected: {path}")  # 打印候选路径
        return  # 直接结束

    print(f"Loading model: {model_path}")  # 打印正在加载的模型
    model = PPO.load(str(model_path), device="auto")  # 加载模型，自动选择设备

    env = gym.make("HalfCheetah-v5", render_mode="human")  # 创建带实时窗口的环境
    try:  # 无论是否异常，最后都关闭环境
        for episode in range(EPISODES):  # 循环观看每一局
            obs, _ = env.reset(seed=EVAL_SEED + episode)  # 用固定 seed 重置环境
            done = False  # 当前局是否结束
            total_reward = 0.0  # 当前局累计回报
            steps = 0  # 当前局步数

            while not done and steps < MAX_STEPS:  # 没结束且没超过最大步数就继续
                action, _ = model.predict(obs, deterministic=True)  # 用确定性策略选择动作
                obs, reward, terminated, truncated, _ = env.step(action)  # 执行动作
                total_reward += float(reward)  # 累加奖励
                steps += 1  # 步数加一
                done = terminated or truncated  # 判断这一局是否结束

            print(  # 打印这一局结果
                f"episode {episode}: return={total_reward:.2f}, steps={steps}"  # 回报和步数
            )
    finally:  # 无论是否异常都执行
        env.close()  # 关闭窗口和环境


if __name__ == "__main__":  # 只有直接运行这个文件时才执行
    main()  # 调用主函数
