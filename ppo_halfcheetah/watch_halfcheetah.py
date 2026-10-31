"""Watch a trained HalfCheetah-v5 model in a real-time MuJoCo window.

In PyCharm: right-click this file -> Run.

It loads the model selected by SEED and USE_BEST.
By default: seed_1/best_model/best_model.zip.
If that does not exist, it falls back to final_model.zip and then the smoke model.
"""

from __future__ import annotations  # 让类型注解延迟求值

import time  # 控制慢放速度
from pathlib import Path  # 处理文件路径

import gymnasium as gym  # 创建 HalfCheetah 环境
import mujoco  # 设置 MuJoCo 跟随镜头
from stable_baselines3 import PPO  # 加载 PPO 模型

PROJECT_DIR = Path(__file__).resolve().parent  # 当前文件所在目录
SEED = 1  # 想观看哪个 seed，就改这里：0、1、2
USE_BEST = True  # True 用 best_model.zip，False 用 final_model.zip


def build_model_candidates(seed: int, use_best: bool) -> tuple[Path, ...]:  # 根据 seed 和模型类型生成候选路径
    logdir = PROJECT_DIR / "logs" / "halfcheetah" / f"seed_{seed}"  # 当前 seed 的日志目录
    candidates: list[Path] = []  # 保存候选模型路径
    if use_best:  # 如果优先使用训练中最好的模型
        candidates.append(logdir / "best_model" / "best_model.zip")  # EvalCallback 保存的 best_model
    candidates.append(logdir / "final_model.zip")  # 训练结束保存的 final_model
    candidates.append(  # 如果正式模型都不存在，就退回 smoke 模型
        PROJECT_DIR / "logs" / "halfcheetah" / "smoke_seed_0" / "final_model.zip"
    )
    return tuple(candidates)  # 转成元组返回


MODEL_CANDIDATES = build_model_candidates(SEED, USE_BEST)  # 生成当前要查找的模型路径列表
EPISODES = 3  # 实时观看跑 3 局
MAX_STEPS = 1000  # 每局最多 1000 步
EVAL_SEED = 1000  # 观看时使用的起始 seed
FOLLOW_CAMERA = True  # 是否开启跟随镜头
SLOWDOWN_SECONDS = 0.05  # 每步额外等待时间，越大越慢
CAMERA_DISTANCE = 5.0  # 相机离躯干的距离
CAMERA_ELEVATION = -20.0  # 相机俯仰角
CAMERA_AZIMUTH = 90.0  # 相机水平角度




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
            if FOLLOW_CAMERA and episode == 0:  # 第一局重置后再创建窗口和跟随镜头
                env.render()  # 重置之后渲染一帧，确保 viewer 已经创建
                viewer = env.unwrapped.mujoco_renderer.viewer  # 取出 MuJoCo 窗口 viewer
                if viewer is not None:  # 如果 viewer 创建成功
                    body_id = mujoco.mj_name2id(  # 找到躯干 body 的编号
                        env.unwrapped.model,  # MuJoCo 模型
                        mujoco.mjtObj.mjOBJ_BODY,  # 查找 body
                        "torso",  # 躯干名字
                    )
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING  # 切换成跟踪相机
                    viewer.cam.trackbodyid = body_id  # 让相机跟踪躯干
                    viewer.cam.distance = CAMERA_DISTANCE  # 设置相机距离
                    viewer.cam.elevation = CAMERA_ELEVATION  # 设置相机俯仰角
                    viewer.cam.azimuth = CAMERA_AZIMUTH  # 设置相机水平角度
            done = False  # 当前局是否结束
            total_reward = 0.0  # 当前局累计回报
            steps = 0  # 当前局步数

            while not done and steps < MAX_STEPS:  # 没结束且没超过最大步数就继续
                action, _ = model.predict(obs, deterministic=True)  # 用确定性策略选择动作
                obs, reward, terminated, truncated, _ = env.step(action)  # 执行动作
                total_reward += float(reward)  # 累加奖励
                steps += 1  # 步数加一
                done = terminated or truncated  # 判断这一局是否结束
                time.sleep(SLOWDOWN_SECONDS)  # 每步暂停一下，方便观察

            print(  # 打印这一局结果
                f"episode {episode}: return={total_reward:.2f}, steps={steps}"  # 回报和步数
            )
    finally:  # 无论是否异常都执行
        env.close()  # 关闭窗口和环境


if __name__ == "__main__":  # 只有直接运行这个文件时才执行
    main()  # 调用主函数
