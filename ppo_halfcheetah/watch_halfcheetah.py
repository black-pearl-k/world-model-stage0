"""Watch any existing HalfCheetah-v5 PPO model in a real-time MuJoCo window.

In PyCharm: right-click this file -> Run.

The script scans all .zip model files under logs/ and models/, lists them,
asks you to choose one by number, and then opens the MuJoCo viewer.

Controls:
  - type a number, then Enter: load that model
  - press Enter directly: load the first model (newest by modification time)
  - type q, then Enter: quit
"""

from __future__ import annotations  # 让类型注解延迟求值

import time  # 控制慢放速度
from datetime import datetime  # 显示模型修改时间
from pathlib import Path  # 处理文件路径

import gymnasium as gym  # 创建 HalfCheetah 环境
import mujoco  # 设置 MuJoCo 跟随镜头
from stable_baselines3 import PPO  # 加载 PPO 模型

PROJECT_DIR = Path(__file__).resolve().parent  # 当前文件所在目录
MODEL_ROOTS = (  # 扫描模型的根目录
    PROJECT_DIR / "logs",  # 训练日志和模型
    PROJECT_DIR / "models",  # 预留模型目录
)
EPISODES = 3  # 选中模型后跑 3 局
MAX_STEPS = 1000  # 每局最多 1000 步
EVAL_SEED = 1000  # 观看时使用的起始 seed
FOLLOW_CAMERA = True  # 是否开启跟随镜头
SLOWDOWN_SECONDS = 0.05  # 每步额外等待时间，越大越慢
CAMERA_DISTANCE = 5.0  # 相机离躯干的距离
CAMERA_ELEVATION = -20.0  # 相机俯仰角
CAMERA_AZIMUTH = 90.0  # 相机水平角度


def find_models() -> list[Path]:  # 找到所有已经存在的模型文件
    models: set[Path] = set()  # 用集合避免重复
    for root in MODEL_ROOTS:  # 依次扫描 logs/ 和 models/
        if not root.exists():  # 如果目录不存在
            continue  # 跳过
        for path in root.rglob("*.zip"):  # 递归查找所有 zip 模型
            if path.is_file():  # 只保留真实文件
                models.add(path)  # 加入集合
    return sorted(  # 按修改时间从新到旧排序
        models,  # 所有模型
        key=lambda path: path.stat().st_mtime,  # 用文件修改时间排序
        reverse=True,  # 最新的排前面
    )


def display_path(path: Path) -> str:  # 显示相对项目目录的路径
    try:
        return str(path.relative_to(PROJECT_DIR))  # 能相对就显示相对路径
    except ValueError:
        return str(path)  # 不在项目目录下就显示绝对路径


def print_models(models: list[Path]) -> None:  # 打印模型列表
    print("\n" + "=" * 78)  # 打印分隔线
    print(f"找到 {len(models)} 个模型（按修改时间从新到旧）")  # 打印模型数量
    print("=" * 78)  # 再打印分隔线
    for index, path in enumerate(models, start=1):  # 从 1 开始编号
        stat = path.stat()  # 读取文件信息
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")  # 格式化修改时间
        size_kb = stat.st_size / 1024  # 文件大小转成 KB
        print(  # 打印一行模型信息
            f"[{index:2d}] {modified}  {size_kb:7.1f} KB  {display_path(path)}"
        )
    print("=" * 78)  # 打印分隔线


def choose_model(models: list[Path]) -> Path | None:  # 让用户选择一个模型
    while True:  # 一直询问直到输入合法
        raw = input(  # 读取用户输入
            f"\n输入模型编号 1-{len(models)}（直接回车=1，q=退出）: "
        ).strip()  # 去掉首尾空格
        if raw.lower() in {"q", "quit", "exit"}:  # 如果用户想退出
            return None  # 返回 None
        if raw == "":  # 如果直接回车
            return models[0]  # 默认选择第一个，也就是最新的模型
        if raw.isdigit():  # 如果输入的是数字
            index = int(raw)  # 转成整数
            if 1 <= index <= len(models):  # 检查编号范围
                return models[index - 1]  # 返回对应模型
        print(f"无效输入：{raw!r}，请输入 1-{len(models)}，或 q 退出。")  # 提示重新输入


def setup_follow_camera(env: gym.Env) -> None:  # 设置 MuJoCo 跟随镜头
    env.render()  # 先渲染一帧，确保 viewer 已经创建
    viewer = env.unwrapped.mujoco_renderer.viewer  # 取出 MuJoCo 窗口 viewer
    if viewer is None:  # 如果 viewer 创建失败
        return  # 直接返回
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


def main() -> None:  # 定义主函数
    models = find_models()  # 扫描所有模型
    if not models:  # 如果一个模型都没有
        print("No model found.")  # 打印提示
        print(f"Scanned: {', '.join(str(root) for root in MODEL_ROOTS)}")  # 打印扫描目录
        print("Run a training script first.")  # 提示先训练
        return  # 直接结束

    print_models(models)  # 打印模型列表
    model_path = choose_model(models)  # 让用户选择
    if model_path is None:  # 如果用户取消
        print("Cancelled.")  # 打印取消信息
        return  # 直接结束

    print(f"\nLoading model: {display_path(model_path)}")  # 打印选中的模型
    model = PPO.load(str(model_path), device="auto")  # 加载选中的 PPO 模型

    env = gym.make("HalfCheetah-v5", render_mode="human")  # 创建带实时窗口的环境
    try:  # 无论是否异常，最后都关闭环境
        for episode in range(EPISODES):  # 循环观看每一局
            obs, _ = env.reset(seed=EVAL_SEED + episode)  # 用固定 seed 重置环境
            if FOLLOW_CAMERA and episode == 0:  # 第一局重置后再创建窗口和跟随镜头
                setup_follow_camera(env)  # 设置跟随镜头
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
