"""Train HalfCheetah-v5 with three independent seeds.

In PyCharm: right-click this file -> Run.
No terminal arguments are needed.
"""

from __future__ import annotations  # 让类型注解延迟求值

from pathlib import Path  # 处理文件路径

from train import PROJECT_DIR, train  # 导入项目目录和训练函数

ENV_ID = "HalfCheetah-v5"  # 使用 HalfCheetah-v5 环境
SEEDS = (0, 1, 2)  # 训练三个独立的 seed
STEPS = 300_000  # 每个 seed 训练 30 万步
EVAL_FREQ = 25_000  # 每 2.5 万步评测一次
EVAL_EPISODES = 5  # 每次评测跑 5 局


def main() -> None:  # 定义主函数
    for seed in SEEDS:  # 依次训练每个 seed
        logdir = PROJECT_DIR / "logs" / "halfcheetah" / f"seed_{seed}"  # 每个 seed 用单独日志目录
        print("\n" + "#" * 70)  # 打印分隔线
        print(f"Training {ENV_ID}, seed={seed}, steps={STEPS}")  # 打印当前训练信息
        print("#" * 70 + "\n")  # 再打印一条分隔线
        train(  # 调用训练函数
            env_id=ENV_ID,  # 传入环境名
            seed=seed,  # 传入当前 seed
            steps=STEPS,  # 传入总步数
            logdir=logdir,  # 传入当前 seed 的日志目录
            eval_freq=EVAL_FREQ,  # 传入评测频率
            eval_episodes=EVAL_EPISODES,  # 传入每次评测局数
        )

    print("\nAll HalfCheetah seeds finished.")  # 所有 seed 训练完成
    print(f"Models and logs are under: {Path(PROJECT_DIR) / 'logs' / 'halfcheetah'}")  # 打印模型和日志根目录


if __name__ == "__main__":  # 只有直接运行这个文件时才执行
    main()  # 调用主函数
