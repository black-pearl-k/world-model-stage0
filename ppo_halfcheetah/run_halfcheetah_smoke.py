"""HalfCheetah-v5 smoke test.

In PyCharm: right-click this file -> Run.
"""

from __future__ import annotations  # 让类型注解延迟求值

from train import PROJECT_DIR, train  # 导入项目目录和训练函数

ENV_ID = "HalfCheetah-v5"  # 使用 HalfCheetah-v5 环境
SEED = 0  # smoke test 只用一个 seed
STEPS = 20_000  # 只跑 2 万步，快速检查流程
EVAL_FREQ = 10_000  # 每 1 万步评测一次
EVAL_EPISODES = 3  # 每次评测跑 3 局，减少评测开销


def main() -> None:  # 定义主函数
    train(  # 调用训练函数
        env_id=ENV_ID,  # 传入环境名
        seed=SEED,  # 传入训练 seed
        steps=STEPS,  # 传入总步数
        logdir=PROJECT_DIR / "logs" / "halfcheetah" / "smoke_seed_0",  # smoke 日志单独存放
        eval_freq=EVAL_FREQ,  # 传入评测频率
        eval_episodes=EVAL_EPISODES,  # 传入每次评测局数
    )


if __name__ == "__main__":  # 只有直接运行这个文件时才执行
    main()  # 调用主函数
