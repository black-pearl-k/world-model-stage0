"""为最终 2.0 m/s 阶段补跑 seed 1 和 seed 2。

在 PyCharm 里右键这个文件 -> Run。

当前只有 seed 0 有完整的 1.95 m/s 前一阶段模型，所以 seed 1/2
都从同一个模型继续：

    logs/halfcheetah_native/seed_0/final_model.zip

注意：这属于“共享初始化的微调”，不是从零开始的独立 seed。
但训练采样、动作噪声和模型更新仍然分别使用 seed 1 / seed 2。

输出：
    logs/halfcheetah_final/seed_1/
    logs/halfcheetah_final/seed_2/
    results/halfcheetah_final_seed_1.json
    results/halfcheetah_final_seed_2.json
"""

from __future__ import annotations  # 让类型注解延迟求值

import run_halfcheetah_final as base  # 复用最终速度阶段的训练逻辑

SEEDS = (1, 2)  # seed 0 已经训练过
SHARED_INITIAL_MODEL = (
    base.PROJECT_DIR
    / "logs"
    / "halfcheetah_native"
    / "seed_0"
    / "final_model.zip"
)


def main() -> None:
    """依次为 seed 1 和 seed 2 训练最终速度模型。"""
    if not SHARED_INITIAL_MODEL.exists():
        raise FileNotFoundError(
            f"Missing shared initial model: {SHARED_INITIAL_MODEL}"
        )

    for seed in SEEDS:
        base.SEED = seed
        base.INITIAL_MODEL = SHARED_INITIAL_MODEL

        base.LOG_DIR = (
            base.PROJECT_DIR / "logs" / "halfcheetah_final" / f"seed_{seed}"
        )
        base.VIDEO_DIR = (
            base.PROJECT_DIR / "videos" / "halfcheetah_final" / f"seed_{seed}"
        )
        base.SNAPSHOT_CSV = (
            base.RESULTS_DIR
            / f"halfcheetah_final_seed_{seed}_snapshots.csv"
        )
        base.FINAL_JSON = (
            base.RESULTS_DIR / f"halfcheetah_final_seed_{seed}.json"
        )
        base.SNAPSHOT_MODEL_DIR = base.LOG_DIR / "snapshots"

        print("\n" + "#" * 78)
        print(
            f"Training final 2.0 m/s stage: seed={seed}, "
            f"initial={SHARED_INITIAL_MODEL}"
        )
        print("#" * 78 + "\n")

        base.main()

    print("\nFinal 2.0 m/s seeds 1 and 2 finished.")
    print(
        "Note: both used the seed 0 1.95 m/s model as shared initialization."
    )


if __name__ == "__main__":
    main()
