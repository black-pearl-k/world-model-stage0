"""画 HalfCheetah 奖励塑形实验的曲线。

在 PyCharm 里右键 -> Run。

输入：
    logs/halfcheetah_gait/seed_0/evaluations.npz
    results/halfcheetah_gait_seed_0_snapshots.csv
    results/halfcheetah_gait_seed_0.json（可选）

输出：
    results/halfcheetah_gait_curve.png

左图是训练中的塑形奖励曲线，右图是各个快照节点的原始 HalfCheetah
回报。两者尺度不同，所以分开画，不要直接比较纵轴大小。
"""

from __future__ import annotations  # 让类型注解延迟求值

import csv  # 读取快照 CSV
import json  # 读取最终评测
from pathlib import Path  # 处理路径

import matplotlib  # 导入 matplotlib

matplotlib.use("Agg")  # 不弹窗，直接保存图片
import matplotlib.pyplot as plt  # 绘图接口
import numpy as np  # 读取 npz 和计算统计量

from train import PROJECT_DIR  # 项目目录


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 EvalCallback 的 evaluations.npz。"""
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]
    mean = results.mean(axis=1)
    if results.shape[1] > 1:
        std = results.std(axis=1, ddof=1)
    else:
        std = np.zeros_like(mean)
    return timesteps, mean, std


def load_snapshots(path: Path) -> tuple[list[int], list[float], list[float]]:
    """读取快照 CSV 里的步数和原始奖励。"""
    steps: list[int] = []
    means: list[float] = []
    stds: list[float] = []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            steps.append(int(row["step"]))
            means.append(float(row["original_mean"]))
            stds.append(float(row["original_std"]))

    return steps, means, stds


def main() -> None:
    curve_path = (
        PROJECT_DIR / "logs" / "halfcheetah_gait" / "seed_0" / "evaluations.npz"
    )
    snapshot_path = (
        PROJECT_DIR / "results" / "halfcheetah_gait_seed_0_snapshots.csv"
    )
    final_path = PROJECT_DIR / "results" / "halfcheetah_gait_seed_0.json"

    if not curve_path.exists() and not snapshot_path.exists():
        print(f"Missing: {curve_path}")
        print(f"Missing: {snapshot_path}")
        print("Run run_halfcheetah_gait.py first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # 左图：训练中的塑形奖励。这个纵轴不能和原始 HalfCheetah 分数直接比。
    ax = axes[0]
    if curve_path.exists():
        steps, mean, std = load_curve(curve_path)
        ax.plot(
            steps,
            mean,
            color="tab:blue",
            linewidth=2.0,
            marker="o",
            markersize=3,
            label="shaped eval",
        )
        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            color="tab:blue",
            alpha=0.15,
        )
        best_index = int(mean.argmax())
        ax.scatter(
            [steps[best_index]],
            [mean[best_index]],
            color="tab:red",
            s=55,
            zorder=5,
            label=(
                f"best shaped: {mean[best_index]:.1f} "
                f"@ {steps[best_index] / 1000:.0f}K"
            ),
        )
        ax.set_title("Shaped training reward")
        ax.legend(loc="lower right")
    else:
        ax.text(
            0.5,
            0.5,
            "evaluations.npz not found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Shaped training reward")

    ax.set_xlabel("environment steps")
    ax.set_ylabel("shaped evaluation return")
    ax.grid(alpha=0.3)

    # 右图：各快照节点的原始 HalfCheetah 回报，用来和 baseline 对比。
    ax = axes[1]
    if snapshot_path.exists():
        snap_steps, snap_mean, snap_std = load_snapshots(snapshot_path)
        ax.plot(
            snap_steps,
            snap_mean,
            color="tab:green",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="snapshot original return",
        )
        ax.fill_between(
            snap_steps,
            np.array(snap_mean) - np.array(snap_std),
            np.array(snap_mean) + np.array(snap_std),
            color="tab:green",
            alpha=0.15,
        )

    if final_path.exists():
        final_result = json.loads(final_path.read_text(encoding="utf-8"))
        ax.axhline(
            final_result["mean"],
            color="tab:red",
            linestyle=":",
            linewidth=1.6,
            label=(
                f"final test: {final_result['mean']:.1f} "
                f"± {final_result['std']:.1f}"
            ),
        )

    ax.set_xlabel("environment steps")
    ax.set_ylabel("original evaluation return")
    ax.set_title("Original HalfCheetah return at snapshots")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.suptitle("PPO HalfCheetah-v5: reward-shaped gait experiment")
    fig.tight_layout()

    out_path = PROJECT_DIR / "results" / "halfcheetah_gait_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
