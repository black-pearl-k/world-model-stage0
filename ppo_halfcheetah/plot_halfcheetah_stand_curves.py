"""画“只站立”实验的曲线和姿态指标。

在 PyCharm 里右键 -> Run。

输入：
    logs/halfcheetah_stand/seed_0/evaluations.npz
    results/halfcheetah_stand_seed_0_snapshots.csv

输出：
    results/halfcheetah_stand_curve.png
"""

from __future__ import annotations  # 让类型注解延迟求值

import csv  # 读取快照 CSV
from pathlib import Path  # 处理路径

import matplotlib  # 导入 matplotlib

matplotlib.use("Agg")  # 不弹窗，直接保存图片
import matplotlib.pyplot as plt  # 绘图接口
import numpy as np  # 读取 npz 和计算统计量

from train import PROJECT_DIR  # 项目目录


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 standing 阶段的 evaluations.npz。"""
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]
    mean = results.mean(axis=1)
    if results.shape[1] > 1:
        std = results.std(axis=1, ddof=1)
    else:
        std = np.zeros_like(mean)
    return timesteps, mean, std


def load_snapshots(path: Path) -> dict[str, list[float]]:
    """读取快照 CSV 中的姿态和接触指标。"""
    columns = {
        "step": [],
        "mean_upright": [],
        "torso_contact_rate": [],
        "head_contact_rate": [],
        "mean_x_velocity": [],
    }

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            for name in columns:
                columns[name].append(float(row[name]))

    return columns


def main() -> None:
    root = PROJECT_DIR
    curve_path = (
        root / "logs" / "halfcheetah_stand" / "seed_0" / "evaluations.npz"
    )
    snapshot_path = (
        root / "results" / "halfcheetah_stand_seed_0_snapshots.csv"
    )

    if not curve_path.exists() and not snapshot_path.exists():
        print(f"Missing: {curve_path}")
        print(f"Missing: {snapshot_path}")
        print("Run run_halfcheetah_stand.py first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 左上：站立奖励。
    ax = axes[0, 0]
    if curve_path.exists():
        steps, mean, std = load_curve(curve_path)
        ax.plot(steps, mean, color="tab:blue", linewidth=2, label="stand eval")
        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            color="tab:blue",
            alpha=0.15,
        )
        ax.legend(loc="lower right")
    ax.set_title("Stand reward")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("stand evaluation return")
    ax.grid(alpha=0.3)

    if snapshot_path.exists():
        snapshots = load_snapshots(snapshot_path)
        steps = snapshots["step"]

        # 右上：身体水平程度。
        ax = axes[0, 1]
        ax.plot(
            steps,
            snapshots["mean_upright"],
            color="tab:green",
            marker="o",
            linewidth=2,
        )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title("Mean cos(rooty): 1 = horizontal")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("mean upright")
        ax.grid(alpha=0.3)

        # 左下：躯干/头碰地比例。
        ax = axes[1, 0]
        ax.plot(
            steps,
            snapshots["torso_contact_rate"],
            color="tab:orange",
            marker="o",
            linewidth=2,
            label="torso contact",
        )
        ax.plot(
            steps,
            snapshots["head_contact_rate"],
            color="tab:red",
            marker="o",
            linewidth=2,
            label="head contact",
        )
        ax.set_ylim(bottom=0)
        ax.set_title("Body / head contact rate")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("fraction of steps")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

        # 右下：前后速度，应该接近 0。
        ax = axes[1, 1]
        ax.plot(
            steps,
            snapshots["mean_x_velocity"],
            color="tab:purple",
            marker="o",
            linewidth=2,
        )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title("Mean x velocity: should be near 0")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("mean x velocity")
        ax.grid(alpha=0.3)
    else:
        for ax in axes[0, 1:]:
            ax.text(
                0.5,
                0.5,
                "snapshot CSV not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        for ax in axes[1, :]:
            ax.text(
                0.5,
                0.5,
                "snapshot CSV not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    fig.suptitle("PPO HalfCheetah-v5: stand-only stage")
    fig.tight_layout()

    out_path = root / "results" / "halfcheetah_stand_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
