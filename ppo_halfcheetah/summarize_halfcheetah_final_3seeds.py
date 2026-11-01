"""汇总最终 2.0 m/s 三个 seed 的评测结果并画总结图。

在 PyCharm 里右键这个文件 -> Run。

输入：
    results/halfcheetah_final_seed_0.json
    results/halfcheetah_final_seed_1.json
    results/halfcheetah_final_seed_2.json
    results/halfcheetah_final_seed_0_snapshots.csv
    results/halfcheetah_final_seed_1_snapshots.csv
    results/halfcheetah_final_seed_2_snapshots.csv
    logs/halfcheetah_final/seed_0/evaluations.npz
    logs/halfcheetah_final/seed_1/evaluations.npz
    logs/halfcheetah_final/seed_2/evaluations.npz

输出：
    results/halfcheetah_final_3seeds_summary.csv
    results/halfcheetah_final_3seeds_summary.json
    results/halfcheetah_final_3seeds_summary.png
"""

from __future__ import annotations  # 让类型注解延迟求值

import csv  # 保存汇总 CSV
import json  # 保存汇总 JSON
from pathlib import Path  # 处理路径

import matplotlib  # 导入 matplotlib

matplotlib.use("Agg")  # 不弹窗，直接保存图片
import matplotlib.pyplot as plt  # 绘图接口
import numpy as np  # 统计和读取 npz

from train import PROJECT_DIR  # 项目目录

SEEDS = (0, 1, 2)
TARGET_SPEED = 2.0


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 EvalCallback 的评测曲线，返回步数和每次 5 局的均值。"""
    data = np.load(path)
    return data["timesteps"], data["results"].mean(axis=1)


def load_snapshots(path: Path) -> dict[str, list[float]]:
    """读取快照 CSV 中的关键指标。"""
    fields = {
        "step": [],
        "speed_mean": [],
        "mean_x_velocity": [],
        "mean_upright": [],
        "torso_contact_rate": [],
        "head_contact_rate": [],
    }
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            for key in fields:
                fields[key].append(float(row[key]))
    return fields


def sample_std(values: list[float]) -> float:
    """样本标准差。"""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    results_dir = PROJECT_DIR / "results"
    logs_dir = PROJECT_DIR / "logs" / "halfcheetah_final"
    results_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    snapshots: dict[int, dict[str, list[float]]] = {}
    all_episodes: dict[int, list[float]] = {}

    for seed in SEEDS:
        json_path = results_dir / f"halfcheetah_final_seed_{seed}.json"
        snap_path = results_dir / f"halfcheetah_final_seed_{seed}_snapshots.csv"
        curve_path = logs_dir / f"seed_{seed}" / "evaluations.npz"

        if not json_path.exists() or not snap_path.exists():
            raise FileNotFoundError(
                f"Missing result files for seed {seed}: {json_path} / {snap_path}"
            )

        final = json.loads(json_path.read_text(encoding="utf-8"))
        episodes = [float(x) for x in final["episodes"]]
        all_episodes[seed] = episodes
        snap = load_snapshots(snap_path)
        snapshots[seed] = snap

        if curve_path.exists():
            curves[seed] = load_curve(curve_path)

        # 取最后一个快照，作为 300K 最终模型的姿态统计
        record = {
            "seed": seed,
            "final_mean": round(float(final["mean"]), 2),
            "final_std": round(float(final["std"]), 2),
            "final_median": round(float(np.median(episodes)), 2),
            "final_min": round(float(np.min(episodes)), 2),
            "final_max": round(float(np.max(episodes)), 2),
            "negative_episodes": int(np.sum(np.array(episodes) < 0)),
            "snapshot_speed_mean": round(float(snap["speed_mean"][-1]), 2),
            "snapshot_x_velocity": round(float(snap["mean_x_velocity"][-1]), 4),
            "snapshot_upright": round(float(snap["mean_upright"][-1]), 4),
            "torso_contact_rate": round(float(snap["torso_contact_rate"][-1]), 4),
            "head_contact_rate": round(float(snap["head_contact_rate"][-1]), 4),
        }
        records.append(record)

    # 汇总三个 seed 的 seed-level 均值
    aggregate = {
        "final_mean": round(float(np.mean([r["final_mean"] for r in records])), 2),
        "final_std_across_seeds": round(
            sample_std([r["final_mean"] for r in records]), 2
        ),
        "x_velocity_mean": round(
            float(np.mean([r["snapshot_x_velocity"] for r in records])), 4
        ),
        "upright_mean": round(
            float(np.mean([r["snapshot_upright"] for r in records])), 4
        ),
        "head_contact_mean": round(
            float(np.mean([r["head_contact_rate"] for r in records])), 4
        ),
        "torso_contact_mean": round(
            float(np.mean([r["torso_contact_rate"] for r in records])), 4
        ),
    }

    summary = {
        "experiment": "HalfCheetah-v5 final 2.0 m/s",
        "target_speed": TARGET_SPEED,
        "protocol": {
            "final_eval_episodes": 10,
            "final_eval_seed": "1000-1009",
            "model_selection": "final_model",
            "shared_initialization": (
                "seed 1/2 used seed 0 native 1.95 m/s model as shared initialization"
            ),
        },
        "seeds": records,
        "aggregate": aggregate,
        "note": (
            "final_mean/final_std include all 10 deterministic test episodes; "
            "negative_episodes counts episodes with return below 0."
        ),
    }

    json_path = results_dir / "halfcheetah_final_3seeds_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = results_dir / "halfcheetah_final_3seeds_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # 画图：2x2 总结图
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 左上：3 个 seed 的训练评测曲线
    ax = axes[0, 0]
    for seed in SEEDS:
        if seed in curves:
            steps, mean = curves[seed]
            ax.plot(steps, mean, linewidth=1.4, alpha=0.8, label=f"seed {seed}")
    if curves:
        common_steps = curves[SEEDS[0]][0]
        matrix = np.stack([curves[s][1] for s in SEEDS if s in curves])
        ax.plot(
            common_steps,
            matrix.mean(axis=0),
            color="black",
            linewidth=2.2,
            label="seed mean",
        )
    ax.set_title("Training curves: speed-shaped eval")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("eval return")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    # 右上：快照 x velocity
    ax = axes[0, 1]
    for seed in SEEDS:
        snap = snapshots[seed]
        ax.plot(
            snap["step"],
            snap["mean_x_velocity"],
            marker="o",
            linewidth=1.6,
            label=f"seed {seed}",
        )
    ax.axhline(TARGET_SPEED, color="gray", linestyle="--", label="target 2.0")
    ax.set_title("Snapshot mean x velocity")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("m/s")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    # 左下：最终 10 局回报的分布
    ax = axes[1, 0]
    positions = np.arange(len(SEEDS)) + 1
    for pos, seed in zip(positions, SEEDS):
        episodes = np.array(all_episodes[seed])
        jitter = np.linspace(-0.08, 0.08, len(episodes))
        ax.scatter(
            np.full(len(episodes), pos) + jitter,
            episodes,
            s=18,
            alpha=0.75,
        )
    ax.boxplot(
        [all_episodes[s] for s in SEEDS],
        positions=positions,
        widths=0.45,
        showfliers=False,
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_title("Final test: 10 episodes per seed")
    ax.set_ylabel("original return")
    ax.grid(alpha=0.3, axis="y")

    # 右下：最终快照的关键指标
    ax = axes[1, 1]
    x = np.arange(len(SEEDS))
    x_velocity = [r["snapshot_x_velocity"] for r in records]
    upright = [r["snapshot_upright"] for r in records]
    ax.bar(x - 0.18, x_velocity, width=0.36, label="x velocity")
    ax.bar(x + 0.18, upright, width=0.36, label="upright")
    ax.axhline(TARGET_SPEED, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylim(0, 2.35)
    ax.set_title("Final snapshot: speed and posture")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right")

    fig.suptitle("HalfCheetah-v5 final 2.0 m/s: 3-seed summary")
    fig.tight_layout()
    png_path = results_dir / "halfcheetah_final_3seeds_summary.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    # 控制台表格
    print("=" * 90)
    print("seed  final_mean  final_std  median   min     max   neg  x_vel  upright  head")
    print("-" * 90)
    for r in records:
        print(
            f"{r['seed']:>3}  {r['final_mean']:>10.2f}  {r['final_std']:>9.2f}  "
            f"{r['final_median']:>7.2f}  {r['final_min']:>7.2f}  "
            f"{r['final_max']:>7.2f}  {r['negative_episodes']:>3}  "
            f"{r['snapshot_x_velocity']:>5.3f}  {r['snapshot_upright']:>7.3f}  "
            f"{r['head_contact_rate']:>5.3f}"
        )
    print("-" * 90)
    print(
        f"3-seed mean return: {aggregate['final_mean']:.2f} "
        f"± {aggregate['final_std_across_seeds']:.2f} "
        f"(std across seed-level means)"
    )
    print(f"3-seed mean x velocity: {aggregate['x_velocity_mean']:.3f} m/s")
    print(f"3-seed mean upright: {aggregate['upright_mean']:.3f}")
    print(f"3-seed mean head contact: {aggregate['head_contact_mean']:.3f}")
    print("=" * 90)
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
