"""Plot the 300K and 1M evaluation curves for HalfCheetah-v5 seed 0.

In PyCharm: right-click this file -> Run.

The curve comes from EvalCallback's evaluations.npz:
  logs/halfcheetah_1m/seed_0/evaluations.npz

Output:
  results/halfcheetah_1m_curve.png
"""

from __future__ import annotations  # 让类型注解延迟求值

import json  # 读取最终评测结果
from pathlib import Path  # 处理文件路径

import matplotlib  # 导入 matplotlib

matplotlib.use("Agg")  # 不弹窗，直接把图片保存到文件
import matplotlib.pyplot as plt  # 导入绘图接口
import numpy as np  # 读取 npz 和计算统计量

from train import PROJECT_DIR  # 导入项目目录


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # 读取一条评测曲线
    data = np.load(path)  # 读取 evaluations.npz
    timesteps = data["timesteps"]  # 每次评测对应的环境步数
    results = data["results"]  # 形状是 [评测次数, 每次评测局数]
    mean = results.mean(axis=1)  # 对每次评测的多个 episode 求平均
    if results.shape[1] > 1:  # 如果每次评测不止一局
        std = results.std(axis=1, ddof=1)  # 计算样本标准差
    else:  # 如果每次评测只有一局
        std = np.zeros_like(mean)  # 标准差设为 0
    return timesteps, mean, std  # 返回横轴、均值、标准差


def main() -> None:  # 定义主函数
    one_m_path = (  # 1M 实验的评测文件
        PROJECT_DIR / "logs" / "halfcheetah_1m" / "seed_0" / "evaluations.npz"
    )
    base_path = (  # 300K baseline 的 seed 0 评测文件
        PROJECT_DIR / "logs" / "halfcheetah" / "seed_0" / "evaluations.npz"
    )
    final_path = PROJECT_DIR / "results" / "halfcheetah_1m_seed_0.json"  # 1M 最终评测 JSON

    if not one_m_path.exists():  # 如果 1M 实验还没训练
        print(f"Missing: {one_m_path}")  # 打印缺失路径
        print("Run run_halfcheetah_1m.py first.")  # 提示先训练
        return  # 直接结束

    fig, ax = plt.subplots(figsize=(9, 5.5))  # 创建 9x5.5 英寸画布

    if base_path.exists():  # 如果 300K baseline 曲线存在
        base_steps, base_mean, base_std = load_curve(base_path)  # 读取 baseline 曲线
        ax.plot(  # 画 300K 曲线
            base_steps,  # 横轴
            base_mean,  # 纵轴均值
            color="gray",  # 灰色
            linestyle="--",  # 虚线
            linewidth=1.5,  # 线宽
            label="300K run (seed 0)",  # 图例
        )
        ax.fill_between(  # 画 300K 的均值±标准差
            base_steps,  # 横轴
            base_mean - base_std,  # 下边界
            base_mean + base_std,  # 上边界
            color="gray",  # 灰色
            alpha=0.10,  # 透明度
        )

    steps, mean, std = load_curve(one_m_path)  # 读取 1M 实验曲线
    ax.plot(  # 画 1M 曲线
        steps,  # 横轴
        mean,  # 纵轴均值
        color="tab:blue",  # 蓝色
        linewidth=2.0,  # 线宽
        marker="o",  # 每个评测点画小圆点
        markersize=3,  # 圆点大小
        label="1M run (seed 0)",  # 图例
    )
    ax.fill_between(  # 画 1M 的均值±标准差
        steps,  # 横轴
        mean - std,  # 下边界
        mean + std,  # 上边界
        color="tab:blue",  # 蓝色
        alpha=0.15,  # 透明度
    )

    best_index = int(mean.argmax())  # 找到 1M 验证曲线的最高点
    best_step = int(steps[best_index])  # 最高点对应的步数
    best_mean = float(mean[best_index])  # 最高点的平均回报
    ax.scatter(  # 标出最高点
        [best_step],  # 横坐标
        [best_mean],  # 纵坐标
        color="tab:red",  # 红色
        s=55,  # 点大小
        zorder=5,  # 画在最上层
        label=f"best eval: {best_mean:.1f} @ {best_step / 1000:.0f}K",  # 图例
    )

    if final_path.exists():  # 如果最终评测结果存在
        final_result = json.loads(final_path.read_text(encoding="utf-8"))  # 读取 JSON
        ax.axhline(  # 画一条水平参考线
            final_result["mean"],  # 最终 10 局平均回报
            color="tab:red",  # 红色
            linestyle=":",  # 点线
            linewidth=1.5,  # 线宽
            alpha=0.8,  # 透明度
            label=(  # 图例
                f"final test: {final_result['mean']:.1f} "
                f"± {final_result['std']:.1f}"
            ),
        )

    ax.set_xlabel("environment steps")  # 横轴标签
    ax.set_ylabel("evaluation return")  # 纵轴标签
    ax.set_title("PPO HalfCheetah-v5 seed 0: 300K vs 1M")  # 标题
    ax.grid(alpha=0.3)  # 浅色网格
    ax.legend(loc="lower right")  # 显示图例
    fig.tight_layout()  # 自动调整布局

    out_path = PROJECT_DIR / "results" / "halfcheetah_1m_curve.png"  # 输出图片路径
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 创建 results 目录
    fig.savefig(out_path, dpi=160)  # 保存图片
    plt.close(fig)  # 关闭画布，释放内存

    print(f"1M best eval: {best_mean:.2f} at {best_step:,} steps")  # 打印最高验证点
    print(f"1M last eval: {mean[-1]:.2f} at {int(steps[-1]):,} steps")  # 打印最后一次验证
    if final_path.exists():  # 如果最终评测存在
        print(  # 打印最终评测
            f"1M final test: {final_result['mean']:.2f} "
            f"± {final_result['std']:.2f}"
        )
    print(f"saved: {out_path}")  # 打印图片保存路径


if __name__ == "__main__":  # 只有直接运行这个文件时才执行
    main()  # 调用主函数
