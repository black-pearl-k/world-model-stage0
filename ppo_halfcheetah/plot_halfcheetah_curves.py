"""Plot evaluation curves for the three HalfCheetah-v5 seeds.

In PyCharm: right-click this file -> Run.
"""

from __future__ import annotations  # 让类型注解延迟求值

from pathlib import Path  # 处理文件路径

import matplotlib  # 导入 matplotlib

matplotlib.use("Agg")  # 使用不弹窗的后端，直接把图片保存到文件
import matplotlib.pyplot as plt  # 导入绘图接口
import numpy as np  # 读取 npz 和计算统计量

from train import PROJECT_DIR  # 导入项目目录


def main() -> None:  # 定义主函数
    eval_files = sorted(  # 找到所有 seed 的评测文件，并排序
        (PROJECT_DIR / "logs" / "halfcheetah").glob("seed_*/evaluations.npz")  # 匹配 seed_0/1/2 下的 evaluations.npz
    )
    if not eval_files:  # 如果一个文件都没有
        print("No evaluations.npz found. Train HalfCheetah first.")  # 提示先训练
        return  # 直接结束

    fig, ax = plt.subplots(figsize=(8, 5))  # 创建 8x5 英寸的画布
    for path in eval_files:  # 依次处理每个 seed 的评测文件
        data = np.load(path)  # 读取 npz 文件
        timesteps = data["timesteps"]  # 取出评测时的环境步数
        results = data["results"]  # 取出每次评测的回报矩阵
        mean = results.mean(axis=1)  # 对每次评测的多个 episode 求平均
        if results.shape[1] > 1:  # 如果每次评测不止一个 episode
            std = results.std(axis=1, ddof=1)  # 计算样本标准差
        else:  # 如果每次评测只有一个 episode
            std = np.zeros_like(mean)  # 标准差设为 0

        label = Path(path).parent.name  # 用 seed_0、seed_1 这样的目录名作为图例
        ax.plot(timesteps, mean, label=label)  # 画平均回报曲线
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.15)  # 画出均值上下一个标准差的阴影

    ax.set_xlabel("environment steps")  # 设置横轴标签
    ax.set_ylabel("evaluation return")  # 设置纵轴标签
    ax.set_title("PPO HalfCheetah-v5: 3 seeds")  # 设置标题
    ax.grid(alpha=0.3)  # 添加浅色网格
    ax.legend()  # 显示图例
    fig.tight_layout()  # 自动调整布局，避免标签被裁掉

    out_path = PROJECT_DIR / "results" / "halfcheetah_curves.png"  # 输出图片路径
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 创建 results 目录
    fig.savefig(out_path, dpi=160)  # 保存图片，分辨率 160 dpi
    plt.close(fig)  # 关闭画布，释放内存
    print(f"saved: {out_path}")  # 打印保存路径


if __name__ == "__main__":  # 只有直接运行这个文件时才执行
    main()  # 调用主函数
