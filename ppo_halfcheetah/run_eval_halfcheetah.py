"""Evaluate the three HalfCheetah-v5 seeds and create a summary table.

In PyCharm: right-click this file -> Run.
"""

from __future__ import annotations  # 让类型注解延迟求值

import csv  # 写 CSV 结果表
import json  # 写 JSON 结果文件
from pathlib import Path  # 处理文件路径

import numpy as np  # 计算均值和标准差

from eval import evaluate_model  # 导入评测函数
from train import PROJECT_DIR  # 导入项目目录

ENV_ID = "HalfCheetah-v5"  # 使用 HalfCheetah-v5 环境
SEEDS = (0, 1, 2)  # 评测三个训练 seed
EPISODES = 10  # 每个模型评测 10 局
EVAL_SEED = 1000  # 最终评测从 seed 1000 开始
MAX_VIDEO_STEPS = 300  # 视频最多录 300 帧


def main() -> None:  # 定义主函数
    results_dir = PROJECT_DIR / "results"  # 结果目录
    videos_dir = PROJECT_DIR / "videos"  # 视频目录
    results_dir.mkdir(parents=True, exist_ok=True)  # 创建结果目录
    videos_dir.mkdir(parents=True, exist_ok=True)  # 创建视频目录

    all_results: list[dict] = []  # 保存三个 seed 的评测结果

    for seed in SEEDS:  # 依次评测每个 seed
        model_path = (  # 拼出当前 seed 的最终模型路径
            PROJECT_DIR / "logs" / "halfcheetah" / f"seed_{seed}" / "final_model.zip"
        )
        if not model_path.exists():  # 如果模型不存在
            print(f"[skip] missing model: {model_path}")  # 打印跳过信息
            print("       run run_halfcheetah_3seeds.py first.")  # 提示先训练
            continue  # 跳过当前 seed

        print(f"\nEvaluating seed {seed} ...")  # 打印当前评测 seed
        video_path = videos_dir / f"halfcheetah_seed_{seed}.mp4" if seed == 0 else None  # 只给 seed 0 录视频
        screenshot_path = (  # 只给 seed 0 存截图
            videos_dir / f"halfcheetah_seed_{seed}.png" if seed == 0 else None
        )

        try:  # 先尝试带渲染评测
            result = evaluate_model(  # 调用评测函数
                model_path=model_path,  # 传入模型路径
                env_id=ENV_ID,  # 传入环境名
                episodes=EPISODES,  # 传入评测局数
                eval_seed=EVAL_SEED,  # 传入评测起始 seed
                video_path=video_path,  # 传入视频路径
                screenshot_path=screenshot_path,  # 传入截图路径
                max_video_steps=MAX_VIDEO_STEPS,  # 限制视频帧数
            )
        except Exception as exc:  # 如果渲染失败
            print(f"[warn] video/screenshot failed: {exc}")  # 打印警告
            print("       retrying evaluation without rendering ...")  # 提示重新无渲染评测
            result = evaluate_model(  # 无渲染再评测一次
                model_path=model_path,  # 传入模型路径
                env_id=ENV_ID,  # 传入环境名
                episodes=EPISODES,  # 传入评测局数
                eval_seed=EVAL_SEED,  # 传入评测起始 seed
            )

        result["train_seed"] = seed  # 把训练 seed 写进结果
        all_results.append(result)  # 保存当前结果

        out_path = results_dir / f"eval_halfcheetah_seed_{seed}.json"  # 当前 seed 的 JSON 路径
        out_path.write_text(  # 写入 JSON 文件
            json.dumps(result, indent=2, ensure_ascii=False),  # 转成带缩进的 JSON
            encoding="utf-8",  # 使用 UTF-8 编码
        )
        print(  # 打印当前 seed 的评测结果
            f"seed {seed}: mean={result['mean']:.2f}, "  # 平均回报
            f"std={result['std']:.2f}, "  # 标准差
            f"episodes={result['episodes']}"  # 每一局的回报列表
        )

    if not all_results:  # 如果一个模型都没评测
        print("\nNo models were evaluated.")  # 打印提示
        return  # 直接结束

    seed_means = [float(r["mean"]) for r in all_results]  # 取出每个 seed 的平均回报
    summary = {  # 汇总结果
        "env": ENV_ID,  # 环境名
        "seeds": [int(r["train_seed"]) for r in all_results],  # 训练 seed 列表
        "seed_means": seed_means,  # 每个 seed 的平均回报
        "mean": float(np.mean(seed_means)),  # 三个 seed 平均值的平均
        "std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,  # 三个 seed 平均值的标准差
        "episodes_per_seed": EPISODES,  # 每个 seed 评测局数
        "eval_seed_start": EVAL_SEED,  # 评测起始 seed
    }

    (results_dir / "halfcheetah_summary.json").write_text(  # 写入汇总 JSON
        json.dumps(summary, indent=2, ensure_ascii=False),  # 转成带缩进的 JSON
        encoding="utf-8",  # 使用 UTF-8 编码
    )

    with (results_dir / "halfcheetah_results.csv").open(  # 打开 CSV 文件
        "w", newline="", encoding="utf-8"  # 写模式，避免多余空行，使用 UTF-8
    ) as f:  # 文件对象命名为 f
        writer = csv.writer(f)  # 创建 CSV 写入器
        writer.writerow(["train_seed", "eval_mean", "eval_std", "episodes"])  # 写表头
        for result in all_results:  # 遍历每个 seed 的结果
            writer.writerow(  # 写一行
                [
                    result["train_seed"],  # 训练 seed
                    f"{result['mean']:.3f}",  # 平均回报，保留三位小数
                    f"{result['std']:.3f}",  # 标准差，保留三位小数
                    EPISODES,  # 评测局数
                ]
            )

    print("\n" + "=" * 60)  # 打印分隔线
    print(f"Env: {ENV_ID}")  # 打印环境名
    print(f"Seeds: {summary['seeds']}")  # 打印 seed 列表
    print(f"Seed means: {[round(x, 2) for x in seed_means]}")  # 打印每个 seed 的平均回报
    print(f"Mean ± Std: {summary['mean']:.2f} ± {summary['std']:.2f}")  # 打印总体均值和标准差
    print(f"Saved: {results_dir / 'halfcheetah_summary.json'}")  # 打印汇总文件路径
    print("=" * 60)  # 再打印一条分隔线


if __name__ == "__main__":  # 只有直接运行这个文件时才执行
    main()  # 调用主函数
