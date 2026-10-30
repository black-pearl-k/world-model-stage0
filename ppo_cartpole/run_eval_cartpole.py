"""Evaluate the three CartPole seeds and create a summary table.

In PyCharm: right-click this file -> Run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from eval import evaluate_model
from train import PROJECT_DIR

ENV_ID = "CartPole-v1"
SEEDS = (0, 1, 2)
EPISODES = 10
EVAL_SEED = 1000


def main() -> None:
    results_dir = PROJECT_DIR / "results"
    videos_dir = PROJECT_DIR / "videos"
    results_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for seed in SEEDS:
        model_path = PROJECT_DIR / "logs" / "cartpole" / f"seed_{seed}" / "final_model.zip"
        if not model_path.exists():
            print(f"[skip] missing model: {model_path}")
            print("       run run_cartpole_3seeds.py first.")
            continue

        print(f"\nEvaluating seed {seed} ...")
        video_path = videos_dir / f"cartpole_seed_{seed}.mp4" if seed == 0 else None
        screenshot_path = videos_dir / f"cartpole_seed_{seed}.png" if seed == 0 else None

        try:
            result = evaluate_model(
                model_path=model_path,
                env_id=ENV_ID,
                episodes=EPISODES,
                eval_seed=EVAL_SEED,
                video_path=video_path,
                screenshot_path=screenshot_path,
            )
        except Exception as exc:
            print(f"[warn] video/screenshot failed: {exc}")
            print("       retrying evaluation without rendering ...")
            result = evaluate_model(
                model_path=model_path,
                env_id=ENV_ID,
                episodes=EPISODES,
                eval_seed=EVAL_SEED,
            )

        result["train_seed"] = seed
        all_results.append(result)

        out_path = results_dir / f"eval_cartpole_seed_{seed}.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"seed {seed}: mean={result['mean']:.2f}, "
            f"std={result['std']:.2f}, "
            f"episodes={result['episodes']}"
        )

    if not all_results:
        print("\nNo models were evaluated.")
        return

    seed_means = [float(r["mean"]) for r in all_results]
    summary = {
        "env": ENV_ID,
        "seeds": [int(r["train_seed"]) for r in all_results],
        "seed_means": seed_means,
        "mean": float(np.mean(seed_means)),
        "std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
        "episodes_per_seed": EPISODES,
        "eval_seed_start": EVAL_SEED,
    }

    (results_dir / "cartpole_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (results_dir / "cartpole_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["train_seed", "eval_mean", "eval_std", "episodes"])
        for result in all_results:
            writer.writerow(
                [
                    result["train_seed"],
                    f"{result['mean']:.3f}",
                    f"{result['std']:.3f}",
                    EPISODES,
                ]
            )

    print("\n" + "=" * 60)
    print(f"Env: {ENV_ID}")
    print(f"Seeds: {summary['seeds']}")
    print(f"Seed means: {[round(x, 2) for x in seed_means]}")
    print(f"Mean ± Std: {summary['mean']:.2f} ± {summary['std']:.2f}")
    print(f"Saved: {results_dir / 'cartpole_summary.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
