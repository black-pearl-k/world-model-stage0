"""Plot evaluation curves for the three CartPole seeds.

In PyCharm: right-click this file -> Run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import PROJECT_DIR


def main() -> None:
    eval_files = sorted(
        (PROJECT_DIR / "logs" / "cartpole").glob("seed_*/evaluations.npz")
    )
    if not eval_files:
        print("No evaluations.npz found. Train CartPole first.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for path in eval_files:
        data = np.load(path)
        timesteps = data["timesteps"]
        results = data["results"]
        mean = results.mean(axis=1)
        if results.shape[1] > 1:
            std = results.std(axis=1, ddof=1)
        else:
            std = np.zeros_like(mean)

        label = Path(path).parent.name
        ax.plot(timesteps, mean, label=label)
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("environment steps")
    ax.set_ylabel("evaluation return")
    ax.set_title("PPO CartPole: 3 seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = PROJECT_DIR / "results" / "cartpole_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
