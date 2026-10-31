"""Train HalfCheetah-v5 with three independent seeds.

In PyCharm: right-click this file -> Run.
No terminal arguments are needed.
"""

from __future__ import annotations

from pathlib import Path

from train import PROJECT_DIR, train

ENV_ID = "HalfCheetah-v5"
SEEDS = (0, 1, 2)
STEPS = 300_000
EVAL_FREQ = 25_000
EVAL_EPISODES = 5


def main() -> None:
    for seed in SEEDS:
        logdir = PROJECT_DIR / "logs" / "halfcheetah" / f"seed_{seed}"
        print("\n" + "#" * 70)
        print(f"Training {ENV_ID}, seed={seed}, steps={STEPS}")
        print("#" * 70 + "\n")
        train(
            env_id=ENV_ID,
            seed=seed,
            steps=STEPS,
            logdir=logdir,
            eval_freq=EVAL_FREQ,
            eval_episodes=EVAL_EPISODES,
        )

    print("\nAll HalfCheetah seeds finished.")
    print(f"Models and logs are under: {Path(PROJECT_DIR) / 'logs' / 'halfcheetah'}")


if __name__ == "__main__":
    main()
