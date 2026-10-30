"""Train CartPole with three independent seeds.

In PyCharm: right-click this file -> Run.
No terminal arguments are needed.
"""

from __future__ import annotations

from pathlib import Path

from train import PROJECT_DIR, train

ENV_ID = "CartPole-v1"
SEEDS = (0, 1, 2)
STEPS = 50_000
EVAL_FREQ = 5_000
EVAL_EPISODES = 10


def main() -> None:
    for seed in SEEDS:
        logdir = PROJECT_DIR / "logs" / "cartpole" / f"seed_{seed}"
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

    print("\nAll CartPole seeds finished.")
    print(f"Models and logs are under: {Path(PROJECT_DIR) / 'logs' / 'cartpole'}")


if __name__ == "__main__":
    main()
