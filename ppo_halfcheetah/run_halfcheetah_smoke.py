"""HalfCheetah-v5 smoke test.

In PyCharm: right-click this file -> Run.
"""

from __future__ import annotations

from train import PROJECT_DIR, train

ENV_ID = "HalfCheetah-v5"
SEED = 0
STEPS = 20_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 3


def main() -> None:
    train(
        env_id=ENV_ID,
        seed=SEED,
        steps=STEPS,
        logdir=PROJECT_DIR / "logs" / "halfcheetah" / "smoke_seed_0",
        eval_freq=EVAL_FREQ,
        eval_episodes=EVAL_EPISODES,
    )


if __name__ == "__main__":
    main()
