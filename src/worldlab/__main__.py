"""Command-line entry point for the built-in WorldLab demo."""

from __future__ import annotations

import argparse

from .demo import run_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the built-in WorldLab demo")
    parser.add_argument("--model", choices=("counter", "random-frame"), default="counter")
    parser.add_argument("--goal", type=int, default=3)
    parser.add_argument("--frame-size", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-policy", action="store_true")
    args = parser.parse_args()

    result = run_demo(
        model=args.model,
        goal=args.goal,
        frame_size=args.frame_size,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        random_policy=args.random_policy,
    )
    print("WorldLab demo")
    print(f"total_reward={result.total_reward}")
    print(f"length={result.length}")
    print(f"terminated={result.terminated}")
    print(f"truncated={result.truncated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
