from __future__ import annotations

from worldlab import build_demo, run_demo
from worldlab.runtime import EnvironmentLoop, LoopConfig


def test_builtin_demo_runs_without_user_defined_components() -> None:
    env, agent = build_demo(goal=3)

    with EnvironmentLoop(
        env,
        agent,
        config=LoopConfig(training=False, deterministic=True),
    ) as loop:
        result = loop.run_episode(seed=0)

    assert result.total_reward == 3.0
    assert result.length == 3
    assert result.terminated is True
    assert result.truncated is False


def test_builtin_demo_function_returns_episode_result() -> None:
    result = run_demo(goal=2, seed=0)

    assert result.total_reward == 2.0
    assert result.final_observation == 2


def test_random_frame_world_model_runs_through_the_same_loop() -> None:
    result = run_demo(
        model="random-frame",
        frame_size=4,
        max_episode_steps=2,
        seed=0,
    )

    assert len(result.final_observation) == 4
    assert result.total_reward == 2.0
    assert result.terminated is False
    assert result.truncated is True
