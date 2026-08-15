"""The smallest runnable WorldLab end-to-end demo."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from worldlab.agents import PolicyAgent
from worldlab.core import Agent, Environment, Policy
from worldlab.data import EpisodeResult
from worldlab.envs import make_counter_environment, make_random_frame_environment
from worldlab.policies import ConstantPolicy, RandomPolicy
from worldlab.runtime import EnvironmentLoop, LoopConfig


def build_demo(
    *,
    model: str = "counter",
    goal: int = 3,
    frame_size: int = 8,
    max_episode_steps: Optional[int] = None,
    random_policy: bool = False,
) -> Tuple[Environment[Any, int], Agent[Any, int]]:
    """Create replaceable default Environment and Agent components."""

    if model == "counter":
        env: Environment[Any, int] = make_counter_environment(
            goal=goal,
            max_episode_steps=max_episode_steps,
        )
    elif model == "random-frame":
        env = make_random_frame_environment(
            frame_size=frame_size,
            max_episode_steps=max_episode_steps or 3,
        )
    else:
        raise ValueError(f"unknown demo model: {model}")
    policy: Policy[int, int]
    if random_policy:
        policy = RandomPolicy(env.action_space)
    else:
        policy = ConstantPolicy(1)
    return env, PolicyAgent(policy)


def run_demo(
    *,
    model: str = "counter",
    goal: int = 3,
    frame_size: int = 8,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = 0,
    random_policy: bool = False,
) -> EpisodeResult[Any]:
    env, agent = build_demo(
        model=model,
        goal=goal,
        frame_size=frame_size,
        max_episode_steps=max_episode_steps,
        random_policy=random_policy,
    )
    with EnvironmentLoop(
        env,
        agent,
        config=LoopConfig(training=False, deterministic=not random_policy),
    ) as loop:
        return loop.run_episode(seed=seed)
