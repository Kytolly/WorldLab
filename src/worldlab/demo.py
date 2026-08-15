"""The smallest runnable WorldLab end-to-end demo."""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

from worldlab.agents import PolicyAgent
from worldlab.core import Agent, Environment, Policy
from worldlab.data import EpisodeResult, Transition
from worldlab.envs import make_counter_environment, make_random_frame_environment
from worldlab.policies import ConstantPolicy, RandomPolicy
from worldlab.runtime import EnvironmentLoop, LoopCallback, LoopConfig, TraceSink


class _DemoStepDelay(LoopCallback[Any, int]):
    """Keep the built-in dashboard demo observable between transitions."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def on_step(
        self,
        episode_index: int,
        step_index: int,
        transition: Transition[Any, int],
    ) -> None:
        del episode_index, step_index, transition
        time.sleep(self.delay_s)


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
    trace: Optional[TraceSink] = None,
    step_delay: float = 0.0,
) -> EpisodeResult[Any]:
    if step_delay < 0.0:
        raise ValueError("step_delay must be non-negative")
    env, agent = build_demo(
        model=model,
        goal=goal,
        frame_size=frame_size,
        max_episode_steps=max_episode_steps,
        random_policy=random_policy,
    )
    callbacks = (_DemoStepDelay(step_delay),) if step_delay > 0.0 else ()
    with EnvironmentLoop(
        env,
        agent,
        config=LoopConfig(training=False, deterministic=not random_policy),
        callbacks=callbacks,
        trace=trace,
    ) as loop:
        return loop.run_episode(seed=seed)


def run_deterministic_demo(
    *,
    goal: int = 3,
    seed: int = 0,
    trace: Optional[TraceSink] = None,
) -> EpisodeResult[Any]:
    """Run the stable counter demo used by automated acceptance checks."""

    return run_demo(
        model="counter",
        goal=goal,
        seed=seed,
        random_policy=False,
        trace=trace,
    )
