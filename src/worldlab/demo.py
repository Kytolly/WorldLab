"""The smallest runnable WorldLab end-to-end demo."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from worldlab.agents import PolicyAgent
from worldlab.core import Agent, Environment, Policy
from worldlab.config import config_to_dict
from worldlab.data import EpisodeResult, Transition
from worldlab.envs import make_counter_environment, make_example_environment
from worldlab.policies import ConstantPolicy, RandomPolicy
from worldlab.runtime import EnvironmentLoop, LoopCallback, LoopConfig, TraceSink
from worldlab.world_models import ExampleWorldModel


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


class _ConfiguredStepDelay(LoopCallback[Any, Any]):
    """Keep a configured chunk run observable between transitions."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def on_step(
        self,
        episode_index: int,
        step_index: int,
        transition: Transition[Any, Any],
    ) -> None:
        del episode_index, step_index, transition
        time.sleep(self.delay_s)


def build_demo(
    *,
    model: str = "counter",
    goal: int = 3,
    max_episode_steps: Optional[int] = None,
    random_policy: bool = False,
) -> Tuple[Environment[Any, int], Agent[Any, int]]:
    """Create replaceable default Environment and Agent components."""

    if model == "counter":
        env: Environment[Any, int] = make_counter_environment(
            goal=goal,
            max_episode_steps=max_episode_steps,
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


def build_configured_demo(
    config: Mapping[str, Any],
) -> tuple[Environment[Any, Any], Agent[Any, Any], Mapping[str, Any]]:
    """Build the v0.2.1 ExampleWorldModel graph from resolved config."""

    values = config_to_dict(config)
    rollout = values["rollout"]
    world_model_config = values["world_model"]
    inference = world_model_config["inference"]
    model = ExampleWorldModel.from_config(
        world_model_config,
        chunk_size=int(rollout["chunk_size"]),
    )
    task_config = values["environment"]["task"]
    environment = make_example_environment(
        model,
        goal=int(task_config["goal"]),
        max_episode_steps=task_config.get("max_episode_steps"),
    )
    action_value = float(values["policy"]["inference"]["action_value"])
    action = np.full(
        (model.chunk_size, int(inference["action_dim"])),
        action_value,
        dtype=np.float32,
    )
    trajectory_length = task_config.get("max_episode_steps") or model.chunk_size
    options = _synthetic_options(model, int(trajectory_length))
    return environment, PolicyAgent(ConstantPolicy(action)), options


def run_configured_demo(
    config: Mapping[str, Any],
    *,
    trace: Optional[TraceSink] = None,
) -> EpisodeResult[Any]:
    """Run one or more episodes using a resolved v0.2.1 configuration."""

    values = config_to_dict(config)
    if values["run"]["mode"] not in {"demo", "rollout"}:
        raise NotImplementedError(
            f"run.mode={values['run']['mode']!r} is reserved for a future trainer"
        )
    for component in ("world_model", "policy"):
        mode = values["training"][component]["mode"]
        if mode != "disabled":
            raise NotImplementedError(
                f"training.{component}.mode={mode!r} is reserved for a future trainer"
            )
    environment, agent, options = build_configured_demo(values)
    runtime = values["runtime"]
    callbacks: tuple[LoopCallback[Any, Any], ...] = ()
    if float(runtime["step_delay_s"]) > 0.0:
        callbacks = (_ConfiguredStepDelay(float(runtime["step_delay_s"])),)
    loop_config = LoopConfig(
        training=bool(runtime["training"]),
        deterministic=bool(runtime["deterministic"]),
        render=bool(runtime["render"]),
        validate_spaces=bool(runtime["validate_spaces"]),
        safety_max_steps=runtime.get("safety_max_steps"),
    )
    result: Optional[EpisodeResult[Any]] = None
    try:
        with EnvironmentLoop(
            environment,
            agent,
            config=loop_config,
            callbacks=callbacks,
            trace=trace,
        ) as loop:
            for episode_index in range(int(values["run"]["episodes"])):
                result = loop.run_episode(
                    episode_index=episode_index,
                    seed=int(values["run"]["seed"]) + episode_index,
                    options=options,
                )
    finally:
        environment.close()
    assert result is not None
    return result


def _synthetic_options(model: ExampleWorldModel, trajectory_length: int) -> Mapping[str, Any]:
    """Create deterministic local fixture arrays without external data coupling."""

    _, views, height, width = model.frame_shape
    length = max(model.chunk_size, trajectory_length)
    eye3 = np.repeat(np.eye(3, dtype=np.float32)[None], views, axis=0)
    eye4 = np.repeat(np.eye(4, dtype=np.float32)[None], views, axis=0)
    c2w = np.repeat(
        np.eye(4, dtype=np.float32)[None, None],
        views * length,
        axis=0,
    ).reshape(views, length, 4, 4)
    return {
        "intrinsic": eye3,
        "extrinsic": eye4,
        "first_frame": np.zeros(model.frame_shape, dtype=np.float32),
        "trajectory": np.zeros((3, views, length, height, width), dtype=np.float32),
        "c2w": c2w,
        "task": "synthetic example",
    }
