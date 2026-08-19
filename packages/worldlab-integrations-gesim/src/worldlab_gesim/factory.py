"""Composition helpers for constructing a GE-Sim closed loop.

Factories create instances and connect framework boundaries. They contain no
episode stepping logic and no network protocol implementation.
"""

from __future__ import annotations

from typing import Any

from worldlab.agents import PolicyAgent
from worldlab.core import Policy, RewardProvider, Simulator, Space
from worldlab.runtime import EnvironmentLoop, LoopConfig

from .base import RewardClient
from .reward import GESimRewardAdapter
from .provider import GESimRewardProvider
from .task import GESimTask
from gesim.types import Observation
from .worldlab_env import GESimSimulatorAdapter, GESimWorldModelBackend, GESimWorldModelEnv


def make_gesim_task(
    *,
    instruction: str,
    observation: Any,
    termination: Any,
    reward: RewardProvider[Any, Any] | None = None,
    reward_client: RewardClient | None = None,
    reward_adapter: GESimRewardAdapter | None = None,
) -> GESimTask:
    """Create a task from already-constructed signal components."""

    if reward is not None and reward_client is not None:
        raise ValueError("pass either reward or reward_client, not both")
    if reward_client is not None:
        reward = GESimRewardProvider(
            reward_client=reward_client,
            adapter=reward_adapter,
        )
    return GESimTask(
        instruction=instruction,
        observation=observation,
        termination=termination,
        reward=reward,
    )


def make_gesim_environment(
    *,
    backend: GESimWorldModelBackend | None = None,
    simulator: Simulator[Observation, Any] | None = None,
    task: GESimTask,
    observation_space: Space[Any],
    action_space: Space[Any],
    chunk_size: int | None = None,
    compress_actions: bool = True,
) -> GESimWorldModelEnv[Any]:
    """Create an environment from either a backend or an existing simulator."""

    if (backend is None) == (simulator is None):
        raise ValueError("pass exactly one of backend or simulator")
    if simulator is None:
        simulator = GESimSimulatorAdapter(
            backend,
            chunk_size=chunk_size,
            compress_actions=compress_actions,
        )
    return GESimWorldModelEnv(
        simulator,
        task,
        observation_space=observation_space,
        action_space=action_space,
    )


def make_gesim_loop(
    *,
    env: GESimWorldModelEnv[Any],
    policy: Policy[Any, Any],
    config: LoopConfig | None = None,
) -> EnvironmentLoop[Any, Any]:
    """Create the Agent and runtime loop after the environment is assembled."""

    return EnvironmentLoop(env, PolicyAgent(policy), config=config)
