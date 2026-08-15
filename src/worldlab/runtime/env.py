"""The minimal WorldLab environment/agent interaction loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Generic, Iterable, List, Mapping, Optional, Tuple, TypeVar

from worldlab.core import Agent, Environment
from worldlab.data import EpisodeResult, Transition

from .callbacks import LoopCallback


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class LoopConfig:
    training: bool = True
    deterministic: bool = False
    render: bool = False
    validate_spaces: bool = True
    safety_max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.safety_max_steps is not None and self.safety_max_steps <= 0:
            raise ValueError("safety_max_steps must be greater than zero")


class EnvironmentLoop(Generic[ObservationT, ActionT]):
    def __init__(
        self,
        env: Environment[ObservationT, ActionT],
        agent: Agent[ObservationT, ActionT],
        *,
        config: Optional[LoopConfig] = None,
        callbacks: Iterable[LoopCallback[ObservationT, ActionT]] = (),
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config or LoopConfig()
        self.callbacks: Tuple[LoopCallback[ObservationT, ActionT], ...] = tuple(callbacks)

    def run_episode(
        self,
        *,
        episode_index: int = 0,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> EpisodeResult[ObservationT]:
        reset = self.env.reset(seed=seed, options=options)
        self._validate_observation(reset.observation)
        self.agent.reset(seed=seed)

        observation = reset.observation
        info = dict(reset.info)
        total_reward = 0.0
        step_index = 0

        for callback in self.callbacks:
            callback.on_episode_start(episode_index, observation, info)
        self._render_if_requested()

        while True:
            policy_output = self.agent.act(
                observation,
                info=info,
                training=self.config.training,
                deterministic=self.config.deterministic,
            )
            self._validate_action(policy_output.action)
            step = self.env.step(policy_output.action)
            self._validate_observation(step.observation)
            step_index += 1

            if (
                self.config.safety_max_steps is not None
                and step_index >= self.config.safety_max_steps
                and not step.done
            ):
                step_info = dict(step.info)
                step_info["worldlab.runtime.safety_limit_reached"] = True
                step = replace(step, truncated=True, info=step_info)

            transition = Transition(
                observation=observation,
                action=policy_output.action,
                reward=float(step.reward),
                next_observation=step.observation,
                terminated=step.terminated,
                truncated=step.truncated,
                info=dict(step.info),
                policy_info=dict(policy_output.info),
            )

            if self.config.training:
                self.agent.observe(transition)
            for callback in self.callbacks:
                callback.on_step(episode_index, step_index, transition)

            total_reward += transition.reward
            observation = transition.next_observation
            info = dict(transition.info)
            self._render_if_requested()

            if transition.done:
                result = EpisodeResult(
                    episode_index=episode_index,
                    total_reward=total_reward,
                    length=step_index,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    final_observation=observation,
                    final_info=info,
                )
                self.agent.end_episode(result, training=self.config.training)
                for callback in self.callbacks:
                    callback.on_episode_end(result)
                return result

    def run(
        self,
        episodes: int,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> List[EpisodeResult[ObservationT]]:
        if episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        return [
            self.run_episode(
                episode_index=index,
                seed=None if seed is None else seed + index,
                options=options,
            )
            for index in range(episodes)
        ]

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "EnvironmentLoop[ObservationT, ActionT]":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _validate_observation(self, observation: ObservationT) -> None:
        if self.config.validate_spaces and not self.env.observation_space.contains(observation):
            raise ValueError("environment produced an observation outside observation_space")

    def _validate_action(self, action: ActionT) -> None:
        if self.config.validate_spaces and not self.env.action_space.contains(action):
            raise ValueError("agent produced an action outside action_space")

    def _render_if_requested(self) -> None:
        if self.config.render:
            self.env.render()
