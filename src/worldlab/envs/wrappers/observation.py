"""Observation-transforming environment wrapper."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.core import Environment, Space
from worldlab.data import ResetResult, StepResult


InnerObservationT = TypeVar("InnerObservationT")
OuterObservationT = TypeVar("OuterObservationT")
ActionT = TypeVar("ActionT")


class ObservationWrapper(Environment[OuterObservationT, ActionT], Generic[InnerObservationT, OuterObservationT, ActionT]):
    def __init__(
        self,
        env: Environment[InnerObservationT, ActionT],
        *,
        observation_space: Space[OuterObservationT],
    ) -> None:
        self.env = env
        self.observation_space = observation_space
        self.action_space = env.action_space

    @abstractmethod
    def observation(self, observation: InnerObservationT) -> OuterObservationT:
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, object]] = None) -> ResetResult[OuterObservationT]:
        result = self.env.reset(seed=seed, options=options)
        return ResetResult(self.observation(result.observation), result.info)

    def step(self, action: ActionT) -> StepResult[OuterObservationT]:
        result = self.env.step(action)
        return StepResult(
            observation=self.observation(result.observation),
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
