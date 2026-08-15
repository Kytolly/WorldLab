"""Action-transforming environment wrapper."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.core import Environment, Space
from worldlab.data import ResetResult, StepResult


ObservationT = TypeVar("ObservationT")
InnerActionT = TypeVar("InnerActionT")
OuterActionT = TypeVar("OuterActionT")


class ActionWrapper(Environment[ObservationT, OuterActionT], Generic[ObservationT, InnerActionT, OuterActionT]):
    def __init__(
        self,
        env: Environment[ObservationT, InnerActionT],
        *,
        action_space: Space[OuterActionT],
    ) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = action_space

    @abstractmethod
    def action(self, action: OuterActionT) -> InnerActionT:
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, object]] = None) -> ResetResult[ObservationT]:
        return self.env.reset(seed=seed, options=options)

    def step(self, action: OuterActionT) -> StepResult[ObservationT]:
        return self.env.step(self.action(action))

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
