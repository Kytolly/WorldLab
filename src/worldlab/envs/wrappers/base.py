"""Base wrapper that preserves the WorldLab Environment interface."""

from __future__ import annotations

from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.core import Environment
from worldlab.data import ResetResult, StepResult


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class EnvironmentWrapper(Environment[ObservationT, ActionT], Generic[ObservationT, ActionT]):
    def __init__(self, env: Environment[ObservationT, ActionT]) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> ResetResult[ObservationT]:
        return self.env.reset(seed=seed, options=options)

    def step(self, action: ActionT) -> StepResult[ObservationT]:
        return self.env.step(action)

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def unwrapped(self) -> Environment[ObservationT, ActionT]:
        return self.env.unwrapped
