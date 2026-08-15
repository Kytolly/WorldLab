"""Episode time-limit wrapper."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional, TypeVar

from worldlab.core import Environment
from worldlab.data import ResetResult, StepResult

from .base import EnvironmentWrapper


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class TimeLimitWrapper(EnvironmentWrapper[ObservationT, ActionT]):
    def __init__(
        self,
        env: Environment[ObservationT, ActionT],
        max_episode_steps: int,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be greater than zero")
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self._episode_done = True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> ResetResult[ObservationT]:
        result = self.env.reset(seed=seed, options=options)
        self.elapsed_steps = 0
        self._episode_done = False
        return result

    def step(self, action: ActionT) -> StepResult[ObservationT]:
        if self._episode_done:
            raise RuntimeError("reset() must be called before step()")

        result = self.env.step(action)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps and not result.done:
            info = dict(result.info)
            info["worldlab.time_limit_reached"] = True
            result = replace(result, truncated=True, info=info)

        self._episode_done = result.done
        return result
