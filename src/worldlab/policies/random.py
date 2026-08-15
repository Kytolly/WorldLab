"""Random action policy."""

from __future__ import annotations

from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.core import Policy, Space
from worldlab.data import PolicyOutput


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class RandomPolicy(Policy[ObservationT, ActionT], Generic[ObservationT, ActionT]):
    def __init__(self, action_space: Space[ActionT]) -> None:
        self.action_space = action_space

    def reset(self, *, seed: Optional[int] = None) -> None:
        self.action_space.seed(seed)

    def act(
        self,
        observation: ObservationT,
        *,
        info: Mapping[str, Any],
        deterministic: bool = False,
    ) -> PolicyOutput[ActionT]:
        del observation, info, deterministic
        return PolicyOutput(self.action_space.sample())
