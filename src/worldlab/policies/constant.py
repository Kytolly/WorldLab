"""A deterministic fixed-action policy for smoke tests and demos."""

from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar

from worldlab.core import Policy
from worldlab.data import PolicyOutput


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class ConstantPolicy(Policy[ObservationT, ActionT], Generic[ObservationT, ActionT]):
    def __init__(self, action: ActionT) -> None:
        self.action = action

    def act(
        self,
        observation: ObservationT,
        *,
        info: Mapping[str, Any],
        deterministic: bool = False,
    ) -> PolicyOutput[ActionT]:
        del observation, info, deterministic
        return PolicyOutput(self.action)
