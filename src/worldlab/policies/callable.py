"""Policy backed by a regular Python callable."""

from __future__ import annotations

from typing import Any, Callable, Generic, Mapping, TypeVar

from worldlab.core import Policy
from worldlab.data import PolicyOutput


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class CallablePolicy(Policy[ObservationT, ActionT], Generic[ObservationT, ActionT]):
    def __init__(
        self,
        function: Callable[[ObservationT, Mapping[str, Any], bool], ActionT],
    ) -> None:
        self.function = function

    def act(
        self,
        observation: ObservationT,
        *,
        info: Mapping[str, Any],
        deterministic: bool = False,
    ) -> PolicyOutput[ActionT]:
        return PolicyOutput(self.function(observation, info, deterministic))
