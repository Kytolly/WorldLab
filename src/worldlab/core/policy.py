"""Policy inference contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.data import PolicyOutput


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class Policy(ABC, Generic[ObservationT, ActionT]):
    def reset(self, *, seed: Optional[int] = None) -> None:
        return None

    @abstractmethod
    def act(
        self,
        observation: ObservationT,
        *,
        info: Mapping[str, Any],
        deterministic: bool = False,
    ) -> PolicyOutput[ActionT]:
        raise NotImplementedError
