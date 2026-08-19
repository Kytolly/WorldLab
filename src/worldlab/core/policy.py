"""Policy inference contract."""

from __future__ import annotations

from typing import Any, Generic, Mapping, Optional, Protocol, TypeVar

from worldlab.data import PolicyOutput


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class Policy(Protocol, Generic[ObservationT, ActionT]):
    def reset(self, *, seed: Optional[int] = None) -> None:
        return None

    def infer(
        self,
        observation: ObservationT,
        *,
        info: Mapping[str, Any],
        deterministic: bool = False,
    ) -> PolicyOutput[ActionT]:
        raise NotImplementedError