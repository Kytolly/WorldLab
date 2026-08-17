"""Framework-neutral World Model inference contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, TypeVar

from worldlab.data import WorldModelContext, WorldModelStepResult


ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class WorldModel(ABC, Generic[ContextT, StateT, ActionT]):
    """Model-side contract for state or action-conditioned chunk inference."""

    @abstractmethod
    def initialize(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> WorldModelContext[ContextT, StateT]:
        raise NotImplementedError

    @abstractmethod
    def sample_step(
        self,
        context: ContextT,
        action: ActionT,
    ) -> WorldModelStepResult[ContextT, StateT]:
        raise NotImplementedError

    def render(self, context: ContextT, state: StateT) -> Any:
        return None

    def close(self) -> None:
        return None
