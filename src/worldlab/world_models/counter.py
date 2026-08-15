"""Deterministic counter World Model used by the smoke demo."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from worldlab.data import WorldModelContext, WorldModelStepResult

from .base import WorldModel


class CounterWorldModel(WorldModel[int, int, int]):
    """Advance an integer state by the selected integer action."""

    def initialize(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> WorldModelContext[int, int]:
        del seed
        state = int((options or {}).get("start", 0))
        return WorldModelContext(context=state, state=state)

    def sample_step(self, context: int, action: int) -> WorldModelStepResult[int, int]:
        state = context + action
        return WorldModelStepResult(context=state, state=state)
