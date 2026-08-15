"""A deterministic toy world model used by ``python -m worldlab``."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from worldlab.core import WorldModel
from worldlab.data import WorldModelContext, WorldModelPrediction


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

    def sample_step(self, context: int, action: int) -> WorldModelPrediction[int, int]:
        state = context + action
        return WorldModelPrediction(context=state, state=state)
