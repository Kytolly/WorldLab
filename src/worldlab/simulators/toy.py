"""A deterministic toy world model used by ``python -m worldlab``."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from worldlab.data import SimulationReset, SimulationStep

from .world_model import WorldModel


class CounterWorldModel(WorldModel[int, int]):
    """Advance an integer state by the selected integer action."""

    def initial_state(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> SimulationReset[int]:
        del seed
        return SimulationReset(int((options or {}).get("start", 0)))

    def predict(self, state: int, action: int) -> SimulationStep[int]:
        return SimulationStep(state + action)
