"""Framework-neutral simulator outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar


StateT = TypeVar("StateT")

# Reserved info keys shared by simulators, tasks, dashboards, and future
# transports.  The namespace avoids collisions with task-specific metadata.
SIMULATION_CHUNK_INDEX = "worldlab.simulation.chunk_index"
SIMULATION_MODEL_LATENCY_S = "worldlab.simulation.model_latency_s"
SIMULATION_FRAMES = "worldlab.simulation.frames"
SIMULATION_STATE = "worldlab.simulation.state"


@dataclass(frozen=True)
class SimulationReset(Generic[StateT]):
    state: StateT
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationStep(Generic[StateT]):
    """Simulator-facing transition before Task reward interpretation.

    ``action`` records the action consumed by the simulator for diagnostics;
    ``frames`` and ``state`` are model outputs. Reward is intentionally absent:
    the Task produces the canonical environment reward.
    """

    state: StateT
    info: Mapping[str, Any] = field(default_factory=dict)
    action: Any = None
    frames: Any = None
