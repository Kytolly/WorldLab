"""Framework-neutral simulator outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar

from ._immutable import freeze_mapping


StateT = TypeVar("StateT")

# Reserved diagnostic keys. ``SimulationStep`` remains modality-neutral;
# these keys are retained for adapter and v0.2 compatibility.
SIMULATION_CHUNK_INDEX = "worldlab.simulation.chunk_index"
SIMULATION_MODEL_LATENCY_S = "worldlab.simulation.model_latency_s"
SIMULATION_FRAMES = "worldlab.simulation.frames"
SIMULATION_STATE = "worldlab.simulation.state"
SIMULATION_OUTPUT = "worldlab.simulation.output"


@dataclass(frozen=True)
class SimulationReset(Generic[StateT]):
    state: StateT
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))


@dataclass(frozen=True)
class SimulationStep(Generic[StateT]):
    """Simulator-facing state transition before Task interpretation.

    Simulator-specific inputs and auxiliary outputs belong in ``info``. The
    generic result only guarantees the new state and diagnostic metadata.
    Reward is intentionally absent: the Task produces the canonical reward.
    """

    state: StateT
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", freeze_mapping(self.info))
