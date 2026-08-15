"""Typed events emitted while an environment/agent loop is running."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Generic, Mapping, Optional, Tuple, TypeVar

from .interaction import EpisodeResult, Transition


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class RuntimeEventKind(str, Enum):
    EPISODE_STARTED = "episode_started"
    POLICY_ACTED = "policy_acted"
    ENVIRONMENT_STEPPED = "environment_stepped"
    TRANSITION_COMMITTED = "transition_committed"
    EPISODE_ENDED = "episode_ended"
    RUNTIME_ERROR = "runtime_error"


class RuntimeStatus(str, Enum):
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RuntimePhase(str, Enum):
    """The operation active when a runtime failure occurred."""

    ENVIRONMENT_RESET = "environment_reset"
    AGENT_RESET = "agent_reset"
    EPISODE_START_CALLBACK = "episode_start_callback"
    POLICY_ACT = "policy_act"
    ENVIRONMENT_STEP = "environment_step"
    TRANSITION_COMMIT = "transition_commit"
    AGENT_OBSERVE = "agent_observe"
    STEP_CALLBACK = "step_callback"
    RENDER = "render"
    AGENT_END_EPISODE = "agent_end_episode"
    EPISODE_END_CALLBACK = "episode_end_callback"


@dataclass(frozen=True)
class RuntimeEvent:
    """Common ordering and timing fields shared by every runtime event."""

    sequence: int
    timestamp_s: float
    monotonic_s: float
    episode_index: int
    step_index: int
    duration_s: float

    kind: ClassVar[RuntimeEventKind]

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("sequence must be greater than zero")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")


@dataclass(frozen=True)
class EpisodeStarted(RuntimeEvent, Generic[ObservationT]):
    kind: ClassVar[RuntimeEventKind] = RuntimeEventKind.EPISODE_STARTED

    seed: Optional[int]
    observation: ObservationT
    info: Mapping[str, Any]


@dataclass(frozen=True)
class PolicyActed(RuntimeEvent, Generic[ObservationT, ActionT]):
    kind: ClassVar[RuntimeEventKind] = RuntimeEventKind.POLICY_ACTED

    observation: ObservationT
    action: ActionT
    policy_info: Mapping[str, Any]
    training: bool
    deterministic: bool


@dataclass(frozen=True)
class EnvironmentStepped(RuntimeEvent, Generic[ObservationT, ActionT]):
    kind: ClassVar[RuntimeEventKind] = RuntimeEventKind.ENVIRONMENT_STEPPED

    action: ActionT
    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


@dataclass(frozen=True)
class TransitionCommitted(RuntimeEvent, Generic[ObservationT, ActionT]):
    kind: ClassVar[RuntimeEventKind] = RuntimeEventKind.TRANSITION_COMMITTED

    transition: Transition[ObservationT, ActionT]
    total_reward: float
    training: bool


@dataclass(frozen=True)
class EpisodeEnded(RuntimeEvent, Generic[ObservationT]):
    kind: ClassVar[RuntimeEventKind] = RuntimeEventKind.EPISODE_ENDED

    result: EpisodeResult[ObservationT]


@dataclass(frozen=True)
class RuntimeErrorEvent(RuntimeEvent):
    kind: ClassVar[RuntimeEventKind] = RuntimeEventKind.RUNTIME_ERROR

    phase: RuntimePhase
    error_type: str
    message: str
    traceback: str


@dataclass(frozen=True)
class TraceDiagnosis:
    """Result of validating a recorded closed-loop event sequence."""

    healthy: bool
    messages: Tuple[str, ...]


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Current observable state derived from the most recent runtime event."""

    status: RuntimeStatus
    sequence: int
    timestamp_s: float
    monotonic_s: float
    episode_index: Optional[int] = None
    step_index: int = 0
    total_reward: float = 0.0
    reward: Optional[float] = None
    terminated: bool = False
    truncated: bool = False
    last_event: Optional[RuntimeEventKind] = None
    phase: Optional[RuntimePhase] = None
    observation: Any = None
    action: Any = None
    next_observation: Any = None
    info: Mapping[str, Any] = field(default_factory=dict)
    policy_info: Mapping[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
