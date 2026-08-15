"""Values exchanged by environments, agents, and runtime components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class Transition(Generic[ObservationT, ActionT]):
    observation: ObservationT
    action: ActionT
    reward: float
    next_observation: ObservationT
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    policy_info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class EpisodeResult(Generic[ObservationT]):
    episode_index: int
    total_reward: float
    length: int
    terminated: bool
    truncated: bool
    final_observation: ObservationT
    final_info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
