"""Deterministic, update-free policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from worldlab.core import Agent, Environment
from worldlab.data import EpisodeResult

from .env import EnvironmentLoop, LoopConfig


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class EvaluationResult(Generic[ObservationT]):
    episodes: List[EpisodeResult[ObservationT]]

    @property
    def mean_reward(self) -> float:
        return sum(item.total_reward for item in self.episodes) / len(self.episodes)

    @property
    def mean_length(self) -> float:
        return sum(item.length for item in self.episodes) / len(self.episodes)


class Evaluator(Generic[ObservationT, ActionT]):
    def __init__(
        self,
        env: Environment[ObservationT, ActionT],
        agent: Agent[ObservationT, ActionT],
        *,
        validate_spaces: bool = True,
        safety_max_steps: Optional[int] = None,
    ) -> None:
        self.loop = EnvironmentLoop(
            env,
            agent,
            config=LoopConfig(
                training=False,
                deterministic=True,
                validate_spaces=validate_spaces,
                safety_max_steps=safety_max_steps,
            ),
        )

    def evaluate(self, episodes: int, *, seed: Optional[int] = None) -> EvaluationResult[ObservationT]:
        return EvaluationResult(self.loop.run(episodes, seed=seed))
