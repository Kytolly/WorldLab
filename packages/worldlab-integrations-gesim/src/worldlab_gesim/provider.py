"""WorldLab reward provider backed by a GE-Sim reward client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from worldlab.core import RewardProvider
from worldlab.data import (
    EvaluatedStepContext,
    RewardResult,
    SIMULATION_FRAMES,
    TaskResetContext,
)

from .base import RewardResult as GESimRewardResult
from .base import RewardClient
from .data import head_view_frames
from .reward import GESimRewardAdapter


TaskResolver = Callable[[EvaluatedStepContext[Any, Any, Any]], str]


GESIM_TASK_INFO = "gesim.task"


class GESimRewardProvider(
    RewardProvider[
        TaskResetContext[Any],
        EvaluatedStepContext[Any, Any, Any],
    ]
):
    """Adapt a GE-Sim ``RewardClient`` into a WorldLab ``RewardProvider``.

    The provider does not hold a ``GESimTask`` instance. By default it reads
    the task string from simulator metadata; a resolver can be supplied for a
    dynamic subtask protocol.
    """

    def __init__(
        self,
        *,
        reward_client: RewardClient,
        task_resolver: TaskResolver | None = None,
        adapter: GESimRewardAdapter | None = None,
    ) -> None:
        if not callable(getattr(reward_client, "evaluate", None)):
            raise TypeError("reward_client must implement evaluate(head_frames, task)")
        self.reward_client = reward_client
        self.task_resolver = task_resolver or _task_from_context
        self.adapter = adapter or GESimRewardAdapter()

    def compute(self, context: EvaluatedStepContext[Any, Any, Any]) -> RewardResult:
        frames = context.step.simulation.info.get(SIMULATION_FRAMES)
        if frames is None:
            raise KeyError(
                f"simulation info is missing GE-Sim frames key {SIMULATION_FRAMES!r}"
            )
        result: GESimRewardResult = self.reward_client.evaluate(
            head_view_frames(frames),
            self.task_resolver(context),
        )
        return self.adapter.adapt(result)


class GESimRewardClientProvider(GESimRewardProvider):
    """Backward-compatible name for :class:`GESimRewardProvider`."""


def _task_from_context(context: EvaluatedStepContext[Any, Any, Any]) -> str:
    task = context.step.simulation.info.get(GESIM_TASK_INFO)
    if not isinstance(task, str) or not task.strip():
        raise KeyError(
            f"simulation info is missing a non-empty {GESIM_TASK_INFO!r} task string"
        )
    return task
