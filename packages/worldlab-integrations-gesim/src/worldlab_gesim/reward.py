"""Conversion from GE-Sim Judge scores to the core reward contract."""

from __future__ import annotations

from typing import Literal

from worldlab.data import RewardResult

from .base import RewardResult as GESimRewardResult


class GESimRewardAdapter:
    """Convert per-frame GE-Sim scores into one chunk-level scalar reward."""

    def __init__(self, *, policy: Literal["last"] = "last") -> None:
        if policy != "last":
            raise ValueError("the only supported GESim reward policy is 'last'")
        self.policy = policy

    def adapt(self, result: GESimRewardResult) -> RewardResult:
        if not isinstance(result, GESimRewardResult):
            raise TypeError("result must be the author-style GE-Sim RewardResult")
        score = float(result.success[-1])
        info: dict[str, object] = {
            "gesim.judge.score_policy": self.policy,
            "gesim.judge.last_frame_score": score,
            "gesim.judge.per_frame_scores": result.success.copy(),
        }
        if result.progress is not None:
            info["gesim.judge.progress"] = result.progress.copy()
        return RewardResult(value=score, info=info)
