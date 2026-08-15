"""Compatibility export for future custom model implementations."""

from worldlab.world_models.base import WorldModel
from worldlab.data import WorldModelStepResult

__all__ = ["WorldModel", "WorldModelStepResult"]
