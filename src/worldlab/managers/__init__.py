"""Composable Task signal managers and typed terms."""

from .observation import ObservationGroups, ObservationManager
from .reward import RewardManager
from .termination import TerminationManager
from .terms import (
    ManagerTermError,
    ObservationTerm,
    ObservationTermSpec,
    RewardTerm,
    RewardTermSpec,
    TerminationTerm,
    TerminationTermSpec,
)

__all__ = [
    "ManagerTermError",
    "ObservationManager",
    "ObservationGroups",
    "ObservationTerm",
    "ObservationTermSpec",
    "RewardManager",
    "RewardTerm",
    "RewardTermSpec",
    "TerminationManager",
    "TerminationTerm",
    "TerminationTermSpec",
]
