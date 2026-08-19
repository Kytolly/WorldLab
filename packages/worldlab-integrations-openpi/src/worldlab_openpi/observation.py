"""Compatibility exports for the shared GE-Sim observation contract."""

from __future__ import annotations

from gesim.policies.openpi import (
    DEFAULT_IMAGE_KEYS,
    DEFAULT_PROMPT_KEY,
    DEFAULT_STATE_KEY,
    build_openpi_payload,
)
from gesim.types import Observation


# Compatibility only. New code should import ``gesim.types.Observation`` so
# world-model output and policy input visibly share one contract.
OpenPIObservation = Observation


__all__ = [
    "DEFAULT_IMAGE_KEYS",
    "DEFAULT_PROMPT_KEY",
    "DEFAULT_STATE_KEY",
    "OpenPIObservation",
    "build_openpi_payload",
]
