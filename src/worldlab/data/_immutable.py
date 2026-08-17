"""Small helpers for immutable public data objects."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a read-only copy of ``value``."""

    return MappingProxyType(dict(value or {}))
